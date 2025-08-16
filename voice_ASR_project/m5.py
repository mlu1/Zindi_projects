#!/usr/bin/env python3
"""
Fine-tune wav2vec 2.0 XLS-R on Swahili speech

• Merges Common Voice 13.0, MLS, and FLEURS into one Dataset
• Normalises text (lower-case, punctuation stripping)
• Optional SpecAugment + noise injection
• Trainer with early stopping and WER monitoring
• Outputs: ./<run_name>/  (model, processor, training logs)

Dependencies  (tested Aug-2025):
  pip install "torch>=2.2.1" "torchaudio>=2.2.1" \
              transformers==4.43.0 datasets==2.20.0 \
              jiwer==3.0.4 accelerate soundfile sentencepiece
"""

# ───────────────────────── CLI & CONFIG ────────────────────────── #
import argparse, re, os, random, numpy as np, torch, torchaudio, soundfile as sf
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Union

parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", default="facebook/wav2vec2-xls-r-300m",
                    help="Base checkpoint to fine-tune")
parser.add_argument("--output_dir", default="wav2vec2-sw-finetuned",
                    help="Folder to save model + processor")
parser.add_argument("--run_name", default="xlsr-sw",
                    help="WANDB / log name (also sub-folder under output_dir)")
parser.add_argument("--train_batch_size", type=int, default=6,
                    help="Per-device batch size")
parser.add_argument("--gradient_accumulation", type=int, default=2,
                    help="Steps to accumulate grads -> effective BS")
parser.add_argument("--num_epochs", type=int, default=10)
parser.add_argument("--learning_rate", type=float, default=3e-5)
parser.add_argument("--eval_steps", type=int, default=1000)
parser.add_argument("--save_steps", type=int, default=1000)
parser.add_argument("--warmup_steps", type=int, default=500)
parser.add_argument("--specaugment", action="store_true",
                    help="Enable SpecAugment + additive noise")
args = parser.parse_args()

# ensure determinism-ish
torch.manual_seed(7); np.random.seed(7); random.seed(7)

# ───────────────────────── Datasets ────────────────────────────── #
from datasets import load_dataset, concatenate_datasets, Audio

def load_split(builder, subset="train"):
    """Return HF dataset with 16 kHz audio column"""
    ds = load_dataset(builder, "sw", split=subset, use_auth_token=True) \
            if builder in ("facebook/multilingual_librispeech",) \
            else load_dataset(builder, split=subset)
    return ds.cast_column("audio", Audio(sampling_rate=16_000))

cv      = load_split("mozilla-foundation/common_voice_13_0")
mls_tr  = load_split("facebook/multilingual_librispeech", "train+validation")
fleurs  = load_split("google/fleurs", "train")

train_ds = concatenate_datasets([cv, mls_tr, fleurs])

# tiny dev set for early-stop (5 000 random examples)
dev_ds   = train_ds.shuffle(seed=42).select(range(5_000))

print(f"🔉  Total training hours: {train_ds.audio.num_frames.sum()/16_000/3600:.1f} h")
print(f"🔉  Dev set hours      : {dev_ds.audio.num_frames.sum()/16_000/3600:.1f} h")

# ──────────────── Text normalisation & augmentation ────────────── #
chars_keep = "abcdefghijklmnopqrstuvwxyz' "
def normalise(batch):
    txt = batch["text"].lower()
    txt = re.sub(r"[^a-z' ]+", " ", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    batch["text"] = txt
    return batch

train_ds = train_ds.map(normalise, num_proc=4)
dev_ds   = dev_ds.map(normalise,   num_proc=2)

if args.specaugment:
    def augment(batch):
        wav = torch.tensor(batch["audio"]["array"])
        wav = torchaudio.transforms.FrequencyMasking(15)(wav)
        wav = torchaudio.transforms.TimeMasking(50)(wav)
        noise = 0.01 * torch.randn_like(wav)
        batch["audio"]["array"] = (wav + noise).numpy()
        return batch
    train_ds = train_ds.map(augment, num_proc=4)

# ───────────────────── Processor & collator ────────────────────── #
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained(
    args.model_name_or_path,
    do_lower_case=True,
    remove_space=False     # keep separator token '|'
)

@dataclass
class DataCollatorCTCWithPadding:
    processor: AutoProcessor
    padding: Union[bool, str] = True
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]):
        audio = [f["audio"]["array"] for f in features]
        batch = self.processor(audio, sampling_rate=16_000,
                               padding=self.padding, return_tensors="pt")
        with self.processor.as_target_processor():
            labels_batch = self.processor(
                [f["text"] for f in features], padding=True, return_tensors="pt")
        batch["labels"] = labels_batch["input_ids"]
        return batch

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

# ───────────────────── Model & training args ───────────────────── #
from transformers import (AutoModelForCTC, TrainingArguments,
                          Trainer, EarlyStoppingCallback)
import evaluate, numpy as np

wer_metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids  = np.argmax(pred.predictions, axis=-1)
    pred_str  = processor.batch_decode(pred_ids)
    # restore pads to compare
    label_ids = np.where(pred.label_ids != -100,
                         pred.label_ids,
                         processor.tokenizer.pad_token_id)
    label_str = processor.batch_decode(label_ids, group_tokens=False)
    return {"wer": wer_metric.compute(predictions=pred_str,
                                      references=label_str)}

model = AutoModelForCTC.from_pretrained(
    args.model_name_or_path,
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
)

training_args = TrainingArguments(
    output_dir           = Path(args.output_dir) / args.run_name,
    run_name             = args.run_name,
    per_device_train_batch_size = args.train_batch_size,
    gradient_accumulation_steps = args.gradient_accumulation,
    evaluation_strategy  = "steps",
    learning_rate        = args.learning_rate,
    warmup_steps         = args.warmup_steps,
    num_train_epochs     = args.num_epochs,
    save_steps           = args.save_steps,
    eval_steps           = args.eval_steps,
    save_total_limit     = 3,
    logging_steps        = 100,
    fp16                 = torch.cuda.is_available(),
    gradient_checkpointing = True,
    dataloader_num_workers = 4,
)

trainer = Trainer(
    model           = model,
    args            = training_args,
    train_dataset   = train_ds,
    eval_dataset    = dev_ds,
    tokenizer       = processor.feature_extractor,
    data_collator   = data_collator,
    compute_metrics = compute_metrics,
    callbacks       = [EarlyStoppingCallback(early_stopping_patience=3)],
)

# ─────────────────────────── TRAIN! ─────────────────────────────── #
trainer.train()

# ───────────────────────── Save artefacts ───────────────────────── #
save_dir = Path(args.output_dir) / args.run_name
trainer.save_model(save_dir)
processor.save_pretrained(save_dir)
print(f"✅  Model + processor saved to {save_dir.resolve()}")


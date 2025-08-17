import re
import regex
import pandas as pd
import numpy as np
import torch

import os
from accelerate import PartialState

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
state = PartialState()
print(state)

from datasets import load_dataset, Features, Value, Audio, DownloadConfig, load_from_disk
from huggingface_hub import login
from transformers import AutoProcessor
import warnings
warnings.filterwarnings('ignore')
import datasets
datasets.logging.set_verbosity_info()

class Config:
  dataset_path = "mozilla-foundation/common_voice_17_0"
  target_language = "sw"
  parent_path = "/tmp2/mluleki/Thesis/voices/Zindi_projects/voice_ASR_project"
  train_dataset_path = os.path.join(parent_path, "processed_hf_train")
  validation_dataset_path = os.path.join(parent_path, "processed_hf_valid")
  processed_train_dataset_path = os.path.join(parent_path, "processed_hf_train_with_labels")
  processed_validation_dataset_path = os.path.join(parent_path, "processed_hf_valid_with_labels")
  model_name = "alamsher/wav2vec2-large-xlsr-53-common-voice-sw"
  batch_size = 16
  num_proc = 8

CFG = Config()


cv_sw_features = Features({
  "client_id": Value("string"),
  "path": Value("string"),
  "sentence_id": Value("string"),
  "sentence": Value("string"),
  "sentence_domain":Value("string"),
  "up_votes": Value("string"), # <- string, not int64
  "down_votes": Value("string"), # <- string, not int64
  "age": Value("string"),
  "gender": Value("string"),
  "variant": Value("string"),
  "locale": Value("string"),
  "segment": Value("string"),
  "accent": Value("string"),
  # keep audio decoded so we get "array" + "sampling_rate"
  "audio":Audio(sampling_rate=16_000, mono=True, decode=True),
})

if not os.path.exists(CFG.train_dataset_path):
  train_dataset = load_dataset(
      CFG.dataset_path,
      CFG.target_language,
      split="train",
      features=cv_sw_features,
      trust_remote_code=True
  )
  train_dataset.save_to_disk(CFG.train_dataset_path)
else:
  train_dataset = load_from_disk(CFG.train_dataset_path)




if not os.path.exists(CFG.validation_dataset_path):
  validation_dataset = load_dataset(
      CFG.dataset_path,
      CFG.target_language,
      split="validation",
      features=cv_sw_features,
      trust_remote_code=True
  )
  validation_dataset.save_to_disk(CFG.validation_dataset_path)
else:
  validation_dataset = load_from_disk(CFG.validation_dataset_path)


print(train_dataset['sentence'][:5])

pattern = regex.compile(r'\p{P}+', re.UNICODE)

def remove_special_characters(batch, col='sentence'):
  batch[col] = [
      re.sub(r'\s+', ' ', pattern.sub(' ', sentence.lower())).strip()
      for sentence in batch[col]
  ]
  return batch

train_dataset = train_dataset.map(remove_special_characters, batched=True, batch_size=1000)
validation_dataset = validation_dataset.map(remove_special_characters, batched=True, batch_size=1000)



def prepare_batch(batch, processor, target_col='sentence'):
    # 2. Extract speech arrays & sample rate
    batch['speech'] = [audio['array'] for audio in batch['audio']]
    batch['sampling_rate'] = [audio['sampling_rate'] for audio in batch['audio']]

    # 3. Process audio into model inputs
    inputs = processor(batch['speech'], sampling_rate=16000, return_tensors="np", padding=True)
    batch['input_values'] = inputs.input_values
    batch['input_length'] = [len(x) for x in batch['input_values']]

    # 4. Tokenize target text
    with processor.as_target_processor():
        batch['labels'] = processor(batch[target_col]).input_ids

    return batch


processor = AutoProcessor.from_pretrained(CFG.model_name)



if not os.path.exists(CFG.processed_train_dataset_path):
  train_dataset = train_dataset.map(
      lambda b: prepare_batch(b, processor),
      batched=True,
      batch_size = CFG.batch_size,
      num_proc = CFG.num_proc,
      remove_columns = train_dataset.column_names
  )
  train_dataset.save_to_disk(CFG.processed_train_dataset_path)
else:
  train_dataset = load_from_disk(CFG.processed_train_dataset_path)



SAMPLE_SIZE = 10000
SEED = 42

train_dataset  = train_dataset.shuffle(seed=SEED).select(range(SAMPLE_SIZE))
train_dataset


if not os.path.exists(CFG.processed_validation_dataset_path):
  validation_dataset = validation_dataset.map(
      lambda b: prepare_batch(b, processor),
      batched=True,
      batch_size = CFG.batch_size,
      num_proc = CFG.num_proc,
      remove_columns = validation_dataset.column_names
  )
  validation_dataset.save_to_disk(CFG.processed_validation_dataset_path)
else:
  validation_dataset = load_from_disk(CFG.processed_validation_dataset_path)




SAMPLE_SIZE = 2000
SEED = 42

validation_dataset  = validation_dataset.shuffle(seed=SEED).select(range(SAMPLE_SIZE))
validation_dataset



from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
  processor: processor
  padding: Union[bool, str] = True

  def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
    input_features = [{"input_values": feature["input_values"]} for feature in features]
    labels_input_ids = [{"input_ids": feature["labels"]} for feature in features]


    batch = self.processor.pad(
        input_features,
        padding=self.padding,
        return_tensors="pt",
    )

    with self.processor.as_target_processor():
        labels_batch = self.processor.pad(
            labels_input_ids,
            padding=self.padding,
            return_tensors="pt",
        )

    labels = labels_batch['input_ids'].masked_fill(labels_batch.attention_mask.ne(1), -100)
    batch['labels'] = labels

    return batch

import evaluate
wer = evaluate.load('wer')



import evaluate
wer_metric = evaluate.load("wer")
def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metric
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


from transformers import AutoModelForCTC, TrainingArguments, Trainer
model = AutoModelForCTC.from_pretrained(
    CFG.model_name,
)
model.freeze_feature_extractor()


import wandb
wandb.login()

training_args = TrainingArguments(
    output_dir=CFG.parent_path,
    per_device_train_batch_size=CFG.batch_size,
    per_device_eval_batch_size=CFG.batch_size,
    gradient_accumulation_steps=2,
    learning_rate=0.01,
    warmup_steps=50,
    eval_strategy = "steps",#epcoch
    num_train_epochs = 10,
    fp16=True,
    save_steps = 50,
    eval_steps = 50,
    logging_steps = 50,
    save_total_limit = 2,
    load_best_model_at_end = True,
    metric_for_best_model = "wer",
    greater_is_better = False,
    # push_to_hub=True)
    )
trainer = Trainer(
    model=model,
    data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor=processor),
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    tokenizer=processor.feature_extractor,
)

trainer.train()


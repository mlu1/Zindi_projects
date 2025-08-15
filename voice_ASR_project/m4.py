#!/usr/bin/env python3
"""
Inference for the Zindi Swahili ASR challenge
• wav2vec 2.0 XLS-R + KenLM shallow-fusion (pyctcdecode)
• Reads HF test split, writes submission.csv
"""
from pathlib import Path
import sys, subprocess, io, argparse, json
import torch, torchaudio, soundfile as sf
import numpy as np, pandas as pd
from datasets import load_dataset, Audio
from transformers import AutoProcessor, AutoModelForCTC
from pyctcdecode import build_ctcdecoder  # ← KenLM beam search
from tqdm import tqdm

# ------------------------------------------------------------------ #
# 1. CLI
# ------------------------------------------------------------------ #
p = argparse.ArgumentParser()
p.add_argument("--model_repo",
               default="thinkKenya/wav2vec2-large-xls-r-300m-sw")
p.add_argument("--kenlm_path", required=True,
               help="Path to .arpa or binary KenLM model")
p.add_argument("--batchsize", type=int, default=8)
p.add_argument("--out_csv", default="submission.csv")
p.add_argument("--hf_cache", default=None)
args = p.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"📟  Device: {DEVICE}")

# ------------------------------------------------------------------ #
# 2. Model + processor
# ------------------------------------------------------------------ #
processor = AutoProcessor.from_pretrained(args.model_repo)
model = AutoModelForCTC.from_pretrained(args.model_repo).to(DEVICE).eval()

# ------------------------------------------------------------------ #
# 3. KenLM decoder (build once)
# ------------------------------------------------------------------ #

# --------------------------------------------------------------- #
# 3.  KenLM decoder  – ensure len(labels)==vocab_size AND unique
# --------------------------------------------------------------- #
from pyctcdecode import build_ctcdecoder

vocab_dict   = processor.tokenizer.get_vocab()              # {token: id}
sorted_vocab = sorted(vocab_dict.items(), key=lambda kv: kv[1])

labels = []
blank_used = False                                          # we allow ONE ""

for token, _ in sorted_vocab:
    if token == "|":                                        # HF's "word-sep"
        labels.append(" ")                                  # literal space
    elif token == processor.tokenizer.pad_token and not blank_used:
        labels.append("")                                   # CTC blank
        blank_used = True
    elif token in {processor.tokenizer.pad_token,
                   processor.tokenizer.eos_token,
                   processor.tokenizer.bos_token,
                   processor.tokenizer.unk_token}:
        # Map *other* specials to unique placeholders
        labels.append(f"<{token.strip('<>')}>")             # e.g. "<unk>"
    else:
        labels.append(token)

# Safety: 1) size match   2) uniqueness
assert len(labels) == model.config.vocab_size, \
    f"Expected {model.config.vocab_size} labels, got {len(labels)}"
assert len(labels) == len(set(labels)), "Duplicate labels still present!"

decoder = build_ctcdecoder(
    labels            = labels,
    kenlm_model_path  = args.kenlm_path,
    alpha             = 0.6,
    beta              = 2.5
)


# ------------------------------------------------------------------ #
# 4. Dataset
# ------------------------------------------------------------------ #
print("⬇️  Loading HF test split …")
ds = load_dataset(
    "sartifyllc/Sartify_ITU_Zindi_Testdataset",
    split="test",
    cache_dir=args.hf_cache
).cast_column("audio", Audio(decode=False))
print(f"   ↳ {len(ds):,} files")

# ------------------------------------------------------------------ #
# 5. Utils
# ------------------------------------------------------------------ #
def bytes_to_16k_np(b: bytes):
    wav, sr = sf.read(io.BytesIO(b))
    if sr != 16000:
        wav = torchaudio.functional.resample(torch.tensor(wav), sr, 16000).numpy()
    return wav

def chunk(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]

# ------------------------------------------------------------------ #
# 6. Inference loop
# ------------------------------------------------------------------ #
print("🚀  Forward + LM beam search …")
results = []
items = [{"fname": ex["filename"], "bytes": ex["audio"]["bytes"]} for ex in ds]

softmax = torch.nn.LogSoftmax(dim=-1)   # log-probs for decoder

for batch in tqdm(list(chunk(items, args.batchsize))):
    wavs = [bytes_to_16k_np(x["bytes"]) for x in batch]

    inputs = processor(
        wavs, sampling_rate=16000, return_tensors="pt", padding=True
    ).to(DEVICE)

    with torch.inference_mode():
        logits = model(inputs.input_values).logits     # (B, T, V)

    log_probs = softmax(logits).cpu().numpy()          # pyctcdecode wants log-probs

    # Beam decode one utterance at a time (decode_batch also works)
    texts = [decoder.decode(lp) for lp in log_probs]

    for ex, txt in zip(batch, texts):
        cleaned = "".join(ch for ch in txt.lower() if ch.isalnum() or ch.isspace()).strip()
        results.append({"filename": ex["fname"], "text": cleaned})

# ------------------------------------------------------------------ #
# 7. Write CSV
# ------------------------------------------------------------------ #
pd.DataFrame(results).to_csv(args.out_csv, index=False)
print(f"✅  {args.out_csv} ready – {len(results):,} rows")


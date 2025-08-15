from pathlib import Path
from huggingface_hub import hf_hub_download

# choose where you want the file to live
my_lm_dir = Path("assets/language_models")      # e.g. ./assets/language_models
my_lm_dir.mkdir(parents=True, exist_ok=True)

lm_path = hf_hub_download(
    repo_id     = "edugp/kenlm",
    filename    = "wikipedia/sw.arpa.bin",
    cache_dir   = my_lm_dir,        # <-- custom location
    resume_download = True
)
print("KenLM stored at:", lm_path)  # → assets/language_models/sw.arpa.bin


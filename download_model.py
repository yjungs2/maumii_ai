# download_model.py
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="yjungs2/trained_klueBERT",
    local_dir="/app/model",
    local_dir_use_symlinks=False
)
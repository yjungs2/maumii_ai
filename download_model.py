# download_model.py
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="dlckdfuf141/korean-emotion-kluebert-v2",
    local_dir="/app/model",
    local_dir_use_symlinks=False
)
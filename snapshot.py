from huggingface_hub import snapshot_download

huggingface_token = "#####################################"

snapshot_download(
    repo_id="mistralai/Mistral-7B-Instruct-v0.1", 
    local_dir="model", 
    local_dir_use_symlinks=False,
    revision="main",
    use_auth_token=huggingface_token  # Pass the token here
)

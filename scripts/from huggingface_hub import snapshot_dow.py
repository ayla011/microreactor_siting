from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="bettergovph/project-noah-hazard-maps",
    repo_type="dataset",
    local_dir="data/raw/hazard/noah",
    ignore_patterns=["*.md"]   # skip readme files
)
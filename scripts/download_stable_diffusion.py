from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import snapshot_download

# Updated to the latest Stable Diffusion 3.5 Large model
MODEL_ID = "stabilityai/stable-diffusion-3.5-large"
DEFAULT_DEST = Path("models") / "stable-diffusion-3.5-large"


def resolve_token() -> str:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN environment variable not found. Visit https://huggingface.co/settings/tokens "
            "to create a token with the required scopes and set it before running this script."
        )
    return token


def download_model(model_id: str = MODEL_ID, dest: Path = DEFAULT_DEST) -> Path:
    dest = dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)

    token = resolve_token()

    print(f"Downloading {model_id} to {dest}...")
    snapshot_download(
        repo_id=model_id,
        local_dir=str(dest),
        local_dir_use_symlinks=False,
        token=token,
        # Exclude some large files if necessary, but usually we want everything for local inference
        # ignore_patterns=["*.safetensors"] # Example if we wanted to exclude safetensors (we don't)
    )
    return dest


def main() -> None:
    target_path = download_model()
    print(f"Stable Diffusion 3.5 Large model files saved to {target_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import snapshot_download

MODEL_ID = "Salesforce/blip-image-captioning-large"
DEFAULT_DEST = Path("models") / "blip-image-captioning-large"


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

    snapshot_download(
        repo_id=model_id,
        local_dir=str(dest),
        local_dir_use_symlinks=False,
        token=token,
    )
    return dest


def main() -> None:
    target_path = download_model()
    print(f"BLIP model files saved to {target_path}")


if __name__ == "__main__":
    main()

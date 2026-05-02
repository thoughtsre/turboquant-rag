import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Union, Optional

_KAGGLE_URL = "https://www.kaggle.com/api/v1/datasets/download/Cornell-University/arxiv"
_REPO_ROOT = Path(__file__).resolve().parent.parent


def download_arxiv(dest_dir: Optional[Union[str, Path]] = None) -> Path:
    """Download and extract the Kaggle arxiv dataset to dest_dir."""
    dest = Path(dest_dir) if dest_dir is not None else _REPO_ROOT / "data" / "raw"
    dest.mkdir(parents=True, exist_ok=True)

    print(f"Downloading arxiv dataset to {dest} ...")
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        with urllib.request.urlopen(_KAGGLE_URL) as response:
            while chunk := response.read(1 << 20):
                tmp.write(chunk)

    try:
        print("Extracting ...")
        with zipfile.ZipFile(tmp_path) as zf:
            zf.extractall(dest)
    finally:
        tmp_path.unlink(missing_ok=True)

    return dest


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download arxiv dataset from Kaggle")
    parser.add_argument("--dest", default=None, help="Override destination directory")
    args = parser.parse_args()
    path = download_arxiv(args.dest)
    print(f"Extracted to {path}")

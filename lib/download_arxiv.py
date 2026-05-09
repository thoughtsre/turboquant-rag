import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Union, Optional
import logging

logger = logging.getLogger(__name__)

_KAGGLE_URL = "https://www.kaggle.com/api/v1/datasets/download/Cornell-University/arxiv"
_REPO_ROOT = Path(__file__).resolve().parent.parent


def download_arxiv(dest_dir: Optional[Union[str, Path]] = None) -> Path:
    """Download and extract the Kaggle arxiv dataset to dest_dir."""
    dest = Path(dest_dir) if dest_dir is not None else _REPO_ROOT / "data" / "raw"
    dest.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading arxiv dataset to {dest} ...")
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        with urllib.request.urlopen(_KAGGLE_URL) as response:
            while chunk := response.read(1 << 20):
                tmp.write(chunk)

    try:
        logger.info("Extracting ...")
        with zipfile.ZipFile(tmp_path) as zf:
            zf.extractall(dest)
    finally:
        tmp_path.unlink(missing_ok=True)

    return dest

def convert_to_parquet(input_path: str | Path) -> Path:
    """Convert the extracted ndjson file to parquet format."""
    import polars as pl

    if isinstance(input_path, str):
        input_path = Path(input_path)

    output_path = input_path.with_suffix(".parquet")
    pl.scan_ndjson(input_path).sink_parquet(output_path)
    logger.info(f"Converted {input_path} to {output_path}")
    return output_path


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    import argparse

    parser = argparse.ArgumentParser(description="Download arxiv dataset from Kaggle")
    parser.add_argument("--dest", default=None, help="Override destination directory")
    args = parser.parse_args()
    path = download_arxiv(args.dest)
    logger.info(f"Extracted to {path}")
    
    convert_to_parquet(path / "arxiv-metadata-oai-snapshot.json")

import polars as pl
import pandas as pd
from pathlib import Path


def sample_data(input_path: str | Path, sample_every: int = 100) -> pd.DataFrame:
    
    if isinstance(input_path, str):
        input_path = Path(input_path)
        
    if not input_path.exists():
        raise FileNotFoundError(f"Input file {input_path} does not exist.")
    
    return (
        pl
        .scan_parquet(input_path)
        .gather_every(sample_every)
        .collect()
        .to_pandas()
    )
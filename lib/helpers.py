import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path
import pyarrow.parquet as pq


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
    
    
def pad_zeros(arr: np.ndarray, target_dim: int) -> np.ndarray:
    """
    Pad the input array with zeros to reach the target dimension.
    If the input array already has the target dimension or more, it is returned unchanged.
    """
    
    _, init_dim = arr.shape
    
    a = np.pad(arr, ((0, 0), (0, target_dim - init_dim)), mode='constant')
    
    return a


def check_parquet(input_filename: str | Path) -> None:
    if isinstance(input_filename, str):
        input_filename = Path(input_filename)
        
    if not input_filename.exists():
        raise FileNotFoundError(f"Input file {input_filename} not found.")
    
    pq_file = pq.ParquetFile(input_filename)
    
    header_text = "Embedding Parquet File Details"
    print(f"\n{header_text}\n{'=' * len(header_text)}")
    print(f"File: {input_filename}")
    print(f"Rows: {pq_file.metadata.num_rows}")
    print("Schema:")
    for field in pq_file.schema_arrow:
        print(f"\tColumn: {field.name}, Type: {field.type}")
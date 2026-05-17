from pathlib import Path
import pickle as pkl
import polars as pl
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from lib.turboquant.turboquant import quantize_embeddings_polar, pack_bits_polar
from lib.helpers import check_parquet


def create_quantized_embeddings(raw_embeddings_path: str | Path, 
                                output_path: str | Path, 
                                codebook_path: str | Path,
                                n_bits: int, 
                                batch_size: int = 30000,
                                max_rows: int | None = None,
                                with_qjl: bool =False) -> None:
    
    if isinstance(raw_embeddings_path, str):
        raw_embeddings_path = Path(raw_embeddings_path)
        
    assert raw_embeddings_path.is_file(), f"Raw embeddings file not found: {raw_embeddings_path}. Run create_embeddings.py first to generate the raw embeddings."
    
    if isinstance(codebook_path, str):
        codebook_path = Path(codebook_path)
    
    assert codebook_path.is_file(), f"Codebook file not found: {codebook_path}"
    
    with open(codebook_path, "rb") as f:
        codebook = pkl.load(f)
        
    if isinstance(output_path, str):
        output_path = output_path.format(n_bits)
        if with_qjl:
            output_path = output_path.replace(".parquet", "_qjl.parquet")
        output_path = Path(output_path)
        
    if with_qjl:
        n_bits -= 1
        
    data = pq.ParquetFile(raw_embeddings_path)
    
    total_rows = data.metadata.num_rows
    
    if max_rows is not None:
        total_rows = min(total_rows, max_rows)
    
    schema = None
    
    pbar = tqdm(total=total_rows, desc="Creating embeddings", unit="rows", smoothing=0)
        
    for i, batch in enumerate(data.iter_batches(batch_size)):
            
        batch = pl.from_arrow(batch)
        
        if (pbar.n + batch.height) > total_rows:
            batch = batch.head(total_rows - pbar.n)
        
        if i == 0:
            raw_embed_dim = batch["embedding"][0].shape[0]
            
            if not np.log2(raw_embed_dim).is_integer():
                rot_embed_dim = 2 ** int(np.ceil(np.log2(raw_embed_dim)))
            else:
                rot_embed_dim = raw_embed_dim
                
            packed_embed_dim = rot_embed_dim // (8 // n_bits) if not with_qjl else rot_embed_dim // (8 // (n_bits+1))
        
        batch = batch.with_columns(
            pl.col("embedding")
            .map_batches(lambda s: quantize_embeddings_polar(s, codebook=codebook, n_bits=n_bits, seed=42), return_dtype=pl.Array(pl.UInt8, rot_embed_dim))
            .alias("quantized_embedding")
        )
            
        if with_qjl:
            pass # to be implemented later
        
        batch = batch.with_columns(
            pl.col("quantized_embedding")
            .map_batches(lambda s: pack_bits_polar(s, n_bits=n_bits), return_dtype=pl.Array(pl.UInt8, packed_embed_dim))
            .alias("packed_embedding")   
        )
        
        batch = batch.select(["id", "packed_embedding"])
        
        if i == 0:
            schema = pa.schema(pa.schema([
                pa.field("id", pa.large_string()),
                pa.field("packed_embedding", pa.list_(pa.uint8(), packed_embed_dim))
            ]))
        
            writer = pq.ParquetWriter(output_path, schema)
            
        writer.write_table(batch.select(pl.col("id"), pl.col("packed_embedding")).to_arrow())
        
        pbar.update(batch.height)
        
        if pbar.n >= total_rows:
            break
            
    pbar.close()
    writer.close()
    
    return 
    

if __name__ == "__main__":
    from argparse import ArgumentParser
    
    parser = ArgumentParser(description="Create quantized embeddings from raw embeddings")
    parser.add_argument("--raw_embeddings_path", type=str, default="./data/processed/arxiv_embeddings.parquet", help="Path to the raw embeddings Parquet file")
    parser.add_argument("--output_path", type=str, default="./data/processed/arxiv_quantized_embeddings_{}bits.parquet", help="Path to the output quantized embeddings Parquet file")
    parser.add_argument("--codebook_path", type=str, default="./data/codebook/384d_codebook.pkl", help="Path to the codebook pickle file")
    parser.add_argument("--n_bits", type=int, default=4, help="Number of bits to quantize each dimension to (e.g. 4 for 4-bit quantization)")
    parser.add_argument("--batch_size", type=int, default=30000, help="Batch size for processing embeddings")
    parser.add_argument("--max_rows", type=int, default=None, help="Maximum number of rows to process (for testing/debugging)")
    parser.add_argument("--with_qjl", action="store_true", help="Whether to apply Quantization-aware Joint Learning (QJL) techniques (not implemented yet)")
    
    args= parser.parse_args()
    
    create_quantized_embeddings(
        raw_embeddings_path=args.raw_embeddings_path,
        output_path=args.output_path,
        codebook_path=args.codebook_path,
        n_bits=args.n_bits,
        batch_size=args.batch_size,
        max_rows=args.max_rows,
        with_qjl=args.with_qjl
    )
    
    check_parquet(args.output_path.format(args.n_bits))
from pathlib import Path
import pickle as pkl
import polars as pl
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from lib.turboquant.turboquant import quantize_embeddings_polar, pack_bits_polar, dequantize_embeddings_polar, rotate_embeddings_polar, pack_qjl_bits_polar
from lib.helpers import check_parquet


def create_quantized_embeddings(raw_embeddings_path: str | Path, 
                                output_path: str | Path, 
                                codebook_path: str | Path,
                                n_bits: int, 
                                mse_seed: int = 42,
                                qjl_seed: int = 24,
                                batch_size: int = 30000,
                                max_rows: int | None = None,
                                with_qjl: bool =False,
                                full_data: bool = False) -> str | Path:
    
    if isinstance(raw_embeddings_path, str):
        raw_embeddings_path = Path(raw_embeddings_path)
        
    assert raw_embeddings_path.is_file(), f"Raw embeddings file not found: {raw_embeddings_path}. Run create_embeddings.py first to generate the raw embeddings."
    
    if isinstance(codebook_path, str):
        codebook_path = Path(codebook_path)
    
    assert codebook_path.is_file(), f"Codebook file not found: {codebook_path}"
    
    with open(codebook_path, "rb") as f:
        codebook = pkl.load(f)
        
    if isinstance(output_path, str):
        output_path = output_path.format(n_bits, mse_seed)
        if with_qjl:
            output_path = output_path.replace(".parquet", f"_qjl_seed={qjl_seed}.parquet")
            
        if full_data:
            output_path = output_path.replace(".parquet", "_full_data.parquet")
            
        output_path = Path(output_path)
    
    assert 8 % n_bits == 0
        
    data = pq.ParquetFile(raw_embeddings_path)
    
    total_rows = data.metadata.num_rows
    
    if max_rows is not None:
        total_rows = min(total_rows, max_rows)
    
    schema = None
    
    pbar = tqdm(total=total_rows, desc="Creating embeddings", unit="rows", smoothing=0)
        
    for i, batch in enumerate(data.iter_batches(batch_size)):
            
        batch = pl.from_arrow(batch)
        
        if (pbar.n + batch.height) > total_rows: # type: ignore
            batch = batch.head(total_rows - pbar.n)
        
        if i == 0:
            raw_embed_dim = batch["embedding"][0].shape[0] # type: ignore
            
            if not np.log2(raw_embed_dim).is_integer():
                rot_embed_dim = 2 ** int(np.ceil(np.log2(raw_embed_dim)))
            else:
                rot_embed_dim = raw_embed_dim
                
            packed_embed_dim = rot_embed_dim // (8 // n_bits)
        
        batch = batch.with_columns( # type: ignore
            pl.col("embedding")
            .map_batches(lambda s: rotate_embeddings_polar(s, seed=mse_seed), return_dtype=pl.Array(pl.Float32, rot_embed_dim))
            .alias("rotated_embedding")
        ).with_columns(
            pl.col("rotated_embedding")
            .map_batches(lambda s: quantize_embeddings_polar(s, codebook=codebook, n_bits=n_bits), return_dtype=pl.Array(pl.UInt8, rot_embed_dim))
            .alias("quantized_embedding")
        ).with_columns(
            pl.col("quantized_embedding")
            .map_batches(lambda s: pack_bits_polar(s, n_bits=n_bits), return_dtype=pl.Array(pl.UInt8, packed_embed_dim))
            .alias("packed_embedding")   
        )
            
        if with_qjl:

            batch = batch.with_columns(
                pl.col("packed_embedding")
                .map_batches(lambda s: dequantize_embeddings_polar(s, codebook=codebook, n_bits=n_bits, seed=mse_seed ,orig_embed_dim=raw_embed_dim))
                .alias("recovered_embedding")
            ).with_columns(
                (pl.col("recovered_embedding") - pl.col("embedding"))
                .alias("residuals")
            ).with_columns(
                (pl.col("residuals")
                .map_batches(lambda s: rotate_embeddings_polar(s, qjl_seed), 
                            return_dtype=pl.Array(pl.Float32, rot_embed_dim))
                .arr.eval(pl.element() > 0)
                .map_batches(pack_qjl_bits_polar, return_dtype=pl.Array(pl.UInt8, rot_embed_dim // 8))
                .alias("packed_qjl_bits")),
                (pl.col("residuals").arr.eval(pl.element()**2).arr.sum()**0.5).alias("gamma")
            )
            
        
        if i == 0:
            
            cols = [
                pa.field("id", pa.large_string()),
                pa.field("packed_embedding", pa.list_(pa.uint8(), packed_embed_dim))
            ]
            
            if with_qjl:
                cols += [
                    pa.field("gamma", pa.float32()),
                    pa.field("packed_qjl_bits", pa.list_(pa.uint8(), rot_embed_dim // 8))
                ]
                
            if full_data:
                cols += [
                    pa.field("embedding", pa.list_(pa.float32(), raw_embed_dim)),
                    pa.field("rotated_embedding", pa.list_(pa.float32(), rot_embed_dim)),
                    pa.field("quantized_embedding", pa.list_(pa.uint8(), rot_embed_dim))
                ]
                
                if with_qjl:
                    
                    cols += [
                    pa.field("residuals", pa.list_(pa.float32(), raw_embed_dim)),
                    pa.field("recovered_embedding", pa.list_(pa.float32(), raw_embed_dim))
                ]
            
            schema = pa.schema(cols)
        
            writer = pq.ParquetWriter(output_path, schema)
            
        writer.write_table(batch.select(*(_.name for _ in cols)).to_arrow())
        
        pbar.update(batch.height)
        
        if pbar.n >= total_rows:
            break
            
    pbar.close()
    writer.close()
    
    return output_path
    

if __name__ == "__main__":
    from argparse import ArgumentParser
    
    parser = ArgumentParser(description="Create quantized embeddings from raw embeddings")
    parser.add_argument("--raw_embeddings_path", type=str, default="./data/processed/arxiv_embeddings.parquet", help="Path to the raw embeddings Parquet file")
    parser.add_argument("--output_path", type=str, default="./data/processed/arxiv_quantized_embeddings_{}bits_seed={}.parquet", help="Path to the output quantized embeddings Parquet file")
    parser.add_argument("--codebook_path", type=str, default="./data/codebook/384d_codebook.pkl", help="Path to the codebook pickle file")
    parser.add_argument("--n_bits", type=int, default=4, help="Number of bits to quantize each dimension to (e.g. 4 for 4-bit quantization)")
    parser.add_argument("--batch_size", type=int, default=30000, help="Batch size for processing embeddings")
    parser.add_argument("--max_rows", type=int, default=None, help="Maximum number of rows to process (for testing/debugging)")
    parser.add_argument("--with_qjl", action="store_true", help="Whether to apply Quantization-aware Joint Learning (QJL) techniques (not implemented yet)")
    parser.add_argument("--full_data", action="store_true", help="Whether to write all data columns")
    
    args= parser.parse_args()
    
    output_path = create_quantized_embeddings(
        raw_embeddings_path=args.raw_embeddings_path,
        output_path=args.output_path,
        codebook_path=args.codebook_path,
        n_bits=args.n_bits,
        batch_size=args.batch_size,
        max_rows=args.max_rows,
        with_qjl=args.with_qjl,
        full_data=args.full_data
    )
    
    check_parquet(output_path)
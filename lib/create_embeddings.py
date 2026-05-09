from argparse import ArgumentParser
import polars as pl
from typing import Optional
from pathlib import Path
from sentence_transformers import SentenceTransformer
from logging import getLogger
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from lib.download_arxiv import download_arxiv

logger = getLogger(__name__)

def create_embeddings(input_filename: str | Path, 
                      output_dir: str | Path, 
                      batch_size: int = 128,
                      num_rows: Optional[int] = None,
                      auto_download: bool = True) -> None:
    
    def embed_batch(s: pl.Series) -> pl.Series:
        texts = s.to_list()
        embeddings = embedder.encode(texts, batch_size=batch_size, precision="float32", show_progress_bar=False).tolist()
        return pl.Series(embeddings, dtype=pl.Array(pl.Float32, embedder.get_embedding_dimension()))
    
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
        
    if isinstance(input_filename, str):
        input_filename = Path(input_filename)
        
    output_dir.mkdir(parents=True, exist_ok=True)
    output_filename = output_dir / "arxiv_embeddings.parquet"
    
    if not input_filename.exists():
        if auto_download:
            print(f"Input file {input_filename} not found. Attempting to download dataset...")
            download_arxiv(dest_dir=input_filename.parent)
        else:
            raise FileNotFoundError(f"Input file {input_filename} not found and auto_download is disabled.")
    else:
        logger.info(f"Found input file {input_filename}, proceeding with embedding creation.")
    
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embed_dim = embedder.get_embedding_dimension()
    
    data = pq.ParquetFile(input_filename)
    total_rows = data.metadata.num_rows
    
    if num_rows is not None:
        total_rows = min(total_rows, num_rows)
    
    schema = pa.schema([
        pa.field("id", pa.large_string()),
        pa.field("content", pa.large_string()),
        pa.field("embedding", pa.list_(pa.float32(), embed_dim))
    ])
    
    pbar = tqdm(total=total_rows, desc="Creating embeddings", unit="rows", smoothing=0)
    
    with pq.ParquetWriter(output_filename, schema) as writer:
        
        buffer = []
        buffer_size = 0
        buffer_max_size = 10000
        
        for batch in data.iter_batches(batch_size, columns=["id", "title", "abstract"]):
            batch_df = (
                pl.from_arrow(batch)
                .filter(pl.col("abstract").str.len_chars() > 150) # type: ignore
            )
            
            batch_len = batch_df.height # type: ignore
            
            if batch_len == 0:
                continue
            
            if (pbar.n + batch_len) > total_rows:
                batch_df = batch_df.head(total_rows - pbar.n)
                batch_len = batch_df.height # type: ignore
            
            batch_df = (
                batch_df.select( # type: ignore
                    pl.col("id"),
                    (pl.lit("Title: ") + pl.col("title") + pl.lit(" | Abstract: ") + pl.col("abstract")).alias("content")
                )
                .with_columns(
                    pl.col("content").map_batches(embed_batch, return_dtype=pl.Array(pl.Float32, embed_dim)).alias("embedding")
                )
            )
            
            buffer.append(batch_df.to_arrow())
            buffer_size += batch_len
            
            if buffer_size > buffer_max_size:
                combined_table = pa.concat_tables(buffer)
                writer.write_table(combined_table)
                buffer = []
                buffer_size = 0
            
            pbar.update(batch_len)
            
        combined_table = pa.concat_tables(buffer)
        writer.write_table(combined_table)
        buffer = []
        buffer_size = 0
            
    pbar.close()
    
    return

def check_embeddings(input_filename: str | Path) -> None:
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

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    
    from time import time
    start_time = time()
    
    parser = ArgumentParser(description="Create embeddings from arxiv dataset")
    parser.add_argument("--input", default="./data/raw/arxiv-metadata-oai-snapshot.parquet", help="Path to input parquet file")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for embedding creation")
    parser.add_argument("--output", default="./data/processed", help="Directory to save output parquet file")
    parser.add_argument("--num_rows", type=int, default=None, help="Number of rows to process (for testing)")
    parser.add_argument("--no_download", action="store_true", help="Disable auto-downloading of dataset if input file is missing")
    
    args = parser.parse_args()
    
    try:
        create_embeddings(
            input_filename=args.input,
            output_dir=args.output,
            num_rows=args.num_rows,
            batch_size=args.batch_size,
            auto_download=not args.no_download
        )
        
    except Exception as e:
        logger.exception("Failed to create embeddings")
        
        raise e
    
    else:
        logger.info("Successfully created embeddings")
        check_embeddings(Path(args.output) / "arxiv_embeddings.parquet")
    
    finally:
        elapsed_time = time() - start_time
        logger.info(f"Total execution time: {elapsed_time:.2f} seconds")

    
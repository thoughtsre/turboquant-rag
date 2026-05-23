import os
import requests
from pathlib import Path
from lib.download_arxiv import *

if __name__ == "__main__":
    
    os.makedirs("../data", exist_ok=True)
    
    print("Downloading raw data from Kaggle")
    path = download_arxiv()
    convert_to_parquet(path / "arxiv-metadata-oai-snapshot.json")
    
    print("Downloading codebook")
    codebook_resp = requests.get("https://github.com/thoughtsre/turboquant-rag/v1/384d_codebook.pkl")
    codebook_resp.raise_for_status()
    Path("../data/codebook").mkdir(exist_ok=True)
    (Path("../data/codebook") / "384d_codebook.pkl").write_bytes(codebook_resp.content)
    
    print("Downloading document embeddings")
    codebook_resp = requests.get("https://github.com/thoughtsre/turboquant-rag/v1/arxiv_quantized_embeddings_4bits_qjl_full_data.parquet")
    codebook_resp.raise_for_status()
    Path("../data/processed").mkdir(exist_ok=True)
    (Path("../data/processed") / "arxiv_quantized_embeddings_4bits_qjl_full_data.parquet").write_bytes(codebook_resp.content)
    
    
    
        
"""
Create a codebook for quantization using k-means clustering on samples drawn from a beta distribution. 
The codebook is generated for different bit widths (1, 2, and 4 bits) and saved as a pickle file in the specified output directory. 
The beta distribution parameters are set based on the embedding dimension to create a suitable range of values for quantization.
"""

from scipy.stats import beta
from scipy.cluster.vq import kmeans
import pickle as pk
from argparse import ArgumentParser
from pathlib import Path

def create_codebook(d: int = 384, bits: int = 4, num_samples: int = int(1e5)):
    
    a = b = (d-1) / 2
    
    samples = beta.rvs(a, b, loc=-1, scale=2, size=num_samples)
    
    centroids, _ = kmeans(samples, int(2**bits))
    
    centroids.sort()
    
    return centroids


if __name__ == "__main__":
    
    parser = ArgumentParser(description="Create a codebook for quantization")
    parser.add_argument("--output_dir", type=str, default="./data/codebook", help="Path to save the codebook")
    parser.add_argument("--embedding_dim", type=int, default=384, help="Dimension of the embeddings")
    parser.add_argument("--num_samples", type=int, default=int(1e6), help="Number of samples to generate for k-means")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    bits = [1, 2, 4]
    
    codebook = dict([(f"{i}bits", create_codebook(d=args.embedding_dim, bits=i, num_samples=args.num_samples)) for i in bits])
    
    with open(output_dir / f"{args.embedding_dim}d_codebook.pkl", "wb") as f:
        pk.dump(codebook, f)
    
    print(f"Codebook created and saved to {output_dir}")
import polars as pl
import numpy as np

from lib.algorithms import fwht_batch


def quantize_embeddings_numpy(rot_emb: np.ndarray, codebook: dict, n_bits: int) -> np.ndarray:
    """Quantize embeddings using NumPy arrays.
    rot_emb should be shape (batch_size, d).
    codebook should be a dict with keys like "4bits" and values of shape (d, 2^n_bits).
    """
    
    return np.argmin(np.abs(rot_emb[:, :, np.newaxis] - codebook[f"{int(n_bits)}bits"].reshape(1, 1, -1)), axis=2).astype(np.uint8)

def pack_bits_numpy(emb: np.ndarray, stride: int) -> np.ndarray:
    """Pack quantized embeddings into bytes using NumPy.
    emb should be shape (batch_size, d).
    stride is how many n_bits fit into 8 bits (e.g. stride=2 for 4 bits, stride=4 for 2 bits).
    """
    
    packed_emb = emb[:, 0::stride].copy()
    
    for i in range(1, stride):
        packed_emb = (packed_emb << stride) | emb[:, i::stride]
    
    return packed_emb


def quantize_embeddings_polar(s: pl.Series, codebook: dict, n_bits: int, seed: int) -> pl.Series:
    """Quantize embeddings using Polars Series and Fast Walsh-Hadamard Transform."""
    
    emb = np.vstack(s.to_numpy().astype(np.float32)) # type: ignore
    rot_emb, _ = fwht_batch(emb, seed=seed, sign_flip=True)
    
    quantized_emb_buckets = quantize_embeddings_numpy(rot_emb, codebook=codebook, n_bits=n_bits)
    
    return pl.Series(quantized_emb_buckets, dtype=pl.Array(pl.UInt8, rot_emb.shape[1]))


def pack_bits_polar(s: pl.Series, n_bits: int) -> pl.Series: 
    """Pack quantized embeddings into bytes using Polars Series."""
    
    if n_bits >= 8:
        raise ValueError("Packing is only needed for n_bits < 8")
    
    assert 8%n_bits == 0, "n_bits must be a divisor of 8 for packing"
    
    stride = int(8 // n_bits)
    
    emb = np.vstack(s.to_numpy().astype(np.uint8)) # type: ignore
    
    packed_emb = pack_bits_numpy(emb, stride=stride)
    
    return pl.Series(packed_emb, dtype=pl.Array(pl.UInt8, packed_emb.shape[1]))



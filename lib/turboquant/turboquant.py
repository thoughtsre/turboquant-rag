import polars as pl
import numpy as np

from lib.algorithms import fwht_batch


def quantize_embeddings(s: pl.Series, codebook: dict, n_bits: int, seed: int) -> pl.Series:
    
    emb = np.vstack(s.to_numpy().astype(np.float32)) # type: ignore
    rot_emb, _ = fwht_batch(emb, seed=seed, sign_flip=True)
    
    quantized_emb_buckets = np.argmin(
        np.abs(rot_emb[:, :, np.newaxis] - codebook[f"{int(n_bits)}bits"].reshape(1, 1, -1)), 
        axis=2).astype(np.uint8)

    return pl.Series(quantized_emb_buckets, dtype=pl.Array(pl.UInt8, rot_emb.shape[1]))

def pack_bits(s: pl.Series, n_bits: int) -> pl.Series: 
    
    if n_bits >= 8:
        raise ValueError("Packing is only needed for n_bits < 8")
    
    assert 8%n_bits == 0, "n_bits must be a divisor of 8 for packing"
    
    stride = int(8 // n_bits)
    
    emb = np.vstack(s.to_numpy().astype(np.uint8)) # type: ignore
    
    packed_emb = emb[:, 0::stride].copy()
    
    for i in range(1, stride):
        packed_emb = (packed_emb << n_bits) | emb[:, i::stride]
    
    return pl.Series(packed_emb, dtype=pl.Array(pl.UInt8, packed_emb.shape[1]))
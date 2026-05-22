import polars as pl
import numpy as np

from lib.algorithms import fwht_batch, generate_sign_flips


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

def rotate_embeddings_polar(s: pl.Series, seed: int) -> pl.Series:
    
    emb = np.vstack(s.to_numpy().astype(np.float32)) # type: ignore
    rot_emb, _ = fwht_batch(emb, seed=seed, sign_flip=True)
    
    return pl.Series(rot_emb, dtype=pl.Array(pl.UInt8, rot_emb.shape[1]))

def quantize_embeddings_polar(s: pl.Series, codebook: dict, n_bits: int) -> pl.Series:
    """Quantize embeddings using Polars Series and Fast Walsh-Hadamard Transform."""
    
    emb = np.vstack(s.to_numpy().astype(np.float32)) # type: ignore
    
    quantized_emb_buckets = quantize_embeddings_numpy(emb, codebook=codebook, n_bits=n_bits)
    
    return pl.Series(quantized_emb_buckets, dtype=pl.Array(pl.UInt8, emb.shape[1]))


def pack_bits_polar(s: pl.Series, n_bits: int) -> pl.Series: 
    """Pack quantized embeddings into bytes using Polars Series."""
    
    if n_bits >= 8:
        raise ValueError("Packing is only needed for n_bits < 8")
    
    assert 8%n_bits == 0, "n_bits must be a divisor of 8 for packing"
    
    stride = int(8 // n_bits)
    
    emb = np.vstack(s.to_numpy().astype(np.uint8)) # type: ignore
    
    packed_emb = pack_bits_numpy(emb, stride=stride)
    
    return pl.Series(packed_emb, dtype=pl.Array(pl.UInt8, packed_emb.shape[1]))


def calculate_centroids_lookup_table(codebook: dict, n_bits: int) -> np.ndarray:
    """Calculate centroids lookup table
    
    This is to allow for quick look up of the corresponding centroid coordinates from
    packed centroid IDs. For example, when b=4, 2 centroid IDs can be packed into a 
    single uint8.
    """
    
    assert 8 & n_bits == 0, "8 should be divisible by n_bits"
    
    n_slots = int(8 / n_bits)
    
    bit_mask = (1 << n_bits) - 1
    
    lut = []
    
    centroid_coords = codebook[f"{int(n_bits)}bits"]
    
    for i in range(int(2**(n_bits * n_slots))):
        
        centroid_ids = [(np.uint(i) >> int(j * n_bits)) & bit_mask for j in range(n_slots-1, -1, -1)]
        
        lut.append(tuple([centroid_coords[_] for _ in centroid_ids]))
        
    return np.array(lut)


def recover_centroid_coords(packed_quantized_embeddings: np.ndarray, n_bits: int, codebook: dict) -> np.ndarray:
    
    lut = calculate_centroids_lookup_table(codebook, n_bits)
    
    n_samples = packed_quantized_embeddings.shape[0]
    
    return lut[packed_quantized_embeddings].reshape((n_samples, -1))


def unrotate_embeddings(rotated_embeddings: np.ndarray, orig_dim: int, seed: int = 42): 
    
    d_signs = generate_sign_flips(seed, rotated_embeddings.shape[1])
    
    unrot_embed = fwht_batch(rotated_embeddings, seed=seed, sign_flip=False)
    
    return (unrot_embed * d_signs)[:, :orig_dim] # type: ignore


def dequantize_embeddings_polar(s: pl.Series, codebook: dict, n_bits: int, seed: int, orig_embed_dim: int) -> pl.Series:
    """De-quantize packed and rotated embeddings using Polars Series."""
    
    packed_quantized_emb = np.vstack(s.to_numpy().astype(np.int8)) # type: ignore
    
    unpacked_centroid_coords =  recover_centroid_coords(packed_quantized_emb, n_bits, codebook)
    
    unrot_emb = unrotate_embeddings(unpacked_centroid_coords, orig_embed_dim, seed)
    
    return pl.Series(unrot_emb, dtype=pl.Array(pl.Float32, unrot_emb.shape[1]))
    

def calculate_query_distance_lookup_table(query_embedding: np.ndarray, n_bits: int, codebook: dict) -> np.ndarray:
    
    assert 8 % n_bits == 0, "8 should be divisible by n_bits"
    
    embed_dim = query_embedding.shape
    
    assert len(embed_dim) == 2, "Query embedding should be a 2D-array"
    assert embed_dim[0] == 1, "There should only be 1 query"
    assert embed_dim[1] % 2 == 0, "Should be multiple of 2 if rotated via FWHT"
    
    lut = []
    
    n_slots = int(8/n_bits)
    
    bit_mask = (1 << n_bits) - 1
    
    for i in range(int(2**(n_bits * n_slots))):
        
        centroid_ids = [(np.uint(i) >> int(j * n_bits)) & bit_mask for j in range(n_slots-1, -1, -1)]
        
        cluster_dists = []
        
        for j in range(int(query_embedding.shape[1] // n_slots)):
        
            cluster_dists.append(np.sum([query_embedding.flatten()[n_slots*j + idx_delta] * codebook[f"{n_bits}bits"][int(c)] for idx_delta, c in enumerate(centroid_ids)]))
            
        lut.append(cluster_dists)
        
    return np.array(lut)


def pack_qjl_bits_polar(s: pl.Series):
    
    qjl_bits = s.to_numpy()
    

    assert len(qjl_bits.shape) == 2, "Expecting a 2D array of qjl bits"
    assert qjl_bits.shape[1] % 8 == 0, "The embedding dimension should be divisible by 8 to ensure clean packing."
    
    packed_qjl_bits = np.packbits(qjl_bits, axis=-1)
    
    return pl.Series(packed_qjl_bits, dtype=pl.Array(pl.UInt8, packed_qjl_bits.shape[1]))
    
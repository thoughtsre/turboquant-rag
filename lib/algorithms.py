import numpy as np
from typing import Tuple

from lib.helpers import pad_zeros


def _fwht(a: np.ndarray, d: int, batch_size: int) -> np.ndarray:
    
    for i in range(np.log2(d).astype(int)):
        stride = 2**i
        # Reshape to isolate the butterfly pairs within each vector
        a = a.reshape(batch_size, -1, 2, stride)
        # Apply the butterfly: sum and difference across the pairs
        a = np.stack([a[:, :, 0, :] + a[:, :, 1, :], 
                      a[:, :, 0, :] - a[:, :, 1, :]], axis=2)
        
    return a.reshape(batch_size, d) / np.sqrt(d)


def generate_sign_flips(seed: int, d: int) -> np.ndarray:
    
    rng = np.random.default_rng(seed)
    return rng.choice([1.0, -1.0], size=d)


def fwht_batch(a: np.ndarray, seed: int = 42, sign_flip: bool = True) -> Tuple[np.ndarray, np.ndarray] | np.ndarray:
    """
    Fast Walsh-Hadamard Transform for a batch of vectors.
    Input 'a' should be shape (batch_size, d), where d is a power of 2.
    """
    a = a.astype(float)
    batch_size, d = a.shape
    
    if np.log2(d) % 1 != 0:
        d_pad = 2**int(np.ceil(np.log2(d)))
        a = pad_zeros(a, d_pad)
        d = d_pad
    
    if sign_flip:
        d_signs = generate_sign_flips(seed, d)
        a = a * d_signs
    
    a = _fwht(a, d, batch_size)
    
    if sign_flip:
        return a, d_signs
    
    return a


def ifwht_batch(a: np.ndarray, d_signs=None) -> np.ndarray:
    """
    Inverse Fast Walsh-Hadamard Transform for a batch of vectors.
    If sign_flip was applied in fwht_batch, the same d_signs must be provided to reverse it.
    """
    batch_size, d = a.shape
    
    a = _fwht(a, d, batch_size)
    
    if d_signs is not None:
        a = a * d_signs
    
    return a
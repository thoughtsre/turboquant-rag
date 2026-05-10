import numpy as np

def fwht_batch(a, seed=42) -> np.ndarray:
    """
    Fast Walsh-Hadamard Transform for a batch of vectors.
    Input 'a' should be shape (batch_size, d), where d is a power of 2.
    """
    a = a.astype(float)
    batch_size, d = a.shape
    
    if np.log2(d) % 1 != 0:
        d_pad = 2**int(np.ceil(np.log2(d)))
        a = np.pad(a, ((0, 0), (0, d_pad - d)), mode='constant')
        d = d_pad
        
    rng = np.random.default_rng(seed)
    d_signs = rng.choice([1.0, -1.0], size=d)
    a = a * d_signs
    
    n = int(np.log2(d))
    
    for i in range(n):
        stride = 2**i
        # Reshape to isolate the butterfly pairs within each vector
        a = a.reshape(batch_size, -1, 2, stride)
        # Apply the butterfly: sum and difference across the pairs
        a = np.stack([a[:, :, 0, :] + a[:, :, 1, :], 
                      a[:, :, 0, :] - a[:, :, 1, :]], axis=2)
    
    return a.reshape(batch_size, d) / np.sqrt(d)
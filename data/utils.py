import hashlib
import pickle

import numpy as np


def array_signature(array):
    length = len(array)
    assert length > 0
    middle = length // 2
    window = min(middle, 10)
    small_array = array[middle-window:middle+window]

    if isinstance(array, np.ndarray):
        assert len(small_array.shape) == 1
        small_array = small_array.tolist()

    assert isinstance(small_array, list)
    return f"{length}-{hashmd5(small_array)}"



def hashmd5(obj):
    """
    Hash an object into a deterministic string
    """
    return hashlib.md5(pickle.dumps(obj)).hexdigest()

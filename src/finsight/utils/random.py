import numpy as np


def create_rng(seed: int = 42) -> np.random.Generator:
    """
    Create a reproducible NumPy random number generator.
    """
    return np.random.default_rng(seed)
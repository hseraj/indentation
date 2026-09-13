import numpy as np

def to_log(physical):
    """Convert physical parameters to log10 space."""
    return np.log10(np.asarray(physical, dtype=float))

def from_log(log_params):
    """Convert log10 parameters back to physical space."""
    return 10.0 ** np.asarray(log_params, dtype=float)

def log_bounds_of(physical_bounds):
    """Convert physical bounds [(lo, hi), ...] to log10 space."""
    return [(np.log10(lo), np.log10(hi)) for (lo, hi) in physical_bounds]

def log_guesses_of(physical_guesses):
    """Convert list of physical guess tuples to log10 space."""
    return [tuple(to_log(g)) for g in physical_guesses]
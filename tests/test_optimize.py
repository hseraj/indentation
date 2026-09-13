import numpy as np
import sys
import os

# Add project root to path to import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from indentation.optimize import to_log, from_log, log_bounds_of, log_guesses_of

def test_round_trip():
    """Test that from_log(to_log(x)) == x."""
    # Physical parameters spanning many magnitudes
    physical_params = np.array([1e-6, 1.5, 1e3, 5e8, 2.2e6])
    
    # Convert to log space
    log_params = to_log(physical_params)
    
    # Convert back to physical space
    recovered_params = from_log(log_params)
    
    # Check if they are exactly the same
    if np.allclose(physical_params, recovered_params):
        print(f"[PASS] test_round_trip (Recovered: {recovered_params})")
    else:
        print(f"[FAIL] test_round_trip (Original: {physical_params}, Recovered: {recovered_params})")
        sys.exit(1)

def test_bounds_and_guesses():
    """Test conversion of bounds and guesses to log space."""
    # Physical bounds
    physical_bounds = [(1e-6, 1e9), (1e-3, 1e3)]
    log_bnds = log_bounds_of(physical_bounds)
    expected_bnds = [(-6.0, 9.0), (-3.0, 3.0)]
    
    if np.allclose(log_bnds, expected_bnds):
        print(f"[PASS] test_bounds_and_guesses (Bounds converted correctly)")
    else:
        print(f"[FAIL] test_bounds_and_guesses (Expected {expected_bnds}, Got {log_bnds})")
        sys.exit(1)
        
    # Physical guesses
    physical_guesses = [(1e6, 1.0), (0.5e6, 5.0)]
    log_guesses = log_guesses_of(physical_guesses)
    expected_guesses = [(6.0, 0.0), (np.log10(0.5e6), np.log10(5.0))]
    
    if np.allclose(log_guesses, expected_guesses):
        print(f"[PASS] test_bounds_and_guesses (Guesses converted correctly)")
    else:
        print(f"[FAIL] test_bounds_and_guesses (Expected {expected_guesses}, Got {log_guesses})")
        sys.exit(1)

if __name__ == "__main__":
    test_round_trip()
    test_bounds_and_guesses()
    print("\nTask 3 (optimize.py) Complete and Verified!")
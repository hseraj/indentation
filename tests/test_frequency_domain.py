import numpy as np
import sys
import os

# Add project root to path to import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from indentation.transforms import calculate_frequency_domain

def test_frequency_domain_math():
    """Test the frequency domain calculation for a Standard Linear Solid model."""
    # SLS Parameters: E1=1e6, E2=2e6, tau=0.5
    # Total E(t) = 1e6 + 2e6*exp(-t/0.5)
    # E_inf = 1e6, E1=2e6, tau=0.5
    params = {"E_inf [Pa]": 1e6, "E1 [Pa]": 2e6, "tau1 [s]": 0.5}
    
    # Test frequencies: 0.1 Hz, 1.0 Hz, 10.0 Hz
    freq_hz = np.array([0.1, 1.0, 10.0])
    
    result = calculate_frequency_domain("Prony_N1", params, freq_hz)
    
    # 1. Check that all keys exist
    required_keys = ["Frequency_Hz", "Angular_Frequency_rads", "Storage_Modulus_Pa", 
                     "Loss_Modulus_Pa", "Magnitude_Pa", "Loss_Tangent", "Damping_Ratio"]
    for key in required_keys:
        assert key in result, f"Missing key in frequency data: {key}"
        
    # 2. Check physical limits
    E_prime = result["Storage_Modulus_Pa"]
    E_double_prime = result["Loss_Modulus_Pa"]
    
    # At very low frequency (0.1 Hz, omega = 0.628 rad/s), E' should approach the relaxed modulus E_inf (1e6)
    # Because E_inf + E1 * (omega*tau)^2 / (1 + (omega*tau)^2) -> E_inf as omega -> 0
    # Actually, E'(0) = E_inf. At 0.1 Hz, it should be close to 1e6.
    if np.abs(E_prime[0] - 1e6) > 5e5:
        print(f"[FAIL] Low-frequency E' should approach E_inf. Got {E_prime[0]}")
        sys.exit(1)
        
    # At very high frequency (10 Hz, omega = 62.8 rad/s), E' should approach the instantaneous modulus E_inf + E1 (3e6)
    # Because E' -> E_inf + E1 as omega -> infinity
    if np.abs(E_prime[2] - 3e6) > 5e5:
        print(f"[FAIL] High-frequency E' should approach E_inf + E1. Got {E_prime[2]}")
        sys.exit(1)
        
    # Loss modulus E'' should be positive
    if np.any(E_double_prime < 0):
        print("[FAIL] Loss modulus E'' must be positive.")
        sys.exit(1)
        
    # Tan(delta) should be positive
    if np.any(result["Loss_Tangent"] < 0):
        print("[FAIL] Loss tangent must be positive.")
        sys.exit(1)

    print("[PASS] test_frequency_domain_math (Limits and sign checks passed)")

if __name__ == "__main__":
    test_frequency_domain_math()
    print("\nFrequency Domain Test Complete and Verified!")
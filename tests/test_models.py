import numpy as np
import sys
import os

# Add project root to path to import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from indentation.viscoelastic.models import MODELS

def test_registry():
    """Test that all 8 models are present in the registry."""
    expected_models = [
        "Elastic_Spring", "Dashpot", "Kelvin_Voigt", "Maxwell",
        "Standard_Linear_Solid", "Prony_N1", "Prony_N2", "Prony_N3"
    ]
    
    if list(MODELS.keys()) == expected_models:
        print(f"[PASS] test_registry (All 8 models present)")
    else:
        print(f"[FAIL] test_registry (Expected {expected_models}, Got {list(MODELS.keys())})")
        sys.exit(1)

def test_model_math():
    """Test the E*(s) and E(t) mathematical builders for a few models."""
    s_arr = np.array([1.0, 2.0, 5.0])
    t_arr = np.array([0.0, 1.0, 2.0])
    
    # 1. Test Maxwell
    E_star_func = MODELS["Maxwell"]["E_star_func"]
    E_relax_func = MODELS["Maxwell"]["E_relax_func"]
    
    E_val, tau_val = 1e6, 0.5
    Es = E_star_func(s_arr, E_val, tau_val)
    Et = E_relax_func(t_arr, E_val, tau_val)
    
    expected_Es = (E_val * tau_val * s_arr) / (1.0 + tau_val * s_arr)
    expected_Et = E_val * np.exp(-t_arr / tau_val)
    
    if np.allclose(Es, expected_Es) and np.allclose(Et, expected_Et):
        print(f"[PASS] test_model_math (Maxwell E* and E(t) correct)")
    else:
        print(f"[FAIL] test_model_math (Maxwell math incorrect)")
        sys.exit(1)
        
    # 2. Test Prony_N2
    E_star_func = MODELS["Prony_N2"]["E_star_func"]
    E_relax_func = MODELS["Prony_N2"]["E_relax_func"]
    
    Ei, E1, t1, E2, t2 = 0.5e6, 1e6, 0.1, 2e6, 5.0
    Es = E_star_func(s_arr, Ei, E1, t1, E2, t2)
    Et = E_relax_func(t_arr, Ei, E1, t1, E2, t2)
    
    expected_Es = Ei + (E1 * t1 * s_arr) / (1.0 + t1 * s_arr) + (E2 * t2 * s_arr) / (1.0 + t2 * s_arr)
    expected_Et = Ei + E1 * np.exp(-t_arr / t1) + E2 * np.exp(-t_arr / t2)
    
    if np.allclose(Es, expected_Es) and np.allclose(Et, expected_Et):
        print(f"[PASS] test_model_math (Prony_N2 E* and E(t) correct)")
    else:
        print(f"[FAIL] test_model_math (Prony_N2 math incorrect)")
        sys.exit(1)

if __name__ == "__main__":
    test_registry()
    test_model_math()
    print("\nTask 5 (models.py) Complete and Verified!")
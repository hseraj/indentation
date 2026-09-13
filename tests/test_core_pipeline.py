import numpy as np
import sys
import os

# Add project root to path to import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from indentation.transforms import compute_force
from indentation.core import run_fit

def test_full_pipeline():
    """Test the core run_fit pipeline for Maxwell model."""
    # 1. Generate Synthetic Data
    t_arr = np.linspace(0, 10, 500)
    dt = t_arr[1] - t_arr[0]
    h_arr = 0.1 * t_arr  # Linear ramp
    C = 1.0
    h_power_exp = 1.5
    
    E_true, tau_true = 1e6, 0.5
    E_relax_func = lambda t: E_true * np.exp(-t / tau_true)
    F_data = compute_force(E_relax_func, t_arr, h_arr, dt, C, h_power_exp)
    
    # 2. Run Pipeline
    S_VALS = np.sort(np.unique(np.logspace(-3, 3, 110)))
    results = run_fit(
        model_name="Maxwell", 
        t_full=t_arr, h_full=h_arr, F_full=F_data, 
        C=C, h_power_exp=h_power_exp, 
        S_VALS=S_VALS, 
        time_window=5.0,
        refine_full_data=True,
        smooth_window=0  # Explicitly disable smoothing for strict math test
    )
    
    # 3. Assertions
    # The best result should be the refined TD fit
    best = results[0]
    
    # Check that we got 3 results (Laplace, TD, Refined TD)
    if len(results) != 3:
        print(f"[FAIL] Expected 3 results, got {len(results)}")
        sys.exit(1)
        
    # Check that Refined TD was the most accurate
    if best["method"] == "Time-Domain (Refined)":
        print(f"[PASS] Best method is Time-Domain (Refined)")
    else:
        print(f"[FAIL] Best method was {best['method']}, expected Time-Domain (Refined)")
        sys.exit(1)
        
    # Check accuracy of the best fit
    if np.isclose(best["parameters"]["E [Pa]"], E_true, rtol=0.01) and np.isclose(best["parameters"]["tau [s]"], tau_true, rtol=0.01):
        print(f"[PASS] Refined TD parameters accurate (E={best['parameters']['E [Pa]']:.2f}, tau={best['parameters']['tau [s]']:.2f})")
    else:
        print(f"[FAIL] Refined TD parameters inaccurate (Got E={best['parameters']['E']}, tau={best['parameters']['tau']})")
        sys.exit(1)

if __name__ == "__main__":
    test_full_pipeline()
    print("\nTask 9 (core.py) Complete and Verified!")
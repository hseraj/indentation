import numpy as np
import sys
import os

# Add project root to path to import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from indentation.transforms import compute_force
from indentation.viscoelastic.fitting import fit_sls_lp, fit_sls_td

def test_sls_pipeline():
    """Test SLS Laplace and Time-Domain fitting with warm-start."""
    # 1. Generate Synthetic Data
    t_arr = np.linspace(0, 10, 500)
    dt = t_arr[1] - t_arr[0]
    h_arr = 0.1 * t_arr  # Linear ramp
    C = 1.0
    h_power_exp = 1.5
    
    E1_true, E2_true, tau_true = 1e6, 2e6, 0.5
    E_relax_func = lambda t: E1_true + E2_true * np.exp(-t / tau_true)
    F_data = compute_force(E_relax_func, t_arr, h_arr, dt, C, h_power_exp)
    
    # 2. Run Laplace Fit (Only used to generate warm-start guesses)
    S_VALS = np.sort(np.unique(np.logspace(-3, 3, 110)))
    lp_res = fit_sls_lp(t_arr, h_arr, F_data, dt, C, h_power_exp, S_VALS)
    print(f"[PASS] SLS Laplace Fit executed (Warm-start E1={lp_res['parameters']['E1 [Pa]']:.2f}, tau={lp_res['parameters']['tau [s]']:.2f})")
    
    # 3. Run Time-Domain Fit with Warm-Start
    td_res = fit_sls_td(t_arr, h_arr, F_data, dt, C, h_power_exp, lp_params=lp_res["parameters"])
    
    td_E1 = td_res["parameters"]["E1 [Pa]"]
    td_E2 = td_res["parameters"]["E2 [Pa]"]
    td_tau = td_res["parameters"]["tau [s]"]
    
    # Time-Domain should be extremely accurate (within 1% relative error)
    if np.isclose(td_E1, E1_true, rtol=0.01) and np.isclose(td_E2, E2_true, rtol=0.01) and np.isclose(td_tau, tau_true, rtol=0.01):
        print(f"[PASS] SLS Time-Domain Fit (E1={td_E1:.2f}, E2={td_E2:.2f}, tau={td_tau:.2f})")
    else:
        print(f"[FAIL] SLS Time-Domain Fit (Expected E1={E1_true}, E2={E2_true}, tau={tau_true}; Got E1={td_E1}, E2={td_E2}, tau={td_tau})")
        sys.exit(1)

if __name__ == "__main__":
    test_sls_pipeline()
    print("\nTask 7 (Nonlinear Fitters) Complete and Verified!")
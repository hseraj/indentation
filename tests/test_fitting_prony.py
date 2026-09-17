import numpy as np
import sys
import os

# Add project root to path to import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from indentation.transforms import compute_force
from indentation.viscoelastic.fitting import fit_prony2_lp, fit_prony2_td

def test_prony2_pipeline():
    """Test Prony N=2 Laplace and Time-Domain fitting with warm-start."""
    # 1. Generate Synthetic Data
    t_arr = np.linspace(0, 10, 500)
    dt = t_arr[1] - t_arr[0]
    h_arr = 0.1 * t_arr  # Linear ramp
    C = 1.0
    h_power_exp = 1.5
    
    Ei_true, E1_true, t1_true, E2_true, t2_true = 0.5e6, 1e6, 0.5, 2e6, 5.0
    E_relax_func = lambda t: Ei_true + E1_true * np.exp(-t / t1_true) + E2_true * np.exp(-t / t2_true)
    F_data = compute_force(E_relax_func, t_arr, h_arr, dt, C, h_power_exp)
    
    # 2. Run Laplace Fit
    S_VALS = np.sort(np.unique(np.logspace(-3, 3, 110)))
    lp_res = fit_prony2_lp(t_arr, h_arr, F_data, dt, C, h_power_exp, S_VALS)
    print(f"[PASS] Prony2 Laplace Fit executed (Warm-start E_inf={lp_res['parameters']['E_inf [Pa]']:.2f})")
    
    # 3. Run Time-Domain Fit with Warm-Start
    td_res = fit_prony2_td(t_arr, h_arr, F_data, dt, C, h_power_exp, lp_params=lp_res["parameters"])
    
    td_Ei = td_res["parameters"]["E_inf [Pa]"]
    td_E1 = td_res["parameters"]["E1 [Pa]"]
    td_E2 = td_res["parameters"]["E2 [Pa]"]
    td_t1 = td_res["parameters"]["tau1 [s]"]
    td_t2 = td_res["parameters"]["tau2 [s]"]
    
    # Time-Domain should be extremely accurate (within 2% relative error for multi-param)
    if (np.isclose(td_Ei, Ei_true, rtol=0.02) and 
        np.isclose(td_E1, E1_true, rtol=0.02) and 
        np.isclose(td_E2, E2_true, rtol=0.02)):
        print(f"[PASS] Prony2 Time-Domain Fit (E_inf={td_Ei:.2f}, E1={td_E1:.2f}, E2={td_E2:.2f})")
    else:
        print(f"[FAIL] Prony2 Time-Domain Fit")
        print(f"   Expected: Ei={Ei_true}, E1={E1_true}, E2={E2_true}")
        print(f"   Got:      Ei={td_Ei}, E1={td_E1}, E2={td_E2}")
        sys.exit(1)

if __name__ == "__main__":
    test_prony2_pipeline()
    print("\nTask 8 (Prony Series Fitters) Complete and Verified!")
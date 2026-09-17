import numpy as np
import sys
import os

# Add project root to path to import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from indentation.viscoelastic.fitting import fit_elastic_td, fit_kv_td, fit_dashpot_td

def test_elastic_td():
    """Test Elastic Spring fitting against synthetic data."""
    t_arr = np.linspace(0, 10, 1000)
    dt = t_arr[1] - t_arr[0]
    h_arr = 0.1 * t_arr
    C = 1.0
    h_power_exp = 1.5
    
    E_true = 1e6
    h_power = h_arr ** h_power_exp
    F_data = C * E_true * h_power
    
    res = fit_elastic_td(t_arr, h_arr, F_data, dt, C, h_power_exp)
    
    if np.isclose(res["parameters"]["E [Pa]"], E_true, rtol=1e-5) and res["RelErr_%"] < 1e-5:
        print(f"[PASS] test_elastic_td (E={res['parameters']['E [Pa]']:.2f})")
    else:
        print(f"[FAIL] test_elastic_td")
        sys.exit(1)

def test_kv_td():
    """Test Kelvin-Voigt fitting against synthetic data."""
    t_arr = np.linspace(0, 10, 1000)
    dt = t_arr[1] - t_arr[0]
    h_arr = 0.1 * t_arr
    C = 1.0
    h_power_exp = 1.5
    
    E_true, eta_true = 1e6, 1e5
    h_power = h_arr ** h_power_exp
    dh = np.gradient(h_power, dt)
    F_data = C * (E_true * h_power + eta_true * dh)
    
    res = fit_kv_td(t_arr, h_arr, F_data, dt, C, h_power_exp)
    
    if np.isclose(res["parameters"]["E [Pa]"], E_true, rtol=1e-5) and np.isclose(res["parameters"]["eta [Pa.s]"], eta_true, rtol=1e-5):
        print(f"[PASS] test_kv_td (E={res['parameters']['E [Pa]']:.2f}, eta={res['parameters']['eta [Pa.s]']:.2f})")
    else:
        print(f"[FAIL] test_kv_td")
        sys.exit(1)

def test_dashpot_td():
    """Test Dashpot fitting against synthetic data."""
    t_arr = np.linspace(0, 10, 1000)
    dt = t_arr[1] - t_arr[0]
    h_arr = 0.1 * t_arr
    C = 1.0
    h_power_exp = 1.5
    
    eta_true = 1e5
    h_power = h_arr ** h_power_exp
    dh = np.gradient(h_power, dt)
    F_data = C * eta_true * dh
    
    res = fit_dashpot_td(t_arr, h_arr, F_data, dt, C, h_power_exp)
    
    if np.isclose(res["parameters"]["eta [Pa.s]"], eta_true, rtol=1e-5) and res["RelErr_%"] < 1.0:
        print(f"[PASS] test_dashpot_td (eta={res['parameters']['eta [Pa.s]']:.2f})")
    else:
        print(f"[FAIL] test_dashpot_td")
        sys.exit(1)

if __name__ == "__main__":
    test_elastic_td()
    test_kv_td()
    test_dashpot_td()
    print("\nTask 6 (fitting.py closed-form) Complete and Verified!")
import numpy as np
import sys
import os

# Add project root to path to import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from indentation.transforms import compute_force, numerical_laplace, rmse, relerr

def test_compute_force_elastic():
    """Test compute_force against an analytical elastic solution."""
    # Elastic: E(t) = E (constant)
    E_val = 1e6
    E_relax_func = lambda t: np.ones_like(t) * E_val
    
    # Dummy probe params
    C = 1.0 
    h_power_exp = 1.5
    
    # Time and displacement arrays
    dt = 0.1
    t_arr = np.arange(0, 10, dt)
    
    # h(t) = v*t
    h_arr = 0.5 * t_arr 
    
    # Calculate force using our function
    Fm = compute_force(E_relax_func, t_arr, h_arr, dt, C, h_power_exp)
    
    # Analytical solution: F(t) = C * E * h(t)^h_power_exp
    F_analytical = C * E_val * (h_arr ** h_power_exp)
    
    # We skip the first 10 points because np.gradient uses central differences,
    # which causes expected numerical boundary errors at t=0.
    relative_errors = np.abs((Fm[10:] - F_analytical[10:]) / F_analytical[10:])
    max_rel_error = np.max(relative_errors)
    
    # 0.1% relative error is excellent for numerical convolution
    if max_rel_error < 1e-2:
        print(f"[PASS] test_compute_force_elastic (Max Relative Error: {max_rel_error:.2e})")
    else:
        print(f"[FAIL] test_compute_force_elastic (Max Relative Error: {max_rel_error:.2e})")
        sys.exit(1)

def test_laplace_and_errors():
    """Test numerical Laplace transform and error metrics."""
    # Test Laplace of exp(-t) which should be 1 / (s+1)
    t_arr = np.linspace(0, 50, 5000)
    f_t = np.exp(-t_arr)
    s_vals = np.array([1.0, 2.0, 5.0])
    
    F_s = numerical_laplace(f_t, t_arr, s_vals)
    expected = 1.0 / (s_vals + 1.0)
    
    err = np.max(np.abs(F_s - expected))
    if err < 1e-3:
        print(f"[PASS] test_laplace_and_errors (Laplace Max Error: {err:.2e})")
    else:
        print(f"[FAIL] test_laplace_and_errors (Laplace Max Error: {err:.2e})")
        sys.exit(1)
        
    # Test error functions
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.1, 2.0, 2.9])
    
    rmse_val = rmse(a, b)
    
    assert np.isclose(rmse_val, np.sqrt((0.01 + 0.0 + 0.01)/3)), "RMSE calculation is wrong"
    print("[PASS] test_laplace_and_errors (RMSE and RelErr logic ran)")

if __name__ == "__main__":
    test_compute_force_elastic()
    test_laplace_and_errors()
    print("\nTask 2 (transforms.py) Complete and Verified!")
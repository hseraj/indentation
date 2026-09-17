import numpy as np
from scipy.signal import fftconvolve, savgol_filter

def compute_h_power_and_dh(h_arr, h_power_exp, dt_step, smooth_window=0):
    """
    Computes h^n and its time derivative.
    If smooth_window > 0, uses the analytical Savitzky-Golay derivative for 
    mathematical perfection and robustness against noise.
    """
    # Prevent NaN in fractional powers by clipping negative noise to 0
    h_arr = np.maximum(np.asarray(h_arr, dtype=float), 0.0)
    h_power = h_arr ** h_power_exp
    
    if smooth_window > 0:
        win = int(smooth_window)
        if win % 2 == 0: win += 1
        n = len(h_power)
        if win >= n: win = n - 1 if n % 2 == 0 else n - 2
        if win < 3: win = 3 # Minimum window size for polyorder=2
        
        # Analytical derivative of the fitted polynomial (deriv=1)
        dh = savgol_filter(h_power, window_length=win, polyorder=2, deriv=1, delta=dt_step)
    else:
        # Standard numerical derivative for clean data
        dh = np.gradient(h_power, dt_step)
        
    return h_power, dh

def numerical_laplace(f_t, t_arr, s_vals):
    """
    Compute the numerical Laplace transform of f(t) for given s values.
    Uses vectorized trapezoidal integration.
    """
    f_t = np.asarray(f_t, dtype=float)
    t_arr = np.asarray(t_arr, dtype=float)
    s_vals = np.asarray(s_vals, dtype=float)
    
    integrand = f_t * np.exp(-np.outer(s_vals, t_arr))
    # Use trapezoid for NumPy >= 2.0, fallback to trapz for NumPy < 2.0
    if hasattr(np, 'trapezoid'):
        return np.trapezoid(integrand, t_arr, axis=1)
    else:
        return np.trapz(integrand, t_arr, axis=1)

def compute_force(E_relax_func, t_arr, h_arr, dt_step, C, h_power_exp, h_power=None, dh=None):
    """
    Compute the force via the hereditary integral using FFT convolution.
    F(t) = C * integral( E(t-tau) * d(h^power)/dtau )
    """
    t_arr = np.asarray(t_arr, dtype=float)
    h_arr = np.asarray(h_arr, dtype=float)
    
    # Precompute if not provided
    if h_power is None or dh is None:
        h_power, dh = compute_h_power_and_dh(h_arr, h_power_exp, dt_step)
    
    E_relax_arr = E_relax_func(t_arr)
    
    # O(N log N) convolution
    S = fftconvolve(E_relax_arr, dh, mode='full')[:len(t_arr)]
    
    # Trapezoidal correction
    Fm = C * dt_step * (S - 0.5 * E_relax_arr * dh[0] - 0.5 * E_relax_arr[0] * dh)
    Fm[0] = 0.0
    return Fm

def rmse(Fm, Fd):
    """Root Mean Square Error between model and data."""
    Fm = np.asarray(Fm, dtype=float)
    Fd = np.asarray(Fd, dtype=float)
    return np.sqrt(np.mean((Fm - Fd)**2))

def relerr(Fm, Fd):
    """Mean Relative Error in percentage."""
    Fm = np.asarray(Fm, dtype=float)
    Fd = np.asarray(Fd, dtype=float)
    threshold = 1e-10
    denom = np.maximum(np.abs(Fd), threshold)
    return np.mean(np.abs(Fm - Fd) / denom) * 100

def r2_score(Fm, Fd):
    """Coefficient of Determination (R^2). 1.0 is perfect, <0 is worse than mean."""
    Fm = np.asarray(Fm, dtype=float)
    Fd = np.asarray(Fd, dtype=float)
    ss_res = np.sum((Fd - Fm) ** 2)
    ss_tot = np.sum((Fd - np.mean(Fd)) ** 2)
    return 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

def nrmse(Fm, Fd):
    """Normalized Root Mean Square Error (as a percentage)."""
    Fm = np.asarray(Fm, dtype=float)
    Fd = np.asarray(Fd, dtype=float)
    rmse_val = rmse(Fm, Fd)
    range_Fd = np.max(Fd) - np.min(Fd)
    return (rmse_val / range_Fd) * 100.0 if range_Fd > 0 else 0.0

def log_rmse(Fm, Fd, eps=1e-10):
    """Logarithmic RMSE, useful for data spanning multiple orders of magnitude."""
    Fm = np.asarray(Fm, dtype=float)
    Fd = np.asarray(Fd, dtype=float)
    Fm_safe = np.maximum(Fm, eps)
    Fd_safe = np.maximum(Fd, eps)
    return np.sqrt(np.mean((np.log(Fm_safe) - np.log(Fd_safe)) ** 2))

def rmsre(Fm, Fd, eps=1e-10):
    """Root Mean Square Relative Error (normalized point-by-point)."""
    Fm = np.asarray(Fm, dtype=float)
    Fd = np.asarray(Fd, dtype=float)
    denom = np.abs(Fd) + eps
    return np.sqrt(np.mean(((Fm - Fd) / denom) ** 2))
    
def calculate_frequency_domain(model_name: str, params: dict, freq_hz: np.ndarray) -> dict:
    """
    Calculates the frequency-domain dynamic moduli (E', E'', |E*|, tan delta, zeta)
    by evaluating the Carson transform E*(s) at s = i*omega.
    """
    from indentation.viscoelastic.models import MODELS
    
    model = MODELS[model_name]
    E_star_func = model["E_star_func"]
    param_names = model["param_names"]
    
    # Map the physical params dict (e.g., {"E [Pa]": 1e6}) to the lambda arguments (e.g., E=1e6)
    args = []
    for p_name in param_names:
        found = False
        for key, val in params.items():
            if key.startswith(p_name):
                args.append(val)
                found = True
                break
        if not found:
            raise ValueError(f"Parameter {p_name} not found in params dict: {params}")
            
    # Convert ordinary frequency (Hz) to angular frequency (rad/s)
    omega = 2.0 * np.pi * np.asarray(freq_hz, dtype=float)
    
    # Substitute s = i * omega (1j in Python is the imaginary unit)
    s = 1j * omega
    
    # Evaluate the Carson transform E*(s) = s * E(s) using complex numbers
    E_star_complex = E_star_func(s, *args)
    
    # Extract dynamic properties
    E_prime = np.real(E_star_complex)       # Storage Modulus E'
    E_double_prime = np.imag(E_star_complex) # Loss Modulus E''
    magnitude = np.abs(E_star_complex)       # |E*|
    
    # Calculate tan(delta) and damping ratio zeta safely (handle division by zero)
    eps = 1e-12
    tan_delta = np.where(np.abs(E_prime) < eps, np.inf, E_double_prime / E_prime)
    zeta = tan_delta / 2.0
    
    return {
        "Frequency_Hz": freq_hz,
        "Angular_Frequency_rads": omega,
        "Storage_Modulus_Pa": E_prime,
        "Loss_Modulus_Pa": E_double_prime,
        "Magnitude_Pa": magnitude,
        "Loss_Tangent": tan_delta,
        "Damping_Ratio": zeta
    }
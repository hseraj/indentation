import numpy as np
from scipy.optimize import minimize
from indentation.transforms import numerical_laplace, compute_force, rmse, relerr, compute_h_power_and_dh
from indentation.optimize import from_log, log_bounds_of, log_guesses_of
from indentation.viscoelastic.models import MODELS

# =============================================================================
# DYNAMIC INITIAL GUESS GENERATOR
# =============================================================================

def generate_dynamic_guesses(model_name, t_arr, h_arr, F_data, C, h_power_exp):
    """Generates scale-invariant initial guesses based on experimental data."""
    h_power = h_arr ** h_power_exp
    max_F = np.max(np.abs(F_data))
    max_h_pow = np.max(np.abs(h_power))
    
    if max_F < 1e-12 or max_h_pow < 1e-12 or C < 1e-12:
        E_est = 1e6
    else:
        E_est = max_F / (C * max_h_pow)
        
    t_total = t_arr[-1] - t_arr[0]
    if t_total <= 0: t_total = 1.0
    tau_est = t_total / 2.0
    eta_est = E_est * t_total
    
    if model_name == "Elastic_Spring":
        return [[E_est], [0.5 * E_est], [2.0 * E_est]]
    elif model_name == "Dashpot":
        return [[eta_est], [0.5 * eta_est], [2.0 * eta_est]]
    elif model_name == "Kelvin_Voigt":
        return [[E_est, eta_est], [0.5 * E_est, 2.0 * eta_est], [2.0 * E_est, 0.5 * eta_est]]
    elif model_name == "Maxwell":
        return [
            [E_est, tau_est], [0.5 * E_est, 2.0 * tau_est], [2.0 * E_est, 0.5 * tau_est], 
            [E_est, 0.1 * tau_est], [E_est, 5.0 * tau_est]
        ]
    elif model_name == "Standard_Linear_Solid":
        return [
            [0.5 * E_est, 0.5 * E_est, tau_est], [0.2 * E_est, 0.8 * E_est, 2.0 * tau_est],
            [0.8 * E_est, 0.2 * E_est, 0.5 * tau_est], [0.5 * E_est, 0.5 * E_est, 0.1 * tau_est],
            [0.5 * E_est, 0.5 * E_est, 5.0 * tau_est]
        ]
    elif model_name == "Prony_N1":
        return [
            [0.5 * E_est, 0.5 * E_est, tau_est], [0.8 * E_est, 0.2 * E_est, 2.0 * tau_est],
            [0.2 * E_est, 0.8 * E_est, 0.5 * tau_est], [0.5 * E_est, 0.5 * E_est, 0.1 * tau_est]
        ]
    elif model_name == "Prony_N2":
        return [
            [0.5 * E_est, 0.25 * E_est, 0.3 * tau_est, 0.25 * E_est, 1.5 * tau_est],
            [0.2 * E_est, 0.4 * E_est, 0.1 * tau_est, 0.4 * E_est, 2.0 * tau_est],
            [0.8 * E_est, 0.1 * E_est, 0.5 * tau_est, 0.1 * E_est, 3.0 * tau_est]
        ]
    elif model_name == "Prony_N3":
        return [
            [0.4 * E_est, 0.2 * E_est, 0.2 * tau_est, 0.2 * E_est, tau_est, 0.2 * E_est, 2.0 * tau_est],
            [0.6 * E_est, 0.1 * E_est, 0.1 * tau_est, 0.2 * E_est, 0.5 * tau_est, 0.1 * E_est, 2.0 * tau_est]
        ]
    return []

# =============================================================================
# CONFIDENCE INTERVAL CALCULATORS
# =============================================================================

def _calc_linear_ci(X, y, params):
    """Calculate CI for linear least squares models."""
    N, p = X.shape
    residuals = y - X @ params
    rss = np.sum(residuals**2)
    sigma2 = rss / (N - p) if N > p else 0.0
    try:
        cov = sigma2 * np.linalg.inv(X.T @ X)
        se = np.sqrt(np.diag(cov))
        ci = 1.96 * se
    except np.linalg.LinAlgError:
        se = np.zeros_like(params)
        ci = np.zeros_like(params)
    return {"Standard_Error": se, "95%_CI": ci}

def _calc_nonlinear_ci(obj, log_params, N, p):
    """Calculate CI for nonlinear models using numerical Hessian."""
    eps = 1e-4
    n_params = len(log_params)
    H = np.zeros((n_params, n_params))
    for i in range(n_params):
        for j in range(n_params):
            e_i = np.zeros(n_params); e_i[i] = eps
            e_j = np.zeros(n_params); e_j[j] = eps
            H[i, j] = (obj(log_params + e_i + e_j) - obj(log_params + e_i - e_j) -
                       obj(log_params - e_i + e_j) + obj(log_params - e_i - e_j)) / (4 * eps**2)
    
    H = (H + H.T) / 2  # Force symmetric
    rss = obj(log_params) * N  # obj is MSE
    sigma2 = rss / (N - p) if N > p else 0.0
    
    try:
        cov_log = 2 * sigma2 * np.linalg.pinv(H)
        se_log = np.sqrt(np.abs(np.diag(cov_log)))
        se_phys = from_log(log_params) * np.log(10) * se_log
        ci_phys = 1.96 * se_phys
        return {"Standard_Error": se_phys, "95%_CI": ci_phys, "cov_log": cov_log}
    except Exception:
        return {"Standard_Error": np.zeros(n_params), "95%_CI": np.zeros(n_params), "cov_log": np.zeros((n_params, n_params))}

def _propagate_eta_ci(E_val, tau_val, eta_val, cov_log, idx_E, idx_tau):
    """
    Propagate standard error for eta = E * tau using the log-space covariance matrix.
    Var(log_eta) = Var(log_E) + Var(log_tau) + 2*Cov(log_E, log_tau)
    """
    var_log_eta = cov_log[idx_E, idx_E] + cov_log[idx_tau, idx_tau] + 2 * cov_log[idx_E, idx_tau]
    se_log_eta = np.sqrt(np.abs(var_log_eta))
    se_eta = eta_val * np.log(10) * se_log_eta
    ci_eta = 1.96 * se_eta
    return se_eta, ci_eta

# =============================================================================
# GENERIC LAPLACE FITTER
# =============================================================================

def fit_laplace_generic(E_star_model_func, initial_guesses, bounds, 
                        t_arr, h_arr, F_data, C, h_power_exp, S_VALS):
    h_power = h_arr ** h_power_exp
    h_power_s = numerical_laplace(h_power, t_arr, S_VALS)
    F_s = numerical_laplace(F_data, t_arr, S_VALS)

    valid = h_power_s > 1e-15
    s_valid = S_VALS[valid]
    # We compute the Carson Transform: s*E(s) = F(s) / (C * H^n(s))
    E_star_exp = F_s[valid] / (C * h_power_s[valid])

    def obj(log_params):
        params = from_log(log_params)
        E_model = E_star_model_func(s_valid, *params)
        E_model_safe = np.maximum(E_model, 1e-10)
        E_data_safe = np.maximum(E_star_exp, 1e-10)
        residuals = np.log(E_model_safe) - np.log(E_data_safe)
        return np.sum(residuals**2)

    log_bnds = log_bounds_of(bounds)
    log_guesses = log_guesses_of(initial_guesses)

    best = None
    best_cost = float('inf')
    for guess in log_guesses:
        res = minimize(obj, guess, method='L-BFGS-B', bounds=log_bnds,
                       options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-10})
        if res.fun < best_cost:
            best_cost = res.fun
            best = res

    best.x = from_log(best.x)
    return best

# =============================================================================
# OBJECTIVE FUNCTION
# =============================================================================

def objective_mse(params, E_relax_func_builder, t_arr, h_arr, F_data, dt_step, C, h_power_exp, h_power, dh):
    E_relax_func = E_relax_func_builder(*params)
    Fm = compute_force(E_relax_func, t_arr, h_arr, dt_step, C, h_power_exp, h_power=h_power, dh=dh)
    return np.mean((Fm - F_data)**2)

# =============================================================================
# CLOSED-FORM FITTERS
# =============================================================================

def fit_elastic_td(t_arr, h_arr, F_data, dt_step, C, h_power_exp, lp_params=None, smooth_window=0):
    h_power, _ = compute_h_power_and_dh(h_arr, h_power_exp, dt_step, smooth_window)
    X = (C * h_power).reshape(-1, 1)
    F_data_safe = np.nan_to_num(F_data)
    E = np.linalg.lstsq(X, F_data_safe, rcond=None)[0][0]
    Fm = C * E * h_power
    Fm[0] = 0.0
    return {"model": "Elastic_Spring", "method": "Time-Domain", "parameters": {"E [Pa]": E}, 
            "F_model": Fm, "RMSE_N": rmse(Fm, F_data), "RelErr_%": relerr(Fm, F_data), "CI": _calc_linear_ci(X, F_data_safe, np.array([E]))}

def fit_elastic_lp(t_arr, h_arr, F_data, dt_step, C, h_power_exp, S_VALS, bounds=None):
    h_power = h_arr ** h_power_exp
    h_power_s = numerical_laplace(h_power, t_arr, S_VALS)
    F_s = numerical_laplace(F_data, t_arr, S_VALS)
    valid = h_power_s > 1e-15
    E_star_exp = F_s[valid] / (C * h_power_s[valid])
    E = np.median(E_star_exp)
    Fm = C * E * (h_arr ** h_power_exp)
    Fm[0] = 0.0
    return {"model": "Elastic_Spring", "method": "Laplace", "parameters": {"E [Pa]": E}, 
            "F_model": Fm, "RMSE_N": rmse(Fm, F_data), "RelErr_%": relerr(Fm, F_data)}

def fit_dashpot_td(t_arr, h_arr, F_data, dt_step, C, h_power_exp, lp_params=None, smooth_window=0):
    _, dh = compute_h_power_and_dh(h_arr, h_power_exp, dt_step, smooth_window)
    X = (C * dh).reshape(-1, 1)
    F_data_safe = np.nan_to_num(F_data)
    eta = np.linalg.lstsq(X, F_data_safe, rcond=None)[0][0]
    Fm = C * eta * dh
    Fm[0] = 0.0
    return {"model": "Dashpot", "method": "Time-Domain", "parameters": {"eta [Pa.s]": eta}, 
            "F_model": Fm, "RMSE_N": rmse(Fm, F_data), "RelErr_%": relerr(Fm, F_data), "CI": _calc_linear_ci(X, F_data_safe, np.array([eta]))}

def fit_dashpot_lp(t_arr, h_arr, F_data, dt_step, C, h_power_exp, S_VALS, bounds=None):
    if bounds is None:
        bounds = MODELS["Dashpot"]["bounds"]
    h_power = h_arr ** h_power_exp
    h_power_s = numerical_laplace(h_power, t_arr, S_VALS)
    F_s = numerical_laplace(F_data, t_arr, S_VALS)
    valid = (h_power_s > 1e-15) & (S_VALS > 1e-6)
    s_valid = S_VALS[valid]
    E_star_exp = F_s[valid] / (C * h_power_s[valid])
    eta_estimates = E_star_exp / s_valid
    mid_range = (s_valid > 0.1) & (s_valid < 10)
    eta = np.median(eta_estimates[mid_range]) if np.any(mid_range) else np.median(eta_estimates)
    eta = np.clip(eta, bounds[0][0], bounds[0][1])
    dh = np.gradient(h_power, dt_step)
    Fm = C * eta * dh
    Fm[0] = 0.0
    return {"model": "Dashpot", "method": "Laplace", "parameters": {"eta [Pa.s]": eta}, 
            "F_model": Fm, "RMSE_N": rmse(Fm, F_data), "RelErr_%": relerr(Fm, F_data)}

def fit_kv_td(t_arr, h_arr, F_data, dt_step, C, h_power_exp, lp_params=None, smooth_window=0):
    h_power, dh = compute_h_power_and_dh(h_arr, h_power_exp, dt_step, smooth_window)
    X = np.column_stack([C * h_power, C * dh])
    F_data_safe = np.nan_to_num(F_data)
    coeffs = np.linalg.lstsq(X, F_data_safe, rcond=None)[0]
    E, eta = coeffs[0], coeffs[1]
    Fm = C * (E * h_power + eta * dh)
    Fm[0] = 0.0
    return {"model": "Kelvin_Voigt", "method": "Time-Domain", "parameters": {"E [Pa]": E, "eta [Pa.s]": eta}, 
            "F_model": Fm, "RMSE_N": rmse(Fm, F_data), "RelErr_%": relerr(Fm, F_data), "CI": _calc_linear_ci(X, F_data_safe, coeffs)}

def fit_kv_lp(t_arr, h_arr, F_data, dt_step, C, h_power_exp, S_VALS, bounds=None):
    E_star_model = MODELS["Kelvin_Voigt"]["E_star_func"]
    initial_guesses = generate_dynamic_guesses("Kelvin_Voigt", t_arr, h_arr, F_data, C, h_power_exp)
    bounds = MODELS["Kelvin_Voigt"]["bounds"]
    res = fit_laplace_generic(E_star_model, initial_guesses, bounds, t_arr, h_arr, F_data, C, h_power_exp, S_VALS)
    E, eta = res.x
    h_power = h_arr ** h_power_exp
    dh = np.gradient(h_power, dt_step)
    Fm = C * (E * h_power + eta * dh)
    Fm[0] = 0.0
    return {"model": "Kelvin_Voigt", "method": "Laplace", "parameters": {"E [Pa]": E, "eta [Pa.s]": eta}, 
            "F_model": Fm, "RMSE_N": rmse(Fm, F_data), "RelErr_%": relerr(Fm, F_data)}

# =============================================================================
# NONLINEAR FITTERS
# =============================================================================

def fit_maxwell_lp(t_arr, h_arr, F_data, dt_step, C, h_power_exp, S_VALS, bounds=None):
    E_star_model = MODELS["Maxwell"]["E_star_func"]
    initial_guesses = generate_dynamic_guesses("Maxwell", t_arr, h_arr, F_data, C, h_power_exp)
    bounds = MODELS["Maxwell"]["bounds"]
    res = fit_laplace_generic(E_star_model, initial_guesses, bounds, t_arr, h_arr, F_data, C, h_power_exp, S_VALS)
    E, tau = res.x
    eta = E * tau
    Fm = compute_force(lambda tv: E * np.exp(-tv / tau), t_arr, h_arr, dt_step, C, h_power_exp)
    return {"model": "Maxwell", "method": "Laplace", "parameters": {"E [Pa]": E, "eta [Pa.s]": eta, "tau [s]": tau}, 
            "F_model": Fm, "RMSE_N": rmse(Fm, F_data), "RelErr_%": relerr(Fm, F_data)}

def fit_maxwell_td(t_arr, h_arr, F_data, dt_step, C, h_power_exp, lp_params=None, smooth_window=0):
    def builder(E, tau): return lambda tv: E * np.exp(-tv / tau)
    h_power, dh = compute_h_power_and_dh(h_arr, h_power_exp, dt_step, smooth_window)
    def obj(log_p): return objective_mse(from_log(log_p), builder, t_arr, h_arr, F_data, dt_step, C, h_power_exp, h_power, dh)

    physical_guesses = generate_dynamic_guesses("Maxwell", t_arr, h_arr, F_data, C, h_power_exp)
    if lp_params: 
        physical_guesses = [(lp_params["E [Pa]"], lp_params["tau [s]"])] + physical_guesses

    bounds = MODELS["Maxwell"]["bounds"]
    best = min([minimize(obj, g, method='L-BFGS-B', bounds=log_bounds_of(bounds), options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-12}) for g in log_guesses_of(physical_guesses)], key=lambda r: r.fun)

    E, tau = from_log(best.x)
    eta = E * tau
    Fm = compute_force(builder(E, tau), t_arr, h_arr, dt_step, C, h_power_exp, h_power=h_power, dh=dh)
    
    ci_dict = _calc_nonlinear_ci(obj, best.x, len(t_arr), 2)
    se_eta, ci_eta = _propagate_eta_ci(E, tau, eta, ci_dict["cov_log"], 0, 1)
    se = [ci_dict["Standard_Error"][0], se_eta, ci_dict["Standard_Error"][1]]
    ci = [ci_dict["95%_CI"][0], ci_eta, ci_dict["95%_CI"][1]]
    ci_dict_final = {"Standard_Error": se, "95%_CI": ci}
    
    return {"model": "Maxwell", "method": "Time-Domain", "parameters": {"E [Pa]": E, "eta [Pa.s]": eta, "tau [s]": tau}, 
            "F_model": Fm, "RMSE_N": rmse(Fm, F_data), "RelErr_%": relerr(Fm, F_data), "CI": ci_dict_final}

def fit_sls_lp(t_arr, h_arr, F_data, dt_step, C, h_power_exp, S_VALS, bounds=None):
    E_star_model = MODELS["Standard_Linear_Solid"]["E_star_func"]
    initial_guesses = generate_dynamic_guesses("Standard_Linear_Solid", t_arr, h_arr, F_data, C, h_power_exp)
    bounds = MODELS["Standard_Linear_Solid"]["bounds"]
    res = fit_laplace_generic(E_star_model, initial_guesses, bounds, t_arr, h_arr, F_data, C, h_power_exp, S_VALS)
    E1, E2, tau = res.x
    eta = E2 * tau
    Fm = compute_force(lambda tv: E1 + E2 * np.exp(-tv / tau), t_arr, h_arr, dt_step, C, h_power_exp)
    return {"model": "Standard_Linear_Solid", "method": "Laplace", "parameters": {"E1 [Pa]": E1, "E2 [Pa]": E2, "eta [Pa.s]": eta, "tau [s]": tau}, 
            "F_model": Fm, "RMSE_N": rmse(Fm, F_data), "RelErr_%": relerr(Fm, F_data)}

def fit_sls_td(t_arr, h_arr, F_data, dt_step, C, h_power_exp, lp_params=None, smooth_window=0):
    def builder(E1, E2, tau): return lambda tv: E1 + E2 * np.exp(-tv / tau)
    h_power, dh = compute_h_power_and_dh(h_arr, h_power_exp, dt_step, smooth_window)
    def obj(log_p): return objective_mse(from_log(log_p), builder, t_arr, h_arr, F_data, dt_step, C, h_power_exp, h_power, dh)

    physical_guesses = generate_dynamic_guesses("Standard_Linear_Solid", t_arr, h_arr, F_data, C, h_power_exp)
    if lp_params: 
        physical_guesses = [(lp_params["E1 [Pa]"], lp_params["E2 [Pa]"], lp_params["tau [s]"])] + physical_guesses

    bounds = MODELS["Standard_Linear_Solid"]["bounds"]
    best = min([minimize(obj, g, method='L-BFGS-B', bounds=log_bounds_of(bounds), options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-12}) for g in log_guesses_of(physical_guesses)], key=lambda r: r.fun)

    E1, E2, tau = from_log(best.x)
    eta = E2 * tau
    Fm = compute_force(builder(E1, E2, tau), t_arr, h_arr, dt_step, C, h_power_exp, h_power=h_power, dh=dh)
    
    ci_dict = _calc_nonlinear_ci(obj, best.x, len(t_arr), 3)
    se_eta, ci_eta = _propagate_eta_ci(E2, tau, eta, ci_dict["cov_log"], 1, 2)
    se = [ci_dict["Standard_Error"][0], ci_dict["Standard_Error"][1], se_eta, ci_dict["Standard_Error"][2]]
    ci = [ci_dict["95%_CI"][0], ci_dict["95%_CI"][1], ci_eta, ci_dict["95%_CI"][2]]
    ci_dict_final = {"Standard_Error": se, "95%_CI": ci}
    
    return {"model": "Standard_Linear_Solid", "method": "Time-Domain", "parameters": {"E1 [Pa]": E1, "E2 [Pa]": E2, "eta [Pa.s]": eta, "tau [s]": tau}, 
            "F_model": Fm, "RMSE_N": rmse(Fm, F_data), "RelErr_%": relerr(Fm, F_data), "CI": ci_dict_final}

# =============================================================================
# PRONY SERIES FITTERS
# =============================================================================

def fit_prony1_lp(t_arr, h_arr, F_data, dt_step, C, h_power_exp, S_VALS, bounds=None):
    E_star_model = MODELS["Prony_N1"]["E_star_func"]
    initial_guesses = generate_dynamic_guesses("Prony_N1", t_arr, h_arr, F_data, C, h_power_exp)
    bounds = MODELS["Prony_N1"]["bounds"]
    res = fit_laplace_generic(E_star_model, initial_guesses, bounds, t_arr, h_arr, F_data, C, h_power_exp, S_VALS)
    Ei, E1, t1 = res.x
    Fm = compute_force(lambda tv: Ei + E1 * np.exp(-tv / t1), t_arr, h_arr, dt_step, C, h_power_exp)
    return {"model": "Prony_N1", "method": "Laplace", "parameters": {"E_inf [Pa]": Ei, "E1 [Pa]": E1, "tau1 [s]": t1}, 
            "F_model": Fm, "RMSE_N": rmse(Fm, F_data), "RelErr_%": relerr(Fm, F_data)}

def fit_prony1_td(t_arr, h_arr, F_data, dt_step, C, h_power_exp, lp_params=None, smooth_window=0):
    def builder(Ei, E1, t1): return lambda tv: Ei + E1 * np.exp(-tv / t1)
    h_power, dh = compute_h_power_and_dh(h_arr, h_power_exp, dt_step, smooth_window)
    def obj(log_p): return objective_mse(from_log(log_p), builder, t_arr, h_arr, F_data, dt_step, C, h_power_exp, h_power, dh)

    physical_guesses = generate_dynamic_guesses("Prony_N1", t_arr, h_arr, F_data, C, h_power_exp)
    if lp_params: 
        physical_guesses = [(lp_params["E_inf [Pa]"], lp_params["E1 [Pa]"], lp_params["tau1 [s]"])] + physical_guesses

    bounds = MODELS["Prony_N1"]["bounds"]
    best = min([minimize(obj, g, method='L-BFGS-B', bounds=log_bounds_of(bounds), options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-12}) for g in log_guesses_of(physical_guesses)], key=lambda r: r.fun)

    Ei, E1, t1 = from_log(best.x)
    Fm = compute_force(builder(Ei, E1, t1), t_arr, h_arr, dt_step, C, h_power_exp, h_power=h_power, dh=dh)
    
    ci_dict = _calc_nonlinear_ci(obj, best.x, len(t_arr), 3)
    return {"model": "Prony_N1", "method": "Time-Domain", "parameters": {"E_inf [Pa]": Ei, "E1 [Pa]": E1, "tau1 [s]": t1}, 
            "F_model": Fm, "RMSE_N": rmse(Fm, F_data), "RelErr_%": relerr(Fm, F_data), "CI": ci_dict}

def fit_prony2_lp(t_arr, h_arr, F_data, dt_step, C, h_power_exp, S_VALS, bounds=None):
    E_star_model = MODELS["Prony_N2"]["E_star_func"]
    initial_guesses = generate_dynamic_guesses("Prony_N2", t_arr, h_arr, F_data, C, h_power_exp)
    bounds = MODELS["Prony_N2"]["bounds"]
    res = fit_laplace_generic(E_star_model, initial_guesses, bounds, t_arr, h_arr, F_data, C, h_power_exp, S_VALS)
    Ei, E1, t1, E2, t2 = res.x
    Fm = compute_force(lambda tv: Ei + E1*np.exp(-tv/t1) + E2*np.exp(-tv/t2), t_arr, h_arr, dt_step, C, h_power_exp)
    return {"model": "Prony_N2", "method": "Laplace", "parameters": {"E_inf [Pa]": Ei, "E1 [Pa]": E1, "tau1 [s]": t1, "E2 [Pa]": E2, "tau2 [s]": t2}, 
            "F_model": Fm, "RMSE_N": rmse(Fm, F_data), "RelErr_%": relerr(Fm, F_data)}

def fit_prony2_td(t_arr, h_arr, F_data, dt_step, C, h_power_exp, lp_params=None, smooth_window=0):
    def builder(Ei, E1, t1, E2, t2): return lambda tv: Ei + E1 * np.exp(-tv / t1) + E2 * np.exp(-tv / t2)
    h_power, dh = compute_h_power_and_dh(h_arr, h_power_exp, dt_step, smooth_window)
    def obj(log_p): return objective_mse(from_log(log_p), builder, t_arr, h_arr, F_data, dt_step, C, h_power_exp, h_power, dh)

    physical_guesses = generate_dynamic_guesses("Prony_N2", t_arr, h_arr, F_data, C, h_power_exp)
    if lp_params: 
        physical_guesses = [(lp_params["E_inf [Pa]"], lp_params["E1 [Pa]"], lp_params["tau1 [s]"], lp_params["E2 [Pa]"], lp_params["tau2 [s]"])] + physical_guesses

    bounds = MODELS["Prony_N2"]["bounds"]
    best = min([minimize(obj, g, method='L-BFGS-B', bounds=log_bounds_of(bounds), options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-12}) for g in log_guesses_of(physical_guesses)], key=lambda r: r.fun)

    Ei, E1, t1, E2, t2 = from_log(best.x)
    Fm = compute_force(builder(Ei, E1, t1, E2, t2), t_arr, h_arr, dt_step, C, h_power_exp, h_power=h_power, dh=dh)
    
    ci_dict = _calc_nonlinear_ci(obj, best.x, len(t_arr), 5)
    return {"model": "Prony_N2", "method": "Time-Domain", "parameters": {"E_inf [Pa]": Ei, "E1 [Pa]": E1, "tau1 [s]": t1, "E2 [Pa]": E2, "tau2 [s]": t2}, 
            "F_model": Fm, "RMSE_N": rmse(Fm, F_data), "RelErr_%": relerr(Fm, F_data), "CI": ci_dict}

def fit_prony3_lp(t_arr, h_arr, F_data, dt_step, C, h_power_exp, S_VALS, bounds=None):
    E_star_model = MODELS["Prony_N3"]["E_star_func"]
    initial_guesses = generate_dynamic_guesses("Prony_N3", t_arr, h_arr, F_data, C, h_power_exp)
    bounds = MODELS["Prony_N3"]["bounds"]
    res = fit_laplace_generic(E_star_model, initial_guesses, bounds, t_arr, h_arr, F_data, C, h_power_exp, S_VALS)
    Ei, E1, t1, E2, t2, E3, t3 = res.x
    Fm = compute_force(lambda tv: Ei + E1*np.exp(-tv/t1) + E2*np.exp(-tv/t2) + E3*np.exp(-tv/t3), t_arr, h_arr, dt_step, C, h_power_exp)
    return {"model": "Prony_N3", "method": "Laplace", "parameters": {"E_inf [Pa]": Ei, "E1 [Pa]": E1, "tau1 [s]": t1, "E2 [Pa]": E2, "tau2 [s]": t2, "E3 [Pa]": E3, "tau3 [s]": t3}, 
            "F_model": Fm, "RMSE_N": rmse(Fm, F_data), "RelErr_%": relerr(Fm, F_data)}

def fit_prony3_td(t_arr, h_arr, F_data, dt_step, C, h_power_exp, lp_params=None, smooth_window=0):
    def builder(Ei, E1, t1, E2, t2, E3, t3): return lambda tv: Ei + E1*np.exp(-tv/t1) + E2*np.exp(-tv/t2) + E3*np.exp(-tv/t3)
    h_power, dh = compute_h_power_and_dh(h_arr, h_power_exp, dt_step, smooth_window)
    def obj(log_p): return objective_mse(from_log(log_p), builder, t_arr, h_arr, F_data, dt_step, C, h_power_exp, h_power, dh)

    physical_guesses = generate_dynamic_guesses("Prony_N3", t_arr, h_arr, F_data, C, h_power_exp)
    if lp_params: 
        physical_guesses = [(lp_params["E_inf [Pa]"], lp_params["E1 [Pa]"], lp_params["tau1 [s]"], lp_params["E2 [Pa]"], lp_params["tau2 [s]"], lp_params["E3 [Pa]"], lp_params["tau3 [s]"])] + physical_guesses

    bounds = MODELS["Prony_N3"]["bounds"]
    best = min([minimize(obj, g, method='L-BFGS-B', bounds=log_bounds_of(bounds), options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-12}) for g in log_guesses_of(physical_guesses)], key=lambda r: r.fun)

    Ei, E1, t1, E2, t2, E3, t3 = from_log(best.x)
    Fm = compute_force(builder(Ei, E1, t1, E2, t2, E3, t3), t_arr, h_arr, dt_step, C, h_power_exp, h_power=h_power, dh=dh)
    
    ci_dict = _calc_nonlinear_ci(obj, best.x, len(t_arr), 7)
    return {"model": "Prony_N3", "method": "Time-Domain", "parameters": {"E_inf [Pa]": Ei, "E1 [Pa]": E1, "tau1 [s]": t1, "E2 [Pa]": E2, "tau2 [s]": t2, "E3 [Pa]": E3, "tau3 [s]": t3}, 
            "F_model": Fm, "RMSE_N": rmse(Fm, F_data), "RelErr_%": relerr(Fm, F_data), "CI": ci_dict}
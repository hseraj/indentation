import numpy as np
from scipy.signal import savgol_filter
from indentation.viscoelastic import fitting
from indentation.transforms import rmse, relerr, r2_score, nrmse, log_rmse, rmsre, compute_h_power_and_dh

FUNCTION_MAP: dict[str, dict[str, object]] = {
    "Elastic_Spring": {"td": fitting.fit_elastic_td, "lp": fitting.fit_elastic_lp},
    "Dashpot": {"td": fitting.fit_dashpot_td, "lp": fitting.fit_dashpot_lp},
    "Kelvin_Voigt": {"td": fitting.fit_kv_td, "lp": fitting.fit_kv_lp},
    "Maxwell": {"td": fitting.fit_maxwell_td, "lp": fitting.fit_maxwell_lp},
    "Standard_Linear_Solid": {"td": fitting.fit_sls_td, "lp": fitting.fit_sls_lp},
    "Prony_N1": {"td": fitting.fit_prony1_td, "lp": fitting.fit_prony1_lp},
    "Prony_N2": {"td": fitting.fit_prony2_td, "lp": fitting.fit_prony2_lp},
    "Prony_N3": {"td": fitting.fit_prony3_td, "lp": fitting.fit_prony3_lp},
}

def compute_full_force(model_name: str, params: dict, t_full: np.ndarray, h_full: np.ndarray, dt_full: float, C: float, h_power_exp: float, smooth_window: int = 0) -> np.ndarray:
    """
    Recalculate force over the full time array using the fitted parameters.
    Uses the analytical SG derivative if smooth_window > 0 for consistency.
    """
    h_power, dh = compute_h_power_and_dh(h_full, h_power_exp, dt_full, smooth_window)
    
    if model_name == "Elastic_Spring":
        Fm = C * params["E [Pa]"] * h_power
        Fm[0] = 0.0
        return Fm
    elif model_name == "Dashpot":
        Fm = C * params["eta [Pa.s]"] * dh
        Fm[0] = 0.0
        return Fm
    elif model_name == "Kelvin_Voigt":
        Fm = C * (params["E [Pa]"] * h_power + params["eta [Pa.s]"] * dh)
        Fm[0] = 0.0
        return Fm
    elif model_name == "Maxwell":
        return fitting.compute_force(lambda tv: params["E [Pa]"] * np.exp(-tv / params["tau [s]"]), t_full, h_full, dt_full, C, h_power_exp, h_power=h_power, dh=dh)
    elif model_name == "Standard_Linear_Solid":
        return fitting.compute_force(lambda tv: params["E1 [Pa]"] + params["E2 [Pa]"] * np.exp(-tv / params["tau [s]"]), t_full, h_full, dt_full, C, h_power_exp, h_power=h_power, dh=dh)
    elif model_name == "Prony_N1":
        return fitting.compute_force(lambda tv: params["E_inf [Pa]"] + params["E1 [Pa]"] * np.exp(-tv / params["tau1 [s]"]), t_full, h_full, dt_full, C, h_power_exp, h_power=h_power, dh=dh)
    elif model_name == "Prony_N2":
        return fitting.compute_force(lambda tv: params["E_inf [Pa]"] + params["E1 [Pa]"] * np.exp(-tv / params["tau1 [s]"]) + params["E2 [Pa]"] * np.exp(-tv / params["tau2 [s]"]), t_full, h_full, dt_full, C, h_power_exp, h_power=h_power, dh=dh)
    elif model_name == "Prony_N3":
        return fitting.compute_force(lambda tv: params["E_inf [Pa]"] + params["E1 [Pa]"] * np.exp(-tv / params["tau1 [s]"]) + params["E2 [Pa]"] * np.exp(-tv / params["tau2 [s]"]) + params["E3 [Pa]"] * np.exp(-tv / params["tau3 [s]"]), t_full, h_full, dt_full, C, h_power_exp, h_power=h_power, dh=dh)
    raise ValueError(f"Unknown model: {model_name}")

def run_fit(model_name: str, t_full: np.ndarray, h_full: np.ndarray, F_full: np.ndarray, C: float, h_power_exp: float, S_VALS: np.ndarray, time_window: float | None = None, refine_full_data: bool = False, smooth_window: int = 0) -> list[dict]:
    """
    Main pipeline to run Laplace and Time-Domain fits.
    """
    if model_name not in FUNCTION_MAP:
        raise ValueError(f"Unknown model: {model_name}")
        
    dt_full = t_full[1] - t_full[0]
    
    # Apply Savitzky-Golay Smoothing to Force to remove load cell noise
    if smooth_window > 0:
        win = int(smooth_window)
        if win % 2 == 0: win += 1
        n = len(t_full)
        if win >= n: win = n - 1 if n % 2 == 0 else n - 2
            
        F_full = savgol_filter(F_full, window_length=win, polyorder=2)
        print(f"[Info] Applied Savitzky-Golay smoothing (window={win}). Force smoothed and Disp derivative uses analytical SG derivative.")
    
    def get_all_metrics(Fm: np.ndarray, Fd: np.ndarray) -> dict:
        return {
            "RMSE_N": rmse(Fm, Fd),
            "RelErr_%": relerr(Fm, Fd),
            "R2": r2_score(Fm, Fd),
            "NRMSE_%": nrmse(Fm, Fd),
            "Log_RMSE": log_rmse(Fm, Fd),
            "RMSRE": rmsre(Fm, Fd)
        }
    
    if time_window is not None and time_window < t_full[-1]:
        end_idx = np.searchsorted(t_full, time_window)
        t_sub, h_sub, F_sub = t_full[:end_idx], h_full[:end_idx], F_full[:end_idx]
    else:
        t_sub, h_sub, F_sub = t_full, h_full, F_full
        
    dt_sub = dt_full
    results = []
    
    lp_params = None
    if "lp" in FUNCTION_MAP[model_name]:
        lp_func = FUNCTION_MAP[model_name]["lp"]
        lp_result = lp_func(t_full, h_full, F_full, dt_full, C, h_power_exp, S_VALS)
        Fm_full_lp = compute_full_force(model_name, lp_result["parameters"], t_full, h_full, dt_full, C, h_power_exp, smooth_window=smooth_window)
        lp_result.update({"F_model": Fm_full_lp})
        lp_result.update(get_all_metrics(Fm_full_lp, F_full))
        results.append(lp_result)
        lp_params = lp_result["parameters"]

    td_func = FUNCTION_MAP[model_name]["td"]
    td_result = td_func(t_sub, h_sub, F_sub, dt_sub, C, h_power_exp, lp_params=lp_params, smooth_window=smooth_window)
    Fm_full_td = compute_full_force(model_name, td_result["parameters"], t_full, h_full, dt_full, C, h_power_exp, smooth_window=smooth_window)
    td_result.update({"F_model": Fm_full_td})
    td_result.update(get_all_metrics(Fm_full_td, F_full))
    results.append(td_result)
    
    if refine_full_data:
        td_params = td_result["parameters"]
        refine_result = td_func(t_full, h_full, F_full, dt_full, C, h_power_exp, lp_params=td_params, smooth_window=smooth_window)
        refine_result["method"] = "Time-Domain (Refined)"
        Fm_refine = compute_full_force(model_name, refine_result["parameters"], t_full, h_full, dt_full, C, h_power_exp, smooth_window=smooth_window)
        refine_result.update({"F_model": Fm_refine})
        refine_result.update(get_all_metrics(Fm_refine, F_full))
        results.append(refine_result)

    results_sorted = sorted(results, key=lambda x: x['RMSE_N'])
    return results_sorted
    
    
import os
import glob
from indentation.io import load_data, save_results
from indentation.plotting import plot_comparison

def run_batch_fit(model_name: str, folder_path: str, C: float, h_power_exp: float, S_VALS: np.ndarray, time_window: float | None = None, refine_full_data: bool = False, smooth_window: int = 0, output_dir: str = "batch_results", time_rate: float | None = None) -> list[dict]:
    """
    Scans a folder for CSV/Excel files and processes them in batch.
    Returns a list of summary dictionaries for the batch summary.
    """
    # Find all CSV and Excel files
    files = glob.glob(os.path.join(folder_path, "*.csv")) + glob.glob(os.path.join(folder_path, "*.xlsx"))
    if not files:
        print(f"No CSV or Excel files found in: {folder_path}")
        return []
        
    batch_summary = []
    
    for filepath in files:
        filename = os.path.basename(filepath)
        print(f"\n{'=' * 70}\n  PROCESSING FILE: {filename}\n{'=' * 70}")
        
        try:
            t_full, h_full, F_full = load_data(filepath, time_rate=time_rate)
            print(f"  Loaded {len(t_full)} points.")
            
            # Run the main pipeline
            results = run_fit(
                model_name=model_name,
                t_full=t_full, h_full=h_full, F_full=F_full,
                C=C, h_power_exp=h_power_exp,
                S_VALS=S_VALS,
                time_window=time_window,
                refine_full_data=refine_full_data,
                smooth_window=smooth_window
            )
            
            best = results[0]
            
            # Save individual results to a subfolder named after the file
            file_stem = os.path.splitext(filename)[0]
            ind_dir = os.path.join(output_dir, file_stem)
            
            save_results(results, model_name, t_full, F_full, output_dir=ind_dir)
            plot_comparison(results, t_full, F_full, model_name, output_dir=ind_dir)
            
            # Collect the best data for the batch summary
            summary = {
                "Filename": filename,
                "Method": best["method"],
                "RMSE_N": best.get("RMSE_N", 0),
                "RelErr_%": best.get("RelErr_%", 0),
                "R2": best.get("R2", 0)
            }
            # Flatten the best parameters into the summary
            for k, v in best['parameters'].items():
                summary[k] = v
                
            batch_summary.append(summary)
            
        except Exception as e:
            print(f"[Error] Failed to process {filename}: {e}")
            
    return batch_summary
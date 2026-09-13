import os
import json
import numpy as np
import pandas as pd

class MissingTimeRateError(Exception):
    """Raised when a 2-column file is loaded without a time_rate."""
    pass

def _get_long_path(path: str) -> str:
    """Bypass Windows 260-character path limit by prepending \\?\."""
    if os.name == 'nt' and not path.startswith('\\\\?\\'):
        return '\\\\?\\' + os.path.abspath(path)
    return path

def load_data(filename: str, time_rate: float | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Robustly load CSV or Excel data.
    
    Handles auto-detection of delimiters (comma/semicolon), European decimals,
    and supports both 2-column (Disp, Force) and 3-column (Time, Disp, Force) formats.
    Automatically resamples data to a uniform time grid if needed.
    
    Parameters
    ----------
    filename : str
        Path to the input file (.csv, .xlsx, or .xls).
    time_rate : float, optional
        Loading rate (displacement per second). Required if the file has 2 columns.
        
    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        A tuple containing (time, displacement, force) arrays.
        
    Raises
    ------
    MissingTimeRateError
        If a 2-column file is loaded without providing `time_rate`.
    ValueError
        If the file format is invalid or data is insufficient.
    """
    try:
        ext = os.path.splitext(filename)[1].lower()
        
        if ext in ['.xls', '.xlsx']:
            df = pd.read_excel(filename, header=None, dtype=str)
        else:
            with open(filename, 'r') as f:
                first_line = f.readline()
            delimiter = ';' if first_line.count(';') > first_line.count(',') else ','
            df = pd.read_csv(filename, sep=delimiter, header=None, dtype=str)
            
        df = df.dropna(how='all')
        df = df.map(lambda x: x.strip().replace(',', '.') if isinstance(x, str) else x)
        df = df.apply(pd.to_numeric, errors='coerce')
        df = df.dropna(how='all')
        
        data = df.to_numpy(dtype=float)
        if data.ndim == 1:
            data = np.reshape(data, (1, -1))
            
        if data.shape[1] < 2 or data.shape[1] > 3:
            raise ValueError(f"Expected 2 or 3 columns, got {data.shape[1]}.")
            
        data = data[~np.isnan(data).any(axis=1)]
        if len(data) < 2:
            raise ValueError("Less than 2 valid data points found.")
            
        if data.shape[1] == 3:
            t_raw, h_raw, F_raw = data[:, 0], data[:, 1], data[:, 2]
            
        elif data.shape[1] == 2:
            # Disp, Force -> Generate Time
            h_raw = data[:, 0]
            F_raw = data[:, 1]
            
            if time_rate is None or time_rate <= 0:
                raise MissingTimeRateError("File has 2 columns (Disp, Force). A 'time_rate' is required.")
            
            # Calculate the sampling time step (dt) from the loading rate (v) and the initial displacement steps.
            # v = dh / dt  =>  dt = dh / v
            # We use the median of the first few positive displacement steps to be robust against noise.
            dh = np.diff(h_raw)
            dh_positive = dh[dh > 1e-12] # Filter out zero-displacement steps (hold phase)
            
            if len(dh_positive) > 0:
                dt = np.median(dh_positive) / time_rate
            else:
                dt = 0.1 # Fallback if no ramp is detected
                
            # Generate time array using the constant time step
            t_raw = np.arange(len(h_raw)) * dt
            
        t_raw, unique_indices = np.unique(t_raw, return_index=True)
        h_raw = h_raw[unique_indices]
        F_raw = F_raw[unique_indices]
        
        dt_raw = np.diff(t_raw)
        if len(dt_raw) > 1 and np.std(dt_raw) > 0.01 * np.mean(dt_raw):
            print(f"[Warning] Non-uniform time steps detected. Resampling to a uniform grid...")
            t_uniform = np.linspace(t_raw[0], t_raw[-1], len(t_raw))
            h_uniform = np.interp(t_uniform, t_raw, h_raw)
            F_uniform = np.interp(t_uniform, t_raw, F_raw)
            return t_uniform, h_uniform, F_uniform
            
        return t_raw, h_raw, F_raw
        
    except MissingTimeRateError:
        raise  # Propagate this specific error to the CLI so it can ask the user
    except Exception as e:
        raise ValueError(f"Failed to load data file: {e}")

def save_results(results: list[dict], model_name: str, t_full: np.ndarray, F_full: np.ndarray, output_dir: str = "prediction_results") -> None:
    """
    Save fitting results (including Confidence Intervals) to JSON, TXT, CSV, and Excel files.
    """
    output_dir = _get_long_path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Save TXT Table
    txt_path = os.path.join(output_dir, "model_comparison_table.txt")
    with open(txt_path, 'w') as f:
        f.write(f"Model: {model_name}\n")
        f.write("=" * 70 + "\n")
        for i, res in enumerate(results):
            best_tag = " (BEST)" if i == 0 else ""
            f.write(f"{res['method']}{best_tag} - Metrics:\n")
            f.write(f"  RMSE   = {res.get('RMSE_N', 0):.6e}\n")
            f.write(f"  RelErr = {res.get('RelErr_%', 0):.4f}%\n")
            f.write(f"  R2     = {res.get('R2', 0):.6f}\n")
            f.write(f"  RMSRE  = {res.get('RMSRE', 0):.6e}\n")
            f.write(f"  Params:\n")
            ci_list = res.get("CI", {}).get("95%_CI")
            for i, (k, v) in enumerate(res['parameters'].items()):
                if ci_list is not None:
                    f.write(f"    {k:<15} = {v:.4e} ± {ci_list[i]:.4e}\n")
                else:
                    f.write(f"    {k:<15} = {v:.4e}    (CI: N/A)\n")
            f.write("\n")
            
    # 2. Save detailed JSON
    json_data = []
    metric_keys = ["RMSE_N", "RelErr_%", "R2", "NRMSE_%", "Log_RMSE", "RMSRE"]
    for r in results:
        temp = {k: v for k, v in r.items() if k != 'F_model'}
        temp['parameters'] = {k: float(v) for k, v in r['parameters'].items()}
        for mk in metric_keys:
            if mk in temp:
                temp[mk] = float(temp[mk])
        if "CI" in r:
            temp["CI"] = {
                "Standard_Error": [float(x) for x in r["CI"]["Standard_Error"]],
                "95%_CI": [float(x) for x in r["CI"]["95%_CI"]]
            }
        json_data.append(temp)
        
    json_path = os.path.join(output_dir, "detailed_results.json")
    with open(json_path, 'w') as f:
        json.dump({"model": model_name, "results": json_data, "best_method": results[0]['method']}, f, indent=2)
        
    # 3. Save CSV and Excel files using Pandas
    summary_data = []
    for r in results:
        row = {
            "Method": r["method"],
            "RMSE_N": r.get("RMSE_N", 0),
            "RelErr_%": r.get("RelErr_%", 0),
            "R2": r.get("R2", 0),
            "NRMSE_%": r.get("NRMSE_%", 0),
            "Log_RMSE": r.get("Log_RMSE", 0),
            "RMSRE": r.get("RMSRE", 0)
        }
        for i, (k, v) in enumerate(r['parameters'].items()):
            row[k] = v
            ci = r.get("CI", {}).get("95%_CI")
            if ci is not None:
                row[f"{k} [95%_CI]"] = ci[i]
            else:
                row[f"{k} [95%_CI]"] = "N/A"
        summary_data.append(row)
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(os.path.join(output_dir, "summary_results.csv"), index=False)
    
    force_data = {"Time": t_full, "Experimental_Force": F_full}
    for r in results:
        force_data[f"F_model_{r['method']}"] = r["F_model"]
    df_force = pd.DataFrame(force_data)
    df_force.to_csv(os.path.join(output_dir, "force_comparison.csv"), index=False)
    
    with pd.ExcelWriter(os.path.join(output_dir, "full_results.xlsx")) as writer:
        df_summary.to_excel(writer, sheet_name="Summary", index=False)
        df_force.to_excel(writer, sheet_name="Force_Data", index=False)
        
    print(f"Results saved to {os.path.abspath(output_dir)}/")
    
    
    
def save_batch_summary(batch_summary: list[dict], output_dir: str = "batch_results") -> None:
    """
    Saves a summary of batch fitting results, including statistical analysis.
    """
    output_dir = _get_long_path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create DataFrame from the list of summary dictionaries
    df = pd.DataFrame(batch_summary)
    
    # Ensure numeric columns are treated as numbers for statistics
    numeric_cols = [c for c in df.columns if c not in ['Filename', 'Method']]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    # Calculate Statistics (Mean and Standard Deviation)
    mean_row = {'Filename': 'AVERAGE', 'Method': ''}
    std_row = {'Filename': 'STD_DEV', 'Method': ''}
    
    for col in numeric_cols:
        mean_row[col] = df[col].mean()
        std_row[col] = df[col].std()
        
    # Append the statistics to the DataFrame
    df = pd.concat([df, pd.DataFrame([mean_row, std_row])], ignore_index=True)
    
    # Save to CSV and Excel
    csv_path = os.path.join(output_dir, "batch_summary.csv")
    df.to_csv(csv_path, index=False)
    
    xlsx_path = os.path.join(output_dir, "batch_summary.xlsx")
    with pd.ExcelWriter(xlsx_path) as writer:
        df.to_excel(writer, sheet_name="Batch Summary", index=False)
        
    print(f"\nBatch summary with statistics saved to {os.path.abspath(output_dir)}/")
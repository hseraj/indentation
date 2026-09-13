import numpy as np
import sys
import os
from indentation.core import run_fit, run_batch_fit
from indentation.io import load_data, save_results, save_batch_summary, MissingTimeRateError
from indentation.plotting import plot_comparison
from indentation.probes import get_probe_geometry
from indentation.viscoelastic.models import MODELS

def get_float_input(prompt, default):
    while True:
        val = input(f"{prompt} [default: {default}]: ").strip()
        if val == "": return default
        try: return float(val)
        except ValueError: print("Invalid input. Please enter a numeric value.")

def get_shared_settings():
    """Gets the model, probe, and optimization settings shared across batch or single runs."""
    # 1. Select Model
    print("\nAvailable Viscoelastic Models:")
    model_keys = list(MODELS.keys())
    for i, key in enumerate(model_keys, 1):
        print(f"  [{i}] {MODELS[key]['display']}")
        
    while True:
        choice = input(f"Select model (1-{len(model_keys)}): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(model_keys):
            model_name = model_keys[int(choice)-1]
            break
        print("Invalid choice.")

    # 2. Probe Type
    print("\n--- Probe Type Selection ---")
    print("  [1] Spherical\n  [2] Cylindrical\n  [3] Conical")
    print("  [4] Four-sided Pyramidal\n  [5] Blunted Four-sided Pyramidal")
    
    probe_map = {
        "1": "Spherical", "2": "Cylindrical", "3": "Conical",
        "4": "Four-sided Pyramidal", "5": "Blunted Four-sided Pyramidal"
    }
    while True:
        choice = input("Select probe type (1-5): ").strip()
        if choice in probe_map: break
        print("Invalid choice.")
        
    probe_type = probe_map[choice]
    params = {}
    params["nu"] = get_float_input("Enter Poisson's ratio (nu)", 0.5)
    if probe_type in ["Spherical", "Cylindrical"]:
        params["R"] = get_float_input("Enter probe radius R in meters", 2.0e-3)
    else:
        params["theta_deg"] = get_float_input("Enter probe angle theta in degrees", 30.0)
        
    C, h_power_exp = get_probe_geometry(probe_type, params)

    # 3. Time Window & S-Values
    time_window = get_float_input("Enter time window for TD optimization (s, 0 for full data)", 0.0)
    if time_window == 0.0: time_window = None
    
    smooth_window = int(get_float_input("Enter smoothing window (0 for no smoothing, e.g., 15)", 0.0))
    
    s_min = get_float_input("Min Laplace variable (s) [1/s]", 1e-3)
    s_max = get_float_input("Max Laplace variable (s) [1/s]", 1e3)
    S_VALS = np.sort(np.unique(np.logspace(np.log10(s_min), np.log10(s_max), 110)))

    refine = input("Run full-data refinement? (y/n): ").strip().lower() == 'y'

    return model_name, C, h_power_exp, S_VALS, time_window, refine, smooth_window

def run_single():
    """Workflow for processing a single data file."""
    while True:
        filepath = input("\nEnter input file path (CSV or Excel): ").strip()
        try:
            try:
                t_full, h_full, F_full = load_data(filepath)
            except MissingTimeRateError:
                print("File has 2 columns (Disp, Force).")
                time_rate = get_float_input("Enter loading velocity (m/s) during the ramp phase", 1.0)
                t_full, h_full, F_full = load_data(filepath, time_rate=time_rate)
            print(f"  Loaded {len(t_full)} points.")
            break
        except Exception as e:
            print(f"Error: {e}")

    model_name, C, h_power_exp, S_VALS, time_window, refine, smooth_window = get_shared_settings()

    print("\nRunning pipeline...")
    
    results = run_fit(
        model_name=model_name,
        t_full=t_full, h_full=h_full, F_full=F_full,
        C=C, h_power_exp=h_power_exp,
        S_VALS=S_VALS,
        time_window=time_window,
        refine_full_data=refine,
        smooth_window=smooth_window
    )

    # Print and Save Results
    print(f"\n{'=' * 70}\n  RESULTS\n{'=' * 70}")
    for i, res in enumerate(results):
        best_tag = " (BEST)" if i == 0 else ""
        print(f"\n  METHOD: {res['method']}{best_tag}")
        print(f"    RMSE         = {res.get('RMSE_N', 0):.6e} N")
        print(f"    RelErr       = {res.get('RelErr_%', 0):.4f}%")
        print(f"    R2           = {res.get('R2', 0):.6f}")
        print(f"    RMSRE        = {res.get('RMSRE', 0):.6e}")
        print(f"    Log_RMSE     = {res.get('Log_RMSE', 0):.6e}")
        print(f"    Parameters:")
        ci_list = res.get("CI", {}).get("95%_CI")
        for i, (k, v) in enumerate(res['parameters'].items()):
            if ci_list is not None:
                print(f"      {k:<15} = {v:.4e} ± {ci_list[i]:.4e}")
            else:
                print(f"      {k:<15} = {v:.4e}    (CI: N/A)")

    out_dir = input("\nEnter output folder path [Press Enter for default 'prediction_results']: ").strip()
    if not out_dir: out_dir = "prediction_results"
        
    save_results(results, model_name, t_full, F_full, output_dir=out_dir)
    plot_comparison(results, t_full, F_full, model_name, output_dir=out_dir)
    print(f"\nComputation complete. Check the '{out_dir}' folder.")

def run_batch():
    """Workflow for processing a folder of data files."""
    folder_path = input("\nEnter the folder path containing your CSV/Excel files: ").strip()
    if not os.path.isdir(folder_path):
        print("Error: Directory not found.")
        return

    out_dir = input("Enter output folder for batch results [Press Enter for default 'batch_results']: ").strip()
    if not out_dir: out_dir = "batch_results"

    model_name, C, h_power_exp, S_VALS, time_window, refine, smooth_window = get_shared_settings()
    
    # Ask for loading rate in case the folder contains 2-column files
    time_rate_str = input("If files have 2 columns, enter loading rate (Press Enter to assume 1.0): ").strip()
    time_rate = float(time_rate_str) if time_rate_str else 1.0

    print(f"\nStarting Batch Processing...")
    
    # Run the batch loop
    batch_summary = run_batch_fit(
        model_name=model_name,
        folder_path=folder_path,
        C=C, h_power_exp=h_power_exp,
        S_VALS=S_VALS,
        time_window=time_window,
        refine_full_data=refine,
        smooth_window=smooth_window,
        output_dir=out_dir,
        time_rate=time_rate
    )

    # Save the final summary with statistics
    if batch_summary:
        save_batch_summary(batch_summary, output_dir=out_dir)
        print(f"\nBatch Computation complete. Check the '{out_dir}' folder.")
    else:
        print("\nNo files were successfully processed.")

def main():
    print("=" * 70)
    print("  INDENTATION VISCOELASTIC MODEL IDENTIFICATION TOOL")
    print("=" * 70)
    
    while True:
        print("\nSelect Mode:")
        print("  [1] Process Single File")
        print("  [2] Process Batch Folder (Multiple Files)")
        print("  [0] Exit")
        
        choice = input("Enter choice (0-2): ").strip()
        
        if choice == '1':
            run_single()
        elif choice == '2':
            run_batch()
        elif choice == '0':
            print("Exiting...")
            break
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    main()
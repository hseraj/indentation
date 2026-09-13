import numpy as np
import sys
import os

# Add project root to path so we can import the indentation package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from indentation import run_fit, save_results, plot_comparison, get_probe_geometry

def main():
    print("--- Running Basic Fit Example ---")
    
    # 1. Generate Synthetic Data (SLS Model)
    print("Generating synthetic SLS data...")
    t_arr = np.linspace(0, 10, 500)
    dt = t_arr[1] - t_arr[0]
    h_arr = 0.1 * t_arr  # Linear ramp indentation
    
    # SLS True Parameters: E1=1e6, E2=2e6, tau=0.5
    def E_relax_true(t): 
        return 1e6 + 2e6 * np.exp(-t / 0.5)
    
    # Compute true force
    C, h_power_exp = get_probe_geometry("Spherical", {"nu": 0.5, "R": 2.0e-3})
    h_power = h_arr ** h_power_exp
    dh = np.gradient(h_power, dt)
    E_arr = E_relax_true(t_arr)
    
    # Hereditary integral for synthetic data
    from scipy.signal import fftconvolve
    S = fftconvolve(E_arr, dh, mode='full')[:len(t_arr)]
    F_data = C * dt * (S - 0.5 * E_arr * dh[0] - 0.5 * E_arr[0] * dh)
    F_data[0] = 0.0
    
    # 2. Run the Identification Pipeline
    print("Running Laplace -> Time-Domain optimization...")
    S_VALS = np.sort(np.unique(np.logspace(-3, 3, 110)))
    
    results = run_fit(
        model_name="Standard_Linear_Solid",
        t_full=t_arr, h_full=h_arr, F_full=F_data,
        C=C, h_power_exp=h_power_exp,
        S_VALS=S_VALS,
        time_window=5.0,       
        refine_full_data=True,
        smooth_window=0        # Analytical SG derivative will be used
    )
    
    # 3. Print and Save Results
    print("\n--- Results ---")
    for res in results:
        print(f"Method: {res['method']} | RelErr: {res['RelErr_%']:.4f}%")
        ci_list = res.get("CI", {}).get("95%_CI")
        for i, (k, v) in enumerate(res['parameters'].items()):
            if ci_list is not None:
                print(f"  {k:<15} = {v:.4e} ± {ci_list[i]:.4e}")
            else:
                print(f"  {k:<15} = {v:.4e}    (CI: N/A)")
            
    # Save to a folder inside the current example directory
    out_dir = os.path.join(os.path.dirname(__file__), "basic_fit output")
    save_results(results, "Standard_Linear_Solid", t_arr, F_data, output_dir=out_dir)
    plot_comparison(results, t_arr, F_data, "Standard_Linear_Solid", output_dir=out_dir)
    
    print("\nExample complete! Check the 'basic_fit output' folder.")

if __name__ == "__main__":
    main()
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
import threading
import os
import numpy as np
import pandas as pd

from indentation.core import run_fit, run_batch_fit
from indentation.io import load_data, save_results, save_batch_summary, MissingTimeRateError
from indentation.plotting import plot_comparison, plot_frequency_domain
from indentation.probes import get_probe_geometry
from indentation.viscoelastic.models import MODELS

class IndentationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Indentation v0.2.0")
        self.root.geometry("800x700")
        
        self.create_widgets()
        
    def create_widgets(self):
        # --- Mode Selection ---
        mode_frame = ttk.LabelFrame(self.root, text="Mode")
        mode_frame.pack(fill="x", padx=10, pady=5)
        self.mode_var = tk.StringVar(value="single")
        ttk.Radiobutton(mode_frame, text="Single File", variable=self.mode_var, value="single", command=self.toggle_mode).pack(side="left", padx=10, pady=5)
        ttk.Radiobutton(mode_frame, text="Batch Folder", variable=self.mode_var, value="batch", command=self.toggle_mode).pack(side="left", padx=10, pady=5)
        
        # --- Paths ---
        path_frame = ttk.LabelFrame(self.root, text="Paths")
        path_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(path_frame, text="Input:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        self.input_path = tk.StringVar()
        ttk.Entry(path_frame, textvariable=self.input_path, width=60).grid(row=0, column=1, padx=5, pady=5)
        self.btn_browse_in = ttk.Button(path_frame, text="Browse...", command=self.browse_input)
        self.btn_browse_in.grid(row=0, column=2, padx=5, pady=5)
        
        ttk.Label(path_frame, text="Output:").grid(row=1, column=0, padx=5, pady=5, sticky="e")
        self.output_path = tk.StringVar()
        ttk.Entry(path_frame, textvariable=self.output_path, width=60).grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(path_frame, text="Browse...", command=self.browse_output).grid(row=1, column=2, padx=5, pady=5)
        
        # --- Parameters ---
        param_frame = ttk.LabelFrame(self.root, text="Parameters")
        param_frame.pack(fill="x", padx=10, pady=5)
        
        # Model
        ttk.Label(param_frame, text="Model:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        self.model_var = tk.StringVar()
        model_names = list(MODELS.keys())
        self.model_cb = ttk.Combobox(param_frame, textvariable=self.model_var, values=model_names, state="readonly", width=25)
        self.model_cb.current(4) # Standard Linear Solid
        self.model_cb.grid(row=0, column=1, padx=5, pady=5)
        
        # Probe
        ttk.Label(param_frame, text="Probe Type:").grid(row=0, column=2, padx=5, pady=5, sticky="e")
        self.probe_var = tk.StringVar()
        probe_names = ["Spherical", "Cylindrical", "Conical", "Four-sided Pyramidal", "Blunted Four-sided Pyramidal"]
        self.probe_cb = ttk.Combobox(param_frame, textvariable=self.probe_var, values=probe_names, state="readonly", width=25)
        self.probe_cb.current(0)
        self.probe_cb.grid(row=0, column=3, padx=5, pady=5)
        
        # Probe Params
        ttk.Label(param_frame, text="Poisson's ratio (nu):").grid(row=1, column=0, padx=5, pady=5, sticky="e")
        self.nu_var = tk.StringVar(value="0.5")
        ttk.Entry(param_frame, textvariable=self.nu_var, width=10).grid(row=1, column=1, sticky="w", padx=5, pady=5)
        
        ttk.Label(param_frame, text="Radius R (m) / Angle (deg):").grid(row=1, column=2, padx=5, pady=5, sticky="e")
        self.geo_var = tk.StringVar(value="2.0e-3")
        ttk.Entry(param_frame, textvariable=self.geo_var, width=10).grid(row=1, column=3, sticky="w", padx=5, pady=5)
        
        # Optimization Params
        ttk.Label(param_frame, text="Time Window (s, 0=full):").grid(row=2, column=0, padx=5, pady=5, sticky="e")
        self.tw_var = tk.StringVar(value="0.0")
        ttk.Entry(param_frame, textvariable=self.tw_var, width=10).grid(row=2, column=1, sticky="w", padx=5, pady=5)
        
        ttk.Label(param_frame, text="Smoothing Window (0=none):").grid(row=2, column=2, padx=5, pady=5, sticky="e")
        self.sw_var = tk.StringVar(value="15")
        ttk.Entry(param_frame, textvariable=self.sw_var, width=10).grid(row=2, column=3, sticky="w", padx=5, pady=5)
        
        # Loading rate for 2-column data
        ttk.Label(param_frame, text="Loading Rate (m/s) [if 2-col]:").grid(row=3, column=0, padx=5, pady=5, sticky="e")
        self.rate_var = tk.StringVar(value="1.0e-4")
        ttk.Entry(param_frame, textvariable=self.rate_var, width=10).grid(row=3, column=1, sticky="w", padx=5, pady=5)
        
        # Max frequency for dynamic modulus calculation
        ttk.Label(param_frame, text="Max Freq (Hz) for E*(ω):").grid(row=3, column=2, padx=5, pady=5, sticky="e")
        self.freq_var = tk.StringVar(value="1000.0")
        ttk.Entry(param_frame, textvariable=self.freq_var, width=10).grid(row=3, column=3, sticky="w", padx=5, pady=5)
        
        # Move Refinement checkbox to row 4 and span it across all columns
        self.refine_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(param_frame, text="Run full-data refinement", variable=self.refine_var).grid(row=4, column=0, columnspan=4, padx=5, pady=5, sticky="w")
        
        # --- Run Button ---
        self.run_btn = ttk.Button(self.root, text="Run Evaluation", command=self.start_thread)
        self.run_btn.pack(pady=10)
        
        # --- Log / Results ---
        log_frame = ttk.LabelFrame(self.root, text="Output Log")
        log_frame.pack(fill="both", expand=True, padx=10, pady=5)
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, height=15)
        self.log_text.pack(fill="both", expand=True, padx=5, pady=5)
        
    def toggle_mode(self):
        if self.mode_var.get() == "single":
            self.btn_browse_in.config(text="Browse File...")
        else:
            self.btn_browse_in.config(text="Browse Folder...")
            
    def browse_input(self):
        if self.mode_var.get() == "single":
            file_path = filedialog.askopenfilename(filetypes=[("CSV/Excel", "*.csv *.xlsx *.xls"), ("All Files", "*.*")])
            if file_path:
                self.input_path.set(file_path)
        else:
            folder_path = filedialog.askdirectory()
            if folder_path:
                self.input_path.set(folder_path)
                
    def browse_output(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.output_path.set(folder_path)
            
    def log(self, message):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()
        
    def start_thread(self):
        self.run_btn.config(state="disabled")
        self.log_text.delete(1.0, tk.END)
        threading.Thread(target=self.process_data, daemon=True).start()
        
    def process_data(self):
        try:
            self.log("Starting processing...")
            model_name = self.model_var.get()
            probe_type = self.probe_var.get()
            nu = float(self.nu_var.get())
            geo_val = float(self.geo_var.get())
            
            if probe_type in ["Spherical", "Cylindrical"]:
                params = {"nu": nu, "R": geo_val}
            else:
                params = {"nu": nu, "theta_deg": geo_val}
                
            C, h_power_exp = get_probe_geometry(probe_type, params)
            
            tw = float(self.tw_var.get())
            if tw == 0.0: tw = None
            sw = int(self.sw_var.get())
            refine = self.refine_var.get()
            
            S_VALS = np.sort(np.unique(np.logspace(-3, 3, 110)))
            out_dir = self.output_path.get() if self.output_path.get() else "gui_results"
            
            if self.mode_var.get() == "single":
                self.log(f"Loading file: {self.input_path.get()}")
                try:
                    t_full, h_full, F_full = load_data(self.input_path.get())
                except MissingTimeRateError:
                    rate = float(self.rate_var.get())
                    self.log(f"Warning: 2-column data detected. Using Loading Rate = {rate} m/s")
                    t_full, h_full, F_full = load_data(self.input_path.get(), time_rate=rate)
                    
                self.log("Running fit...")
                
                freq_max = float(self.freq_var.get())
                results = run_fit(
                    model_name=model_name,
                    t_full=t_full, h_full=h_full, F_full=F_full,
                    C=C, h_power_exp=h_power_exp,
                    S_VALS=S_VALS, time_window=tw,
                    refine_full_data=refine, smooth_window=sw,
                    freq_max=freq_max
                )
                
                self.log("\n--- RESULTS ---")
                for i, res in enumerate(results):
                    best_tag = " (BEST)" if i == 0 else ""
                    self.log(f"Method: {res['method']}{best_tag} | RMSE: {res.get('RMSE_N', 0):.4e} | R2: {res.get('R2', 0):.4f}")
                    ci_list = res.get("CI", {}).get("95%_CI")
                    for j, (k, v) in enumerate(res['parameters'].items()):
                        if ci_list is not None:
                            self.log(f"  {k:<15} = {v:.4e} ± {ci_list[j]:.4e}")
                        else:
                            self.log(f"  {k:<15} = {v:.4e}    (CI: N/A)")
                            
                save_results(results, model_name, t_full, F_full, output_dir=out_dir)
                plot_comparison(results, t_full, h_full, F_full, model_name, output_dir=out_dir)
                plot_frequency_domain(results, model_name, output_dir=out_dir)
                self.log(f"\nComplete! Results saved to: {os.path.abspath(out_dir)}")
                
            else:
                self.log(f"Starting batch processing on folder: {self.input_path.get()}")
                rate = float(self.rate_var.get())
                
                freq_max = float(self.freq_var.get())
                batch_summary = run_batch_fit(
                    model_name=model_name,
                    folder_path=self.input_path.get(),
                    C=C, h_power_exp=h_power_exp,
                    S_VALS=S_VALS, time_window=tw,
                    refine_full_data=refine, smooth_window=sw,
                    output_dir=out_dir,
                    time_rate=rate,
                    freq_max=freq_max
                )
                if batch_summary:
                    save_batch_summary(batch_summary, output_dir=out_dir)
                    self.log(f"\nBatch Complete! Summary saved to: {os.path.abspath(out_dir)}")
                else:
                    self.log("No files processed.")
                    
        except Exception as e:
            self.log(f"\n[ERROR] {str(e)}")
        finally:
            self.run_btn.config(state="normal")

def main():
    root = tk.Tk()
    app = IndentationGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
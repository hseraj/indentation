#!/usr/bin/env python3
"""
================================================================================
Viscoelastic Indentation Test Data Generator
================================================================================

Generates a comprehensive dataset of simulated indentation tests to benchmark
the `indentation` package.

Features:
- Uses the `indentation` package's own FFT convolution to generate true force.
- Tests 7 models (SLS, Maxwell, Kelvin-Voigt, Dashpot, Prony N1/N2/N3).
- Tests 5 probes (Spherical, Cylindrical, Conical, Pyramidal, Blunted Pyramidal).
- Tests Uniform and Non-Uniform (adaptive) time sampling.
- Applies realistic heteroscedastic noise to Force and Displacement.
- Organizes files into a strict hierarchy with parameter-based names.
- Generates a `metadata.json` for every test condition.
"""

import numpy as np
import os
import sys
import json
import csv

# Add project root to path so we can import the indentation package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from indentation.transforms import compute_force
from indentation.probes import get_probe_geometry

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "simulated_data2")

NOISE_LEVELS = [0.0, 0.02, 0.05]  # 0%, 2%, 5%
FORCE_NOISE_FLOOR = 1e-5  # 10 µN floor
DISP_NOISE_STD = 1e-6     # 1 µm standard deviation

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_filename_params(model_name, params):
    """Converts physical parameters into a readable filename."""
    parts = []
    if model_name == "Maxwell":
        parts.append(f"E_{params['E [Pa]']/1e6:.1f}MPa")
        parts.append(f"tau_{params['tau [s]']:.1f}s")
    elif model_name == "Standard_Linear_Solid":
        parts.append(f"E1_{params['E1 [Pa]']/1e6:.1f}MPa")
        parts.append(f"E2_{params['E2 [Pa]']/1e6:.1f}MPa")
        parts.append(f"tau_{params['tau [s]']:.1f}s")
    elif model_name == "Kelvin_Voigt":
        parts.append(f"E_{params['E [Pa]']/1e6:.1f}MPa")
        parts.append(f"eta_{params['eta [Pa.s]']/1e6:.1f}MPa_s")
    elif model_name == "Dashpot":
        parts.append(f"eta_{params['eta [Pa.s]']/1e6:.1f}MPa_s")
        
    elif model_name == "Prony_N1":
        parts.append(f"Einf_{params['E_inf [Pa]']/1e6:.1f}")
        parts.append(f"E1_{params['E1 [Pa]']/1e6:.1f}")
        parts.append(f"t1_{params['tau1 [s]']:.1f}")
    elif model_name == "Prony_N2":
        parts.append(f"Einf_{params['E_inf [Pa]']/1e6:.1f}")
        parts.append(f"E1_{params['E1 [Pa]']/1e6:.1f}")
        parts.append(f"t1_{params['tau1 [s]']:.1f}")
        parts.append(f"E2_{params['E2 [Pa]']/1e6:.1f}")
        parts.append(f"t2_{params['tau2 [s]']:.1f}")
    elif model_name == "Prony_N3":
        parts.append(f"Einf_{params['E_inf [Pa]']/1e6:.1f}")
        parts.append(f"E1_{params['E1 [Pa]']/1e6:.1f}")
        parts.append(f"t1_{params['tau1 [s]']:.2f}")
        parts.append(f"E2_{params['E2 [Pa]']/1e6:.1f}")
        parts.append(f"t2_{params['tau2 [s]']:.1f}")
        parts.append(f"E3_{params['E3 [Pa]']/1e6:.1f}")
        parts.append(f"t3_{params['tau3 [s]']:.1f}")
    else:
        parts.append("params")
    
    # Replace dots with 'p' for filesystem safety (e.g., 1.0 -> 1p0)
    name = "_".join(parts)
    return name.replace(".", "p") + ".csv"

def generate_time_array(sampling_type, t_ramp, t_total, dt):
    """Generates uniform or adaptive non-uniform time arrays."""
    if sampling_type == "Uniform":
        return np.arange(0, t_total + dt, dt)
    elif sampling_type == "NonUniform":
        # Fast sampling during ramp, 10x slower during hold
        dt_ramp = dt
        dt_hold = dt * 10.0
        
        t_ramp_arr = np.arange(0, t_ramp, dt_ramp)
        t_hold_arr = np.arange(t_ramp, t_total + dt_hold, dt_hold)
        
        t = np.unique(np.concatenate((t_ramp_arr, t_hold_arr)))
        return t
    else:
        raise ValueError("Unknown sampling type")

def add_noise_realistic(F_clean, h_clean, noise_fraction, seed=None):
    """Applies heteroscedastic force noise and constant disp noise."""
    if noise_fraction == 0:
        return F_clean.copy(), h_clean.copy()
    
    if seed is not None:
        np.random.seed(seed)
        
    sigma_F = noise_fraction * np.abs(F_clean) + FORCE_NOISE_FLOOR
    F_noisy = F_clean + np.random.normal(0, sigma_F)
    
    h_noise = np.random.normal(0, DISP_NOISE_STD, size=h_clean.shape)
    h_noisy = np.maximum(h_clean + h_noise, 0)  # Prevent negative displacement
    
    return F_noisy, h_noisy

# =============================================================================
# TEST CASE DEFINITIONS
# =============================================================================

def get_test_cases():
    """Returns a list of dictionaries defining all test conditions."""
    cases = []
    
    # Define Models and Parameter Combinations
    models = {
        "Standard_Linear_Solid": [
            {"E1 [Pa]": 1.0e6, "E2 [Pa]": 2.0e6, "tau [s]": 0.5},
            {"E1 [Pa]": 0.5e6, "E2 [Pa]": 1.0e6, "tau [s]": 1.0},
            {"E1 [Pa]": 2.0e6, "E2 [Pa]": 3.0e6, "tau [s]": 2.0}
        ],
        "Maxwell": [
            {"E [Pa]": 1.0e6, "tau [s]": 0.5},
            {"E [Pa]": 2.0e6, "tau [s]": 1.0},
            {"E [Pa]": 0.5e6, "tau [s]": 2.0}
        ],
        "Kelvin_Voigt": [
            {"E [Pa]": 1.0e6, "eta [Pa.s]": 0.5e6},
            {"E [Pa]": 2.0e6, "eta [Pa.s]": 1.0e6},
            {"E [Pa]": 0.5e6, "eta [Pa.s]": 2.0e6}
        ],
        "Dashpot": [
            {"eta [Pa.s]": 0.5e6},
            {"eta [Pa.s]": 1.0e6},
            {"eta [Pa.s]": 2.0e6}
        ],
        "Prony_N1": [
            {"E_inf [Pa]": 0.5e6, "E1 [Pa]": 1.5e6, "tau1 [s]": 1.0},
            {"E_inf [Pa]": 1.0e6, "E1 [Pa]": 2.0e6, "tau1 [s]": 0.5}
        ],
        "Prony_N2": [
            {"E_inf [Pa]": 0.3e6, "E1 [Pa]": 1.0e6, "tau1 [s]": 0.1, "E2 [Pa]": 2.0e6, "tau2 [s]": 5.0},
            {"E_inf [Pa]": 0.5e6, "E1 [Pa]": 1.5e6, "tau1 [s]": 0.5, "E2 [Pa]": 1.0e6, "tau2 [s]": 10.0}
        ],
        "Prony_N3": [
            {"E_inf [Pa]": 0.2e6, "E1 [Pa]": 1.0e6, "tau1 [s]": 0.05, "E2 [Pa]": 1.5e6, "tau2 [s]": 1.0, "E3 [Pa]": 0.5e6, "tau3 [s]": 10.0},
            {"E_inf [Pa]": 0.3e6, "E1 [Pa]": 2.0e6, "tau1 [s]": 0.1,  "E2 [Pa]": 1.0e6, "tau2 [s]": 2.0, "E3 [Pa]": 0.3e6, "tau3 [s]": 15.0}
        ]
    }
    
    # Define Probes
    probes = {
        "Spherical": {"nu": 0.5, "R": 2.0e-3},
        "Cylindrical": {"nu": 0.5, "R": 2.0e-3},
        "Conical": {"nu": 0.5, "theta_deg": 30.0},
        "Four-sided Pyramidal": {"nu": 0.5, "theta_deg": 30.0},
        "Blunted Four-sided Pyramidal": {"nu": 0.5, "theta_deg": 30.0}
    }
    
    # Define Conditions (Sampling Type, h_max, Total Time, Ramp Time, dt)
    conditions = [
        {"sampling": "Uniform",    "h_max": 1.0e-3, "t_total": 600.0, "t_ramp": 10.0, "dt": 0.05},
        {"sampling": "NonUniform", "h_max": 2.0e-3, "t_total": 300.0, "t_ramp": 5.0,  "dt": 0.05}
    ]
    
    # Build the cartesian product of all cases
    for model_name, param_list in models.items():
        for probe_name, probe_params in probes.items():
            for cond in conditions:
                for params in param_list:
                    cases.append({
                        "model": model_name,
                        "probe": probe_name,
                        "probe_params": probe_params,
                        "sampling": cond["sampling"],
                        "h_max": cond["h_max"],
                        "t_total": cond["t_total"],
                        "t_ramp": cond["t_ramp"],
                        "dt": cond["dt"],
                        "params": params
                    })
    return cases

# =============================================================================
# MAIN GENERATION LOGIC
# =============================================================================

def main():
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    test_cases = get_test_cases()
    generated_count = 0
    
    print("=" * 70)
    print("VISCOELASTIC INDENTATION DATASET GENERATOR")
    print(f"Total Base Cases: {len(test_cases)}")
    print(f"Noise Levels: {[f'{int(n*100)}%' for n in NOISE_LEVELS]}")
    print("=" * 70)
    
    for i, case in enumerate(test_cases):
        # 1. Define folder paths
        cond_name = f"{case['sampling']}_h{case['h_max']*1e3:.1f}mm_T{case['t_total']:.0f}s_Ramp{case['t_ramp']:.0f}s".replace(".", "p")
        
        base_folder = os.path.join(
            BASE_OUTPUT_DIR,
            f"Model_{case['model']}",
            f"Probe_{case['probe']}",
            cond_name
        )
        os.makedirs(base_folder, exist_ok=True)
        
        # 2. Generate Time and Displacement arrays
        # Use a fine uniform grid to compute the true force, then sample if NonUniform
        dt_fine = 0.01
        t_fine = np.arange(0, case['t_total'] + dt_fine, dt_fine)
        h_fine = np.where(t_fine <= case['t_ramp'], 
                          (case['h_max'] / case['t_ramp']) * t_fine, 
                          case['h_max'])
        
        # 3. Get Probe Geometry
        C, h_power_exp = get_probe_geometry(case['probe'], case['probe_params'])
        
        # 4. Calculate True Force using analytical derivatives for perfection
        p = case['params']
        
        # Calculate exact analytical derivative of h^n for the ramp-and-hold profile
        v = case['h_max'] / case['t_ramp']  # Velocity during ramp
        h_power_fine = np.where(t_fine <= case['t_ramp'], 
                                (v * t_fine) ** h_power_exp, 
                                case['h_max'] ** h_power_exp)
        
        # Analytical derivative: d/dt(v*t)^n = n * v^n * t^(n-1)
        # At t=0, this is 0 for n>1 (Spherical/Conical/Pyramidal) and v for n=1 (Cylindrical)
        dh_power_fine = np.where(t_fine <= case['t_ramp'],
                                 h_power_exp * (v ** h_power_exp) * (t_fine ** (h_power_exp - 1)),
                                 0.0) # Derivative is 0 during hold
        
        if case['model'] == "Standard_Linear_Solid":
            E_relax_func = lambda t: p["E1 [Pa]"] + p["E2 [Pa]"] * np.exp(-t / p["tau [s]"])
            # Pass the exact analytical h_power and dh to compute_force for perfection
            F_fine = compute_force(E_relax_func, t_fine, h_fine, dt_fine, C, h_power_exp, 
                                   h_power=h_power_fine, dh=dh_power_fine)
            
        elif case['model'] == "Maxwell":
            E_relax_func = lambda t: p["E [Pa]"] * np.exp(-t / p["tau [s]"])
            F_fine = compute_force(E_relax_func, t_fine, h_fine, dt_fine, C, h_power_exp,
                                   h_power=h_power_fine, dh=dh_power_fine)
            
        elif case['model'] == "Prony_N1":
            E_relax_func = lambda t: p["E_inf [Pa]"] + p["E1 [Pa]"] * np.exp(-t / p["tau1 [s]"])
            F_fine = compute_force(E_relax_func, t_fine, h_fine, dt_fine, C, h_power_exp,
                                   h_power=h_power_fine, dh=dh_power_fine)
                                   
        elif case['model'] == "Prony_N2":
            E_relax_func = lambda t: p["E_inf [Pa]"] + p["E1 [Pa]"] * np.exp(-t / p["tau1 [s]"]) + \
                                       p["E2 [Pa]"] * np.exp(-t / p["tau2 [s]"])
            F_fine = compute_force(E_relax_func, t_fine, h_fine, dt_fine, C, h_power_exp,
                                   h_power=h_power_fine, dh=dh_power_fine)
                                   
        elif case['model'] == "Prony_N3":
            E_relax_func = lambda t: p["E_inf [Pa]"] + p["E1 [Pa]"] * np.exp(-t / p["tau1 [s]"]) + \
                                       p["E2 [Pa]"] * np.exp(-t / p["tau2 [s]"]) + \
                                       p["E3 [Pa]"] * np.exp(-t / p["tau3 [s]"])
            F_fine = compute_force(E_relax_func, t_fine, h_fine, dt_fine, C, h_power_exp,
                                   h_power=h_power_fine, dh=dh_power_fine)
            
        elif case['model'] == "Kelvin_Voigt":
            # Analytical: F = C * (E * h^n + eta * dh^n/dt)
            F_fine = C * (p["E [Pa]"] * h_power_fine + p["eta [Pa.s]"] * dh_power_fine)
            
        elif case['model'] == "Dashpot":
            # Analytical: F = C * eta * dh^n/dt
            F_fine = C * p["eta [Pa.s]"] * dh_power_fine
        
        # 6. Apply Sampling (Uniform or NonUniform)
        if case['sampling'] == "Uniform":
            # Sample every 'dt' seconds using integer slicing to avoid floating point errors
            step = int(case['dt'] / dt_fine)
            indices = np.arange(0, len(t_fine), step)
            t_clean = t_fine[indices]
            h_clean = h_fine[indices]
            F_clean = F_fine[indices]
        else:
            # Adaptive sampling: fast during ramp, slow during hold
            t_ramp_arr = np.arange(0, case['t_ramp'] + case['dt'], case['dt'])
            t_hold_arr = np.arange(case['t_ramp'] + case['dt']*10, case['t_total'] + case['dt']*10, case['dt']*10)
            t_clean = np.unique(np.concatenate((t_ramp_arr, t_hold_arr)))
            # Interpolate clean h and F to the non-uniform time grid
            h_clean = np.interp(t_clean, t_fine, h_fine)
            F_clean = np.interp(t_clean, t_fine, F_fine)
            
        # 7. Generate Files for each Noise Level
        metadata_files = {}
        for noise_level in NOISE_LEVELS:
            noise_pct = int(noise_level * 100)
            noise_folder = os.path.join(base_folder, f"Noise_{noise_pct}pct")
            os.makedirs(noise_folder, exist_ok=True)
            
            # Use deterministic seed based on case index and noise level
            F_noisy, h_noisy = add_noise_realistic(F_clean, h_clean, noise_level, seed=i*100+noise_pct)
            
            # Format filename based on params
            fname = get_filename_params(case['model'], case['params'])
            fpath = os.path.join(noise_folder, fname)
            
            # Save CSV
            with open(fpath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Time [s]', 'Displacement [m]', 'Force [N]'])
                for j in range(len(t_clean)):
                    writer.writerow([f'{t_clean[j]:.6f}', f'{h_noisy[j]:.10e}', f'{F_noisy[j]:.10e}'])
            
            generated_count += 1
            
            # Store metadata for this file
            if noise_pct == 0:
                metadata_files[fname] = case['params']
                
        # 8. Save metadata.json for the condition
        meta_path = os.path.join(base_folder, "metadata.json")
        with open(meta_path, 'w') as f:
            json.dump({
                "Condition": cond_name,
                "Probe": case['probe'],
                "Probe_Params": case['probe_params'],
                "Model": case['model'],
                "Files": metadata_files
            }, f, indent=2)
            
        if (i + 1) % 20 == 0 or i == len(test_cases) - 1:
            print(f"  Processed {i + 1}/{len(test_cases)} base conditions... ({generated_count} total files)")
            
    print("\n" + "=" * 70)
    print(f"COMPLETE: Generated {generated_count} CSV files in {BASE_OUTPUT_DIR}")
    print("=" * 70)

if __name__ == "__main__":
    main()
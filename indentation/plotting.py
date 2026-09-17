import os
import numpy as np
import matplotlib.pyplot as plt

def _get_long_path(path: str) -> str:
    r"""Bypass Windows 260-character path limit by prepending \\?\."""
    if os.name == 'nt' and not path.startswith('\\\\?\\'):
        return '\\\\?\\' + os.path.abspath(path)
    return path
    

def plot_comparison(results, t_full, h_full, F_full, model_name, output_dir="prediction_results"):
    """
    Generates a comprehensive 3x2 figure:
    Top-Left: All Methods vs. Experimental Force
    Top-Right: Extracted Relaxation Modulus E(t) vs Time
    Mid-Left: Best Method vs. Experimental Force
    Mid-Right: Force vs. Displacement (F-h)
    Bottom-Left: Absolute Residuals of the Best Method
    Bottom-Right: Relative Error (%) of the Best Method
    """
    output_dir = _get_long_path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 16))
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12']
    
    best = results[0]
    
    # Panel 1 (Top-Left): All Methods vs Experimental
    ax1 = axes[0, 0]
    ax1.plot(t_full, F_full, 'k-', lw=3, label='Experimental Data', alpha=0.8)
    for i, res in enumerate(results):
        c = colors[i % len(colors)]
        ls = ':' if res == best else '--'
        lw = 2.5 if res == best else 1.5
        alpha = 0.9 if res == best else 0.6
        label = f"{res['method']} (BEST)" if res == best else res['method']
        ax1.plot(t_full, res['F_model'], ls, lw=lw, alpha=alpha, color=c, label=label)
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Force [N]')
    ax1.set_title(f"All Methods Fit: {model_name}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Panel 2 (Top-Right): Extracted Relaxation Modulus E(t)
    ax2 = axes[0, 1]
    from indentation.viscoelastic.models import MODELS
    model_info = MODELS[best["model"]]
    E_relax_func = model_info["E_relax_func"]
    
    args = []
    for p_name in model_info["param_names"]:
        for key, val in best['parameters'].items():
            if key.startswith(p_name):
                args.append(val)
                break
                
    if args:
        E_t_best = E_relax_func(t_full, *args)
        ax2.plot(t_full, E_t_best, 'b-', lw=2.5, label=f"Extracted E(t) ({best['method']})")
        
        ax2.set_xlabel('Time [s]')
        ax2.set_ylabel('Relaxation Modulus [Pa]')
        ax2.set_title('Extracted Material Modulus')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'E(t) plot not available', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Extracted Material Modulus')
        
    # Panel 3 (Middle-Left): Best Method Only vs Experimental
    ax3 = axes[1, 0]
    ax3.plot(t_full, F_full, 'ko', markersize=4, label='Experimental Data', alpha=0.6)
    ax3.plot(t_full, best['F_model'], 'r-', lw=2.5, label=f"Best Fit ({best['method']})")
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Force [N]')
    ax3.set_title(f"Best Method Fit: {best['method']}")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Panel 4 (Middle-Right): Force vs. Displacement (F-h)
    ax4 = axes[1, 1]
    ax4.plot(h_full, F_full, 'ko', markersize=4, label='Experimental Data', alpha=0.6)
    ax4.plot(h_full, best['F_model'], 'r-', lw=2.5, label=f"Best Fit ({best['method']})")
    ax4.set_xlabel('Displacement [m]')
    ax4.set_ylabel('Force [N]')
    ax4.set_title('Force vs. Displacement (F-h)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Panel 5 (Bottom-Left): Absolute Residuals of the Best Method
    ax5 = axes[2, 0]
    residuals = best['F_model'] - F_full
    ax5.plot(t_full, residuals, lw=1.5, color='r', label='Residuals')
    ax5.axhline(0, color='k', ls='-', alpha=0.5)
    ax5.set_xlabel('Time [s]')
    ax5.set_ylabel('Residual Force [N]')
    ax5.set_title(f"Absolute Residuals ({best['method']})")
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Panel 6 (Bottom-Right): Relative Error (%) of the Best Method
    ax6 = axes[2, 1]
    eps = 1e-12
    rel_err = np.abs((best['F_model'] - F_full) / np.maximum(np.abs(F_full), eps)) * 100
    ax6.plot(t_full, rel_err, lw=1.5, color='m', label='Relative Error (%)')
    ax6.axhline(0, color='k', ls='-', alpha=0.5)
    # Cap the y-axis to a reasonable limit (e.g., 0 to 20%) to avoid huge spikes at t=0
    upper_limit = np.percentile(rel_err[t_full > t_full[0]], 95) * 1.5 if len(rel_err[t_full > t_full[0]]) > 0 else 10
    ax6.set_ylim(0, min(max(upper_limit, 5), 100)) 
    ax6.set_xlabel('Time [s]')
    ax6.set_ylabel('Relative Error [%]')
    ax6.set_title(f"Relative Error ({best['method']})")
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    plot_path = os.path.join(output_dir, 'comparison_plots.png')
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)  # Close to free up memory
    
    print(f"Plot saved to {os.path.abspath(plot_path)}")
    return plot_path
    
def plot_frequency_domain(results, model_name, output_dir="prediction_results"):
    """
    Generates a comprehensive 2x2 figure for frequency-domain analysis:
    1. Master Curve (E', E'', |E*| vs Frequency)
    2. Tan(delta) and Damping Ratio vs Frequency
    3. Cole-Cole Plot (E'' vs E')
    4. Time-Frequency Equivalence (E(t) and E'(omega) vs Inverse Frequency)
    """
    output_dir = _get_long_path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    if not results or "Frequency_Data" not in results[0]:
        return  # Skip if no frequency data exists
        
    best = results[0]
    freq_data = best["Frequency_Data"]
    
    f = np.array(freq_data["Frequency_Hz"])
    omega = np.array(freq_data["Angular_Frequency_rads"])
    E_prime = np.array(freq_data["Storage_Modulus_Pa"])
    E_double_prime = np.array(freq_data["Loss_Modulus_Pa"])
    mag = np.array(freq_data["Magnitude_Pa"])
    tan_delta = np.array(freq_data["Loss_Tangent"])
    
    # Calculate E(t) from the model parameters to plot against inverse frequency
    from indentation.viscoelastic.models import MODELS
    model = MODELS[model_name]
    E_relax_func = model["E_relax_func"]
    args = []
    for p_name in model["param_names"]:
        for key, val in best["parameters"].items():
            if key.startswith(p_name):
                args.append(val)
                break
                
    # Equivalent time scale: t = 1 / omega
    t_eq = 1.0 / omega
    E_t = E_relax_func(t_eq, *args)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f"Frequency Domain Master Curves - {model_name} ({best['method']})", fontsize=16, fontweight='bold')
    
    eps = 1e-12
    
    # Plot 1: Master Curve (Log-Log)
    ax1 = axes[0, 0]
    ax1.loglog(omega, np.maximum(E_prime, eps), 'b-', lw=2, label="Storage Modulus E'(ω)")
    ax1.loglog(omega, np.maximum(E_double_prime, eps), 'r-', lw=2, label="Loss Modulus E''(ω)")
    ax1.loglog(omega, np.maximum(mag, eps), 'k--', lw=1.5, label="Magnitude |E*|")
    ax1.set_xlabel('Angular Frequency, ω [rad/s]')
    ax1.set_ylabel('Modulus [Pa]')
    ax1.set_title('Dynamic Modulus Master Curve')
    ax1.legend()
    ax1.grid(True, which="both", ls="-", alpha=0.2)
    
    # Plot 2: Tan(delta) and Damping Ratio (Log-Log)
    ax2 = axes[0, 1]
    ax2.loglog(omega, np.maximum(tan_delta, eps), 'g-', lw=2, label="Loss Tangent (tan δ)")
    ax2.loglog(omega, np.maximum(tan_delta/2.0, eps), 'm--', lw=1.5, label="Damping Ratio (ζ)")
    ax2.set_xlabel('Angular Frequency, ω [rad/s]')
    ax2.set_ylabel('Damping Parameters')
    ax2.set_title('Damping Characteristics')
    ax2.legend()
    ax2.grid(True, which="both", ls="-", alpha=0.2)
    
    # Plot 3: Cole-Cole Plot (E'' vs E')
    ax3 = axes[1, 0]
    ax3.plot(E_prime, E_double_prime, 'c-o', lw=2, markersize=4)
    ax3.set_xlabel("Storage Modulus E' [Pa]")
    ax3.set_ylabel("Loss Modulus E'' [Pa]")
    ax3.set_title('Cole-Cole Plot')
    ax3.grid(True, ls="-", alpha=0.2)
    
    # Plot 4: Time-Frequency Equivalence (E(t) and E' vs Inverse Frequency)
    ax4 = axes[1, 1]
    # Plot E(t) vs equivalent time
    ax4.loglog(np.maximum(t_eq, eps), np.maximum(E_t, eps), 'b-', lw=2, label="Relaxation Modulus E(t)")
    # Plot E' vs equivalent time (1/omega)
    ax4.loglog(np.maximum(t_eq, eps), np.maximum(E_prime, eps), 'r--', lw=2, label="Storage Modulus E'(ω) vs 1/ω")
    ax4.set_xlabel('Inverse Angular Frequency (1/ω ≈ t) [s]')
    ax4.set_ylabel('Modulus [Pa]')
    ax4.set_title('Time-Frequency Equivalence')
    ax4.legend()
    ax4.grid(True, which="both", ls="-", alpha=0.2)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_path = os.path.join(output_dir, 'frequency_master_plots.png')
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)
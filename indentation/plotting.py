import os
import numpy as np
import matplotlib.pyplot as plt

def _get_long_path(path: str) -> str:
    """Bypass Windows 260-character path limit by prepending \\?\."""
    if os.name == 'nt' and not path.startswith('\\\\?\\'):
        return '\\\\?\\' + os.path.abspath(path)
    return path
    

def plot_comparison(results, t_full, F_full, model_name, output_dir="prediction_results"):
    """
    Generates a 2-panel plot:
    1. Experimental vs. Model Force
    2. Residuals of the best fit
    """
    output_dir = _get_long_path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12']
    
    # Left Plot: Force vs Time
    axes[0].plot(t_full, F_full, 'k-', lw=2.5, label='Experimental Data')
    best = results[0]
    
    for i, res in enumerate(results):
        c = colors[i % len(colors)]
        ls = ':' if res == best else '--'
        lw = 2.5 if res == best else 1.5
        alpha = 0.7 if res == best else 0.6
        label = f"{res['method']} (BEST)" if res == best else res['method']
        axes[0].plot(t_full, res['F_model'], ls, lw=lw, alpha=alpha, color=c, label=label)
        
    axes[0].set_xlabel('Time [s]')
    axes[0].set_ylabel('Force [N]')
    axes[0].set_title(f"Model Fit: {model_name}")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Right Plot: Residuals
    axes[1].plot(t_full, best['F_model'] - F_full, lw=1.5, color='r', label='Residuals (Best)')
    axes[1].axhline(0, color='k', ls='-', alpha=0.3)
    axes[1].set_xlabel('Time [s]')
    axes[1].set_ylabel('Residual Force [N]')
    axes[1].set_title(f"Residuals ({best['method']})")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    plot_path = os.path.join(output_dir, 'comparison_plots.png')
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)  # Close to free up memory
    
    print(f"Plot saved to {os.path.abspath(plot_path)}")
    return plot_path
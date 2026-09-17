import numpy as np
import sys
import os
import shutil
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from indentation.plotting import plot_comparison, plot_frequency_domain

def test_plotting():
    """Test plot generation."""
    test_dir = "test_plotting_temp"
    os.makedirs(test_dir, exist_ok=True)
    
    # Dummy data
    t_full = np.linspace(0, 10, 100)
    h_full = 0.1 * t_full  # Added dummy displacement
    F_full = 100 * t_full
    
    # Dummy results (Now includes model and parameters for the E(t) plot)
    results = [
        {
            "model": "Elastic_Spring",
            "method": "Time-Domain",
            "parameters": {"E [Pa]": 1e6},
            "F_model": F_full * 0.98,  # Slightly off
        },
        {
            "model": "Elastic_Spring",
            "method": "Laplace",
            "parameters": {"E [Pa]": 0.95e6},
            "F_model": F_full * 0.95,  # More off
        }
    ]
    
    # Pass h_full into the function call
    plot_path = plot_comparison(results, t_full, h_full, F_full, "Elastic_Spring", output_dir=test_dir)
    
    if not os.path.exists(plot_path):
        print("[FAIL] Plot file was not created.")
        sys.exit(1)
        
    file_size = os.path.getsize(plot_path)
    if file_size > 100:
        print(f"[PASS] Plot generated successfully ({file_size} bytes)")
    else:
        print(f"[FAIL] Plot file is too small ({file_size} bytes)")
        sys.exit(1)
        
    # Cleanup
    shutil.rmtree(test_dir)

if __name__ == "__main__":
    test_plotting()
    print("\nTask 11 (Plotting) Complete and Verified!")
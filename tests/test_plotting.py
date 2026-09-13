import numpy as np
import sys
import os
import shutil
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from indentation.plotting import plot_comparison

def test_plotting():
    """Test plot generation."""
    test_dir = "test_plotting_temp"
    os.makedirs(test_dir, exist_ok=True)
    
    # Dummy data
    t_full = np.linspace(0, 10, 100)
    F_full = 100 * t_full
    
    # Dummy results
    results = [
        {
            "method": "Time-Domain",
            "F_model": F_full * 0.98,  # Slightly off
        },
        {
            "method": "Laplace",
            "F_model": F_full * 0.95,  # More off
        }
    ]
    
    plot_path = plot_comparison(results, t_full, F_full, "DummyModel", output_dir=test_dir)
    
    if not os.path.exists(plot_path):
        print("[FAIL] Plot file was not created.")
        sys.exit(1)
        
    file_size = os.path.getsize(plot_path)
    if file_size > 100:
        print(f"[PASS] Plot generated successfully ({file_size} bytes)")
    else:
        print(f"[FAIL] Plot file is too small ({file_size} bytes)")
        sys.exit(1)
        
    shutil.rmtree(test_dir)

if __name__ == "__main__":
    test_plotting()
    print("\nTask 11 (plotting.py) Complete and Verified!")
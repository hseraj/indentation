import numpy as np
import sys
import os
import json
import shutil

# Add project root to path to import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from indentation.io import load_data, save_results

def test_io2():
    """Test CSV loading, NaN handling, and non-uniform time resampling."""
    test_dir = "test_io2_temp"
    csv_path = os.path.join(test_dir, "dummy_data.csv")
    
    os.makedirs(test_dir, exist_ok=True)
    
    # 1. Write a dummy CSV file with NON-UNIFORM time steps and a NaN row
    with open(csv_path, 'w') as f:
        f.write("Time,Displacement,Force\n") # Header
        f.write("0.0,0.0,0.0\n")
        f.write("0.1,0.1,100.0\n")
        f.write("0.5,0.2,200.0\n")   # Large time gap
        f.write("0.6,nan,nan\n")     # NaN row to be skipped
        f.write("0.62,0.3,300.0\n") # Small time gap
        f.write("1.0,0.4,400.0\n")
        
    # 2. Load CSV and verify resampling
    try:
        t, h, F = load_data(csv_path)
    except Exception as e:
        print(f"[FAIL] load_data raised an exception: {e}")
        sys.exit(1)
        
    # We should have 5 points (NaN removed). Original times: 0.0, 0.1, 0.5, 0.62, 1.0
    if len(t) == 5:
        print(f"[PASS] load_data (Loaded {len(t)} points, skipped header/NaN correctly)")
    else:
        print(f"[FAIL] load_data (Expected 5 points, got {len(t)})")
        sys.exit(1)
        
    # Verify time steps are now perfectly uniform
    dt_steps = np.diff(t)
    if np.allclose(dt_steps, dt_steps[0]):
        print(f"[PASS] load_data (Time steps successfully resampled to uniform dt={dt_steps[0]:.4f})")
    else:
        print(f"[FAIL] load_data (Time steps are non-uniform: {dt_steps})")
        sys.exit(1)
        
    # 3. Save dummy results
    dummy_results = [
        {
            "model": "TestModel",
            "method": "Time-Domain",
            "parameters": {"E": 1e6, "tau": 0.5},
            "F_model": np.array([0.0, 100.0, 200.0]),
            "RMSE_N": 1.5,
            "RelErr_%": 2.5
        }
    ]
    
    # Add dummy arrays
    t_dummy = np.array([0.0, 1.0, 2.0])
    F_dummy = np.array([0.0, 100.0, 200.0])
    
    # Update the save_results call
    out_dir = os.path.join(test_dir, "output")
    save_results(dummy_results, "TestModel", t_dummy, F_dummy, output_dir=out_dir)
    
    # 4. Verify output files exist
    txt_exists = os.path.exists(os.path.join(out_dir, "model_comparison_table.txt"))
    json_exists = os.path.exists(os.path.join(out_dir, "detailed_results.json"))
    
    if txt_exists and json_exists:
        print("[PASS] save_results (TXT and JSON files created)")
    else:
        print(f"[FAIL] save_results (TXT exists: {txt_exists}, JSON exists: {json_exists})")
        sys.exit(1)
        
    if os.path.exists(os.path.join(out_dir, "summary_results.csv")):
        print("[PASS] save_results (CSV file created)")
    else:
        print("[FAIL] save_results (CSV file missing)")
        sys.exit(1)
        
    
    # 5. Verify JSON contents
    with open(os.path.join(out_dir, "detailed_results.json"), 'r') as f:
        data = json.load(f)
        if "F_model" not in data["results"][0] and data["results"][0]["parameters"]["E"] == 1000000.0:
            print("[PASS] save_results (F_model excluded from JSON, E converted to float)")
        else:
            print("[FAIL] save_results (JSON formatting error)")
            sys.exit(1)
            
    # Cleanup
    shutil.rmtree(test_dir)

if __name__ == "__main__":
    test_io2()
    print("\nTask 2 (Non-Uniform Time Fix) Complete and Verified!")
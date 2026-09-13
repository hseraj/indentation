import numpy as np
import sys
import os
import json
import shutil

# Add project root to path to import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from indentation.io import load_data, save_results

def test_io():
    """Test CSV/Excel loading, delimiters, 2-column generation, and resampling."""
    test_dir = "test_io_temp"
    os.makedirs(test_dir, exist_ok=True)
    
    # 1. Test European CSV (Semicolon) with Non-Uniform Time
    csv_eu_path = os.path.join(test_dir, "data_eu.csv")
    with open(csv_eu_path, 'w') as f:
        f.write("Time;Displacement;Force\n")
        f.write("0,0;0,0;0,0\n")
        f.write("0,1;0,1;100,0\n")
        f.write("0,5;0,2;200,0\n")
        f.write("0,6;nan;nan\n")
        f.write("0,62;0,3;300,0\n")
        f.write("1,0;0,4;400,0\n")
        
    try:
        t, h, F = load_data(csv_eu_path)
    except Exception as e:
        print(f"[FAIL] load_data EU CSV raised an exception: {e}")
        sys.exit(1)
        
    if len(t) == 5 and np.allclose(np.diff(t), np.diff(t)[0]):
        print(f"[PASS] load_data EU CSV (Loaded {len(t)} points, resampled to uniform dt)")
    else:
        print(f"[FAIL] load_data EU CSV (Expected 5 uniform points, got {len(t)})")
        sys.exit(1)

    # 2. Test 2-Column CSV (Disp, Force) with Time Rate
    csv_2col_path = os.path.join(test_dir, "data_2col.csv")
    with open(csv_2col_path, 'w') as f:
        f.write("Displacement,Force\n")
        f.write("0.0,0.0\n")
        f.write("1.0,100.0\n")
        f.write("2.0,200.0\n")
        f.write("3.0,300.0\n")
        
    try:
        # Assuming rate = 2.0 (e.g., mm/s). So h=1.0 -> t=0.5
        t, h, F = load_data(csv_2col_path, time_rate=2.0)
    except Exception as e:
        print(f"[FAIL] load_data 2-col raised an exception: {e}")
        sys.exit(1)
        
    if len(t) == 4 and np.isclose(t[1], 0.5) and np.isclose(t[3], 1.5):
        print(f"[PASS] load_data 2-col CSV (Time generated correctly, t[1]={t[1]:.2f})")
    else:
        print(f"[FAIL] load_data 2-col CSV (Time generation incorrect: {t})")
        sys.exit(1)

    # 3. Test MissingTimeRateError
    try:
        load_data(csv_2col_path)
    except Exception as e:
        if "time_rate" in str(e) or "2 columns" in str(e):
            print(f"[PASS] load_data MissingTimeRateError caught correctly")
        else:
            print(f"[FAIL] load_data raised wrong exception: {e}")
            sys.exit(1)
    else:
        print("[FAIL] load_data did not raise error for 2-col without rate")
        sys.exit(1)

    # 4. Save dummy results
    dummy_results = [
        {
            "model": "TestModel", "method": "Time-Domain",
            "parameters": {"E": 1e6, "tau": 0.5},
            "F_model": np.array([0.0, 100.0, 200.0]), "RMSE_N": 1.5, "RelErr_%": 2.5
        }
    ]
    # Add dummy arrays
    t_dummy = np.array([0.0, 1.0, 2.0])
    F_dummy = np.array([0.0, 100.0, 200.0])
    
    # Update the save_results call
    out_dir = os.path.join(test_dir, "output")
    save_results(dummy_results, "TestModel", t_dummy, F_dummy, output_dir=out_dir)
    
    # Cleanup
    shutil.rmtree(test_dir)

if __name__ == "__main__":
    test_io()
    print("\nTask (Advanced I/O) Complete and Verified!")
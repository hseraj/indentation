import numpy as np
import sys
import os

# Add project root to path to import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from indentation.probes import get_probe_geometry

def test_probes():
    """Test all 5 probe geometries against their analytical formulas."""
    nu = 0.5
    R = 2.0e-3
    theta_deg = 30.0
    theta_rad = np.radians(theta_deg)
    
    # 1. Spherical
    C, exp = get_probe_geometry("Spherical", {"nu": nu, "R": R})
    expected_C = (4.0 * np.sqrt(R)) / (3.0 * (1.0 - nu**2))
    if np.isclose(C, expected_C) and exp == 1.5:
        print(f"[PASS] Spherical (C={C:.4f}, exp={exp})")
    else:
        print(f"[FAIL] Spherical (Expected C={expected_C:.4f}, exp=1.5; Got C={C:.4f}, exp={exp})")
        sys.exit(1)
        
    # 2. Cylindrical
    C, exp = get_probe_geometry("Cylindrical", {"nu": nu, "R": R})
    expected_C = (2.0 * R) / (1.0 - nu**2)
    if np.isclose(C, expected_C) and exp == 1.0:
        print(f"[PASS] Cylindrical (C={C:.4f}, exp={exp})")
    else:
        print(f"[FAIL] Cylindrical (Expected C={expected_C:.4f}, exp=1.0; Got C={C:.4f}, exp={exp})")
        sys.exit(1)
        
    # 3. Conical
    C, exp = get_probe_geometry("Conical", {"nu": nu, "theta_deg": theta_deg})
    expected_C = (2.0 * np.tan(theta_rad)) / (np.pi * (1.0 - nu**2))
    if np.isclose(C, expected_C) and exp == 2.0:
        print(f"[PASS] Conical (C={C:.4f}, exp={exp})")
    else:
        print(f"[FAIL] Conical (Expected C={expected_C:.4f}, exp=2.0; Got C={C:.4f}, exp={exp})")
        sys.exit(1)
        
    # 4. Four-sided Pyramidal
    C, exp = get_probe_geometry("Four-sided Pyramidal", {"nu": nu, "theta_deg": theta_deg})
    expected_C = (3.0 * np.tan(theta_rad)) / (4.0 * (1.0 - nu**2))
    if np.isclose(C, expected_C) and exp == 2.0:
        print(f"[PASS] Four-sided Pyramidal (C={C:.4f}, exp={exp})")
    else:
        print(f"[FAIL] Four-sided Pyramidal (Expected C={expected_C:.4f}, exp=2.0; Got C={C:.4f}, exp={exp})")
        sys.exit(1)
        
    # 5. Blunted Four-sided Pyramidal
    C, exp = get_probe_geometry("Blunted Four-sided Pyramidal", {"nu": nu, "theta_deg": theta_deg})
    expected_C = np.tan(theta_rad) / (np.sqrt(2) * (1.0 - nu**2))
    if np.isclose(C, expected_C) and exp == 2.0:
        print(f"[PASS] Blunted Four-sided Pyramidal (C={C:.4f}, exp={exp})")
    else:
        print(f"[FAIL] Blunted Four-sided Pyramidal (Expected C={expected_C:.4f}, exp=2.0; Got C={C:.4f}, exp={exp})")
        sys.exit(1)

    # Test invalid inputs
    try:
        get_probe_geometry("Spherical", {"nu": 0.9, "R": 1.0})
    except ValueError:
        pass # This is expected
    else:
        print("[FAIL] Did not raise ValueError for invalid nu")
        sys.exit(1)

if __name__ == "__main__":
    test_probes()
    print("\nTask 4 (probes.py) Complete and Verified!")
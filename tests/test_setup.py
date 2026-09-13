
import sys
import subprocess

def test_package_imports():
    """Test that the package can be imported."""
    try:
        import indentation
        print("[PASS] Package 'indentation' imported successfully.")
    except ImportError as e:
        print(f"[FAIL] Failed to import package: {e}")
        sys.exit(1)

def test_dependencies():
    """Test that required external packages are available."""
    try:
        import numpy
        import scipy
        import matplotlib
        print("[PASS] All dependencies (numpy, scipy, matplotlib) are available.")
    except ImportError as e:
        print(f"[FAIL] Missing dependency: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_package_imports()
    test_dependencies()
    print("\nTask 1 Setup Complete and Verified!")
# Indentation

A Python package for evaluating parameters of viscoelastic models in indentation tests. It uses a fast Laplace-domain estimation as a warm-start for a highly accurate Time-Domain (hereditary integral) optimization, achieving scale-invariant fitting via log10-parameter transformation.

Whether you are characterizing soft hydrogels, biological tissues, or stiff polymers, this package provides a rigorous, mathematically robust pipeline to extract viscoelastic properties from raw experimental data.

## Mathematical Background

The package solves the inverse problem of finding the relaxation modulus $E(t)$ that best fits the experimental indentation force $F(t)$. The forward model is based on the Boltzmann superposition principle (hereditary integral) for a rigid indenter of arbitrary geometry:

$$ F(t) = C \int_0^t E(t-\tau) \frac{d}{d\tau} [h(\tau)]^n d\tau $$

Where:
- $C$ and $n$ are the contact coefficient and displacement exponent, respectively (dependent on probe geometry).
- $h(t)$ is the indenter displacement.
- $E(t)$ is the material's relaxation modulus.

To avoid numerical instability and slow convergence, the package optimizes the parameters in log10-space ($x = \log_{10} p$), ensuring scale invariance across magnitudes of stiffness and viscosity.

## Probe Geometries and Contact Mechanics

The package supports five standard probe geometries. The relationship between force and displacement is defined as $F(t) \propto C \cdot h(t)^n$. The equations for the contact coefficient $C$ (assuming a Poisson's ratio $\nu$) are implemented as follows [1–4]:

- **Spherical**: $C = \frac{4 \sqrt{R}}{3(1-\nu^2)}$, $n = 1.5$
- **Cylindrical**: $C = \frac{2 R}{1-\nu^2}$, $n = 1.0$
- **Conical**: $C = \frac{2 \tan\theta}{\pi(1-\nu^2)}$, $n = 2.0$
- **Four-sided Pyramidal**: $C = \frac{3 \tan\theta}{4(1-\nu^2)}$, $n = 2.0$
- **Blunted Four-sided Pyramidal**: $C = \frac{\tan\theta}{\sqrt{2}(1-\nu^2)}$, $n = 2.0$

## Viscoelastic Models

The relaxation modulus $E(t)$ is parameterized using standard rheological models. The package also utilizes the Carson transform $s \cdot E(s)$ (where $E(s)$ is the Laplace transform of $E(t)$) for the Laplace-domain fitting. The implemented models are:

- **Elastic Spring**: $E(t) = E$
- **Dashpot**: $E(t) = \eta \delta(t)$
- **Kelvin-Voigt**: $E(t) = E H(t) + \eta \delta(t)$
- **Maxwell**: $E(t) = E \exp(-t/\tau)$
- **Standard Linear Solid (SLS)**: $E(t) = E_1 + E_2 \exp(-t/\tau)$
- **Prony Series (N=1, 2, 3)**: $E(t) = E_\infty + \sum_{i=1}^{N} E_i \exp(-t/\tau_i)$

*(Note: $H(t)$ is the Heaviside step function and $\delta(t)$ is the Dirac delta function).*

## Key Features

- **8 Rheological Models**: Elastic Spring, Dashpot, Kelvin-Voigt, Maxwell, Standard Linear Solid (SLS), and generalized Prony Series (N=1, 2, 3).
- **5 Probe Geometries**: Spherical, Cylindrical, Conical, Four-sided Pyramidal, and Blunted Four-sided Pyramidal.
- **Two-Stage Optimization Pipeline**: 
  1. **Laplace Domain**: Computes the Carson transform to rapidly estimate a global solution without initial user guesses.
  2. **Time-Domain**: Uses the Laplace solution as a warm-start for a highly accurate FFT-based hereditary integral optimization.
- **Scale-Invariant Fitting**: Log10-parameter optimization ensures stable convergence across magnitudes, preventing failures when fitting soft (kPa) or stiff (GPa) materials.
- **Robust Noise Handling**: Utilizes analytical Savitzky-Golay derivatives (`deriv=1`) to mathematically suppress high-frequency noise in the displacement signal without introducing numerical discretization errors.
- **Statistical Rigor**: Automatically calculates and reports Standard Error and 95% Confidence Intervals (CI) for all fitted Time-Domain parameters using the covariance matrix and log-space Hessian.
- **Comprehensive I/O**:
  - Handles European CSVs (semicolons/commas) and Excel (`.xlsx`/`.xls`) files natively.
  - Auto-detects and resamples non-uniform time steps to a uniform grid.
  - Supports 2-column data (Displacement, Force) with auto-generation of the time array based on a constant loading rate.
- **Batch Processing**: Process an entire folder of indentation tests at once. Generates individual plots/JSONs for each file, plus a master `batch_summary.csv` and `.xlsx` containing the Mean and Standard Deviation of all parameters.
- **Multiple User Interfaces**: Interactive CLI, Python API, and a standalone Graphical User Interface (GUI).

## Installation

### Prerequisites
- Python >= 3.11
- NumPy, SciPy, Matplotlib, Pandas, OpenPyxl

### Setup
Clone the repository and install locally in editable mode:

```bash
git clone https://github.com/hseraj/indentation.git
cd indentation
pip install -e .
```

## Usage

The package offers three ways to process your data, depending on your workflow:

### 1. Graphical User Interface (GUI)
Ideal for researchers without a programming background. 

- **Standalone Executables (No Python Required):** 
  Pre-compiled standalone executables for **Windows, macOS, and Linux** are automatically built and provided on the [GitHub Releases page](https://github.com/hseraj/indentation/releases). 
  Simply download the file for your operating system, double-click to run, and start analyzing your data immediately without installing Python or any dependencies.
  
- **Run from Python:**
  If you prefer using Python, you can launch the GUI directly from your terminal:
  ```bash
  python -m indentation.gui
  ```
  *(Developers can rebuild the executables from source using PyInstaller: `pyinstaller --noconsole --onefile --name Indentation indentation/gui.py`)*

### 2. Interactive CLI
Run the interactive command-line tool:
```bash
python -m indentation.cli
```
The CLI will prompt you to choose between Single File processing or Batch Folder processing, guiding you through model selection, probe parameters, and smoothing options.

### 3. Python API
For full programmatic control and integration into custom data analysis scripts. See `examples/viscoelastic/basic_fit.py` and `examples/viscoelastic/basic_fit2.py` for a full tutorial.

```python
import numpy as np
from indentation import run_fit, get_probe_geometry, load_data

# 1. Define probe
C, h_power_exp = get_probe_geometry("Spherical", {"nu": 0.5, "R": 2.0e-3})

# 2. Load your data (t, h, F)
# If your file has 2 columns (Disp, Force), provide the loading rate:
# t_arr, h_arr, F_data = load_data("data.csv", time_rate=1.0e-4)
t_arr, h_arr, F_data = load_data("data.csv")

# 3. Run pipeline
S_VALS = np.sort(np.unique(np.logspace(-3, 3, 110)))
results = run_fit(
    model_name="Maxwell",
    t_full=t_arr, h_full=h_arr, F_full=F_data,
    C=C, h_power_exp=h_power_exp,
    S_VALS=S_VALS,
    smooth_window=15, # Use analytical SG derivative for noisy data
    refine_full_data=True
)

# 4. Get best parameters
best = results[0]
print(f"Best method: {best['method']}")
print(f"Parameters: {best['parameters']}")
print(f"95% CI: {best.get('CI', {}).get('95%_CI')}")
```

## Outputs

For every processed file, the package generates a comprehensive set of outputs in your designated folder:
- **`model_comparison_table.txt`**: A clean, human-readable summary of metrics and parameters (with `±` CI).
- **`detailed_results.json`**: Machine-readable output containing all parameters, metrics, and covariance data.
- **`summary_results.csv` & `full_results.xlsx` & `force_comparison.csv`**: Tabular data of the summary and a sheet containing the Time, Experimental Force, and Model Force arrays for easy plotting in Origin or MATLAB.
- **`comparison_plots.png`**: A high-quality 2-panel matplotlib figure showing the Force vs. Time fit and the Residuals.

## Benchmark Dataset Generator

To validate the package, a mathematically flawless synthetic data generator is included in `examples/generate_simulated_data.py` and `examples/generate_simulated_data2.py`. It generates an exhaustive dataset testing 7 models, 5 probes, uniform/non-uniform sampling, and 3 noise levels (0%, 2%, 5%) using analytical derivatives to avoid boundary errors.

## Assumptions & Limitations

- **Linear Viscoelasticity (LVE):** The package assumes the material behaves linearly within the applied strain range.
- **Frictionless Contact:** Assumes no sliding friction between the indenter and the sample.
- **Instantaneous Models:** Models that rely purely on instantaneous derivatives (Dashpot, Kelvin-Voigt) are highly sensitive to displacement noise because differentiation acts as a high-pass filter. Using the `smooth_window` parameter applies an analytical Savitzky-Golay derivative, ensuring robust fitting even for these models.

## Project Structure

The repository is organized to separate the core mathematical engine from the user interfaces, tests, and examples:

```text
indentation_project/
│
├── indentation/                  # The core Python package
│   ├── __init__.py               # Public API exports
│   ├── core.py                   # Main pipeline (run_fit) & batch loop
│   ├── io.py                     # Robust CSV/Excel loading & saving
│   ├── plotting.py               # Matplotlib comparison & residual plots
│   ├── cli.py                    # Interactive Single/Batch command-line menu
│   ├── transforms.py             # FFT convolution, analytical SG derivatives, metrics
│   ├── optimize.py               # Log10 <-> Physical space conversions
│   ├── probes.py                 # Probe geometry contact coefficients
│   ├── gui.py                    # Tkinter GUI application source code
│   └── viscoelastic/
│       ├── __init__.py
│       ├── models.py             # 8 rheological models registry
│       └── fitting.py            # Laplace & Time-Domain fitters
│
├── tests/                        # Pytest scripts for continuous integration
│   ├── test_setup.py             # Validates package installation and dependencies
│   ├── test_transforms.py        # Validates FFT convolution and SG derivatives
│   ├── test_optimize.py          # Validates Log10 <-> Physical space conversions
│   ├── test_probes.py            # Validates contact coefficients for all 5 probes
│   ├── test_models.py            # Validates the registry of the 8 viscoelastic models
│   ├── test_io.py                # Validates CSV/Excel parsing, European decimals, and 2-col data
│   ├── test_io2.py               # Additional I/O tests (non-uniform time resampling)
│   ├── test_fitting_closed.py    # Validates Elastic, Dashpot, Kelvin-Voigt closed-form fitting
│   ├── test_fitting_nonlinear.py # Validates Maxwell & SLS optimization with warm-start
│   ├── test_fitting_prony.py     # Validates Prony N=1, N=2, N=3 fitting pipelines
│   └── test_core_pipeline.py     # End-to-end validation of the Laplace -> TD -> Refinement logic
│
├── examples/                     # Tutorials, benchmark datasets, and outputs
│   ├── .gitkeep
│   └── viscoelastic/                    # All viscoelastic specific examples and benchmarks
│       ├── basic_fit.py                 # Python API tutorial (clean synthetic SLS data)
│       ├── basic_fit2.py                # Python API tutorial (noisy data & smoothing application)
│       ├── generate_simulated_data.py   # Benchmark generator (Dashpot, KV, Maxwell, SLS for Spherical/Conical)
│       ├── generate_simulated_data2.py  # Benchmark generator (Prony series for Cylindrical/Pyramidal probes)
│       ├── basic_fit output/            # Output folder generated by basic_fit.py
│       ├── basic_fit2 output/           # Output folder generated by basic_fit2.py
│       ├── simulated_data/              # Benchmark datasets & verified fitting results (Dashpot, KV, Maxwell, SLS)
│       └── simulated_data2/             # Benchmark datasets & verified fitting results (Prony series)
│
├── .github/                             # GitHub Actions CI for auto-building Windows/Mac/Linux executables
│   └── workflows/
│       └── build_release.yml
│
├── .gitignore
├── CITATION.cff                  # GitHub citation metadata
├── LICENSE                       # MIT License
├── pyproject.toml                # Package configuration and dependencies
└── README.md                     # Documentation
```

## References

[1] I.N. Sneddon, *The relation between load and penetration in the axisymmetric boussinesq problem for a punch of arbitrary profile*, International Journal of Engineering Science 3(1) (1965) 47-57.
[2] G.M. Pharr, W.C. Oliver, F.R. Brotzen, *On the generality of the relationship among contact stiffness, contact area, and elastic modulus during indentation*, Journal of Materials Research 7(3) (1992) 613-617.
[3] J. Alcaraz, L. Buscemi, M. Grabulosa, X. Trepat, B. Fabry, R. Farré, D. Navajas, *Microrheology of Human Lung Epithelial Cells Measured by Atomic Force Microscopy*, Biophysical Journal 84(3) (2003) 2071-2079.
[4] F. Rico, P. Roca-Cusachs, N. Gavara, R. Farré, M. Rotger, D. Navajas, *Probing mechanical properties of living cells by atomic force microscopy with blunted pyramidal cantilever tips*, Physical Review E 72(2) (2005) 021914.

## Citation

If you use this software in your research, please cite the associated article (currently a preprint):

```bibtex
@article{azadi2025indentation,
  title={Unpacking Variations of Elastic Modulus in Time and Frequency Domain Detected by Indentation: Review of Models, Errors, Experimental Insights and Nonlinearity Considerations},
  author={Azadi, Mojtaba and Seraj, Hasan and Sokolov, Igor},
  year={2025},
  url={https://dx.doi.org/10.2139/ssrn.5464116}
}
```

## Acknowledgements
This software was developed with the assistance of AI tools for code generation, debugging, and documentation drafting. The scientific methodology, mathematical formulations, and final validation were conducted and reviewed by the authors.

## License
MIT License. Copyright (c) 2026 Hasan Seraj, Mojtaba Azadi, Igor Sokolov.

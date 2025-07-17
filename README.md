# IAM-XED
**Independent Atom Model for X-ray and Electron Diffraction**

A Python package for calculating X-ray diffraction (XRD) and ultrafast electron diffraction (UED) signals from molecular geometries and trajectories using the Independent Atom Model (IAM) approximation.

## Features

- **XRD calculations** with optional inelastic scattering
- **UED calculations** with pair distribution function (PDF) analysis
- **Static and time-resolved** diffraction simulations
- **Ensemble averaging** for multiple trajectories
- **Difference calculations** for pump-probe experiments
- **Flexible input formats** (single geometries, trajectories, ensembles)

## Installation

### From Source (Recommended)
```bash
git clone https://github.com/blevine37/IAM-XED.git
cd IAM-XED
pip install -e .
```

The `-e` flag installs in editable mode for development. For regular use, omit this flag:
```bash
pip install .
```

### Dependencies
Ensure you have Python ≥ 3.7 and the required packages:
```bash
pip install numpy scipy matplotlib tqdm pytest
```

### Testing
Run the tests to ensure everything is 
working correctly. 

```bash
pytest -v
```

## Quick start


### Input files: Signal Geometries (`--signal-geoms`)
*All geometries must be in XYZ format with coordinates in Angstroms.*


**Single XYZ file:**
- **Static** (default): Takes the geometry or averages all geometries in the file and calculates static signal
- **Time-resolved** (`--calculation-type time-resolved`): Treats all geometries in the file as a trajectory with `--timestep` intervals

**Directory of XYZ files:**
- **Static** (default): Averages first geometry from each file  
- **Time-resolved** (`--calculation-type time-resolved`): Each XYZ file represents a trajectory with `--timestep` intervals to be averaged as an ensemble member

**Examples:**
```bash
# Single molecule
iamxed --xrd --signal-geoms molecule.xyz

# Trajectory 
iamxed --ued --calculation-type time-resolved --signal-geoms traj.xyz

# Ensemble averaging
iamxed --xrd --calculation-type time-resolved --signal-geoms ./ensemble_dir/
```

## Usage

See all available options:
```bash
iamxed --help
```

### XRD Calculations

**Single Geometry:**
```bash
iamxed --xrd --signal-geoms molecule.xyz
```
Calculates scattering intensity I(q) as a function of momentum transfer q (Bohr⁻¹)

**Difference Signal:**
```bash
iamxed --xrd --signal-geoms excited.xyz --reference-geoms ground.xyz
```
Calculates the relative difference signal: ΔI/I₀ = (I₁-I₀)/I₀ × 100% (I₁ - signal-geoms, I₀ - reference-geoms)

**Inlcuding Inelastic Scattering:**
```bash
iamxed --xrd --signal-geoms molecule.xyz --inelastic
```
Includes Compton scattering using Szaloki parameters

**Time-resolved Trajectory Calulation:**
```bash
iamxed --xrd --signal-geoms trajectory.xyz --calculation-type time-resolved --qmin 0.0 --qmax 10.0 --npoints 100 --timestep 40
```
Calculates the time-resolved relative difference scattering signal ΔI/I₀(q,t) against the t=0 frame. Momentum coordinate divided to 100 points goes from 0.0 to 10.0 Bh-1. Timestep is assumed 40 a.t.u.

**Time-resolved Ensemble Calulation:**
```bash
iamxed --xrd --signal-geoms ./ensemble_dir/ --calculation-type time-resolved --qmin 0.0 --qmax 10.0 --npoints 100 --timestep 40 --tmax 500
```
Calculates the same signal as in trajectory case, averaging over all trajectories in the ./ensemble_dir/ folder up to 500 fs.

### UED Calculations
*Momentum coordinate in plots and export is labeled 's' for UED*

**Single Geometry:**
```bash
iamxed --ued --signal-geoms molecule.xyz
```
Calculates the modified scattering intensity sM(s) = s × I_mol/I_at and real-space pair distribution function PDF(r) =  r × ∫[s_min to s_max] sM(s) × sin(s × r) × exp(-alpha×s^2) ds. 


**Difference Signal:**
```bash
iamxed --ued --signal-geoms excited.xyz --reference-geoms ground.xyz
```
Calculates the relative difference signal: ΔI/I₀ = (I₁-I₀)/I₀ × 100% (I₁ - signal-geoms, I₀ - reference-geoms, I = sM(s)) and ΔPDF(r) = PDF₁(r)-PDF₀(r)

**Time-Resolved Trajectory Calculation:**
```bash
iamxed --ued --calculation-type time-resolved --signal-geoms trajectory.xyz --timestep 40 --fwhm 100 --pdf-alpha 0.04
```
Calculates time-resolved relative difference signal ΔI/I₀(s,t) and ΔPDF(r,t) against the t=0 frame. Timestep is set to 40 a.t.u., additional temporal smoothing with 100 fs FWHM Gaussian function, alpha smearing parameter at 0.04 Å².

**Time-resolved Ensemble Calulation:**
```bash
iamxed --ued --calculation-type time-resolved --signal-geoms ./ensemble_dir/ --timestep 40 --fwhm 100 --pdf-alpha 0.04
```
Calculates the same signal as in trajectory case, averaging over all trajectories in the ./ensemble_dir/ folder.

### Key Options

| Option | Description | Default |
|--------|-------------|---------|
| `--calculation-type` | `static` or `time-resolved` | `static` |
| `--qmin`, `--qmax` | Momentum transfer range (Bohr⁻¹) | 5.29e-9, 5.29 |
| `--npoints` | Number of q-points | 200 |
| `--timestep` | Time step (atomic time units) | 10.0 |
| `--fwhm` | Temporal smoothing FWHM (fs) | 150.0 |
| `--pdf-alpha` | PDF damping parameter (Å²) | 0.04 |
| `--tmax` | Maximum time (fs) | None |
| `--export` | Export data filename | None |
| `--plot-units` | `bohr-1` or `angstrom-1` | `bohr-1` |
| `--plot-flip` | Flip x and y axis | False |


## Output Files

### Static Calculations
- `export.txt`: Signal data with units in header
- `export_PDF.txt`: PDF data (UED only)

### Time-Resolved Calculations  
- `export.npz`: Binary archive containing:
  - `times`, `times_smooth`: Time axis (fs)
  - `q`/`s`: Momentum transfer axis (Bohr⁻¹) 
  - `signal_raw`, `signal_smooth`: Diffraction signals
  - `r`, `pdfs_raw`, `pdfs_smooth`: PDF data (UED only)
  - `metadata`: Command and units information

```python
# Load time-resolved data
import numpy as np
data = np.load('results.npz')
metadata = data['metadata']  # Command and units
times = data['times']        # Time points (fs)
signal = data['signal_raw']  # Raw signal
```

## Code Structure

```
src/iamxed/
├── iamxed.py      # Main entry point and CLI
├── physics.py     # Calculator classes (XRD/UED)
├── io_utils.py    # File I/O and argument parsing
├── plotting.py    # Visualization functions
├── XSF/           # X-ray scattering form factors
└── ESF/           # Electron scattering form factors
```

## Dependencies

- **Python** ≥ 3.7
- **NumPy** - Array operations
- **SciPy** - Interpolation and filtering  
- **Matplotlib** - Plotting
- **tqdm** - Progress bars
- **pytest** - Testing (development)

## Python API

You can use IAM-XED as a library in your Python scripts:

```python
from iamxed import iamxed
from argparse import Namespace

params = {
    "signal_geoms": "./ensemble/",
    "reference_geoms": None,
    "calculation_type": "time-resolved",
    "ued": True,
    "xrd": False,
    "inelastic": False,
    "qmin": 0,
    "qmax": 4,
    "npoints": 200,
    "timestep": 20.0,
    "fwhm": 120.0,
    "pdf_alpha": 0.04,
    "tmax": False,
    "export": "ued_ensemble",
    "log_to_file": False,
    "plot_disable": True,
    "plot_flip": False,
    "plot_units": "bohr-1",
    "debug": True
}

params = Namespace(**params)

iamxed(params)
```

## License

MIT License - see LICENSE file for details.

## Authors

- **Jiri Suchan** 
- **Jiri Janos**

## Citation

If you use IAM-XED in your research, please cite:

```bibtex
@software{iam_xed,
  title = {IAM-XED: Independent Atom Model for X-ray and Electron Diffraction},
  author = {Suchan, Jiri and Janos, Jiri},
  url = {https://github.com/blevine37/IAM-XED},
  year = {2025}
}
```

The IAM parameters for XRD reference:
```
Prince, E. (Ed.). (2004). International Tables for Crystallography, Volume C: Mathematical, physical and chemical tables. Springer Science & Business Media. 
ISBN 1-4020-1900-9 
```
The IAM parameters for UED were calculated using the ELSEPA program assuming 3.7 MeV electron:
```
https://github.com/eScatter/elsepa (commit 98862ff)
Salvat, F., Jablonski, A., & Powell, C. J. (2005). ELSEPA—Dirac partial-wave calculation of elastic scattering of electrons and positrons by atoms, positive ions and molecules. Computer physics communications, 165(2), 157-190.
```
The inelastic contribution parameters for UED reference:
```
Szalóki, I. (1996). Empirical equations for atomic form factor and incoherent scattering functions. X‐Ray Spectrometry, 25(1), 21-28.
```

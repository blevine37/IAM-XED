# IAM-XED: Independent Atom Model for X-ray and Electron Diffraction

A Python package for calculating X-ray diffraction (XRD) and ultrafast electron diffraction (UED) signals from molecular geometries and trajectories using the Independent Atom Model (IAM) approximation.

Authors: **Jiří Suchan** and **Jiří Janoš**

## Features

- **XRD calculations** with optional inelastic Compton scattering
- **UED calculations** with pair distribution function (PDF) analysis
- **Static and time-resolved** diffraction simulations
- **Ensemble averaging** for multiple trajectories
- **Difference calculations** for pump-probe experiments
- **Flexible input formats** (single geometries, trajectories, ensembles)

## Installation

### From PyPI (Recommended)
IAM-XED is available on PyPI and can be installed simply using pip:
```bash
pip install iamxed
```
For installation into an isolated Python environment, we recomment using `pipx` or `uv`.

### From Source 
Local installation, mainly for development purposes, can be done by cloning the repository and installing it with pip:
```bash
git clone https://github.com/blevine37/IAM-XED.git
cd IAM-XED
pip install .
```

Editable installation can be achieved with `-e` flag which allows you to modify the source code without reinstalling the package:
```bash
pip install -e .
```

### Dependencies
IAM-XED requires Python ≥ 3.7 and several packages: numpy, scipy, matplotlib, tqdm and pytest (for testing). Installation with pip will automatically handle these dependencies. If you want to install them manually, you can use the following command:
```bash
pip install numpy scipy matplotlib tqdm pytest
```

### Testing
Run the tests to ensure everything is 
working correctly by launching pytest in the root directory (IAM-XED) of the repository:

```bash
pytest -v
```

## Quick start
IAM-XED is called in the command line with input specified in form of flags.
Three main specification govern the type of calculation:
- XRD or UED calculation (`--xrd` or `--ued`)
- static or time-resolved calculation (`--calculation-type static` or `--calculation-type time-resolved`)
- input geometries for calculating the signal in XYZ format and Angstrom units (`--signal-geoms <path>`)

Input geometries can be provided as a single XYZ file or a directory containing multiple XYZ files. 
Simple examples of how to use IAM-XED for different types of calculations are shown below:
```bash
# Single molecule
iamxed --xrd --calculation-type static --signal-geoms molecule.xyz

# Trajectory 
iamxed --ued --calculation-type time-resolved --signal-geoms traj.xyz

# Averaging over a set of trajectories in a directory
iamxed --xrd --calculation-type time-resolved --signal-geoms ./dir_with_trajs/
```

## Types of calculations available
Two types of calculations are supported: **static** and **time-resolved**. While static calculations average over all geometries in the input file or directory, time-resolved calculations treat the input files as trajectories.
Both modes are compatible with UED and XRD (possibly with inelastic Compton scattering). Details about the modes are summarized below.

### Static Calculations

Static calculations compute the average signal over all provided geometries. This is useful for obtaining a single diffraction pattern or PDF from a static structure or an ensemble of structures.
If reference geometries are provided, the difference signal is calculated as a relative change from the reference signal.

### Time-resolved Calculations

Time-resolved calculations treat the input geometries as a trajectories, computing the signal for each time frame along the trajctory. 
In time-resolved mode, only difference signals can be calculated, comparing the signal at each time frame to the first frame (t=0). This is useful for simulating pump-probe experiments or tracking changes in the structure over time. 
Currently, different reference than the first frame cannot be specified, but this feature may be added in the future.


## Input files: 
IAM-XED requires as an input signal geometries (used for calculating the singal) and optionally reference geometries (used for calculating reference for the difference signal).
The input geometries can be provided  in various formats: a single XYZ file or a directory containing multiple XYZ files. 
If directory is provided IAM-XED searches for all files with `.xyz` extension within.
The geometries must be in **XYZ format** with coordinates in **Angstroms**.

### Signal Geometries (`--signal-geoms`)
#### Static calculations
- **Single XYZ file**: Averages all geometries in the file and calculates static signal.
- **Directory of XYZ files**: Averages over first geometries from each XYZ file in the directory and calculates static signal.

#### Time-resolved calculations
- **Single XYZ file**: Treats all geometries in the file as a trajectory with `--timestep` intervals. 
- **Directory of XYZ files**: Each XYZ file represents a trajectory with `--timestep` intervals. The signal is average over all geometries in each time frame. Note that trajectories shorter than the longest trajectory or the specified maximum time `--tmax` will be padded with zeros, i.e., they contribute only up to the time they reached and don't contribute to the ensemble for longer times.

### Reference Geometries (`--reference-geoms`)
Reference geometries are used for calculating the difference signal in static calculations, the reference in time-resolved calculations is always the first time frame.
#### Static calculations
- **Single XYZ file**: Averages all geometries in the file and calculates static signal.
- **Directory of XYZ files**: Averages over first geometries from each XYZ file in the directory and calculates static signal.


## Key Options

| Option | Description                                                                  | Default                                 |
|--------|------------------------------------------------------------------------------|-----------------------------------------|
| `--calculation-type` | `static` or `time-resolved`                                                  | `static`                                |
| `--signal-geoms` | Path to single XYZ file or directory of XYZ files                            | None (required parameter)               |
| `--reference-geoms` | Path to reference XYZ file for difference signal (available only for static) | None                                    |
| `--ued` | Enable ultrafast electron diffraction calculation                            | False (mutually exclusive with `--xrd`) |
| `--xrd` | Enable X-ray diffraction calculation                                         | False (mutually exclusive with `--ued`) |
| `--inelastic` | Include inelastic scattering (XRD only)                                      | False                                   |
| `--timestep` | Time step (atomic time units)                                                | 10.0                                    |
| `--tmax` | Maximum time considered (fs)                                                 | None (up to the longest trajectory)     |
| `--fwhm` | FWHM parameter for Gaussian temporal convolution (fs)                        | 150.0                                   |
| `--pdf-alpha` | PDF damping parameter (Å²)                                                | 0.04                                    |
| `--qmin`, `--qmax` | Momentum transfer range $q$ (or $s$) (Bohr⁻¹)                           | $5.29\cdot 10^{-9}$, $5.29$             |
| `--npoints` | Number of $q$-points                                                         | 200                                     |
| `--log-to-file` | Log output to file along with console                                        | False                                   |
| `--plot-disable` | Disable plotting of results                                                  | False                                   |
| `--export` | Export data by providing a filename                                          | None                                    |
| `--plot-units` | `bohr-1` or `angstrom-1`                                                     | `bohr-1`                                |
| `--plot-flip` | Flip x and y axis in plots                                                   | False                                   |
| `--debug` | Enable debug output                                                          | False                                   |

More details on each option can be found in the help message (`iamxed --help`).


## Usage

### XRD Calculations

*Momentum coordinate in plots and export is labeled 'q' for XRD*

**Single Geometry:**
```bash
iamxed --xrd --signal-geoms molecule.xyz
```
Calculates scattering intensity $I(q)$ as a function of momentum transfer $q$ (Bohr⁻¹).

**Difference Signal from Single Geometry:**
```bash
iamxed --xrd --signal-geoms excited.xyz --reference-geoms ground.xyz
```
Calculates the relative difference signal: $\Delta I/I_0 = (I_1-I_0)/I_0 \cdot 100\%$ ($I_1$ - signal-geoms, $I_0$ - reference-geoms).

**Inlcuding Inelastic Scattering for XRD:**
```bash
iamxed --xrd --signal-geoms molecule.xyz --inelastic
```
Includes Compton scattering using Szaloki parameters.

**Time-resolved Single Trajectory Calulation:**
```bash
iamxed --xrd --signal-geoms trajectory.xyz --calculation-type time-resolved --qmin 0.0 --qmax 10.0 --npoints 100 --timestep 40
```
Calculates the time-resolved relative difference scattering signal $\Delta I/I_0 (q,t)$ against the t=0 frame. Momentum coordinate divided to 100 points goes from 0.0 to 10.0 Bohr⁻¹. Timestep is assumed 40 a.t.u.

**Time-resolved Ensemble Calulation:**
```bash
iamxed --xrd --signal-geoms ./ensemble_dir/ --calculation-type time-resolved --qmin 0.0 --qmax 10.0 --npoints 100 --timestep 40 --tmax 500
```
Calculates the same signal as in trajectory case, averaging over all trajectories in the `./ensemble_dir/` folder up to 500 fs.

### UED Calculations
*Momentum coordinate in plots and export is labeled 's' for UED*

**Single Geometry:**
```bash
iamxed --ued --signal-geoms molecule.xyz
```
Calculates the modified scattering intensity $sM(s) = s \cdot I_\mathrm{mol}/I_\mathrm{at}$ and **real-space pair distribution function (PDF)** $P(r) =  r  \int_{s_{min}}^{s_{max}} sM(s) \sin(s r) \exp(-\alpha s^2) \mathrm{d}s$. 


**Difference Signal from Single Trajectory:**
```bash
iamxed --ued --signal-geoms excited.xyz --reference-geoms ground.xyz
```
Calculates the relative difference signal: $\Delta I/I_0 = (I_1-I_0)/I_0 \cdot 100\%$ ($I_1$ - signal-geoms, $I_0$ - reference-geoms) and $\Delta P(r) = P_1(r)-P_0(r)$.

**Time-Resolved Single Trajectory Calculation:**
```bash
iamxed --ued --calculation-type time-resolved --signal-geoms trajectory.xyz --timestep 40 --fwhm 100 --pdf-alpha 0.04
```
Calculates time-resolved relative difference signal $\Delta I/I_0 (q,t)$ and $\Delta P(q,t)$ against the $t=0$ frame. Timestep is set to 40 a.t.u., additional temporal smoothing with 100 fs FWHM Gaussian function, $\alpha$ smearing parameter at 0.04 Å².

**Time-resolved Ensemble Calulation:**
```bash
iamxed --ued --calculation-type time-resolved --signal-geoms ./ensemble_dir/ --timestep 40 --fwhm 100 --pdf-alpha 0.04
```
Calculates the same signal as in trajectory case, averaging over all trajectories in the `./ensemble_dir/` folder.

## Output Files

### Static Calculations
- `export.txt`: Signal data with units in header
- `export_PDF.txt`: PDF data (UED only)

### Time-Resolved Calculations  
- `export.npz`: Binary archive containing:
  - `times`, `times_smooth`: Time axis (fs), smooth refers to convoluted data
  - `q`/`s`: Momentum transfer axis (Bohr⁻¹) 
  - `signal_raw`, `signal_smooth`: Diffraction signals, smooth refers to convoluted data
  - `r`, `pdfs_raw`, `pdfs_smooth`: PDF data (UED only), smooth refers to convoluted data
  - `metadata`: Command and units information

The binary archive can be loaded in Python using NumPy:
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

## Python API

You can use IAM-XED as a library in your Python scripts. The input is provided  instead of a list of flags in command line as a `argparse.Namespace` object, which can be created from a dictionary of parameters. Note that the list of arguments must contain all the parameters, no defaults will be assumed! The main function to call is then `iamxed(params)`. An example of how to use IAM-XED as a library is shown below:

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

MIT License - see [LICENSE](https://github.com/blevine37/IAM-XED/tree/labels?tab=License-1-ov-file) file for details.


## Citation

If you use IAM-XED in your research, please cite:

```bibtex
@software{iam_xed,
  title = {IAM-XED: Independent Atom Model for X-ray and Electron Diffraction},
  author = {Suchan, Jiří and Janoš, Jiří},
  url = {https://github.com/blevine37/IAM-XED},
  year = {2025}
}
```

The IAM parameters for XRD reference:
>Prince, E. (Ed.). (2004). International Tables for Crystallography, Volume C: Mathematical, physical and chemical tables. Springer Science & Business Media. ISBN 1-4020-1900-9 

The IAM parameters for UED were calculated using the [ELSEPA program](https://github.com/eScatter/elsepa) (commit [98862ff](https://github.com/eScatter/elsepa/commit/98862ff7fb56fb430ffdf9f0e311dcc399c7490e)) assuming 3.7 MeV electron kinetic energy:
>Salvat, F., Jablonski, A., & Powell, C. J. (2005). ELSEPA—Dirac partial-wave calculation of elastic scattering of electrons and positrons by atoms, positive ions and molecules. Computer physics communications, 165(2), 157-190.

The inelastic contribution parameters for UED reference:
>Szalóki, I. (1996). Empirical equations for atomic form factor and incoherent scattering functions. X‐Ray Spectrometry, 25(1), 21-28.

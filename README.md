# IAM-XED
# Independent Atom Model - X-ray and Electron Diffraction code

A Python package for calculating X-ray diffraction (XRD) and ultrafast electron diffraction (UED) patterns from molecular geometries and trajectories based on the Independent Atom Model (IAM).

## Installation
The package can be installed from the source code. Clone the repository and run the following command in the root directory:
```bash
git clone git@github.com:blevine37/IAM-XED.git
cd IAM-XED
pip install -e .
```
Flag `-e` installs the package in editable mode, allowing you to modify the source code without reinstalling. 
For common users, we recommend not using this flag.

## Testing
Run the tests to ensure everything is working correctly. Make sure you have pytest installed, then run in the root directory:
```bash
pytest -v
```

## Usage
Once the package is installed, you can use iamxed as a script
```bash
iamxed --help
```

### Single Geometry XRD Calculation
```bash
iamxed --xrd --signal-geoms molecule.xyz
```
Calculating the scattering signal in momentum coordinate q

### Static difference XRD Calculation
Calculating the scattering signal as a relative difference 
```bash
iamxed --xrd --signal-geoms molecule.xyz --reference-geoms molecule0.xyz
```

### Single Geometry UED Calculation
```bash
iamxed --ued --signal-geoms molecule.xyz
```
Calculating the scattering signal as a modified scattering intensity sM(s)=s*Imol/Iat

### Static difference XRD Calculation
```bash
iamxed --ued --signal-geoms molecule.xyz --reference-geoms molecule0.xyz
```
Caclulating the difference signal as \Delta sM(s) = s*(\Delta Imol(s))/Iat(s)

### Time-Resolved UED Calculation
```bash
iamxed --ued --signal-geoms trajectory.xyz --calculation-type time-resolved
```

### Additional Options
- `--inelastic`: Include inelastic atomic contribution for XRD
- `--fwhm`: FWHM for temporal Gaussian smoothing (fs)
- `--pdf-alpha`: Damping parameter for PDF Fourier transform
- `--plot`: Show plots
- `--export`: Export calculated data to file
- `--qmin`, `--qmax`, `--npoints`: Control q-grid parameters

### Calling IAM-XED from Python
You can also use IAM-XED as a library in your Python code. Here is an example of how to use it:

```python
from iamxed import iamxed
from argparse import Namespace

params = {
    "signal_geoms": "./ensemble/",
    "reference_geoms": None,
    "calculation_type": "time-resolved",
    "ued": False,
    "xrd": True,
    "inelastic": True,
    "qmin": 0,
    "qmax": 4,
    "npoints": 100,
    "timestep": 40.0,
    "export": "xrd_ensemble",
    "log_to_file": False,
    "debug": True,
}

params = Namespace(**params)

iamxed(params)
```

## Code Structure
The package is located in the `src/iamxed` directory and contains the following files:
- `xed.py`: Main entry point and CLI
- `physics.py`: Core physics calculations and calculator classes
- `io_utils.py`: File I/O and utility functions
- `plotting.py`: Plotting and visualization functions

## Dependencies

- Python >= 3.7
- NumPy 
- SciPy 
- Matplotlib 
- Pytest

## License
MIT License

## Authors

Jiri Suchan
Jiri Janos

## Citation


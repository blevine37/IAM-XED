# IAM-XED
# Independent Atom Model - X-ray and Electron Diffraction (XED) Calculator

A Python package for calculating X-ray diffraction (XRD) and ultrafast electron diffraction (UED) patterns from molecular geometries and trajectories.

## Installation

## Usage

### Single Geometry XRD Calculation
```bash
python xed.py --xrd --signal-geoms molecule.xyz
```

### Time-Resolved UED Calculation
```bash
python xed.py --ued --signal-geoms trajectory.xyz --calculation-type time-resolved
```

### Additional Options
- `--inelastic`: Include inelastic atomic contribution for XRD
- `--fwhm`: FWHM for temporal Gaussian smoothing (fs)
- `--pdf-alpha`: Damping parameter for PDF Fourier transform
- `--plot`: Show plots
- `--export`: Export calculated data to file
- `--qmin`, `--qmax`, `--npoints`: Control q-grid parameters

## File Structure

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

## Testing
```bash
pytest -v test/test_xed.py
```

## License


## Authors

Jiri Suchan
Jiri Janos

## Citation


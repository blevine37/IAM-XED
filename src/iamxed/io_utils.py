"""
Input/Output utilities for XED (X-ray/Electron Diffraction) calculations.
Handles file reading, logging, and argument parsing.
"""
import numpy as np
import logging
import argparse
import os
import sys
from typing import List, Tuple, Optional

def output_logger(file_output: bool = True, debug: bool = False) -> logging.Logger:
    """Set up the logger for output messages."""
    global logger

    logger = logging.getLogger()
    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(message)s')
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Output file handler
    if file_output:
        file_handler = logging.FileHandler('xed.out', mode='w')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def read_xyz(filename: str) -> Tuple[List[str], np.ndarray]:
    """Read a single geometry from an XYZ file."""
    atoms = []
    coordinates = []
    with open(filename) as xyz:
        n_atoms = int(xyz.readline())
        _ = xyz.readline()
        for line in xyz:
            if len(line.strip()) == 0:
                break
            atom, x, y, z = line.split()
            atoms.append(atom)
            coordinates.append([float(x), float(y), float(z)])
    coordinates = [[w * 1.88973 for w in ww] for ww in coordinates]
    coordinates = np.asarray(coordinates)
    if n_atoms != len(coordinates):
        raise ValueError('Number of atoms in xyz file does not match the number of lines.')
    return atoms, coordinates

def read_xyz_trajectory(filename: str) -> Tuple[List[str], np.ndarray]:
    """Read multiple geometries from an XYZ trajectory file."""
    atoms: List[str] = []
    trajectory = []
    with open(filename, 'r') as f:
        first_frame = True
        while True:
            line = f.readline()
            if not line:
                break
            try:
                n_atoms = int(line.strip())
            except ValueError:
                raise ValueError(f"Expected number of atoms at start of frame, got: {line}")
            _ = f.readline()
            frame_atoms = []
            frame_coords = []
            for _ in range(n_atoms):
                parts = f.readline().split()
                if len(parts) != 4:
                    raise ValueError(f"Invalid atom line: {parts}")
                atom, x, y, z = parts
                frame_atoms.append(atom)
                frame_coords.append([float(x), float(y), float(z)])
            if first_frame:
                atoms = frame_atoms
                first_frame = False
            elif atoms != frame_atoms:
                raise ValueError("Atom labels don't match across frames.")
            frame_coords = [[w * 1.88973 for w in xyz] for xyz in frame_coords]
            trajectory.append(frame_coords)
    coordinates = np.array(trajectory)
    return atoms, coordinates

def find_xyz_files(directory: str) -> List[str]:
    """Find all XYZ files in a directory."""
    return sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.xyz')])

def is_trajectory_file(filename: str) -> bool:
    """Check if an XYZ file contains multiple frames."""
    try:
        with open(filename, 'r') as f:
            # Read first frame
            n_atoms = int(f.readline())
            _ = f.readline()
            for _ in range(n_atoms):
                _ = f.readline()
            # Check if there's another frame
            line = f.readline()
            if not line:  # EOF
                return False
            try:
                _ = int(line.strip())  # Try to read number of atoms for next frame
                return True
            except ValueError:
                return False
    except (ValueError, IOError):
        return False

def get_elements_from_input(signal_geoms: str) -> List[str]:
    """Get unique elements from input geometries.
    
    Args:
        signal_geoms: Path to XYZ file or directory
        
    Returns:
        List of unique element symbols in alphabetical order
    """
    elements = set()
    if os.path.isfile(signal_geoms):
        if is_trajectory_file(signal_geoms):
            atoms, _ = read_xyz_trajectory(signal_geoms)
        else:
            atoms, _ = read_xyz(signal_geoms)
        elements.update(atoms)
    elif os.path.isdir(signal_geoms):
        xyz_files = find_xyz_files(signal_geoms)
        if not xyz_files:
            raise FileNotFoundError('No XYZ files found in directory.')
        # Check if first file is a trajectory
        if is_trajectory_file(xyz_files[0]):
            atoms, _ = read_xyz_trajectory(xyz_files[0])
        else:
            atoms, _ = read_xyz(xyz_files[0])
        elements.update(atoms)
    else:
        raise FileNotFoundError('Signal geometry file not found.')
    return sorted(set(elements))

def parse_cmd_args(args: Optional[List[str]] = None):
    """Parse command line arguments.
    
    Args:
        args: Optional list of command line arguments. If None, uses sys.argv[1:]
    """
    parser = argparse.ArgumentParser(description='X-ray and Electron Diffraction (XED) Calculation Script')
    
    # General options
    general_sec = parser.add_argument_group("General options")
    general_sec.add_argument('--signal-geoms', type=str, required=True,
                           help='Geometries for calculating signal (xyz file or directory)')
    general_sec.add_argument('--reference-geoms', type=str,
                           help='Reference geometries for difference calculation')
    general_sec.add_argument('--ensemble-dir', type=str,
                           help='Directory containing ensemble xyz files')
    general_sec.add_argument('--timestep', type=float, default=10.0,
                           help='Timestep between frames (atomic units)')
    
    # Calculation type option
    general_sec.add_argument('--calculation-type', type=str, choices=['static', 'time-resolved'], default='static',
        help='Either perform static (average/ensemble) or time-resolved (2D map) calculation.')
    
    # Signal type options
    signal_sec = parser.add_argument_group("Signal type options")
    signal_sec.add_argument('--ued', action='store_true',
                           help='Performs UED calculation')
    signal_sec.add_argument('--xrd', action='store_true',
                           help='Performs XRD calculation')
    signal_sec.add_argument('--inelastic', action='store_true',
                           help='Include inelastic atomic contribution for XRD')
    
    # Processing options
    proc_sec = parser.add_argument_group("Processing options")
    proc_sec.add_argument('--fwhm', type=float, default=150.0,
                         help='FWHM for temporal Gaussian smoothing (fs)')
    proc_sec.add_argument('--pdf-alpha', type=float, default=0.04,
                         help='Damping parameter for PDF Fourier transform')
    
    # Output options
    out_sec = parser.add_argument_group("Output options")
    out_sec.add_argument('--log-to-file', action='store_true',
                        help="Save output to 'xed.out'")
    out_sec.add_argument('--debug', action='store_true',
                        help="Print debug output")
    out_sec.add_argument('--plot', action='store_true',
                        help='Plot the results')
    out_sec.add_argument('--export', type=str,
                        help='Export calculated data to file')
    out_sec.add_argument('--plot-units', type=str, default='bohr-1', choices=['bohr-1', 'angstrom-1'],
                        help="Units for plotting the q axis: 'bohr-1' (default) or 'angstrom-1'")
    
    # Grid parameters
    grid_sec = parser.add_argument_group("Grid parameters")
    grid_sec.add_argument('--qmin', type=float, default=5.29e-9,
                         help='Minimum q value (Bohr^-1)')
    grid_sec.add_argument('--qmax', type=float, default=5.29,
                         help='Maximum q value (Bohr^-1)')
    grid_sec.add_argument('--npoints', type=int, default=200,
                         help='Number of q points')
    
    # Parse arguments
    parsed_args = parser.parse_args(args)
    
    # If help was requested, show help and exit immediately
    if '--help' in (args or sys.argv[1:]):
        parser.print_help()
        sys.exit(0)
    
    # Set calculation_type directly from argument (no auto-detection)
    # parsed_args.calculation_type is already set by argparse
    return parsed_args 
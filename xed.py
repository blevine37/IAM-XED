#!/usr/bin/env python3
"""
Main script for X-ray and Electron Diffraction (XED) calculations.
Handles command line interface and orchestrates calculations.
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple

from io_utils import parse_cmd_args, output_logger, get_elements_from_input
from physics import XRDDiffractionCalculator, UEDDiffractionCalculator
from plotting import plot_static, plot_time_resolved

def main():
    """Main function for XED calculations."""
    # Parse command line arguments
    args = parse_cmd_args()
    
    # Set up logger
    logger = output_logger(args.log_to_file, args.debug)
    
    # Validate input arguments
    if not args.xrd and not args.ued:
        logger.error("Must specify either --xrd or --ued")
        return 1
    if args.xrd and args.ued:
        logger.error("Cannot specify both --xrd and --ued")
        return 1
    if not args.signal_geoms:
        logger.error("Must specify --signal-geoms")
        return 1
    if not hasattr(args, 'calculation_type') or args.calculation_type is None:
        logger.error("Must specify --calculation-type (static, time-resolved, etc.)")
        return 1

    # Print basic info about mode and signal type
    mode_msg = f"Mode: {args.calculation_type} | Signal: {'XRD' if args.xrd else 'UED'}"
    print(f"[IAM-XED] {mode_msg}")
        
    # Get unique elements from input geometries
    try:
        elements = get_elements_from_input(args.signal_geoms)
    except FileNotFoundError as e:
        logger.error(str(e))
        return 1
        
    # Initialize appropriate calculator
    if args.xrd:
        calculator = XRDDiffractionCalculator(
            q_start=args.qmin,
            q_end=args.qmax,
            num_point=args.npoints,
            elements=elements,
            inelastic=args.inelastic
        )
    else:  # UED
        calculator = UEDDiffractionCalculator(
            q_start=args.qmin,
            q_end=args.qmax,
            num_point=args.npoints,
            elements=elements
        )
        
    # Perform calculation based on type
    try:
        if args.reference_geoms:
            # Always do difference calculation if reference is provided
            # Compute reference signal (average if multi-geometry)
            if args.calculation_type == 'static':
                q, diff_signal, r, diff_pdf = calculator.calc_difference(args.signal_geoms, args.reference_geoms)
                if args.plot_units == 'angstrom-1' and not args.xrd:
                    diff_signal = diff_signal / 0.529177
                if args.plot:
                    plot_static(q, diff_signal, args.xrd, is_difference=True, plot_units=args.plot_units, r=r, pdf=diff_pdf)
                if args.export:
                    cmd_options = ' '.join(sys.argv[1:])
                    header = f"# xed.py {cmd_options}"
                    np.savetxt(args.export, np.column_stack((q, diff_signal)), header=header, comments='')
            elif args.calculation_type == 'time-resolved':
                times, q, signal_raw, signal_smooth = calculator.calc_trajectory(
                    args.signal_geoms,
                    timestep_au=args.timestep,
                    fwhm_fs=args.fwhm
                )
                # Get reference signal for time-resolved
                q_ref, ref_signal, _, _ = calculator.calc_single(args.reference_geoms)
                # Subtract static reference from all time frames
                diff_raw = (signal_raw - ref_signal[:, None]) / ref_signal[:, None] * 100
                diff_smooth = (signal_smooth - ref_signal[:, None]) / ref_signal[:, None] * 100
                if args.plot:
                    q_plot = q * 1.88973 if args.plot_units == 'angstrom-1' else q
                    plot_time_resolved(times, q_plot, diff_raw, args.xrd, plot_units=args.plot_units, smoothed=False, fwhm_fs=args.fwhm)
                    plot_time_resolved(times, q_plot, diff_smooth, args.xrd, plot_units=args.plot_units, smoothed=True, fwhm_fs=args.fwhm)
                if args.export:
                    np.savez(args.export, times=times, q=q, signal_raw=diff_raw, signal_smooth=diff_smooth)
        else:
            if args.calculation_type == 'static':
                q, signal, r, pdf = calculator.calc_single(args.signal_geoms)
                if args.plot_units == 'angstrom-1' and not args.xrd:
                    signal = signal / 0.529177
                if args.plot:
                    plot_static(q, signal, args.xrd, plot_units=args.plot_units, r=r, pdf=pdf)
                if args.export:
                    cmd_options = ' '.join(sys.argv[1:])
                    header = f"# xed.py {cmd_options}"
                    np.savetxt(args.export, np.column_stack((q, signal)), header=header, comments='')
            elif args.calculation_type == 'time-resolved':
                times, q, signal_raw, signal_smooth = calculator.calc_trajectory(
                    args.signal_geoms,
                    timestep_au=args.timestep,
                    fwhm_fs=args.fwhm
                )
                if args.plot:
                    q_plot = q * 1.88973 if args.plot_units == 'angstrom-1' else q
                    plot_time_resolved(times, q_plot, signal_raw, args.xrd, plot_units=args.plot_units, smoothed=False, fwhm_fs=args.fwhm)
                    plot_time_resolved(times, q_plot, signal_smooth, args.xrd, plot_units=args.plot_units, smoothed=True, fwhm_fs=args.fwhm)
                if args.export:
                    np.savez(args.export, times=times, q=q, signal_raw=signal_raw, signal_smooth=signal_smooth)
    except Exception as e:
        logger.error(f"Error during calculation: {str(e)}")
        return 1
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main()) 
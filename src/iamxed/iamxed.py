#!/usr/bin/env python3
"""
Main script for X-ray and Electron Diffraction (XED) calculations.
Handles command line interface and orchestrates calculations.
"""
import numpy as np

from .io_utils import parse_cmd_args, output_logger, get_elements_from_input
from .physics import XRDDiffractionCalculator, UEDDiffractionCalculator
from .plotting import plot_static, plot_time_resolved, plot_time_resolved_pdf

def main():
    global logger

    from sys import argv

    """Main function for XED calculations."""
    # Parse command line arguments
    args = parse_cmd_args()
    
    # Set up logger
    logger = output_logger(args.log_to_file, args.debug)

    # Print code header and copyright
    logger.info('\n###################'\
                '\n###  IAM-XED   ###'\
                '\n###################\n')
    logger.info("Independent Atom Model code for Ultrafast Electron and X-Ray Diffraction.\n"
                "Copyright (c) 2025 Authors.\n")

    # Validate input arguments
    if not args.xrd and not args.ued:
        logger.error("ERROR: No signal type specified. Use --xrd or --ued or both to specify the signal type.")
        return 1
    if args.xrd and args.ued:
        logger.error("ERROR: Cannot specify both --xrd and --ued.")
        return 1
    if not args.signal_geoms:
        logger.error("ERROR: No signal geometries provided. Use --signal-geoms to specify the geometry file or folder.")
        return 1
    if not hasattr(args, 'calculation_type') or args.calculation_type is None:
        logger.error("ERROR: Must specify --calculation-type (static, time-resolved)")
        return 1

    # Check if signal_geoms is a file or directory
    import os
    if not os.path.exists(args.signal_geoms):
        logger.error(f"ERROR: Signal geometries path {args.signal_geoms} does not exist.")
        return 1
    else:
        if os.path.isfile(args.signal_geoms):
            signal_geom_type = 'file'
        elif os.path.isdir(args.signal_geoms):
            signal_geom_type = 'directory'
        else:
            logger.error('ERROR: Signal geometries path is neither a file nor a directory.')
            return 1

    if args.reference_geoms is None:
        ref_calc = False
    else:
        if not os.path.exists(args.reference_geoms):
            logger.error(f"ERROR: Reference geometries path {args.reference_geoms} does not exist.")
            return 1
        if not os.path.isfile(args.reference_geoms):
            logger.error('ERROR: Reference geometries path is not a file, only single geometry file is supported for reference geometries.')
            return 1
        ref_calc = True

    # Print input parameters
    logger.info('INPUT PARAMETERS\n----------------')
    for key, value in vars(args).items():
        add = ''
        if key == 'timestep':
            add = 'a.t.u.'
        elif key == 'signal_geoms':
            add = f'({signal_geom_type})'
        elif key in ['qmin', 'qmax']:
            add = '1/Bohr'
        logger.info(f"- {key:20s}: {value}  {add}")

    # Print calculation intro
    output = '\nCALCULATION\n-----------\n '
    if args.calculation_type == 'static':
        output += 'Static '
    elif args.calculation_type == 'time-resolved':
        output += 'Time-resolved '
    if args.ued and args.xrd:
        output += 'UED and XRD calculation will be performed.'
    elif args.ued:
        output += 'UED calculation will be performed.'
    elif args.xrd:
        output += 'XRD calculation will be performed.'
    logger.info(output)

    if args.xrd:
        if args.inelastic:
            logger.info(' Inelastic atomic contribution will be included in XRD calculation.')
        else:
            logger.info(' Inelastic atomic contribution will NOT be included in XRD calculation.')

    if signal_geom_type == 'file':
        logger.info(f' Signal geometries will be read from file ({args.signal_geoms}) - no ensemble calculation.')
    elif signal_geom_type == 'directory':
        logger.info(f' Signal geometries will be read from directory ({args.signal_geoms}) - ensemble calculation will be performed.')

    if ref_calc:
        logger.info(' Reference provided, difference calculation will be performed.')
    else:
        logger.info(' No reference provided, only signal calculation will be performed.')

    # Get unique elements from input geometries
    try:
        elements = get_elements_from_input(args.signal_geoms)
    except FileNotFoundError as e:
        logger.error(str(e))
        return 1
    logger.info(f"Elements found in input: {elements}")

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
    logger.info("Calculator initialized.")

    # Perform calculation based on type
    try:
        if args.reference_geoms:
            if args.calculation_type == 'static':
                logger.info('Performing static difference calculation...')
                q, diff_signal, r, diff_pdf = calculator.calc_difference(args.signal_geoms, args.reference_geoms)
                if args.plot:
                    logger.info('Plotting static difference signal...')
                    plot_static(q, diff_signal, args.xrd, is_difference=True, plot_units=args.plot_units, r=r, pdf=diff_pdf)
                if args.export:
                    logger.info(f'Exporting static difference data to {args.export}...')
                    cmd_options = ' '.join(argv[1:])
                    header = f"# iamxed {cmd_options}"
                    np.savetxt(args.export, np.column_stack((q, diff_signal)), header=header, comments='')
            elif args.calculation_type == 'time-resolved':
                logger.info('Performing time-resolved difference calculation...')
                times, q, signal_raw, signal_smooth, r, pdfs_raw, pdfs_smooth = calculator.calc_trajectory(
                    args.signal_geoms,
                    timestep_au=args.timestep,
                    fwhm_fs=args.fwhm,
                    pdf_alpha=args.pdf_alpha
                )
                q_ref, ref_signal, r_ref, ref_pdf = calculator.calc_single(args.reference_geoms)
                diff_raw = (signal_raw - ref_signal[:, None]) / ref_signal[:, None] * 100
                diff_smooth = (signal_smooth - ref_signal[:, None]) / ref_signal[:, None] * 100
                if args.plot:
                    logger.info('Plotting time-resolved difference signal...')
                    plot_time_resolved(times, q, diff_raw, args.xrd, plot_units=args.plot_units, smoothed=False, fwhm_fs=args.fwhm)
                    plot_time_resolved(times, q, diff_smooth, args.xrd, plot_units=args.plot_units, smoothed=True, fwhm_fs=args.fwhm)
                    if not args.xrd and r_ref is not None and ref_pdf is not None:  # Plot PDFs for UED only
                        logger.info('Plotting time-resolved PDFs...')
                        plot_time_resolved_pdf(times, r, pdfs_raw, smoothed=False, fwhm_fs=args.fwhm)
                        plot_time_resolved_pdf(times, r, pdfs_smooth, smoothed=True, fwhm_fs=args.fwhm)
                if args.export:
                    logger.info(f'Exporting time-resolved difference data to {args.export}...')
                    if not args.xrd:  # Include PDFs for UED only
                        np.savez(args.export, times=times, q=q, signal_raw=diff_raw, signal_smooth=diff_smooth,
                               r=r, pdfs_raw=pdfs_raw, pdfs_smooth=pdfs_smooth)
                    else:
                        np.savez(args.export, times=times, q=q, signal_raw=diff_raw, signal_smooth=diff_smooth)
        else:
            if args.calculation_type == 'static':
                logger.info('Performing static calculation...')
                q, signal, r, pdf = calculator.calc_single(args.signal_geoms)
                if args.plot:
                    logger.info('Plotting static signal...')
                    plot_static(q, signal, args.xrd, plot_units=args.plot_units, r=r, pdf=pdf)
                if args.export:
                    logger.info(f'Exporting static data to {args.export}...')
                    cmd_options = ' '.join(argv[1:])
                    header = f"# xed.py {cmd_options}"
                    np.savetxt(args.export, np.column_stack((q, signal)), header=header, comments='')
            elif args.calculation_type == 'time-resolved':
                logger.info('Performing time-resolved calculation...')
                # Both XRD and UED calculators now return the same number of values
                times, q, signal_raw, signal_smooth, r, pdfs_raw, pdfs_smooth = calculator.calc_trajectory(
                    args.signal_geoms,
                    timestep_au=args.timestep,
                    fwhm_fs=args.fwhm,
                    pdf_alpha=args.pdf_alpha
                )
                
                # Get smoothed time axis for smoothed data
                _, times_smooth = calculator.gaussian_smooth_2d_time(signal_raw, times, args.fwhm)
                
                if args.plot:
                    logger.info('Plotting time-resolved signal...')
                    plot_time_resolved(times, q, signal_raw, args.xrd, plot_units=args.plot_units, smoothed=False, fwhm_fs=args.fwhm)
                    plot_time_resolved(times_smooth, q, signal_smooth, args.xrd, plot_units=args.plot_units, smoothed=True, fwhm_fs=args.fwhm)
                    if not args.xrd:  # Plot PDFs for UED only
                        logger.info('Plotting time-resolved PDFs...')
                        plot_time_resolved_pdf(times, r, pdfs_raw, smoothed=False, fwhm_fs=args.fwhm)
                        plot_time_resolved_pdf(times_smooth, r, pdfs_smooth, smoothed=True, fwhm_fs=args.fwhm)
                if args.export:
                    logger.info(f'Exporting time-resolved data to {args.export}...')
                    if not args.xrd:  # Include PDFs for UED only
                        np.savez(args.export, times=times, times_smooth=times_smooth, q=q, signal_raw=signal_raw, signal_smooth=signal_smooth,
                               r=r, pdfs_raw=pdfs_raw, pdfs_smooth=pdfs_smooth)
                    else:
                        np.savez(args.export, times=times, times_smooth=times_smooth, q=q, signal_raw=signal_raw, signal_smooth=signal_smooth)
    except Exception as e:
        logger.error(f"Error during calculation: {str(e)}")
        return 1
    logger.info('Calculation complete.')
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main())
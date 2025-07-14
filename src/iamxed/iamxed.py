#!/usr/bin/env python3
"""
Main script for X-ray and Electron Diffraction (XED) calculations.
Handles command line interface and orchestrates calculations.
"""
import numpy as np
import os
from argparse import Namespace

from .io_utils import parse_cmd_args, output_logger, get_elements_from_input
from .physics import XRDDiffractionCalculator, UEDDiffractionCalculator
from .plotting import plot_static, plot_time_resolved, plot_time_resolved_pdf


def main():
    """main() function that contains argument parsing and the main iamxed calculation. It's purpose is to have argument
    parsing and the consecutive calculation separated such that iamxed can be called in python scripts with the input
    provided as args Namespace."""

    # Parse command line arguments
    args = parse_cmd_args()

    iamxed(args)

def iamxed(args: Namespace):
    global logger
    from sys import argv

    """IAMXED function for x-ray and electron diffraction calculations."""
    # check that args are Namespace in case iamxed is called from python script
    if not isinstance(args, Namespace):
        raise TypeError("Expected args for iamxed() to be a Namespace object.")

    # Set up logger
    logger = output_logger(args.log_to_file, args.debug)

    # Print code header
    logger.info('\n###################'\
                '\n###  IAM-XED   ###'\
                '\n###################\n')
    logger.info("Independent Atom Model code for X-ray and ultrafast Electron Diffraction.\n"
                "Copyright (c) 2025 Suchan J., Janos J.\n")

    # Determine geometry types
    signal_geom_type = 'file' if os.path.isfile(args.signal_geoms) else 'directory'
    ref_calc = False
    if args.reference_geoms is not None:
        ref_calc = True
        ref_geom_type = 'file' if os.path.isfile(args.reference_geoms) else 'directory'

    # Print input parameters
    logger.info('INPUT PARAMETERS\n----------------')
    for key, value in vars(args).items():
        add = ''
        if key == 'timestep':
            add = 'a.t.u.'
        elif key == 'signal_geoms':
            add = f'({signal_geom_type})'
        elif key == 'reference_geoms' and value is not None:
            add = f'({ref_geom_type})'
        elif key in ['qmin', 'qmax']:
            add = '1/Bohr'
        logger.info(f"- {key:20s}: {value}  {add}")

    # Print calculation intro
    output = '\nCALCULATION\n-----------\n '
    if args.calculation_type == 'static':
        output += 'Static '
    elif args.calculation_type == 'time-resolved':
        output += 'Time-resolved '
    if args.ued:
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
        logger.info(f' Signal geometries will be read from file ({args.signal_geoms})')
    elif signal_geom_type == 'directory':
        logger.info(f' Signal geometries will be read from directory ({args.signal_geoms}) - ensemble calculation will be performed.')

    if ref_calc:
        logger.info(f' Reference provided ({args.reference_geoms}), difference calculation will be performed.')
        if ref_geom_type == 'directory':
            logger.info(' Reference is a directory - ensemble averaging will be used for reference.')
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
                    plot_static(q, diff_signal, args.xrd, is_difference=True, plot_units=args.plot_units, r=r, pdf=diff_pdf, plot_flip=args.plot_flip)
                if args.export:
                    logger.info(f'Exporting static difference data to {args.export}...')
                    cmd_options = ' '.join(argv[1:])
                    header = f"# iamxed {cmd_options}"
                    np.savetxt(args.export, np.column_stack((q, diff_signal)), header=header, comments='')
            elif args.calculation_type == 'time-resolved':
                logger.error('ERROR: Time-resolved calculations with a reference are not supported.')
                return 1
        else:
            if args.calculation_type == 'static':
                logger.info('Performing static calculation...')
                q, signal, r, pdf = calculator.calc_single(args.signal_geoms)
                if args.plot:
                    logger.info('Plotting static signal...')
                    plot_static(q, signal, args.xrd, plot_units=args.plot_units, r=r, pdf=pdf, plot_flip=args.plot_flip)
                if args.export:
                    logger.info(f'Exporting static data to {args.export}...')
                    cmd_options = ' '.join(argv[1:])
                    header = f"# xed.py {cmd_options}"
                    np.savetxt(args.export, np.column_stack((q, signal)), header=header, comments='')
            elif args.calculation_type == 'time-resolved':
                logger.info('Performing time-resolved calculation...')
                if signal_geom_type == 'directory':
                    times, q, signal_raw, signal_smooth, r, pdfs_raw, pdfs_smooth = calculator.calc_ensemble(
                        args.signal_geoms,
                        timestep_au=args.timestep,
                        fwhm_fs=args.fwhm,
                        pdf_alpha=args.pdf_alpha,
                        tmax_fs=args.tmax
                    )
                else:
                    times, q, signal_raw, signal_smooth, r, pdfs_raw, pdfs_smooth = calculator.calc_trajectory(
                        args.signal_geoms,
                        timestep_au=args.timestep,
                        fwhm_fs=args.fwhm,
                        pdf_alpha=args.pdf_alpha,
                        tmax_fs=args.tmax
                    )
                # Get smoothed time axis for smoothed data
                _, times_smooth = calculator.gaussian_smooth_2d_time(signal_raw, times, args.fwhm)
                if args.plot:
                    logger.info('Plotting time-resolved signal...')
                    plot_time_resolved(times, q, signal_raw, args.xrd, plot_units=args.plot_units, smoothed=False, fwhm_fs=args.fwhm, plot_flip=args.plot_flip)
                    plot_time_resolved(times_smooth, q, signal_smooth, args.xrd, plot_units=args.plot_units, smoothed=True, fwhm_fs=args.fwhm, plot_flip=args.plot_flip)
                    if not args.xrd:  # Plot PDFs for UED only
                        logger.info('Plotting time-resolved PDFs...')
                        plot_time_resolved_pdf(times, r, pdfs_raw, smoothed=False, fwhm_fs=args.fwhm, plot_flip=args.plot_flip)
                        plot_time_resolved_pdf(times_smooth, r, pdfs_smooth, smoothed=True, fwhm_fs=args.fwhm, plot_flip=args.plot_flip)
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
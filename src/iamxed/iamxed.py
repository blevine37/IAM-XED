#!/usr/bin/env python3
"""
Main script for X-ray and Electron Diffraction (XED) calculations.
Handles command line interface and orchestrates calculations.
"""
import numpy as np
import os
from argparse import Namespace

from .io_utils import parse_cmd_args, output_logger, get_elements_from_input, export_static_data, export_tr_data, find_xyz_files
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
    """IAMXED function for x-ray and electron diffraction calculations."""

    global logger
    from sys import argv

    ### function ###
    def print_input_parameters(args: Namespace):
        # Print code header
        logger.info('\n  ###################' \
                    '\n  ###   IAM-XED   ###' \
                    '\n  ###################\n')
        logger.info("Independent Atom Model code for X-ray and ultrafast Electron Diffraction.\n"
                    "Copyright (c) 2025 Suchan J., Janos J.\n")

        logger.info('INPUT PARAMETERS\n----------------')
        for key, value in vars(args).items():
            # skip parameters that do not impact the calculation to show only relevant parameters
            if args.calculation_type == 'static' and key in ['fwhm', 'tmax', 'timestep']:
                continue
            if args.ued and key in ['xrd', 'inelastic']:
                continue
            elif args.xrd and key in ['ued']:
                continue

            add = ''
            if key == 'timestep':
                add = 'a.t.u.'
            elif key == 'tmax' and value is not None:
                add = 'fs'
            elif key == 'fwhm':
                add = 'fs'
            elif key == 'pdf_alpha':
                add = 'Ang^2'
            elif key == 'signal_geoms':
                add = f'({signal_geom_type})'
            elif key == 'reference_geoms' and value is not None:
                add = f'({ref_geom_type})'
            elif key in ['qmin', 'qmax']:
                add = '1/Bohr'
            logger.info(f"- {key:20s}: {value}  {add}")

        # Print calculation introduction
        output = '\nCALCULATION\n-----------\n'
        if args.calculation_type == 'static':
            output += 'Static '
        elif args.calculation_type == 'time-resolved':
            output += 'Time-resolved '
        if args.ued:
            output += 'UED calculation will be performed.'
        elif args.xrd:
            output += 'XRD calculation'
            if args.inelastic:
                output += ' including inelastic (Compton) scattering contribution will be performed.'
            else:
                output += ' will be performed.'
        logger.info(output)

        #todo: this must be polished

        # print how signal and reference geometries will be read and treated
        if signal_geom_type == 'file':
            if args.calculation_type == 'static':
                logger.info(f'Signal geometries will be read from file ({args.signal_geoms})')
            elif args.calculation_type == 'time-resolved':
                logger.info(f'Trajectory will be read from file ({args.signal_geoms})')
        elif signal_geom_type == 'directory':
            if args.calculation_type == 'static':
                logger.info(f"Signal geometries will be read as only the first geometries in all XYZ files in "
                            f"'{args.signal_geoms}' directory.")
            elif args.calculation_type == 'time-resolved':
                logger.info(
                    f'Trajectories will be read from directory ({args.signal_geoms}).')

        if signal_geom_type == 'directory':
            logger.info(
                'List of XYZ files found in signal directory:\n* ' + '\n* '.join(find_xyz_files(args.signal_geoms)))

        # todo: does it make sense to have calculations of time-resolved with reference?
        if args.reference_geoms:
            logger.info(f'Reference provided ({args.reference_geoms}), difference calculation will be performed.')
            if ref_geom_type == 'directory':
                logger.info(f"Reference geometries will be read as only the first geometries in all XYZ files in "
                            f"'{args.signal_geoms}' directory.")
                logger.info('List of files found in reference directory:\n* ' + '\n* '.join(
                    find_xyz_files(args.reference_geoms)))
        else:
            if args.calculation_type == 'time-resolved':
                logger.info('No reference provided for time-resolved calculation -> first geometries all trajectories will be used as reference.')
            else:
                logger.info('No reference provided, only signal calculation will be performed.')

    ### code ###
    # Set up logger
    logger = output_logger(args.log_to_file, args.debug)

    # check that args are Namespace in case iamxed is called from python script
    if not isinstance(args, Namespace):
        logger.error("ERROR: Expected args for iamxed() to be a Namespace object.")
        raise TypeError("Expected args for iamxed() to be a Namespace object.")

    # Determine geometry types
    signal_geom_type = 'file' if os.path.isfile(args.signal_geoms) else 'directory'
    if args.reference_geoms is not None:
        ref_geom_type = 'file' if os.path.isfile(args.reference_geoms) else 'directory' # todo: check if this is correct, maybe it should be 'directory' if a directory is provided

    # Print header and input parameters
    print_input_parameters(args)

    # Get unique elements from input geometries
    try:
        elements = get_elements_from_input(args.signal_geoms)
    except FileNotFoundError as e:
        logger.error('ERROR: ' + str(e))
        return 1
    logger.info(f"Elements found in input: {', '.join(elements)}")

    # Initialize appropriate calculator
    if args.xrd:
        calculator = XRDDiffractionCalculator(
            q_start=args.qmin,
            q_end=args.qmax,
            num_point=args.npoints,
            elements=elements,
            inelastic=args.inelastic
        )
    elif args.ued:
        calculator = UEDDiffractionCalculator(
            q_start=args.qmin,
            q_end=args.qmax,
            num_point=args.npoints,
            elements=elements,
            ued_energy_ev=3.7e6
        )
    logger.debug("[DEBUG]: Calculator initialized.")

    # Perform calculation based on type
    try:
        if args.reference_geoms:
            if args.calculation_type == 'static':
                logger.info('Performing static difference calculation...')
                q, diff_signal, r, diff_pdf = calculator.calc_difference(args.signal_geoms, args.reference_geoms)
                if args.export:
                    export_static_data(filename=args.export, flags_list=argv[1:], q=q, signal=diff_signal)
            elif args.calculation_type == 'time-resolved':
                logger.error('ERROR: Time-resolved calculations with a reference are not supported.')
                return 1
        else:
            if args.calculation_type == 'static':
                logger.info('Performing static calculation...')
                q, signal, r, pdf = calculator.calc_single(args.signal_geoms)
                if args.export:
                    export_static_data(filename=args.export, flags_list=argv[1:], q=q, signal=signal)
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
                if args.export:
                    if args.ued:  # Include PDFs for UED only
                        export_tr_data(args=args, flags_list=argv[1:], times=times, times_smooth=times_smooth, q=q,
                            signal_raw=signal_raw, signal_smooth=signal_smooth, r=r, pdfs_raw=pdfs_raw,
                            pdfs_smooth=pdfs_smooth)
                    elif args.xrd:
                        export_tr_data(args=args, flags_list=argv[1:], times=times, times_smooth=times_smooth, q=q,
                            signal_raw=signal_raw, signal_smooth=signal_smooth)
    except Exception as e:
        logger.error(f"ERROR: Calculation issued exception: {str(e)}")
        return 1
    logger.info('IAM-XED calculation complete!')

    # plot results
    if args.plot:
        logger.info("\nPLOTTING\n--------")
        try:
            if args.reference_geoms:
                if args.calculation_type == 'static':
                    logger.info('Plotting static difference signal...')
                    plot_static(q, diff_signal, args.xrd, is_difference=True, plot_units=args.plot_units, r=r,
                        pdf=diff_pdf, plot_flip=args.plot_flip)
            else:
                if args.calculation_type == 'static':
                    logger.info('Plotting static signal...')
                    plot_static(q, signal, args.xrd, plot_units=args.plot_units, r=r, pdf=pdf, plot_flip=args.plot_flip)
                elif args.calculation_type == 'time-resolved':
                    logger.info('Plotting time-resolved signal...')
                    plot_time_resolved(times, q, signal_raw, args.xrd, plot_units=args.plot_units, smoothed=False,
                        fwhm_fs=args.fwhm, plot_flip=args.plot_flip)
                    plot_time_resolved(times_smooth, q, signal_smooth, args.xrd, plot_units=args.plot_units,
                        smoothed=True, fwhm_fs=args.fwhm, plot_flip=args.plot_flip)
                    if not args.xrd:  # Plot PDFs for UED only
                        logger.info('Plotting time-resolved PDFs...')
                        plot_time_resolved_pdf(times, r, pdfs_raw, smoothed=False, fwhm_fs=args.fwhm,
                            plot_flip=args.plot_flip)
                        plot_time_resolved_pdf(times_smooth, r, pdfs_smooth, smoothed=True, fwhm_fs=args.fwhm,
                            plot_flip=args.plot_flip)
        except Exception as e:
            logger.error(f"ERROR: Plotting issued exception: {str(e)}")
            return 1
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main())
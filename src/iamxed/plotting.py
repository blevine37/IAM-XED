"""
Plotting utilities for XED (X-ray/Electron Diffraction) calculations.
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from matplotlib.colors import TwoSlopeNorm
from logging import getLogger

logger = getLogger("my_logger") # getting logger

def plot_static(q: np.ndarray, signal: np.ndarray, is_xrd: bool, is_difference: bool = False, plot_units: str = 'bohr-1', r: Optional[np.ndarray] = None, pdf: Optional[np.ndarray] = None, plot_flip: bool = False) -> None:
    """Plot static diffraction pattern, and rPDF if provided.
    Args:
        q: Q-values in atomic units
        signal: Diffraction signal (in Bohr^-1 for UED)
        is_xrd: True if XRD, False if UED
        is_difference: True if plotting difference signal
        plot_units: 'bohr-1' or 'angstrom-1'
        r: r grid for rPDF (optional)
        pdf: rPDF values (optional)
        plot_flip: Whether to flip x and y axes
    """

    pdf_present = (r is not None) and (pdf is not None)

    # Convert units for momentum transfer coordinate if needed
    if plot_units == 'angstrom-1':
        q_plot = q * 1.88973
        signal_plot = signal #not converting / 0.529177 if not is_xrd else signal
        if is_xrd:
            x_label = '$q$ (Å$^{-1}$)' # XRD naming custom
        else:
            x_label = '$s$ (Å$^{-1}$)' # UED naming custom
    else:
        q_plot = q
        signal_plot = signal
        if is_xrd:
            x_label = '$q$ (Bohr$^{-1}$)'
        else:
            x_label = '$s$ (Bohr$^{-1}$)'

    # get qmin and qmax from converted coordinates
    qmin, qmax = np.min(q_plot), np.max(q_plot)

    # get rmin and rmax if rPDF is present
    if pdf_present:
        rmin, rmax = np.min(r), np.max(r)

    ### labels and titles
    # pdf
    if pdf_present:
        if is_difference:
            title_pdf = 'Difference Pair Distribution Function ($\Delta$rPDF)'
            label_pdf = '$\Delta$rPDF (arb. units)'
        else:
            title_pdf = 'Pair Distribution Function (rPDF)'
            label_pdf = 'rPDF (arb. units)'

    #  intensity
    if is_xrd:
        if is_difference:
            label_i = '$\Delta I/I_0$ (%)'
            title_i = 'XRD Relative Difference Signal'
        else:
            label_i = '$I(q)$ (arb. units)'
            title_i = 'XRD Signal Inetnsity'
    else:
        if is_difference:
            label_i = '$\Delta I/I_0$ (%)'
            title_i = 'UED Relative Difference Signal'
        else:
            label_i = '$I(s)$ (arb. units)'
            title_i = 'UED Signal Intensity'

    # initialize plot depending on rPDF presence
    if pdf_present:
        fig, axs = plt.subplots(1,2, figsize=(5*2, 5), gridspec_kw={'width_ratios': [1, 1]})
        ax_I = axs[0]
        ax_pdf = axs[1]
    else:
        fig, axs = plt.subplots(1,1, figsize=(5, 5))
        ax_I = axs

    if plot_flip:
        ax_I.plot(signal_plot, q_plot, linewidth=1.5)
        # ax_I.fill_betweenx(q_plot, signal_plot, alpha=0.1, linewidth=1.5)
        ax_I.axvline(0, color='k', linewidth=1)
        ax_I.set_ylabel(x_label)
        ax_I.set_xlabel(label_i)
        ax_I.set_ylim(qmin, qmax)
        if is_difference: ax_I.set_xlim(-1.1*np.max(np.abs(signal_plot)), 1.1*np.max(np.abs(signal_plot)))
    else:
        ax_I.plot(q_plot, signal_plot, linewidth=1.5)
        # ax_I.fill_between(q_plot, signal_plot, 0, alpha=0.1, linewidth=1.5)
        ax_I.axhline(0, color='black', linewidth=0.5)
        ax_I.set_xlabel(x_label)
        ax_I.set_ylabel(label_i)
        ax_I.set_xlim(qmin, qmax)
        if is_difference: ax_I.set_ylim(-1.1*np.max(np.abs(signal_plot)), 1.1*np.max(np.abs(signal_plot)))

    ax_I.set_title(title_i)
    ax_I.grid(True, alpha=0.3)

    # Plot rPDF if provided
    if pdf_present:
        # Then handle plot orientation
        if plot_flip:
            ax_pdf.plot(pdf, r, linewidth=1.5)
            # ax_pdf.fill_betweenx(r, pdf, alpha=0.1, linewidth=1.5)
            ax_pdf.axvline(0, color='k', linewidth=1)
            ax_pdf.set_ylabel('$r$ (Å)')
            ax_pdf.set_xlabel(label_pdf)
            ax_pdf.set_ylim(rmin, rmax)
            if is_difference: ax_pdf.set_xlim(-1.1*np.max(np.abs(pdf)), 1.1*np.max(np.abs(pdf)))

        else:
            ax_pdf.plot(r, pdf, linewidth=1.5)
            # ax_pdf.fill_between(r, pdf, pdf*0, alpha=0.1, linewidth=1.5)
            ax_pdf.axhline(0, color='k', linewidth=1)
            ax_pdf.set_xlabel('$r$ (Å)')
            ax_pdf.set_ylabel(label_pdf)
            ax_pdf.set_xlim(rmin, rmax)
            if is_difference: ax_pdf.set_ylim(-1.1*np.max(np.abs(pdf)), 1.1*np.max(np.abs(pdf)))

        ax_pdf.set_title(title_pdf)
        ax_pdf.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.get_current_fig_manager().set_window_title(title_i)
    logger.info(f"Opening plot: '{title_i}'")
    plt.savefig(f'iamxed_static.png', dpi=300)
    logger.info("Plot saved as 'iamxed_static.png'")
    plt.show()


def plot_time_resolved(times: np.ndarray, times_smooth: np.ndarray, q: np.ndarray, signal: np.ndarray,
                       signal_smooth: np.ndarray, r: np.ndarray, pdf: np.ndarray, pdf_smooth: np.ndarray, is_xrd: bool,
                       plot_units: str = 'bohr-1', fwhm_fs: float = 150.0, plot_flip: bool = False) -> None:
    """Plot time-resolved diffraction signal (raw or smoothed).
    Args:
        times: Time points in fs
        times_smooth: Time points in fs for convoluted signal
        q: Q-values in atomic units
        signal: Diffraction signal (in Bohr^-1 for UED)
        signal_smooth: Convoluted diffraction signal (in Bohr^-1 for UED)
        r: R-grid for rPDF (optional)
        pdf: rPDF values (optional)
        pdf_smooth: Convoluted rPDF values (optional)
        is_xrd: True if XRD, False if UED
        plot_units: 'bohr-1' or 'angstrom-1'
        smoothed: Whether this is smoothed data
        fwhm_fs: FWHM in fs for smoothed data
        plot_flip: Whether to flip x and y axes
    """

    pdf_present = (r is not None) and (pdf is not None)

    if plot_units == 'angstrom-1':
        q_plot = q * 1.88973
        if is_xrd:
            x_label = '$q$ (Å$^{-1}$)' # XRD naming custom
        else:
            x_label = '$s$ (Å$^{-1}$)' # UED naming custom
    else:
        q_plot = q
        if is_xrd:
            x_label = '$q$ (Bohr$^{-1}$)'
        else:
            x_label = '$s$ (Bohr$^{-1}$)'

    # time minmax
    t_min_smooth, t_max_smooth = times_smooth.min(), times_smooth.max()
    t_min, t_max = times.min(), times.max()

    # setting up plot
    if pdf_present:
        fig, axs = plt.subplots(2, 2, figsize=(9, 8))
        ax_i = axs[0, 0]
        ax_i_smooth = axs[1, 0]
        ax_pdf = axs[0, 1]
        ax_pdf_smooth = axs[1, 1]
    else:
        fig, axs = plt.subplots(1, 2, figsize=(11, 5))
        ax_i = axs[0]
        ax_i_smooth = axs[1]

    ### plot raw dI/I data ###
    signal_plot = signal
    vlim = np.nanmax(np.abs(signal_plot))
    divnorm = TwoSlopeNorm(vmin=-vlim, vcenter=0., vmax=vlim)
    title_i = f'Time-Resolved {"XRD" if is_xrd else "UED"} Signal'

    if plot_flip:
        # Transpose data, swap axes: x=time, y=q
        extent = (t_min, t_max, q_plot.min(), q_plot.max())
        im = ax_i.imshow(signal_plot, extent=extent, aspect='auto', origin='lower', cmap='RdBu_r', norm=divnorm)
        ax_i.set_xlabel('Time (fs)')
        ax_i.set_ylabel(x_label)
    else:
        extent = (q_plot.min(), q_plot.max(), t_min, t_max)
        im = ax_i.imshow(signal_plot.T, extent=extent, aspect='auto', origin='lower', cmap='RdBu_r', norm=divnorm)
        ax_i.set_xlabel(x_label)
        ax_i.set_ylabel('Time (fs)')
    plt.colorbar(im, label=f'$\Delta I/I_0$ (%)')# if is_xrd else f'ΔsM(q) {sm_unit}')
    ax_i.set_title(title_i)

    ### plot convoluted dI/I data ###
    signal_plot = signal_smooth
    vlim = np.nanmax(np.abs(signal_plot))
    divnorm = TwoSlopeNorm(vmin=-vlim, vcenter=0., vmax=vlim)
    title_i = f'Convoluted TR-{"XRD" if is_xrd else "UED"} Signal (FWHM={fwhm_fs} fs)'

    if plot_flip:
        # Transpose data, swap axes: x=time, y=q
        extent = (t_min_smooth, t_max_smooth, q_plot.min(), q_plot.max())
        im = ax_i_smooth.imshow(signal_plot, extent=extent, aspect='auto', origin='lower', cmap='RdBu_r', norm=divnorm)
        ax_i_smooth.set_xlabel('Time (fs)')
        ax_i_smooth.set_ylabel(x_label)
        ax_i_smooth.axvline(0, color='grey', linestyle='-', lw=0.5)
    else:
        extent = (q_plot.min(), q_plot.max(), t_min_smooth, t_max_smooth)
        im = ax_i_smooth.imshow(signal_plot.T, extent=extent, aspect='auto', origin='lower', cmap='RdBu_r', norm=divnorm)
        ax_i_smooth.set_xlabel(x_label)
        ax_i_smooth.set_ylabel('Time (fs)')
        ax_i_smooth.axhline(0, color='grey', linestyle='-', lw=0.5)
    plt.colorbar(im, label=f'$\Delta I/I_0$ (%)')
    ax_i_smooth.set_title(title_i)

    ### plot rPDF if provided ###
    if pdf_present:
        ### raw rPDF ###
        signal_plot = pdf
        vlim = np.nanmax(np.abs(signal_plot))
        divnorm = TwoSlopeNorm(vmin=-vlim, vcenter=0., vmax=vlim)
        title_pdf = f'Time-Resolved {"XRD" if is_xrd else "UED"} $\Delta$rPDF'

        if plot_flip:
            # Transpose data, swap axes: x=time, y=q
            extent = (t_min, t_max, r.min(), r.max())
            im = ax_pdf.imshow(signal_plot, extent=extent, aspect='auto', origin='lower', cmap='RdBu_r', norm=divnorm)
            ax_pdf.set_xlabel('Time (fs)')
            ax_pdf.set_ylabel('$r$ (Å)')
        else:
            extent = (r.min(), r.max(), t_min, t_max)
            im = ax_pdf.imshow(signal_plot.T, extent=extent, aspect='auto', origin='lower', cmap='RdBu_r', norm=divnorm)
            ax_pdf.set_xlabel('$r$ (Å)')
            ax_pdf.set_ylabel('Time (fs)')
        plt.colorbar(im, label=f'$\Delta$rPDF (arb. units)')  # if is_xrd else f'ΔsM(q) {sm_unit}')
        ax_pdf.set_title(title_pdf)

        ### convoluted rPDF ###
        signal_plot = pdf_smooth
        vlim = np.nanmax(np.abs(signal_plot))
        divnorm = TwoSlopeNorm(vmin=-vlim, vcenter=0., vmax=vlim)
        title_pdf = f'Convoluted TR-{"XRD" if is_xrd else "UED"} $\Delta$rPDF'

        if plot_flip:
            # Transpose data, swap axes: x=time, y=q
            extent = (t_min_smooth, t_max_smooth, r.min(), r.max())
            im = ax_pdf_smooth.imshow(signal_plot, extent=extent, aspect='auto', origin='lower', cmap='RdBu_r', norm=divnorm)
            ax_pdf_smooth.set_xlabel('Time (fs)')
            ax_pdf_smooth.set_ylabel('$r$ (Å)')
            ax_pdf_smooth.axvline(0, color='grey', linestyle='-', lw=0.5)
        else:
            extent = (r.min(), r.max(), t_min_smooth, t_max_smooth)
            im = ax_pdf_smooth.imshow(signal_plot.T, extent=extent, aspect='auto', origin='lower', cmap='RdBu_r', norm=divnorm)
            ax_pdf_smooth.set_xlabel('$r$ (Å)')
            ax_pdf_smooth.set_ylabel('Time (fs)')
            ax_pdf_smooth.axhline(0, color='grey', linestyle='-', lw=0.5)
        plt.colorbar(im, label='$\Delta$rPDF (arb. units)')  # if is_xrd else f'ΔsM(q) {sm_unit}')
        ax_pdf_smooth.set_title(title_pdf)

    plt.tight_layout()
    plot_title = 'Time-Resolved ' + ('XRD' if is_xrd else 'UED') + ' Signal'
    plt.get_current_fig_manager().set_window_title(plot_title)
    logger.info(f"Opening plot: '{plot_title}'")
    plt.savefig(f'iamxed_time-resolved.png', dpi=300)
    logger.info("Plot saved as 'iamxed_time-resolved.png'")
    plt.show()

    ### IN CASE WE WANT TO PLOT A SERIES OF LINES IN THIME, WE CAN DO IT THIS WAY IN FUTURE ###
    # step = np.max([1, int(np.ceil(len(times_smooth)/20))])
    # colors = plt.cm.viridis(np.linspace(0, 1, len(times_smooth[::step])))
    # for i, signal in enumerate(pdf_smooth.T[::step,:]):
    #     label = f'{times_smooth[i*step]:.1f} fs' if i%2==0 else None
    #     plt.plot(r, signal, color=colors[i], label=label)
    # plt.legend()
    # plt.show()
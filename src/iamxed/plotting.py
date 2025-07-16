"""
Plotting utilities for XED (X-ray/Electron Diffraction) calculations.
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from matplotlib.colors import TwoSlopeNorm

def plot_static(q: np.ndarray, signal: np.ndarray, is_xrd: bool, is_difference: bool = False, plot_units: str = 'bohr-1', r: Optional[np.ndarray] = None, pdf: Optional[np.ndarray] = None, plot_flip: bool = False) -> None:
    """Plot static diffraction pattern, and PDF if provided.
    Args:
        q: Q-values in atomic units
        signal: Diffraction signal (in Bohr^-1 for UED)
        is_xrd: True if XRD, False if UED
        is_difference: True if plotting difference signal
        plot_units: 'bohr-1' or 'angstrom-1'
        r: r grid for PDF (optional)
        pdf: PDF values (optional)
        plot_flip: Whether to flip x and y axes
    """
    # Convert units for momentum transfer coordinate if needed
    if plot_units == 'angstrom-1':
        q_plot = q * 1.88973
        signal_plot = signal #not converting / 0.529177 if not is_xrd else signal
        if is_xrd:
            x_label = 'q (Å⁻¹)' # XRD naming custom
        else:
            x_label = 's (Å⁻¹)' # UED naming custom
    else:
        q_plot = q
        signal_plot = signal
        if is_xrd:
            x_label = 'q (Bohr⁻¹)'
        else:
            x_label = 's (Bohr⁻¹)'

    plt.figure(figsize=(10, 6))
    if plot_flip:
        plt.plot(signal_plot, q_plot, 'k-', linewidth=1.5)
        plt.ylabel(x_label)
    else:
        plt.plot(q_plot, signal_plot, 'k-', linewidth=1.5)
        plt.xlabel(x_label)
    #  Adding units for signal coordinate
    if is_xrd:
        if is_difference:
            label = 'ΔI/I₀ (%)'
            title = 'XRD Relative Difference Signal'
        else:
            label = 'I(q)'
            title = 'XRD Signal'
    else:
        sm_unit = '(Bohr⁻¹)'
        if is_difference:
            #label = f'ΔsM(s) {sm_unit}'
            label = 'ΔI/I₀ (%)'
            title = 'UED Relative Difference Signal'
        else:
            label = f'sM(s) {sm_unit}'
            title = 'UED Signal'
    if plot_flip:
        plt.xlabel(label)
    else:
        plt.ylabel(label)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'signal.png', dpi=300)
    plt.show()
    # Plot PDF if provided
    if (r is not None) and (pdf is not None):
        plt.figure(figsize=(10, 6))
        
        # Determine labels and title based on difference type first
        if is_difference:
            title = 'Difference Pair Distribution Function (ΔPDF)'
            label = 'ΔPDF(r) (arb. units)'
        else:
            title = 'Pair Distribution Function (PDF)'
            label = 'PDF(r) (arb. units)'
        
        # Then handle plot orientation
        if plot_flip:
            plt.plot(pdf, r, 'b-', linewidth=1.5)
            plt.ylabel('r (Å)')
            plt.xlabel(label)
        else:
            plt.plot(r, pdf, 'b-', linewidth=1.5)
            plt.xlabel('r (Å)')
            plt.ylabel(label)
        
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'pdf.png', dpi=300)
        plt.show()

def plot_time_resolved(times: np.ndarray, q: np.ndarray, signal: np.ndarray, is_xrd: bool, plot_units: str = 'bohr-1', smoothed: bool = False, fwhm_fs: float = 150.0, plot_flip: bool = False) -> None:
    """Plot time-resolved diffraction signal (raw or smoothed).
    Args:
        times: Time points in fs
        q: Q-values in atomic units
        signal: Diffraction signal (in Bohr^-1 for UED)
        is_xrd: True if XRD, False if UED
        plot_units: 'bohr-1' or 'angstrom-1'
        smoothed: Whether this is smoothed data
        fwhm_fs: FWHM in fs for smoothed data
        plot_flip: Whether to flip x and y axes
    """
    if plot_units == 'angstrom-1':
        q_plot = q * 1.88973
        signal_plot = signal #not converting / 0.529177 if not is_xrd else signal
        if is_xrd:
            x_label = 'q (Å⁻¹)' # XRD naming custom
        else:
            x_label = 's (Å⁻¹)' # UED naming custom
        #sm_unit = '(Å⁻¹)' if not is_xrd else ''
    else:
        q_plot = q
        signal_plot = signal
        if is_xrd:
            x_label = 'q (Bohr⁻¹)'
        else:
            x_label = 's (Bohr⁻¹)'
        #sm_unit = '(Bohr⁻¹)' if not is_xrd else ''
    vlim = np.nanmax(np.abs(signal_plot))
    divnorm = TwoSlopeNorm(vmin=-vlim, vcenter=0., vmax=vlim)
    plt.figure(figsize=(10, 6))
    t_min = times.min() #if smoothed else 0
    t_max = times.max()
    if plot_flip:
        # Transpose data, swap axes: x=q, y=time
        extent = (q_plot.min(), q_plot.max(), t_min, t_max)
        im = plt.imshow(signal_plot.T, extent=extent, aspect='auto', origin='lower', cmap='RdBu_r', norm=divnorm)
        plt.xlabel(x_label)
        plt.ylabel('Time (fs)')
        plt.colorbar(im, label=f'ΔI/I₀ (%)')# if is_xrd else f'ΔsM(q) {sm_unit}')
    else:
        extent = (t_min, t_max, q_plot.min(), q_plot.max())
        im = plt.imshow(signal_plot, extent=extent, aspect='auto', origin='lower', cmap='RdBu_r', norm=divnorm)
        plt.xlabel('Time (fs)')
        plt.ylabel(x_label)
        plt.colorbar(im, label=f'ΔI/I₀ (%)')# if is_xrd else f'ΔsM(q) {sm_unit}')
    if smoothed:
        plt.title(f'Time-Resolved {"XRD" if is_xrd else "UED"} Signal (Smoothed, FWHM={fwhm_fs} fs)')
    else:
        plt.title(f'Time-Resolved {"XRD" if is_xrd else "UED"} Signal')
    plt.tight_layout()
    plt.savefig(f'time_resolved_signal{"_smoothed" if smoothed else ""}.png', dpi=300)
    plt.show()

def plot_time_resolved_pdf(times: np.ndarray, r: np.ndarray, pdfs: np.ndarray, smoothed: bool = False, fwhm_fs: float = 150.0, plot_flip: bool = False) -> None:
    """Plot time-resolved PDF data.
    Args:
        times: Time points in fs
        r: R-grid in Angstroms
        pdfs: PDF data array (shape: [r_points, time_points])
        smoothed: Whether this is smoothed data (affects plot title)
        fwhm_fs: FWHM of Gaussian smoothing in fs (for plot title)
        plot_flip: Whether to flip x and y axes
    """
    plt.figure(figsize=(10, 6))
    vlim = np.nanmax(np.abs(pdfs))
    divnorm = TwoSlopeNorm(vmin=-vlim, vcenter=0., vmax=vlim)
    t_min = times.min() if smoothed else 0
    t_max = times.max()
    if plot_flip:
        # Transpose data, swap axes: x=r, y=time
        extent = (r.min(), r.max(), t_min, t_max)
        im = plt.imshow(pdfs.T, extent=extent, aspect='auto', origin='lower', cmap='RdBu_r', norm=divnorm)
        plt.xlabel('r (Å)')
        plt.ylabel('Time (fs)')
        plt.colorbar(im, label='ΔPDF(r) (arb. units)')
    else:
        extent = (t_min, t_max, r.min(), r.max())
        im = plt.imshow(pdfs, extent=extent, aspect='auto', origin='lower', cmap='RdBu_r', norm=divnorm)
        plt.xlabel('Time (fs)')
        plt.ylabel('r (Å)')
        plt.colorbar(im, label='ΔPDF(r) (arb. units)')
    title = 'Time-Resolved ΔPDF'
    if smoothed:
        title += f' (Smoothed, FWHM = {fwhm_fs:.0f} fs)'
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f'time_resolved_pdf{"_smoothed" if smoothed else ""}.png', dpi=300)
    plt.show() 
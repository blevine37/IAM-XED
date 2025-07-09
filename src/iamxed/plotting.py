"""
Plotting utilities for XED (X-ray/Electron Diffraction) calculations.
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from matplotlib.colors import TwoSlopeNorm

def plot_static(q: np.ndarray, signal: np.ndarray, is_xrd: bool, is_difference: bool = False, plot_units: str = 'bohr-1', r: Optional[np.ndarray] = None, pdf: Optional[np.ndarray] = None) -> None:
    """Plot static diffraction pattern, and PDF if provided.
    
    Args:
        q: Q-values in atomic units
        signal: Diffraction signal (in Bohr^-1 for UED)
        is_xrd: True if XRD, False if UED
        is_difference: True if plotting difference signal
        plot_units: 'bohr-1' or 'angstrom-1'
        r: r grid for PDF (optional)
        pdf: PDF values (optional)
    """
    # Convert units if needed
    if plot_units == 'angstrom-1':
        q_plot = q * 1.88973
        # Convert signal to Angstrom^-1 for UED
        signal_plot = signal / 0.529177 if not is_xrd else signal
        x_label = 'q (Å⁻¹)'
    else:
        q_plot = q
        signal_plot = signal
        x_label = 'q (Bohr⁻¹)'
    
    plt.figure(figsize=(10, 6))
    plt.plot(q_plot, signal_plot, 'k-', linewidth=1.5)
    plt.xlabel(x_label)
    if is_xrd:
        if is_difference:
            plt.ylabel('ΔI/I₀ (%)')
            plt.title('XRD Difference Pattern')
        else:
            plt.ylabel('I(q)')
            plt.title('XRD Pattern')
    else:
        # UED: add units to sM(q) label
        if plot_units == 'angstrom-1':
            sm_unit = '(Å⁻¹)'
        else:
            sm_unit = '(Bohr⁻¹)'
        if is_difference:
            plt.ylabel(f'ΔsM(q) {sm_unit}')
            plt.title('UED Difference Pattern')
        else:
            plt.ylabel(f'sM(q) {sm_unit}')
            plt.title('UED Pattern')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    # Plot PDF if provided
    if (r is not None) and (pdf is not None):
        plt.figure(figsize=(10, 6))
        plt.plot(r, pdf, 'b-', linewidth=1.5)
        plt.xlabel('r (Å)')
        plt.ylabel('P(r)')
        plt.title('Pair Distribution Function (PDF)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

def plot_time_resolved(times: np.ndarray, q: np.ndarray, signal: np.ndarray, is_xrd: bool, plot_units: str = 'bohr-1', smoothed: bool = False, fwhm_fs: float = 150.0) -> None:
    """Plot time-resolved diffraction pattern (unsmoothed or smoothed).
    
    Args:
        times: Time points in fs
        q: Q-values in atomic units
        signal: Diffraction signal (in Bohr^-1 for UED)
        is_xrd: True if XRD, False if UED
        plot_units: 'bohr-1' or 'angstrom-1'
        smoothed: Whether this is smoothed data
        fwhm_fs: FWHM in fs for smoothed data
    """
    # Convert units if needed
    if plot_units == 'angstrom-1':
        q_plot = q * 1.88973
        # Convert signal to Angstrom^-1 for UED
        signal_plot = signal / 0.529177 if not is_xrd else signal
        y_label = 'q (Å⁻¹)'
        sm_unit = '(Å⁻¹)' if not is_xrd else ''
    else:
        q_plot = q
        signal_plot = signal
        y_label = 'q (Bohr⁻¹)'
        sm_unit = '(Bohr⁻¹)' if not is_xrd else ''
    
    # Diverging normalization centered at zero
    vlim = np.nanmax(np.abs(signal_plot))
    divnorm = TwoSlopeNorm(vmin=-vlim, vcenter=0., vmax=vlim)
    plt.figure(figsize=(10, 6))
    
    # For smoothed data, we want to show the full time range including negative times
    # For raw data, we start at t=0
    t_min = times.min() if smoothed else 0
    extent = (t_min, times.max(), q_plot.min(), q_plot.max())
    im = plt.imshow(signal_plot, extent=extent, aspect='auto', origin='lower', cmap='RdBu_r', norm=divnorm)
    plt.colorbar(im, label=f'ΔI/I₀ (%)' if is_xrd else f'ΔsM(q) {sm_unit}')
    plt.xlabel('Time (fs)')
    plt.ylabel(y_label)
    if smoothed:
        plt.title(f'Time-Resolved {"XRD" if is_xrd else "UED"} Pattern (Smoothed, FWHM={fwhm_fs} fs)')
    else:
        plt.title(f'Time-Resolved {"XRD" if is_xrd else "UED"} Pattern (Unsmoothed)')
    plt.tight_layout()
    plt.show()

def plot_time_resolved_pdf(times: np.ndarray, r: np.ndarray, pdfs: np.ndarray, smoothed: bool = False, fwhm_fs: float = 150.0) -> None:
    """Plot time-resolved PDF data.
    
    Args:
        times: Time points in fs
        r: R-grid in Angstroms
        pdfs: PDF data array (shape: [r_points, time_points])
        smoothed: Whether this is smoothed data (affects plot title)
        fwhm_fs: FWHM of Gaussian smoothing in fs (for plot title)
    """
    plt.figure(figsize=(10, 6))
    # Use diverging normalization centered at zero for better visualization
    vlim = np.nanmax(np.abs(pdfs))
    divnorm = TwoSlopeNorm(vmin=-vlim, vcenter=0., vmax=vlim)
    
    # For smoothed data, we want to show the full time range including negative times
    # For raw data, we start at t=0
    t_min = times.min() if smoothed else 0
    extent = (t_min, times.max(), r.min(), r.max())
    im = plt.imshow(pdfs, extent=extent, aspect='auto', origin='lower', cmap='RdBu_r', norm=divnorm)
    plt.colorbar(im, label='ΔPDF(r) (arb. units)')
    plt.xlabel('Time (fs)')
    plt.ylabel('r (Å)')
    title = 'Time-Resolved PDF'
    if smoothed:
        title += f' (Smoothed, FWHM = {fwhm_fs:.0f} fs)'
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f'time_resolved_pdf{"_smoothed" if smoothed else ""}.png', dpi=300)
    plt.show() 
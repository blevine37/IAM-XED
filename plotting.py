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
        signal: Diffraction signal
        is_xrd: True if XRD, False if UED
        is_difference: True if plotting difference signal
        plot_units: 'bohr-1' or 'angstrom-1'
        r: r grid for PDF (optional)
        pdf: PDF values (optional)
    """
    if plot_units == 'angstrom-1':
        q_plot = q * 1.88973
        x_label = 'q (Å⁻¹)'
    else:
        q_plot = q
        x_label = 'q (Bohr⁻¹)'
    plt.figure(figsize=(10, 6))
    plt.plot(q_plot, signal, 'k-', linewidth=1.5)
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
    """Plot time-resolved diffraction pattern (unsmoothed or smoothed)."""
    if plot_units == 'angstrom-1':
        q_plot = q * 1.88973
        y_label = 'q (Å⁻¹)'
    else:
        q_plot = q
        y_label = 'q (Bohr⁻¹)'
    # Diverging normalization centered at zero
    vlim = np.nanmax(np.abs(signal))
    divnorm = TwoSlopeNorm(vmin=-vlim, vcenter=0., vmax=vlim)
    plt.figure(figsize=(10, 6))
    extent = (times.min(), times.max(), q_plot.min(), q_plot.max())
    im = plt.imshow(signal, extent=extent, aspect='auto', origin='lower', cmap='RdBu_r', norm=divnorm)
    plt.colorbar(im, label='ΔI/I₀ (%)' if is_xrd else 'ΔsM(q)')
    plt.xlabel('Time (fs)')
    plt.ylabel(y_label)
    if smoothed:
        plt.title(f'Time-Resolved {"XRD" if is_xrd else "UED"} Pattern (Smoothed, FWHM={fwhm_fs} fs)')
    else:
        plt.title(f'Time-Resolved {"XRD" if is_xrd else "UED"} Pattern (Unsmoothed)')
    plt.tight_layout()
    plt.show() 
"""
Physics calculations for X-ray and Electron Diffraction (XED).
Contains base calculator class and specific implementations for XRD and UED.
"""
import numpy as np
from scipy.interpolate import interp1d
import scipy.ndimage
import os
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable

from .io_utils import read_xyz, read_xyz_trajectory, find_xyz_files, is_trajectory_file
from .XSF.xsf_data_elastic import XSF_DATA
from .ESF.esf_data import ESF_DATA

# Physical constants
ANG_TO_BH = 1.8897259886
BH_TO_ANG = 1 / ANG_TO_BH
S_TO_Q = 4 * np.pi
CM_TO_BOHR = 188972598.85789

class BaseDiffractionCalculator(ABC):
    """Base class for diffraction calculations."""
    
    def __init__(self, q_start: float, q_end: float, num_point: int, elements: List[str]):
        self.q_start = q_start
        self.q_end = q_end
        self.num_point = num_point
        self.elements = elements
        self.qfit = np.linspace(q_start, q_end, num=num_point)
        self.form_factors = None
        
    @abstractmethod
    def load_form_factors(self):
        """Load form factors for the calculation."""
        pass
        
    def calc_molecular_intensity(self, aafs: List[np.ndarray], coords: np.ndarray) -> np.ndarray:
        """Calculate molecular intensity."""
        # XRD is real-valued, UED is complex
        Imol = np.zeros_like(self.qfit, dtype=float if isinstance(self, XRDDiffractionCalculator) else complex)
        for i, (i_aaf, i_p) in enumerate(zip(aafs, coords)):
            for j, (j_aaf, j_p) in enumerate(zip(aafs, coords)):
                if i == j:
                    continue
                r_ij = np.linalg.norm(i_p - j_p)
                qr = self.qfit * r_ij
                sinc_term = np.sinc(qr / np.pi)
                if isinstance(self, UEDDiffractionCalculator):
                    # For UED, keep complex conjugate for proper phase handling
                    Imol += np.conjugate(i_aaf) * j_aaf * sinc_term
                else:
                    Imol += i_aaf * j_aaf * sinc_term
        return Imol

    @staticmethod
    def gaussian_smooth_2d_time(Z: np.ndarray, times: np.ndarray, fwhm_fs: float) -> tuple[np.ndarray, np.ndarray]:
        """Apply Gaussian smoothing in time dimension for time-resolved signals.
        
        Implements time-resolved signal smoothing that:
        1. Shows full smoothing effects before t=0 (includes left padding)
        2. Discards edge effects at the end of the signal
        3. Uses FWHM-based Gaussian window
        4. Preserves signal normalization
        
        Args:
            Z: 2D array of shape (n_q, n_t) to smooth along time axis
            times: Time points in femtoseconds
            fwhm_fs: Full Width at Half Maximum of Gaussian kernel in femtoseconds
            
        Returns:
            tuple[np.ndarray, np.ndarray]: (smoothed signal, extended time axis)
            Time axis includes negative times to show full smoothing window
        """
        # Convert FWHM to sigma (standard deviation)
        sigma_fs = fwhm_fs / 2.355
        
        # Get time step and calculate padding
        dt = times[1] - times[0] if len(times) > 1 else 1.0
        sigma_steps = sigma_fs / dt
        pad_width = int(np.ceil(3 * sigma_steps))  # 3 sigma padding

        # Pad on both sides with edge values
        Z_padded = np.pad(Z, ((0, 0), (pad_width, pad_width)), mode='edge')
        
        # Apply Gaussian filter
        Z_smooth = scipy.ndimage.gaussian_filter1d(
            Z_padded, 
            sigma=sigma_steps,
            axis=1,
            mode='nearest'
        )
        
        # Keep left padding but discard right padding
        Z_smooth = Z_smooth[:, :len(times) + pad_width]
        
        # Create time axis including negative times
        times_extended = np.concatenate([
            times[0] + dt * np.arange(-pad_width, 0),   # Left padding
            times,                                      # Original times
        ])
        
        return Z_smooth, times_extended

    @staticmethod
    def FT(s: np.ndarray, T: np.ndarray, alpha: float) -> np.ndarray:
        """Fourier transform for PDF calculation.
        
        Args:
            s: Q-grid in Angstrom^-1
            T: Signal to transform
            alpha: Damping parameter in Angstrom^2
            
        Returns:
            PDF on same grid as input
        """
        T = np.nan_to_num(T)
        Tr = np.empty_like(T)
        for pos, k in enumerate(s):
            Tr[pos] = k * np.trapz(T * np.sin(s * k) * np.exp(-alpha * s**2), x=s)
        return Tr

class XRDDiffractionCalculator(BaseDiffractionCalculator):
    """Calculate XRD patterns using IAM approximation."""
    
    def __init__(self, q_start: float, q_end: float, num_point: int, elements: List[str], 
                 inelastic: bool = False):
        """Initialize XRD calculator.
        
        Args:
            q_start: Starting q value in atomic units
            q_end: Ending q value in atomic units
            num_point: Number of q points
            elements: List of elements to load form factors for
            inelastic: Whether to include inelastic scattering
        """
        super().__init__(q_start, q_end, num_point, elements)
        self.inelastic = inelastic
        self.Szaloki_params = {}
        if inelastic:
            self.load_Szaloki_params()
            
    def load_Szaloki_params(self):
        """Load Szaloki parameters for inelastic scattering."""
        xsf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'XSF')
        self.Szaloki_params = np.loadtxt(os.path.join(xsf_path, 'Szaloki_params_inelastic.csv'), delimiter=',')
        
    def load_form_factors(self):
        """Load XRD form factors from xsf_data_elastic.py."""
        self.form_factors = {}
        for el in self.elements:
            if el not in XSF_DATA:
                raise ValueError(f"XRD form factor data not found for element '{el}' in XSF_DATA")
            
            # Get data from XSF_DATA
            data = XSF_DATA[el]
            
            # Convert units: sin(theta)/lambda in Ang^-1 to q in atomic units
            q_vals = data[:, 0] * S_TO_Q / ANG_TO_BH
            f_vals = data[:, 1]
            
            # Create interpolation function
            f = interp1d(q_vals, f_vals, kind='cubic', bounds_error=False, fill_value=0)
            self.form_factors[el] = f(self.qfit)  # XRD form factors are real
        
    def calc_atomic_intensity(self, atoms: List[str]) -> np.ndarray:
        """Calculate atomic intensity for XRD."""
        Iat = np.zeros_like(self.qfit)
        for atom in atoms:
            ff = self.form_factors[atom]
            Iat += ff ** 2
            if self.inelastic and self.Szaloki_params is not None:
                # Add inelastic contribution
                atn = self.get_atomic_number(atom)
                inel = self.calc_inelastic(atn)
                Iat += inel
        return Iat
        
    def get_atomic_number(self, element: str) -> int:
        """Get atomic number for element."""
        # Atomic number (1-based) to element symbol mapping for Z=1-98
        ELEMENTS = [
            None,  # 0-index placeholder
            'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
            'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
            'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
            'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
            'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
            'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
            'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
            'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
            'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
            'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf'
        ]
        
        try:
            return ELEMENTS.index(element)
        except ValueError:
            raise ValueError(f"Element '{element}' not found in periodic table (Z=1-98)")
        
    def calc_inelastic(self, atomic_number: int) -> np.ndarray:
        """Calculate inelastic scattering for given atomic number."""
        if self.Szaloki_params is None:
            return np.zeros_like(self.qfit)
        params = self.Szaloki_params[atomic_number-1, :]
        Z, d1, d2, d3, q1, t1, t2, t3, *_ = params
        return self._calc_inel(Z, d1, d2, d3, q1, t1, t2, t3, self.qfit * ANG_TO_BH / (4 * np.pi))
        
    @staticmethod
    def _calc_inel(Z, d1, d2, d3, q1, t1, t2, t3, q):
        """Helper for inelastic calculation."""
        s = np.zeros_like(q)
        s1 = XRDDiffractionCalculator._calc_s1(q, d1, d2, d3)
        s2 = XRDDiffractionCalculator._calc_s2(q, Z, d1, d2, d3, q1, t1, t2, t3)
        s[q < q1] = s1[q < q1]
        s[q >= q1] = s2[q >= q1]
        return s
        
    @staticmethod
    def _calc_s1(q, d1, d2, d3):
        s1 = np.zeros_like(q)
        for i, d in enumerate([d1, d2, d3]):
            s1 += d * (np.exp(q) - 1) ** (i + 1)
        return s1
        
    @staticmethod
    def _calc_s2(q, Z, d1, d2, d3, q1, t1, t2, t3):
        s1 = XRDDiffractionCalculator._calc_s1(q, d1, d2, d3)
        s1q1 = XRDDiffractionCalculator._calc_s1(q1, d1, d2, d3)
        g1 = 1 - np.exp(t1 * (q1 - q))
        g2 = 1 - np.exp(t3 * (q1 - q))
        return (Z - s1q1 - t2) * g1 + t2 * g2 + s1q1

    def calc_single(self, geom_file: str, pdf_alpha: float = 0.04) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """Calculate single geometry XRD pattern, or average over all geometries in a directory or trajectory file."""
        import os
        if os.path.isdir(geom_file):
            xyz_files = find_xyz_files(geom_file)
            if not xyz_files:
                raise ValueError(f'No XYZ files found in directory: {geom_file}')
            signals = []
            for f in xyz_files:
                atoms, coords = read_xyz(f)  #Note: Currently expects single frame files
                if not self.form_factors:
                    self.load_form_factors()
                Iat = self.calc_atomic_intensity(atoms)
                Imol = self.calc_molecular_intensity([self.form_factors[a] for a in atoms], coords)
                Itot = Iat + Imol
                signals.append(Itot)
            avg_signal = np.mean(signals, axis=0)
            return self.qfit, avg_signal, None, None
        elif is_trajectory_file(geom_file):
            atoms, trajectory = read_xyz_trajectory(geom_file)
            if not self.form_factors:
                self.load_form_factors()
            signals = []
            for coords in trajectory:
                Iat = self.calc_atomic_intensity(atoms)
                Imol = self.calc_molecular_intensity([self.form_factors[a] for a in atoms], coords)
                Itot = Iat + Imol
                signals.append(Itot)
            avg_signal = np.mean(signals, axis=0)
            return self.qfit, avg_signal, None, None
        else:
            atoms, coords = read_xyz(geom_file)
            if not self.form_factors:
                self.load_form_factors()
            Iat = self.calc_atomic_intensity(atoms)
            Imol = self.calc_molecular_intensity([self.form_factors[a] for a in atoms], coords)
            Itot = Iat + Imol
            return self.qfit, Itot, None, None

    def calc_difference(self, geom1: str, geom2: str, pdf_alpha: float = 0.04) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """Calculate difference between two geometries.
        
        For XRD, returns relative difference (I1-I2)/I2 * 100 as percentage.
        Atom order does not need to match, only the sets of elements must be the same.
        If either geom1 or geom2 is a directory or trajectory file, ensemble averaging will be performed.
        """
        # Get signals for both inputs using calc_single
        _, I1, _, _ = self.calc_single(geom1, pdf_alpha)
        _, I2, _, _ = self.calc_single(geom2, pdf_alpha)
        
        # Calculate relative difference in percent
        diff = (I1 - I2) / I2 * 100
        return self.qfit, diff, None, None

    def calc_trajectory(self, trajfile: str, timestep_au: float = 10.0, fwhm_fs: float = 150.0, pdf_alpha: float = 0.04, tmax_fs: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Calculate time-resolved XRD pattern from trajectory, returning both unsmoothed and smoothed signals.
        For XRD, PDFs are not calculated, so dummy arrays are returned for compatibility with UED.
        
        Args:
            trajfile: Path to trajectory file
            timestep_au: Time step in atomic units
            fwhm_fs: FWHM of Gaussian smoothing in fs
            pdf_alpha: Damping parameter for PDF calculation (unused in XRD)
            tmax_fs: Maximum time to calculate up to in femtoseconds
        
        Returns:
            times: Time points in fs
            q: Q-grid in atomic units
            signal_raw: Raw signal (not smoothed)
            signal_smooth: Gaussian smoothed signal
            r: R-grid for PDF (dummy for XRD)
            pdfs_raw: Raw PDFs (dummy for XRD)
            pdfs_smooth: Gaussian smoothed PDFs (dummy for XRD)
        """
        atoms, trajectory = read_xyz_trajectory(trajfile)
        if not self.form_factors:
            self.load_form_factors()
            
        # Calculate reference (t=0) intensities
        Iat0 = self.calc_atomic_intensity(atoms)
        Imol0 = self.calc_molecular_intensity([self.form_factors[a] for a in atoms], trajectory[0])
        I0 = Iat0 + Imol0
        
        signals = []
        dt_fs = timestep_au / 40 * 0.9675537016  # Convert timestep to fs
        if tmax_fs is not None:
            n_frames = min(len(trajectory), int(np.floor(tmax_fs / dt_fs)) + 1)
        else:
            n_frames = len(trajectory)
        #Loop over frames
        for i, coords in enumerate(tqdm(trajectory[:n_frames], desc='Trajectory frames', total=n_frames, mininterval=0, dynamic_ncols=True)):
            # Check if we've reached the time limit
            current_time = i * dt_fs
            if tmax_fs is not None and current_time > tmax_fs:
                break
                
            # Calculate current frame intensities
            #Iat = self.calc_atomic_intensity(atoms)
            Imol = self.calc_molecular_intensity([self.form_factors[a] for a in atoms], coords)
            I = Iat0 + Imol
            
            # Calculate relative difference in percent
            rel = (I - I0) / I0 * 100
            signals.append(rel)
            
        signal_raw = np.array(signals).T
        
        # Calculate time axis for the frames we actually processed
        times = np.arange(len(signals)) * dt_fs
        
        # Apply temporal smoothing
        signal_smooth, times_smooth = self.gaussian_smooth_2d_time(signal_raw, times, fwhm_fs)
        
        # Return dummy PDF arrays for compatibility with UED
        r = np.array([0.0])  # Dummy r grid
        pdfs_raw = np.zeros((1, signal_raw.shape[1]))  # Dummy raw PDFs
        pdfs_smooth = np.zeros((1, signal_smooth.shape[1]))  # Dummy smoothed PDFs
        
        return times_smooth, self.qfit, signal_raw, signal_smooth, r, pdfs_raw, pdfs_smooth

    def calc_ensemble(self, xyz_dir: str, timestep_au: float = 10.0, fwhm_fs: float = 150.0, pdf_alpha: float = 0.04, tmax_fs: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Calculate ensemble average of trajectories.
        
        Returns relative differences (I(t)-I(0))/I(0) * 100 as percentage.
        For each time point, average over all available trajectories.
        
        Args:
            xyz_dir: Directory containing XYZ trajectory files
            timestep_au: Time step in atomic units
            fwhm_fs: FWHM of Gaussian smoothing in fs
            pdf_alpha: Damping parameter for PDF calculation (unused in XRD)
            tmax_fs: Maximum time to calculate up to in femtoseconds
        """
        xyz_files = find_xyz_files(xyz_dir)
        if not xyz_files:
            raise ValueError(f"No XYZ files found in directory: {xyz_dir}")
        all_Imol = []
        all_Imol0 = []
        max_frames = 0
        dt_fs = timestep_au / 40 * 0.9675537016
        Iat0 = None
        for idx, xyz_file in enumerate(tqdm(xyz_files, desc='Ensemble files')):
            atoms, trajectory = read_xyz_trajectory(xyz_file)
            if not self.form_factors:
                self.load_form_factors()
            if Iat0 is None:
                Iat0 = self.calc_atomic_intensity(atoms)
            Imol0 = self.calc_molecular_intensity([self.form_factors[a] for a in atoms], trajectory[0])
            all_Imol0.append(Imol0)
            Imol_traj = []
            if tmax_fs is not None:
                n_frames = min(len(trajectory), int(np.floor(tmax_fs / dt_fs)) + 1)
            else:
                n_frames = len(trajectory)
            #Loop over frames
            for i, frame in enumerate(tqdm(trajectory[:n_frames], desc='Frames', leave=False, total=n_frames, mininterval=0, dynamic_ncols=True)):
                # Check if we've reached the time limit
                current_time = i * dt_fs
                if tmax_fs is not None and current_time > tmax_fs:
                    break
                Imol = self.calc_molecular_intensity([self.form_factors[a] for a in atoms], frame)
                Imol_traj.append(Imol)
            Imol_traj = np.array(Imol_traj).T
            all_Imol.append(Imol_traj)
            max_frames = max(max_frames, Imol_traj.shape[1])
        padded_Imol = []
        for Imol in all_Imol:
            if Imol.shape[1] < max_frames:
                pad_width = ((0, 0), (0, max_frames - Imol.shape[1]))
                padded = np.pad(Imol, pad_width, mode='constant', constant_values=np.nan)
            else:
                padded = Imol
            padded_Imol.append(padded)
        stacked_Imol = np.stack(padded_Imol, axis=0)
        mean_Imol = np.nanmean(stacked_Imol, axis=0)
        mean_Imol0 = np.nanmean(np.stack(all_Imol0, axis=0), axis=0)
        if Iat0 is None:
            raise ValueError("No valid trajectories found to compute atomic intensity (Iat0).")
        numerator = (Iat0[:, None] + mean_Imol) - (Iat0[:, None] + mean_Imol0[:, None])
        denominator = (Iat0[:, None] + mean_Imol0[:, None])
        signal_raw = numerator / denominator * 100
        times = np.arange(max_frames) * dt_fs
        signal_smooth, times_smooth = self.gaussian_smooth_2d_time(signal_raw, times, fwhm_fs)
        
        # Return dummy PDF arrays for compatibility with UED
        r = np.array([0.0])  # Dummy r grid
        pdfs_raw = np.zeros((1, signal_raw.shape[1]))  # Dummy raw PDFs
        pdfs_smooth = np.zeros((1, signal_smooth.shape[1]))  # Dummy smoothed PDFs
        
        return times, self.qfit, signal_raw, signal_smooth, r, pdfs_raw, pdfs_smooth

class UEDDiffractionCalculator(BaseDiffractionCalculator):
    """UED-specific calculator implementation."""
    
    def __init__(self, q_start: float, q_end: float, num_point: int, elements: List[str],
                 ued_energy_ev: float = 3.7e6):
        super().__init__(q_start, q_end, num_point, elements)
        self.ued_energy_ev = ued_energy_ev
        self.elekin_ha = ued_energy_ev / 27.2114
        self.k = (self.elekin_ha * (self.elekin_ha + 2 * 137 ** 2)) ** 0.5 / 137
        
    def load_form_factors(self):
        """Load UED form factors from esf_data.py."""
        self.form_factors = {}
        for el in self.elements:
            if el not in ESF_DATA:
                raise ValueError(f"UED scattering data not found for element '{el}' in ESF_DATA")
            
            # Get data from ESF_DATA
            data = ESF_DATA[el]
            theta_vals = data[:, 0]  # degrees
            reF_vals = data[:, 1] * CM_TO_BOHR  # Convert cm to bohr
            imF_vals = data[:, 2] * CM_TO_BOHR  # Convert cm to bohr
            
            # Convert q to theta for interpolation
            with np.errstate(invalid='ignore'):
                thetafit = 2 * np.arcsin(np.clip(self.qfit / (2 * self.k), -1, 1)) * 180 / np.pi
                
            # Interpolate real and imag parts
            fre = interp1d(theta_vals, reF_vals, kind='cubic', bounds_error=False, fill_value=0)
            fim = interp1d(theta_vals, imF_vals, kind='cubic', bounds_error=False, fill_value=0)
            yre = fre(thetafit)
            yim = fim(thetafit)
            self.form_factors[el] = yre + 1j * yim  # UED form factors are complex

    def calc_atomic_intensity(self, atoms: List[str]) -> np.ndarray:
        """Calculate atomic intensity for UED."""
        Iat = np.zeros_like(self.qfit, dtype=complex)
        for atom in atoms:
            ff = self.form_factors[atom]  # Complex form factor
            Iat += ff * np.conjugate(ff)  # Multiply by conjugate to get real intensity
        return np.real(Iat)  # Return real part since intensity must be real
            
    def calc_single(self, geom_file: str, pdf_alpha: float = 0.04) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """Calculate single geometry UED pattern and PDF, or average over all geometries in a directory or trajectory file."""
        import os
        from .io_utils import is_trajectory_file, read_xyz_trajectory, find_xyz_files
        if os.path.isdir(geom_file):
            xyz_files = find_xyz_files(geom_file)
            if not xyz_files:
                raise ValueError(f'No XYZ files found in directory: {geom_file}')
            signals = []
            pdfs = []
            for f in xyz_files:
                atoms, coords = read_xyz(f) #Note: Currently expects single frame files
                if not self.form_factors:
                    self.load_form_factors()
                Iat = self.calc_atomic_intensity(atoms)
                Imol = self.calc_molecular_intensity([self.form_factors[a] for a in atoms], coords)
                sm = self.qfit * (Imol / Iat)
                sm_real = np.real(sm)
                q_ang = self.qfit / BH_TO_ANG
                sm_ang = sm_real / BH_TO_ANG
                r = q_ang.copy()  # Use same grid for r as q
                pdf = self.FT(q_ang, sm_ang, pdf_alpha)
                signals.append(sm_real)
                pdfs.append(pdf)
            avg_signal = np.mean(signals, axis=0)
            avg_pdf = np.mean(pdfs, axis=0)
            return self.qfit, avg_signal, r, avg_pdf
        elif is_trajectory_file(geom_file):
            atoms, trajectory = read_xyz_trajectory(geom_file)
            if not self.form_factors:
                self.load_form_factors()
            signals = []
            pdfs = []
            Iat = self.calc_atomic_intensity(atoms)
            for coords in trajectory:
                #Iat = self.calc_atomic_intensity(atoms)
                Imol = self.calc_molecular_intensity([self.form_factors[a] for a in atoms], coords)
                sm = self.qfit * (Imol / Iat)
                sm_real = np.real(sm)
                q_ang = self.qfit / BH_TO_ANG
                sm_ang = sm_real / BH_TO_ANG
                r = q_ang.copy()  # Use same grid for r as q
                pdf = self.FT(q_ang, sm_ang, pdf_alpha)
                signals.append(sm_real)
                pdfs.append(pdf)
            avg_signal = np.mean(signals, axis=0)
            avg_pdf = np.mean(pdfs, axis=0)
            return self.qfit, avg_signal, r, avg_pdf
        else:
            atoms, coords = read_xyz(geom_file)
            if not self.form_factors:
                self.load_form_factors()
            Iat = self.calc_atomic_intensity(atoms)
            Imol = self.calc_molecular_intensity([self.form_factors[a] for a in atoms], coords)
            sm = self.qfit * (Imol / Iat)
            sm_real = np.real(sm)
            q_ang = self.qfit / BH_TO_ANG
            sm_ang = sm_real / BH_TO_ANG
            r = q_ang.copy()  # Use same grid for r as q
            pdf = self.FT(q_ang, sm_ang, pdf_alpha)
            return self.qfit, sm_real, r, pdf

    def calc_difference(self, geom1: str, geom2: str, pdf_alpha: float = 0.04) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """Calculate difference between two geometries and its PDF.
        
        Atom order does not need to match, only the sets of elements must be the same.
        If either geom1 or geom2 is a directory or trajectory file, ensemble averaging will be performed.
        """
        # Get signals and PDFs for both inputs using calc_single
        _, sm1, r1, pdf1 = self.calc_single(geom1, pdf_alpha)
        _, sm2, r2, pdf2 = self.calc_single(geom2, pdf_alpha)
        
        # Calculate difference signal and PDF
        sm_diff = sm1 - sm2
        
        q_ang = r1 / 0.529177
        sm_diff_ang = sm_diff / 0.529177
        r_diff = r1  # Use same grid for r as q
        pdf_diff = self.FT(q_ang, sm_diff_ang, pdf_alpha)
        return self.qfit, sm_diff, r_diff, pdf_diff

    def calc_trajectory(self, trajfile: str, timestep_au: float = 10.0, fwhm_fs: float = 150.0, pdf_alpha: float = 0.04, tmax_fs: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Calculate time-resolved UED pattern from trajectory, returning both unsmoothed and smoothed signals and their PDFs.
        
        Args:
            trajfile: Path to trajectory file
            timestep_au: Time step in atomic units
            fwhm_fs: FWHM of Gaussian smoothing in fs
            pdf_alpha: Damping parameter for PDF calculation
            tmax_fs: Maximum time to calculate up to in femtoseconds
        
        Returns:
            times: Time points in fs
            q: Q-grid in atomic units
            signal_raw: Raw signal (not smoothed)
            signal_smooth: Gaussian smoothed signal
            r: R-grid for PDF in Angstroms
            pdfs_raw: Raw PDFs (not smoothed)
            pdfs_smooth: Gaussian smoothed PDFs
        """
        atoms, trajectory = read_xyz_trajectory(trajfile)
        if not self.form_factors:
            self.load_form_factors()
        Iat = self.calc_atomic_intensity(atoms)
        sm0 = self.qfit * (self.calc_molecular_intensity([self.form_factors[a] for a in atoms], trajectory[0]) / Iat)
        sm0 = np.real(sm0)
        signals = []
        pdfs = []
        
        # Calculate q grid in Angstroms for PDF
        q_ang = self.qfit / BH_TO_ANG
        r = q_ang.copy()
        
        dt_fs = timestep_au / 40 * 0.9675537016  # Convert timestep to fs
        if tmax_fs is not None:
            n_frames = min(len(trajectory), int(np.floor(tmax_fs / dt_fs)) + 1)
        else:
            n_frames = len(trajectory)
        for i, coords in enumerate(tqdm(trajectory[:n_frames], desc='Trajectory frames', total=n_frames, mininterval=0, dynamic_ncols=True)):
            # Check if we've reached the time limit
            current_time = i * dt_fs
            if tmax_fs is not None and current_time > tmax_fs:
                break
                
            Imol = self.calc_molecular_intensity([self.form_factors[a] for a in atoms], coords)
            sm = self.qfit * (Imol / Iat)
            sm = np.real(sm)
            rel = sm - sm0
            
            # Calculate PDF for this frame using provided alpha
            sm_ang = rel / BH_TO_ANG  # Convert to Angstrom^-1 for PDF calculation
            pdf = self.FT(q_ang, sm_ang, pdf_alpha)
            
            signals.append(rel)
            pdfs.append(pdf)
            
        signal_raw = np.array(signals).T
        pdfs_raw = np.array(pdfs).T      # Shape: [r_points, time_points]
        
        # Calculate time axis for the frames we actually processed
        times = np.arange(len(signals)) * dt_fs
        
        # Smooth signals and PDFs separately
        signal_smooth, times_smooth = self.gaussian_smooth_2d_time(signal_raw, times, fwhm_fs)
        pdfs_smooth, _ = self.gaussian_smooth_2d_time(pdfs_raw, times, fwhm_fs)
        
        return times, self.qfit, signal_raw, signal_smooth, r, pdfs_raw, pdfs_smooth

    def calc_ensemble(self, xyz_dir: str, timestep_au: float = 10.0, fwhm_fs: float = 150.0, pdf_alpha: float = 0.04, tmax_fs: Optional[float] = None) -> tuple:
        """Calculate ensemble-averaged signal and PDF from a directory of trajectories, matching the interface of calc_trajectory.
        For each time point, average over all available trajectories.
        
        Args:
            xyz_dir: Directory containing XYZ trajectory files
            timestep_au: Time step in atomic units
            fwhm_fs: FWHM of Gaussian smoothing in fs
            pdf_alpha: Damping parameter for PDF calculation
            tmax_fs: Maximum time to calculate up to in femtoseconds
        """
        xyz_files = find_xyz_files(xyz_dir)
        if not xyz_files:
            raise ValueError(f"No XYZ files found in directory: {xyz_dir}")
        all_Imol = []
        all_Imol0 = []
        all_pdfs = []
        max_frames = 0
        dt_fs = timestep_au / 40 * 0.9675537016
        sfit = self.qfit
        q_ang = self.qfit / BH_TO_ANG
        r = q_ang.copy()
        Iat0 = None
        all_s = []  # Will hold s_k(t) for each trajectory
        for idx, xyz_file in enumerate(tqdm(xyz_files, desc='Ensemble files')):
            atoms, trajectory = read_xyz_trajectory(xyz_file)
            if not self.form_factors:
                self.load_form_factors()
            if Iat0 is None:
                Iat0 = self.calc_atomic_intensity(atoms)
            s_traj = []
            if tmax_fs is not None:
                n_frames = min(len(trajectory), int(np.floor(tmax_fs / dt_fs)) + 1)
            else:
                n_frames = len(trajectory)
            for i, coords in enumerate(tqdm(trajectory[:n_frames], desc='Frames', leave=False, total=n_frames, mininterval=0, dynamic_ncols=True)):
                current_time = i * dt_fs
                if tmax_fs is not None and current_time > tmax_fs:
                    break
                Imol = self.calc_molecular_intensity([self.form_factors[a] for a in atoms], coords)
                s = sfit * (Imol / Iat0)
                s_traj.append(s)
            s_traj = np.array(s_traj).T  # [q, t]
            all_s.append(s_traj)
            max_frames = max(max_frames, s_traj.shape[1])
        # Pad all s to max_frames with NaN
        padded_s = []
        for s in all_s:
            if s.shape[1] < max_frames:
                pad_width = ((0, 0), (0, max_frames - s.shape[1]))
                padded = np.pad(s, pad_width, mode='constant', constant_values=np.nan)
            else:
                padded = s
            padded_s.append(padded)
        stacked_s = np.stack(padded_s, axis=0)  # [n_traj, q, t]
        mean_s = np.nanmean(stacked_s, axis=0)   # [q, t]
        mean_s0 = np.nanmean(stacked_s[:, :, 0], axis=0)  # [q]
        # Final signal: mean_s(t) - mean_s(0)
        signal_raw = np.real(mean_s - mean_s0[:, None])
        # Now calculate PDF from the final signal
        sm_ang = signal_raw / BH_TO_ANG  # Convert to Angstrom^-1 for PDF calculation
        pdfs_raw = np.empty((len(q_ang), signal_raw.shape[1]))
        for t in range(signal_raw.shape[1]):
            pdfs_raw[:, t] = self.FT(q_ang, sm_ang[:, t], pdf_alpha)
        pdfs_raw = np.real(pdfs_raw)
        times = np.arange(max_frames) * dt_fs
        signal_smooth, times_smooth = self.gaussian_smooth_2d_time(signal_raw, times, fwhm_fs)
        pdfs_smooth, _ = self.gaussian_smooth_2d_time(pdfs_raw, times, fwhm_fs)
        return times, self.qfit, signal_raw, signal_smooth, r, pdfs_raw, pdfs_smooth 
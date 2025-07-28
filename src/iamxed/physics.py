"""
Physics calculations for X-ray and Electron Diffraction (XED).
Contains base calculator class and specific implementations for XRD and UED.
"""
import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
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
from logging import getLogger

logger = getLogger("my_logger") # getting logger

# Physical constants
ANG_TO_BH = 1.8897259886
BH_TO_ANG = 1 / ANG_TO_BH
CM_TO_BOHR = 188972598.85789
AU_TO_FS = 0.02418884254

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
        pass # placeholder for abstract method which is implemented in subclasses
        
    def calc_molecular_intensity(self, aafs: List[np.ndarray], coords: np.ndarray) -> np.ndarray:
        """Calculate molecular intensity."""
        Imol = np.zeros_like(self.qfit, dtype=float)
        for i, (i_aaf, i_p) in enumerate(zip(aafs, coords)):
            for j, (j_aaf, j_p) in enumerate(zip(aafs, coords)):
                if j <= i:
                    continue
                r_ij = np.linalg.norm(i_p - j_p)
                qr = self.qfit * r_ij
                sinc_term = np.sinc(qr / np.pi) # qr / np.pi to remove normalization factor
                Imol += 2 * np.real(np.conjugate(i_aaf) * j_aaf * sinc_term) # np.conjugate is used for UED to handle complex form factors but does not affect floats in XRD
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
        Z_smooth = gaussian_filter1d(
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
    def FT(r: np.ndarray, s: np.ndarray, T: np.ndarray, alpha: float) -> np.ndarray:
        """Fourier transform for rPDF calculation.
        
        Args:
            s: Q-grid in Angstrom^-1
            T: Signal to transform
            alpha: Damping parameter in Angstrom^2
            
        Returns:
            rPDF on same grid as input
        """
        logger.debug("[DEBUG]: Entering Fourier transform for rPDF calculation")
        T = np.nan_to_num(T)
        Tr = np.empty_like(T)
        for pos, k in enumerate(r):
            Tr[pos] = k * np.trapz(T * np.sin(s * k) * np.exp(-alpha * s**2), x=s)


        ### THE FOLLOWING COMMENTED OUT CODE IS AND ALTERNATIVE IMPLEMENTATION USING DST ###
        # Discrete Sine Transform (DST) can massively speed up calculations but it has some artifacts.
        # Using numerical integration with np.trapz is more accurate but slower.
        # Currently, we use np.trapz for accuracy but we intend to add flag for fast calculations.
        #
        # from scipy.fft import dst
        #
        # s2 = np.linspace(0, 5000, 500000)
        # damped_T = np.interp(s2, s, T*np.exp(-alpha*s**2))  # Interpolate T to new s grid
        #
        # Tr2 = np.fft.fftshift(dst(damped_T, type=2, norm='ortho'))  # Discrete sine transform
        # r2 = np.pi*np.fft.fftshift(np.fft.fftfreq(len(s2), (s2[1] - s2[0])))  # Frequency grid for FFT
        # Tr2 = np.interp(r, r2, Tr2)  # Interpolate back to original r grid
        # Tr2 *= r  # Scale by r

        logger.debug("[DEBUG]: Finished Fourier transform for rPDF calculation")

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
        super().__init__(q_start, q_end, num_point, elements) # initialize base class
        self.inelastic = inelastic
        self.Szaloki_params = None
        self.load_form_factors()
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
                logger.error(f"ERROR: XRD form factor data not found for element '{el}' in XSF_DATA")
                raise ValueError(f"XRD form factor data not found for element '{el}' in XSF_DATA")

            # Get data from XSF_DATA
            data = XSF_DATA[el]
            
            # Convert units: sin(theta)/lambda in Ang^-1 to q in atomic units
            q_vals = data[:, 0] * (4 * np.pi) / ANG_TO_BH # 4 * np.pi is the conversion factor for Szaloki's definition of q
            f_vals = data[:, 1]
            
            # Create interpolation function
            f = interp1d(q_vals, f_vals, kind='cubic', bounds_error=False, fill_value=0)
            self.form_factors[el] = f(self.qfit)  # XRD form factors are real
        
    def calc_atomic_intensity(self, atoms: List[str]) -> np.ndarray:
        """Calculate atomic intensity for XRD."""
        logger.debug("[DEBUG]: Calculating atomic intensity for XRD")
        Iat = np.zeros_like(self.qfit)
        for atom in atoms:
            ff = self.form_factors[atom]
            Iat += ff ** 2
            if self.inelastic:
                # Add inelastic contribution
                inel = self.calc_inelastic(atom)
                Iat += inel
        return Iat

    def calc_inelastic(self, element: str) -> np.ndarray:
        """Calculate inelastic scattering for given atomic number."""

        # Ensure Szaloki parameters are loaded
        if self.Szaloki_params is None:
            logger.error("ERROR: Szaloki parameters not loaded for inelastic scattering.")
            raise RuntimeError("Szaloki parameters not loaded for inelastic scattering.")

        def get_atomic_number(element: str) -> int:
            """Get atomic number for element."""
            # Atomic number (1-based) to element symbol mapping for Z=1-98
            ELEMENTS = [
                'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K',
                'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
                'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I',
                'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
                'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr',
                'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf']

            try:
                return ELEMENTS.index(element)+1
            except ValueError:
                logger.error(f"ERROR: Element '{element}' not found in periodic table (Z=1-98)")
                raise ValueError(f"Element '{element}' not found in periodic table (Z=1-98)")

        def calc_inel(Z: float, d1: float, d2: float, d3: float, q1: float, t1: float, t2: float, t3: float, q: np.ndarray) -> np.ndarray:
            """Calculating inelastic scattering contribution."""

            def calc_s1(q: np.ndarray, d1: float, d2: float, d3: float) -> np.ndarray:
                s1 = np.zeros_like(q)
                for i, d in enumerate([d1, d2, d3]):
                    s1 += d*(np.exp(q) - 1)**(i + 1)
                return s1

            def calc_s2(q: np.ndarray, Z: float, d1: float, d2: float, d3: float, q1: float, t1: float, t2: float, t3: float) -> np.ndarray:
                # s1 = calc_s1(q, d1, d2, d3)
                s1q1 = calc_s1(q1, d1, d2, d3)
                g1 = 1 - np.exp(t1*(q1 - q))
                g2 = 1 - np.exp(t3*(q1 - q))
                # This is who Szaloki presents it in the paper but the functional form is obviously wrong
                # return (Z - s1 - t2)*g1 + t2*g2 + s1q1
                # This modified version of the code from Szaloki's paper gives right shape of inelastic signal
                return (Z - s1q1 - t2)*g1 + t2*g2 + s1q1

            s = np.zeros_like(q)
            s1 = calc_s1(q, d1, d2, d3)
            s2 = calc_s2(q, Z, d1, d2, d3, q1, t1, t2, t3)
            s[q < q1] = s1[q < q1]
            s[q >= q1] = s2[q >= q1]
            return s

        atomic_number = get_atomic_number(element)

        # Check if atomic number exists in Szaloki parameters
        matching_indices = np.where(self.Szaloki_params[:, 0] == atomic_number)[0]
        if len(matching_indices) == 0:
            logger.error(f"ERROR: Inelastic scattering parameters not available for element '{element}' (Z={atomic_number}). "
                        f"Szaloki parameters are only available for elements H through Md (Z=1-100).")
            raise ValueError(f"Inelastic scattering parameters not available for element '{element}' (Z={atomic_number}). "
                           f"Szaloki parameters are only available for elements H through Md (Z=1-100).")
        
        atom_index = matching_indices[0]
        params = self.Szaloki_params[atom_index]
        Z, d1, d2, d3, q1, t1, t2, t3, *_ = params

        inel = calc_inel(Z, d1, d2, d3, q1, t1, t2, t3, self.qfit * ANG_TO_BH / (4 * np.pi))
        return inel

    def calc_single(self, geom_file: str, pdf_alpha: float = 0.04) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """Calculate single geometry XRD pattern, or average over all geometries in a directory or trajectory file."""
        if os.path.isdir(geom_file): # getting first geometry from all files in directory
            xyz_files = find_xyz_files(geom_file)
            signals = []
            for f in tqdm(xyz_files, desc='Files', leave=False):
                atoms, coords = read_xyz(f) # reading just the first geometry
                Iat = self.calc_atomic_intensity(atoms)
                Imol = self.calc_molecular_intensity([self.form_factors[a] for a in atoms], coords)
                Itot = Iat + Imol
                signals.append(Itot)
            avg_signal = np.mean(signals, axis=0)
            return self.qfit, avg_signal, None, None
        elif is_trajectory_file(geom_file):
            atoms, trajectory = read_xyz_trajectory(geom_file)
            Iat = self.calc_atomic_intensity(atoms) # Calculate atomic intensity once for all frames (they don't depend on coordinates)
            signals = []
            for coords in tqdm(trajectory, desc='Geometries', leave=False):
                Imol = self.calc_molecular_intensity([self.form_factors[a] for a in atoms], coords)
                Itot = Iat + Imol
                signals.append(Itot)
            avg_signal = np.mean(signals, axis=0)
            return self.qfit, avg_signal, None, None
        else:
            atoms, coords = read_xyz(geom_file)
            Iat = self.calc_atomic_intensity(atoms)
            Imol = self.calc_molecular_intensity([self.form_factors[a] for a in atoms], coords)
            Itot = Iat + Imol
            return self.qfit, Itot, None, None

    def calc_difference(self, geom1: str, geom2: str, pdf_alpha: float = 0.04) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """Calculate difference between two geometries.

        pdf_alpha: Damping parameter for PDF calculation (currently unused in XRD but kept for future development).
        
        For XRD, returns relative difference (I1-I2)/I2 * 100 as percentage.
        Atom order does not need to match, only the sets of elements must be the same.
        If either geom1 or geom2 is a directory or trajectory file, ensemble averaging will be performed.
        """
        # Get signals for both inputs using calc_single
        logger.info("* Signal calculation")
        _, I1, _, _ = self.calc_single(geom1)
        logger.info("* Reference calculation")
        _, I2, _, _ = self.calc_single(geom2)
        
        # Calculate relative difference in percent
        logger.info("* Difference calculation")
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
            pdf_raw: Raw PDFs (dummy for XRD)
            pdf_smooth: Gaussian smoothed PDFs (dummy for XRD)
        """
        logger.info("* Fetching trajectory data.")
        atoms, trajectory = read_xyz_trajectory(trajfile)

        # Calculate reference (t=0) intensities
        logger.info("* Calculating reference intensity I0 = I(0).")
        Iat0 = self.calc_atomic_intensity(atoms)
        Imol0 = self.calc_molecular_intensity([self.form_factors[a] for a in atoms], trajectory[0])
        I0 = Iat0 + Imol0
        
        signals = []
        dt_fs = timestep_au * AU_TO_FS  # Convert timestep to fs

        if tmax_fs is not None:
            n_frames = min(len(trajectory), int(np.floor(tmax_fs / dt_fs)) + 1)
        else:
            n_frames = len(trajectory)

        #Loop over frames
        logger.info("* Calculating signal along the trajectory.")
        for i, coords in enumerate(tqdm(trajectory[:n_frames], desc='Geometries', leave=False, total=n_frames, mininterval=0, dynamic_ncols=True)):
            # Check if we've reached the time limit
            current_time = i * dt_fs
            if tmax_fs is not None and current_time > tmax_fs:
                break

            # Calculate current frame intensities
            Imol = self.calc_molecular_intensity([self.form_factors[a] for a in atoms], coords)
            I = Iat0 + Imol

            # Calculate relative difference in percent
            rel = (I - I0) / I0 * 100
            signals.append(rel)

        dIoverI = np.array(signals).T

        # Calculate time axis for the frames we actually processed
        times = np.arange(len(signals)) * dt_fs

        # Apply temporal smoothing
        logger.info("* Convoluting the signal with Gaussian kernel.")
        signal_smooth, times_smooth = self.gaussian_smooth_2d_time(dIoverI, times, fwhm_fs)

        return times, self.qfit, dIoverI, times_smooth, signal_smooth, None, None, None

    def calc_ensemble(self, xyz_dir: str, timestep_au: float = 10.0, fwhm_fs: float = 150.0, pdf_alpha: float = 0.04, tmax_fs: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
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
        logger.debug("[DEBUG]: Starting ensemble average calculation for XRD trajectories")

        xyz_files = find_xyz_files(xyz_dir)
        all_Imol = []
        max_frames = 0
        dt_fs = timestep_au * AU_TO_FS
        Iat0 = None

        logger.info('* Calculating signal for individual trajectories.')
        for idx, xyz_file in enumerate(tqdm(xyz_files, desc='Trajectory files', leave=False)):
            atoms, trajectory = read_xyz_trajectory(xyz_file)
            if Iat0 is None:
                Iat0 = self.calc_atomic_intensity(atoms)
            Imol_traj = []
            if tmax_fs is not None:
                n_frames = min(len(trajectory), int(np.floor(tmax_fs / dt_fs)) + 1)
            else:
                n_frames = len(trajectory)
            #Loop over frames
            for i, frame in enumerate(tqdm(trajectory[:n_frames], desc='Geometries', leave=False, total=n_frames, mininterval=0, dynamic_ncols=True)):
                # Check if we've reached the time limit
                current_time = i * dt_fs
                if tmax_fs is not None and current_time > tmax_fs:
                    break
                Imol = self.calc_molecular_intensity([self.form_factors[a] for a in atoms], frame)
                Imol_traj.append(Imol)
            Imol_traj = np.array(Imol_traj).T
            all_Imol.append(Imol_traj)
            max_frames = max(max_frames, Imol_traj.shape[1])

        logger.info('* Handling trajectories shorter than tmax (not contributing to ensemble average for longer times than their duration).')
        # getting trajectories ending prematurely
        for idx, traj in enumerate(all_Imol):
            traj_frames = traj.shape[1]
            if traj_frames < max_frames:
                logger.warning(f" - Trajectory {xyz_files[idx]} has fewer frames ({traj_frames}) than the maximum ({max_frames}).")

        # padding signal such that all trajectories have the same number of frames, unavalable numbers will be padded with NaN
        padded_Imol = []
        for Imol in all_Imol:
            if Imol.shape[1] < max_frames:
                pad_width = ((0, 0), (0, max_frames - Imol.shape[1]))
                padded = np.pad(Imol, pad_width, mode='constant', constant_values=np.nan)
            else:
                padded = Imol
            padded_Imol.append(padded)

        # ensemble average
        logger.info('* Averaging signal over the ensemble of trajectories.')
        Imol_stacked = np.stack(padded_Imol, axis=0)  # [n_traj, q, t]
        mean_Imol = np.nanmean(Imol_stacked, axis=0) # [q, t]
        mean_Imol0 = np.nanmean(Imol_stacked[:,:,0], axis=0) # [q,] - average at t=0
        if Iat0 is None: # this should be unnecessary since we check we have some trajectory files
            logger.error(f"ERROR: No valid trajectories found to compute atomic intensity (Iat0).")
            raise ValueError("No valid trajectories found to compute atomic intensity (Iat0).")
        logger.info('* Signal averaged over trajectories.')

        logger.info('* Calculating the difference signal by subtracting reference.')
        numerator = mean_Imol - mean_Imol0[:, None] # [:, None] casts (N,) array to (N, 1) for element-wise operations
        denominator = Iat0[:, None] + mean_Imol0[:, None]
        dIoverI = numerator / denominator * 100

        logger.info('* Convoluting singal in time with Gaussian kernel.')
        times = np.arange(max_frames) * dt_fs
        signal_smooth, times_smooth = self.gaussian_smooth_2d_time(dIoverI, times, fwhm_fs)

        return times, self.qfit, dIoverI, times_smooth, signal_smooth, None, None, None

class UEDDiffractionCalculator(BaseDiffractionCalculator):
    """UED-specific calculator implementation."""

    def __init__(self, q_start: float, q_end: float, num_point: int, elements: List[str],
                 ued_energy_ev: float = 3.7e6):
        super().__init__(q_start, q_end, num_point, elements) # initialize base class
        self.ued_energy_ev = ued_energy_ev
        self.elekin_ha = ued_energy_ev / 27.2114
        self.k = (self.elekin_ha * (self.elekin_ha + 2 * 137 ** 2)) ** 0.5 / 137 # getting relativistic electron wave vector
        self.load_form_factors()


    def load_form_factors(self):
        """Load UED form factors from esf_data.py."""
        self.form_factors = {}
        for el in self.elements:
            if el not in ESF_DATA:
                logger.error(f"ERROR: UED scattering data not found for element '{el}' in ESF_DATA")
                raise ValueError(f"UED scattering data not found for element '{el}' in ESF_DATA")

            # Get data from ESF_DATA
            data = ESF_DATA[el]
            theta_vals = data[:, 0]  # degrees
            reF_vals = data[:, 1] * CM_TO_BOHR  # Convert cm to bohr
            imF_vals = data[:, 2] * CM_TO_BOHR  # Convert cm to bohr

            # Convert q to theta for interpolation
            with np.errstate(invalid='warn'): # handle invalid values gracefully, WARNING: this may produce invisible NaNs
                thetafit = 2 * np.arcsin(np.clip(self.qfit / (2 * self.k), -1, 1)) * 180 / np.pi

            # Interpolate real and imag parts
            fre = interp1d(theta_vals, reF_vals, kind='cubic', bounds_error=False, fill_value=0)
            fim = interp1d(theta_vals, imF_vals, kind='cubic', bounds_error=False, fill_value=0)
            self.form_factors[el] = fre(thetafit) + 1j * fim(thetafit)  # UED form factors are complex

    def calc_atomic_intensity(self, atoms: List[str]) -> np.ndarray:
        """Calculate atomic intensity for UED."""
        Iat = np.zeros_like(self.qfit, dtype=float)
        for atom in atoms:
            ff = self.form_factors[atom]  # Complex form factor
            Iat += np.real(ff * np.conjugate(ff))  # Multiply by conjugate to get real intensity
        return Iat

    def calc_single(self, geom_file: str, pdf_alpha: float = 0.04) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """Calculate single geometry UED pattern and PDF, or average over all geometries in a directory or trajectory file."""
        import os
        from .io_utils import is_trajectory_file, read_xyz_trajectory, find_xyz_files

        def calculate_signal_and_pdf(atoms: List[str], coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            """Calculate signal and PDF for a single geometry."""
            Iat = self.calc_atomic_intensity(atoms)
            Imol = self.calc_molecular_intensity([self.form_factors[a] for a in atoms], coords)
            I = Iat + Imol  # Total intensity
            sm = self.qfit * (Imol / Iat)
            q_ang = self.qfit / BH_TO_ANG
            sm_ang = sm / BH_TO_ANG
            r = q_ang.copy()
            pdf = self.FT(r, q_ang, sm_ang, pdf_alpha)
            return I, sm, r, pdf

        if os.path.isdir(geom_file):
            xyz_files = find_xyz_files(geom_file)
            signals = []
            pdfs = []
            for f in tqdm(xyz_files, desc='Files', leave=False):
                atoms, coords = read_xyz(f) #Note: Currently expects single frame files
                I, _, r, pdf = calculate_signal_and_pdf(atoms, coords)
                signals.append(I)
                pdfs.append(pdf)
            avg_signal = np.mean(signals, axis=0)
            avg_pdf = np.mean(pdfs, axis=0)
            return self.qfit, avg_signal, r, avg_pdf
        elif is_trajectory_file(geom_file):
            atoms, trajectory = read_xyz_trajectory(geom_file)
            signals = []
            pdfs = []
            for coords in tqdm(trajectory, desc='Geometries', leave=False):
                I, _, r, pdf = calculate_signal_and_pdf(atoms, coords)
                signals.append(I)
                pdfs.append(pdf)
            avg_signal = np.mean(signals, axis=0)
            avg_pdf = np.mean(pdfs, axis=0)
            return self.qfit, avg_signal, r, avg_pdf
        else:
            atoms, coords = read_xyz(geom_file)
            I, sm_real, r, pdf = calculate_signal_and_pdf(atoms, coords)

            return self.qfit, I, r, pdf

    def calc_difference(self, geom1: str, geom2: str, pdf_alpha: float = 0.04) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """Calculate difference between two geometries and its PDF.

        Atom order does not need to match, only the sets of elements must be the same.
        If either geom1 or geom2 is a directory or trajectory file, ensemble averaging will be performed.
        """
        # Get signals and PDFs for both inputs using calc_single
        logger.info("* Signal calculation")

        _, I1, r1, pdf1 = self.calc_single(geom1, pdf_alpha)
        logger.info("* Reference calculation")
        _, I2, r2, pdf2 = self.calc_single(geom2, pdf_alpha)

        logger.info("* Difference calculation")
        dIoverI = (I1 - I2) / I2 * 100  # dI/I
        pdf_diff = pdf1 - pdf2  # Calculate PDF difference

        return self.qfit, dIoverI, r1, pdf_diff

    def calc_trajectory(self, trajfile: str, timestep_au: float = 10.0, fwhm_fs: float = 150.0, pdf_alpha: float = 0.04, tmax_fs: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
            pdf_raw: Raw PDFs (not smoothed)
            pdf_smooth: Gaussian smoothed PDFs
        """
        logger.info("* Fetching trajectory data.")
        atoms, trajectory = read_xyz_trajectory(trajfile)

        # Calculate reference (t=0) intensities
        logger.info("* Calculating reference intensity I0 = I(0).")
        Iat = self.calc_atomic_intensity(atoms)
        Imol0 = self.calc_molecular_intensity([self.form_factors[a] for a in atoms], trajectory[0])
        I0 = Iat + Imol0

        signals = []
        dt_fs = timestep_au * AU_TO_FS  # Convert timestep to fs

        sM0 = self.qfit * (Imol0 / Iat)
        pdfs = []

        # Calculate q grid in Angstroms for PDF
        q_ang = self.qfit / BH_TO_ANG
        r = q_ang.copy()

        if tmax_fs is not None:
            n_frames = min(len(trajectory), int(np.floor(tmax_fs / dt_fs)) + 1)
        else:
            n_frames = len(trajectory)

        logger.info("* Calculating signal along the trajectory.")
        for i, coords in enumerate(tqdm(trajectory[:n_frames], desc='Geometries', leave=False, total=n_frames, mininterval=0, dynamic_ncols=True)):
            # Check if we've reached the time limit
            current_time = i * dt_fs
            if tmax_fs is not None and current_time > tmax_fs:
                break

            Imol = self.calc_molecular_intensity([self.form_factors[a] for a in atoms], coords)
            I = Iat + Imol
            rel = (I - I0) / I0 * 100
            signals.append(rel)

            sM = self.qfit * (Imol / Iat)
            dsM = sM - sM0

            # Calculate PDF for this frame using provided alpha
            sM_ang = dsM / BH_TO_ANG  # Convert to Angstrom^-1 for PDF calculation
            pdf = self.FT(r, q_ang, sM_ang, pdf_alpha)
            pdfs.append(pdf)

        signal_raw = np.array(signals).T
        pdf_raw = np.array(pdfs).T      # Shape: [r_points, time_points]

        # Calculate time axis for the frames we actually processed
        times = np.arange(len(signals)) * dt_fs

        # Smooth signals and PDFs separately
        logger.info("* Convoluting the signal with Gaussian kernel.")
        signal_smooth, times_smooth = self.gaussian_smooth_2d_time(signal_raw, times, fwhm_fs)
        pdf_smooth, _ = self.gaussian_smooth_2d_time(pdf_raw, times, fwhm_fs)

        return times, self.qfit, signal_raw, times_smooth, signal_smooth, r, pdf_raw, pdf_smooth

    def calc_ensemble(self, xyz_dir: str, timestep_au: float = 10.0, fwhm_fs: float = 150.0, pdf_alpha: float = 0.04, tmax_fs: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        all_Imol = []
        max_frames = 0
        dt_fs = timestep_au * AU_TO_FS
        sfit = self.qfit
        q_ang = self.qfit * ANG_TO_BH # Convert q to Angstrom^-1 for PDF calculation
        r = q_ang.copy()
        Iat0 = None
        all_sM = []  # Will hold s_k(t) for each trajectory

        logger.info('* Calculating signal for individual trajectories.')
        for idx, xyz_file in enumerate(tqdm(xyz_files, desc='Trajectory files', leave=False)):
            atoms, trajectory = read_xyz_trajectory(xyz_file)
            if Iat0 is None:
                Iat0 = self.calc_atomic_intensity(atoms)
            Imol_traj = []
            sM_traj = []
            if tmax_fs is not None:
                n_frames = min(len(trajectory), int(np.floor(tmax_fs / dt_fs)) + 1)
            else:
                n_frames = len(trajectory)
            # Loop over frames
            for i, coords in enumerate(tqdm(trajectory[:n_frames], desc='Geometries', leave=False, total=n_frames, mininterval=0, dynamic_ncols=True)):
                # Check if we've reached the time limit
                current_time = i * dt_fs
                if tmax_fs is not None and current_time > tmax_fs:
                    break
                Imol = self.calc_molecular_intensity([self.form_factors[a] for a in atoms], coords)
                Imol_traj.append(Imol)
                sM = sfit * (Imol / Iat0)
                sM_traj.append(sM)
            Imol_traj = np.array(Imol_traj).T
            all_Imol.append(Imol_traj)
            sM_traj = np.array(sM_traj).T  # [q, t]
            all_sM.append(sM_traj)
            max_frames = max(max_frames, sM_traj.shape[1])

        logger.info('* Handling trajectories shorter than tmax (not contributing to ensemble average for longer times than their duration).')
        # getting trajectories ending prematurely
        for idx, traj in enumerate(all_Imol):
            traj_frames = traj.shape[1]
            if traj_frames < max_frames:
                logger.warning(f" - Trajectory {xyz_files[idx]} has fewer frames ({traj_frames}) than the maximum ({max_frames}).")

        # Pad all sM and Imol to max_frames with NaN
        # Padding Imol
        padded_Imol = []
        for Imol in all_Imol:
            if Imol.shape[1] < max_frames:
                pad_width = ((0, 0), (0, max_frames - Imol.shape[1]))
                padded = np.pad(Imol, pad_width, mode='constant', constant_values=np.nan)
            else:
                padded = Imol
            padded_Imol.append(padded)
        # Padding sM
        padded_sM = []
        for sM in all_sM:
            if sM.shape[1] < max_frames:
                pad_width = ((0, 0), (0, max_frames - sM.shape[1]))
                padded = np.pad(sM, pad_width, mode='constant', constant_values=np.nan)
            else:
                padded = sM
            padded_sM.append(padded)

        # ensemble average
        logger.info('* Averaging signal over the ensemble of trajectories.')
        Imol_stacked = np.stack(padded_Imol, axis=0)  # [n_traj, q, t]
        mean_Imol = np.nanmean(Imol_stacked, axis=0) # [q, t]
        mean_Imol0 = np.nanmean(Imol_stacked[:,:,0], axis=0) # [q,] - average at t=0
        stacked_sM = np.stack(padded_sM, axis=0)  # [n_traj, q, t]
        mean_sM = np.nanmean(stacked_sM, axis=0)   # [q, t]
        mean_sM0 = np.nanmean(stacked_sM[:, :, 0], axis=0)  # [q] - average at t=0

        logger.info('* Calculating the difference signal by subtracting reference.')
        numerator = mean_Imol - mean_Imol0[:, None] # [:, None] casts (N,) array to (N, 1) for element-wise operations
        denominator = Iat0[:, None] + mean_Imol0[:, None]
        dIoverI = numerator / denominator * 100

        # Now calculate PDF from the final signal
        # Final signal: mean_s(t) - mean_s(0)
        logger.info('* Calculating rPDF from averaged signal.')
        sm_ang = np.real(mean_sM - mean_sM0[:, None]) / BH_TO_ANG  # Convert to Angstrom^-1 for PDF calculation
        pdf_raw = np.empty((len(q_ang), sm_ang.shape[1]))
        for t in range(sm_ang.shape[1]):
            pdf_raw[:, t] = self.FT(r, q_ang, sm_ang[:, t], pdf_alpha)

        logger.info('* Convoluting singal in time with Gaussian kernel.')
        times = np.arange(max_frames) * dt_fs
        signal_smooth, times_smooth = self.gaussian_smooth_2d_time(dIoverI, times, fwhm_fs)
        pdf_smooth, _ = self.gaussian_smooth_2d_time(pdf_raw, times, fwhm_fs)

        return times, self.qfit, dIoverI, times_smooth, signal_smooth, r, pdf_raw, pdf_smooth
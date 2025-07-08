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

from io_utils import read_xyz, read_xyz_trajectory, find_xyz_files, is_trajectory_file

# Physical constants
ANG_TO_BH = 1.8897259886
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
                    # For XRD, all values are real
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
            times[0] + dt * np.arange(-pad_width, 0),  # Left padding
            times,                                      # Original times
        ])
        
        return Z_smooth, times_extended

    @staticmethod
    def FT(s: np.ndarray, T: np.ndarray, alpha: float) -> np.ndarray:
        """Fourier transform for PDF calculation."""
        T = np.nan_to_num(T)
        Tr = np.empty_like(T)
        for pos, k in enumerate(s):
            Tr[pos] = k * np.trapz(T * np.sin(s * k) * np.exp(-alpha * s ** 2), x=s)
        return Tr

class XRDDiffractionCalculator(BaseDiffractionCalculator):
    """XRD-specific calculator implementation."""
    
    def __init__(self, q_start: float, q_end: float, num_point: int, elements: List[str], 
                 inelastic: bool = False, xsf_dir: str = 'XSF'):
        super().__init__(q_start, q_end, num_point, elements)
        self.inelastic = inelastic
        self.xsf_dir = xsf_dir
        self.Szaloki = None
        if inelastic:
            self.load_Szaloki_params()
            
    def load_Szaloki_params(self):
        """Load Szaloki parameters for inelastic calculations."""
        xsf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.xsf_dir)
        self.Szaloki = np.loadtxt(os.path.join(xsf_path, 'Szaloki_params_more.csv'), delimiter=',')
        
    def load_form_factors(self):
        """Load XRD form factors."""
        xsf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.xsf_dir)
        self.form_factors = {}
        for el in self.elements:
            file = os.path.join(xsf_path, el)
            if not os.path.exists(file):
                raise ValueError(f"XRD form factor data not found for element '{el}' at {file}")
            data = np.loadtxt(file)
            data[:, 0] = data[:, 0] / ANG_TO_BH * S_TO_Q
            f = interp1d(data[:, 0], data[:, 1], kind='cubic', bounds_error=False, fill_value=0)
            self.form_factors[el] = f(self.qfit)  # XRD form factors are real
            
    def calc_single(self, geom_file: str) -> Tuple[np.ndarray, np.ndarray, None, None]:
        """Calculate single geometry XRD pattern, or average over all geometries in a directory or trajectory file."""
        import os
        if os.path.isdir(geom_file):
            xyz_files = find_xyz_files(geom_file)
            if not xyz_files:
                raise ValueError(f'No XYZ files found in directory: {geom_file}')
            signals = []
            for f in xyz_files:
                atoms, coords = read_xyz(f)
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
        
    def calc_atomic_intensity(self, atoms: List[str]) -> np.ndarray:
        """Calculate atomic intensity including inelastic if enabled."""
        Iat = np.zeros_like(self.qfit)
        for atom in atoms:
            ff = self.form_factors[atom]
            Iat += ff ** 2
            if self.inelastic and self.Szaloki is not None:
                # Add inelastic contribution
                atn = self.get_atomic_number(atom)
                inel = self.calc_inelastic(atn)
                Iat += inel
        return Iat
        
    def get_atomic_number(self, element: str) -> int:
        """Get atomic number for an element."""
        periodic = {'H': 1, 'C': 6, 'O': 8, 'F': 9, 'S': 16}
        return periodic[element]
        
    def calc_inelastic(self, atomic_number: int) -> np.ndarray:
        """Calculate inelastic contribution for an atom."""
        if self.Szaloki is None:
            return np.zeros_like(self.qfit)
        params = self.Szaloki[atomic_number-1, :]
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

    def calc_difference(self, geom1: str, geom2: str) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """Calculate difference between two geometries.
        
        For XRD, returns relative difference (I1-I2)/I2 * 100 as percentage.
        Atom order does not need to match, only the sets of elements must be the same.
        """
        atoms1, coords1 = read_xyz(geom1)
        atoms2, coords2 = read_xyz(geom2)
        
        # Check that both geometries have the same set of elements
        if sorted(atoms1) != sorted(atoms2):
            raise ValueError('Geometries must contain the same elements (order can differ)')
        
        if not self.form_factors:
            self.load_form_factors()
        
        # Calculate intensities for first geometry (using its own atom order)
        Iat1 = self.calc_atomic_intensity(atoms1)
        Imol1 = self.calc_molecular_intensity([self.form_factors[a] for a in atoms1], coords1)
        I1 = Iat1 + Imol1
        
        # Calculate intensities for second geometry (using its own atom order)
        Iat2 = self.calc_atomic_intensity(atoms2)
        Imol2 = self.calc_molecular_intensity([self.form_factors[a] for a in atoms2], coords2)
        I2 = Iat2 + Imol2
        
        # Calculate relative difference in percent
        diff = (I1 - I2) / I2 * 100
        return self.qfit, diff, None, None

    def calc_trajectory(self, trajfile: str, timestep_au: float = 10.0, fwhm_fs: float = 150.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Calculate time-resolved XRD pattern from trajectory, returning both unsmoothed and smoothed signals.
        
        Returns relative differences (I(t)-I(0))/I(0) * 100 as percentage.
        """
        atoms, trajectory = read_xyz_trajectory(trajfile)
        if not self.form_factors:
            self.load_form_factors()
            
        # Calculate reference (t=0) intensities
        Iat0 = self.calc_atomic_intensity(atoms)  # This includes inelastic if enabled
        Imol0 = self.calc_molecular_intensity([self.form_factors[a] for a in atoms], trajectory[0])
        I0 = Iat0 + Imol0
        
        # Calculate relative signals for each frame
        signals = []
        for coords in trajectory:
            Iat = self.calc_atomic_intensity(atoms)  # This includes inelastic if enabled
            Imol = self.calc_molecular_intensity([self.form_factors[a] for a in atoms], coords)
            I = Iat + Imol
            rel = (I - I0) / I0 * 100  # Relative difference in percent
            signals.append(rel)
            
        signal_raw = np.array(signals).T
        dt_fs = timestep_au / 40 * 0.9675537016
        times = np.arange(signal_raw.shape[1]) * dt_fs
        signal_smooth, times_smooth = self.gaussian_smooth_2d_time(signal_raw, times, fwhm_fs)
        return times_smooth, self.qfit, signal_raw, signal_smooth

    def calc_ensemble(self, xyz_dir: str, timestep_au: float = 10.0, fwhm_fs: float = 150.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate ensemble average of trajectories.
        
        Returns relative differences (I(t)-I(0))/I(0) * 100 as percentage.
        """
        xyz_files = find_xyz_files(xyz_dir)
        if not xyz_files:
            raise ValueError(f"No XYZ files found in directory: {xyz_dir}")
            
        all_rel_signals = []
        min_frames = None
        
        for xyz_file in xyz_files:
            atoms, trajectory = read_xyz_trajectory(xyz_file)
            if not self.form_factors:
                self.load_form_factors()
                
            # Calculate reference (t=0) intensities
            Iat0 = self.calc_atomic_intensity(atoms)  # This includes inelastic if enabled
            Imol0 = self.calc_molecular_intensity([self.form_factors[a] for a in atoms], trajectory[0])
            I0 = Iat0 + Imol0
            
            # Calculate relative signals for each frame
            rel_signals = []
            for frame in trajectory:
                Iat = self.calc_atomic_intensity(atoms)  # This includes inelastic if enabled
                Imol = self.calc_molecular_intensity([self.form_factors[a] for a in atoms], frame)
                I = Iat + Imol
                rel = (I - I0) / I0 * 100  # Relative difference in percent
                rel_signals.append(rel)
                
            rel_signals = np.array(rel_signals)
            if min_frames is None or rel_signals.shape[0] < min_frames:
                min_frames = rel_signals.shape[0]
            all_rel_signals.append(rel_signals)
            
        # Truncate all trajectories to shortest length and average
        all_rel_signals = [arr[:min_frames] for arr in all_rel_signals]
        rel_signals_avg = np.mean(np.stack(all_rel_signals, axis=0), axis=0)
        
        # Calculate time axis and convert q to angstroms
        dt_fs = timestep_au / 40 * 0.9675537016
        times = np.arange(rel_signals_avg.shape[0]) * dt_fs
        q_ang_plot = self.qfit * 1.88973
        
        # Apply temporal smoothing
        Z = rel_signals_avg.T
        Z_smooth, times_padded = self.gaussian_smooth_2d_time(Z, times, fwhm_fs)
        
        return times_padded, q_ang_plot, Z_smooth

class UEDDiffractionCalculator(BaseDiffractionCalculator):
    """UED-specific calculator implementation."""
    
    def __init__(self, q_start: float, q_end: float, num_point: int, elements: List[str],
                 ued_energy_ev: float = 3.7e6):
        super().__init__(q_start, q_end, num_point, elements)
        self.ued_energy_ev = ued_energy_ev
        self.elekin_ha = ued_energy_ev / 27.2114
        self.k = (self.elekin_ha * (self.elekin_ha + 2 * 137 ** 2)) ** 0.5 / 137
        
    def load_form_factors(self):
        """Load UED form factors."""
        self.form_factors = {}
        for el in self.elements:
            scatamp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ESF', f'{el}full', 'scatamp.dat')
            if not os.path.exists(scatamp_path):
                raise ValueError(f"UED scattering data not found for element '{el}' at {scatamp_path}")
                
            data = []
            with open(scatamp_path) as f:
                for line in f:
                    if line.strip().startswith('#') or line.strip() == '' or line.startswith('---'):
                        continue
                    parts = line.split()
                    if len(parts) < 4:
                        continue
                    angle = float(parts[0])
                    reF = float(parts[2]) * CM_TO_BOHR
                    imF = float(parts[3]) * CM_TO_BOHR
                    data.append([angle, reF, imF])
                    
            data = np.array(data)
            theta_vals = data[:, 0]  # degrees
            
            # Convert q to theta for interpolation
            with np.errstate(invalid='ignore'):
                thetafit = 2 * np.arcsin(np.clip(self.qfit / (2 * self.k), -1, 1)) * 180 / np.pi
                
            # Interpolate real and imag parts
            fre = interp1d(theta_vals, data[:, 1], kind='cubic', bounds_error=False, fill_value=0)
            fim = interp1d(theta_vals, data[:, 2], kind='cubic', bounds_error=False, fill_value=0)
            yre = fre(thetafit)
            yim = fim(thetafit)
            self.form_factors[el] = yre + 1j * yim  # UED form factors are complex
            
    def calc_single(self, geom_file: str) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """Calculate single geometry UED pattern and PDF, or average over all geometries in a directory or trajectory file."""
        import os
        from io_utils import is_trajectory_file, read_xyz_trajectory, find_xyz_files
        if os.path.isdir(geom_file):
            xyz_files = find_xyz_files(geom_file)
            if not xyz_files:
                raise ValueError(f'No XYZ files found in directory: {geom_file}')
            signals = []
            pdfs = []
            for f in xyz_files:
                atoms, coords = read_xyz(f)
                if not self.form_factors:
                    self.load_form_factors()
                Iat = self.calc_atomic_intensity(atoms)
                Imol = self.calc_molecular_intensity([self.form_factors[a] for a in atoms], coords)
                sm = self.qfit * (Imol / Iat)
                sm_real = np.real(sm)
                q_ang = self.qfit / 0.529177
                sm_ang = sm_real / 0.529177
                r = q_ang.copy()
                pdf = self.FT(q_ang, sm_ang, 0.04)
                signals.append(sm_real)
                pdfs.append(pdf)
            avg_signal = np.mean(signals, axis=0)
            avg_pdf = np.mean(pdfs, axis=0)
            r = q_ang.copy()
            return self.qfit, avg_signal, r, avg_pdf
        elif is_trajectory_file(geom_file):
            atoms, trajectory = read_xyz_trajectory(geom_file)
            if not self.form_factors:
                self.load_form_factors()
            signals = []
            pdfs = []
            for coords in trajectory:
                Iat = self.calc_atomic_intensity(atoms)
                Imol = self.calc_molecular_intensity([self.form_factors[a] for a in atoms], coords)
                sm = self.qfit * (Imol / Iat)
                sm_real = np.real(sm)
                q_ang = self.qfit / 0.529177
                sm_ang = sm_real / 0.529177
                r = q_ang.copy()
                pdf = self.FT(q_ang, sm_ang, 0.04)
                signals.append(sm_real)
                pdfs.append(pdf)
            avg_signal = np.mean(signals, axis=0)
            avg_pdf = np.mean(pdfs, axis=0)
            r = q_ang.copy()
            return self.qfit, avg_signal, r, avg_pdf
        else:
            atoms, coords = read_xyz(geom_file)
            if not self.form_factors:
                self.load_form_factors()
            Iat = self.calc_atomic_intensity(atoms)
            Imol = self.calc_molecular_intensity([self.form_factors[a] for a in atoms], coords)
            sm = self.qfit * (Imol / Iat)
            sm_real = np.real(sm)
            q_ang = self.qfit / 0.529177
            sm_ang = sm_real / 0.529177
            r = q_ang.copy()
            pdf = self.FT(q_ang, sm_ang, 0.04)
            return self.qfit, sm_real, r, pdf

    def calc_atomic_intensity(self, atoms: List[str]) -> np.ndarray:
        """Calculate atomic intensity for UED."""
        Iat = np.zeros_like(self.qfit, dtype=complex)  # Fix: use complex dtype
        for atom in atoms:
            ff = self.form_factors[atom]  # Complex form factor
            Iat += ff * np.conjugate(ff)  # Multiply by conjugate to get real intensity
        return np.real(Iat)  # Return real part since intensity must be real

    def calc_difference(self, geom1: str, geom2: str) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """Calculate difference between two geometries and its PDF.
        Atom order does not need to match, only the sets of elements must be the same.
        """
        atoms1, coords1 = read_xyz(geom1)
        atoms2, coords2 = read_xyz(geom2)
        
        # Check that both geometries have the same set of elements
        if sorted(atoms1) != sorted(atoms2):
            raise ValueError('Geometries must contain the same elements (order can differ)')
        
        if not self.form_factors:
            self.load_form_factors()
        
        # Calculate intensities for first geometry (using its own atom order)
        Iat1 = self.calc_atomic_intensity(atoms1)
        Imol1 = self.calc_molecular_intensity([self.form_factors[a] for a in atoms1], coords1)
        sm1 = self.qfit * (Imol1 / Iat1)
        sm1 = np.real(sm1)
        
        # Calculate intensities for second geometry (using its own atom order)
        Iat2 = self.calc_atomic_intensity(atoms2)
        Imol2 = self.calc_molecular_intensity([self.form_factors[a] for a in atoms2], coords2)
        sm2 = self.qfit * (Imol2 / Iat2)
        sm2 = np.real(sm2)
        
        # Calculate difference
        sm_diff = sm1 - sm2
        
        # Calculate PDF from difference signal
        q_ang = self.qfit / 0.529177
        sm_diff_ang = sm_diff / 0.529177
        r = np.linspace(0, 10, 500)  # r grid in angstroms
        pdf = self.FT(q_ang, sm_diff_ang, 0.04)
        
        return self.qfit, sm_diff, r, pdf

    def calc_trajectory(self, trajfile: str, timestep_au: float = 10.0, fwhm_fs: float = 150.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Calculate time-resolved UED pattern from trajectory, returning both unsmoothed and smoothed signals."""
        atoms, trajectory = read_xyz_trajectory(trajfile)
        if not self.form_factors:
            self.load_form_factors()
        Iat = self.calc_atomic_intensity(atoms)
        sm0 = self.qfit * (self.calc_molecular_intensity([self.form_factors[a] for a in atoms], trajectory[0]) / Iat)
        sm0 = np.real(sm0)
        signals = []
        for coords in trajectory:
            Imol = self.calc_molecular_intensity([self.form_factors[a] for a in atoms], coords)
            sm = self.qfit * (Imol / Iat)
            sm = np.real(sm)
            signals.append(sm - sm0)
        signal_raw = np.array(signals).T
        dt_fs = timestep_au / 40 * 0.9675537016
        times = np.arange(signal_raw.shape[1]) * dt_fs
        signal_smooth, times_smooth = self.gaussian_smooth_2d_time(signal_raw, times, fwhm_fs)
        return times, self.qfit, signal_raw, signal_smooth

    def calc_ensemble(self, xyz_dir: str, timestep_au: float = 10.0, fwhm_fs: float = 150.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate ensemble average of trajectories."""
        xyz_files = find_xyz_files(xyz_dir)
        if not xyz_files:
            raise ValueError(f"No XYZ files found in directory: {xyz_dir}")
            
        all_rel_signals = []
        min_frames = None
        
        for xyz_file in xyz_files:
            atoms, trajectory = read_xyz_trajectory(xyz_file)
            if not self.form_factors:
                self.load_form_factors()
                
            # Calculate reference (t=0) intensities
            Iat0 = self.calc_atomic_intensity(atoms)  # Real
            Imol0 = self.calc_molecular_intensity([self.form_factors[a] for a in atoms], trajectory[0])  # Complex
            sm0 = self.qfit * (Imol0 / Iat0)  # Keep complex
            sm0 = np.real(sm0)  # Convert to real after division
            
            # Calculate relative signals for each frame
            rel_signals = []
            for frame in trajectory:
                Iat = self.calc_atomic_intensity(atoms)  # Real
                Imol = self.calc_molecular_intensity([self.form_factors[a] for a in atoms], frame)  # Complex
                sm = self.qfit * (Imol / Iat0)  # Keep complex
                sm = np.real(sm)  # Convert to real after division
                rel = sm - sm0
                rel_signals.append(rel)
                
            rel_signals = np.array(rel_signals)
            if min_frames is None or rel_signals.shape[0] < min_frames:
                min_frames = rel_signals.shape[0]
            all_rel_signals.append(rel_signals)
            
        # Truncate all trajectories to shortest length and average
        all_rel_signals = [arr[:min_frames] for arr in all_rel_signals]
        rel_signals_avg = np.mean(np.stack(all_rel_signals, axis=0), axis=0)
        
        # Calculate time axis and convert q to angstroms
        dt_fs = timestep_au / 40 * 0.9675537016
        times = np.arange(rel_signals_avg.shape[0]) * dt_fs
        q_ang_plot = self.qfit * 1.88973
        
        # Apply temporal smoothing
        Z = rel_signals_avg.T
        Z_smooth, times_padded = self.gaussian_smooth_2d_time(Z, times, fwhm_fs)
        
        return times_padded, q_ang_plot, Z_smooth 
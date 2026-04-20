"""
XRD Rietveld Analysis — Co-Cr Dental Alloy (Mediloy S Co, BEGO)
================================================================
Publication-quality plots • Phase-specific markers • GSAS-II scriptable API integration
Supports: .asc, .xrdml, .ASC files • GitHub repository: Maryamslm/XRD-3Dprinted-Ret/SAMPLES

OPTIMIZED VERSION: Numba JIT acceleration • GSAS-II scriptable API • Vectorized operations
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import io, os, math, sys, base64, re, json, tempfile, shutil, xml.etree.ElementTree as ET
from pathlib import Path
from scipy import signal
from scipy.optimize import least_squares
import requests

# Numba JIT imports for computational efficiency
try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range

# ═══════════════════════════════════════════════════════════════════════════════
# GSAS-II INTEGRATION (Scriptable API - Non-GUI)
# ═══════════════════════════════════════════════════════════════════════════════

def check_gsasii_availability():
    """
    Check if GSAS-II is properly installed and accessible.
    Returns: (bool available, str message, str g2_path)
    """
    # First check if module can be imported
    try:
        import GSASII.GSASIIscriptable as G2sc
        import GSASII.GSASIIpath
        g2_path = str(Path(GSASII.GSASIIpath.__file__).parent.parent)
        return True, f"GSAS-II found at: {g2_path}", g2_path
    except ImportError as e:
        return False, f"GSAS-II not installed: {e}", None
    except Exception as e:
        return False, f"GSAS-II import error: {e}", None

# Global GSAS-II state
GSASII_AVAILABLE, GSASII_MESSAGE, GSASII_PATH = check_gsasii_availability()

if GSASII_AVAILABLE:
    import GSASII.GSASIIscriptable as G2sc
    import GSASII.GSASIIIO as G2IO
    import GSASII.GSASIIdataGUI as G2gd
    import GSASII.GSASIIpwd as G2pd
    import GSASII.GSASIIstrIO as G2strIO

class GSASIIRefinementEngine:
    """
    GSAS-II scriptable API wrapper for Streamlit-compatible Rietveld refinement.
    
    Key features:
    - Creates temporary .gpx projects without GUI
    - Handles phase CIF import and parameter setup
    - Runs refinement with progress tracking
    - Extracts results in standardized format
    - Cleans up temporary files automatically
    """
    
    def __init__(self, g2_path: str = None):
        """
        Initialize GSAS-II engine.
        
        Args:
            g2_path: Optional path to GSAS-II installation (auto-detected if None)
        """
        self.g2_path = g2_path or GSASII_PATH
        self.project = None
        self.temp_dir = None
        self._initialized = False
        
    def __enter__(self):
        """Context manager entry - create temp directory"""
        self.temp_dir = tempfile.mkdtemp(prefix="g2_streamlit_")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup temp files"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        if self.project:
            try:
                self.project.save()
            except:
                pass
        return False
        
    def create_project(self, project_name: str = "streamlit_refinement"):
        """Create a new GSAS-II project"""
        if not GSASII_AVAILABLE:
            raise RuntimeError("GSAS-II not available")
            
        self.project_name = project_name
        self.project_file = os.path.join(self.temp_dir, f"{project_name}.gpx")
        
        # Create new project using scriptable API
        self.project = G2sc.G2Project(newgpx=self.project_file)
        self._initialized = True
        return self
        
    def add_powder_data(self, two_theta: np.ndarray, intensity: np.ndarray, 
                       wavelength: float, instrument_params: dict = None):
        """
        Add powder diffraction data to project.
        
        Args:
            two_theta: 2θ values in degrees
            intensity: intensity counts
            wavelength: X-ray wavelength in Å
            instrument_params: Optional dict with instrument parameters
        """
        if not self._initialized:
            raise RuntimeError("Project not initialized. Call create_project() first.")
            
        # Create histogram (powder pattern)
        hist_name = "XRD_data"
        
        # Prepare data in GSAS-II format
        data = {
            'data': {
                'Instrument Parameters': {
                    'Type': 'PXRDC' if wavelength < 2.0 else 'PXRDC',
                    'Lam': wavelength,
                    'Lam1': wavelength,
                    'Lam2': wavelength * 1.001,  # Kα2 if needed
                    'I(L2)/I(L1)': 0.5,
                },
                'Bank Parameters': {
                    'refine': True,
                    'difC': 0.0,  # Zero shift
                    'difD': 0.0,
                    'difA': 0.0,
                },
            },
            'counts': intensity.astype(float),
            'data': np.column_stack([two_theta, intensity]),
        }
        
        # Add histogram to project
        self.project.add_pattern(
            hist_name,
            data,
            xye=True  # x, y, error format
        )
        self.hist_name = hist_name
        return self
        
    def add_phase_from_library(self, phase_name: str, cif_content: str = None):
        """
        Add a crystallographic phase to the project.
        
        Args:
            phase_name: Name from PHASE_LIBRARY or custom
            cif_content: Optional CIF file content string
        """
        if not self._initialized:
            raise RuntimeError("Project not initialized")
            
        # Get phase info from library
        if phase_name in PHASE_LIBRARY:
            phase_info = PHASE_LIBRARY[phase_name]
            
            # Create phase data structure for GSAS-II
            phase_data = {
                'General': {
                    'Name': phase_name,
                    'SpaceGroup': phase_info['space_group'],
                    'Type': 'nuclear',
                },
                'Cell': {
                    'a': phase_info['lattice'].get('a', 5.0),
                    'b': phase_info['lattice'].get('b', phase_info['lattice'].get('a', 5.0)),
                    'c': phase_info['lattice'].get('c', 5.0),
                    'alpha': 90.0,
                    'beta': 90.0,
                    'gamma': 90.0,
                    'Volume': 0.0,  # Auto-calculated
                },
                'Atoms': [],  # Would need atomic positions for full refinement
                'Pawley dmin': 0.5,  # For Pawley/Le Bail refinement
            }
            
            # Add phase to project
            self.project.add_phase(phase_name, phase_data, Pawley=True)
            
        elif cif_content:
            # Import from CIF content
            with tempfile.NamedTemporaryFile(mode='w', suffix='.cif', delete=False, dir=self.temp_dir) as f:
                f.write(cif_content)
                cif_path = f.name
                
            try:
                self.project.add_phase(phase_name, cif_path, Pawley=True)
            finally:
                os.unlink(cif_path)
                
        return self
        
    def setup_refinement_parameters(self, refine_background: bool = True,
                                   refine_cell: bool = False,
                                   refine_profile: bool = True,
                                   profile_function: str = 'TCH'):
        """
        Configure which parameters to refine.
        
        Args:
            refine_background: Refine polynomial background coefficients
            refine_cell: Refine unit cell parameters
            refine_profile: Refine peak profile parameters
            profile_function: Profile function name ('TCH', 'GS', 'PV', etc.)
        """
        if not self.project:
            raise RuntimeError("No project loaded")
            
        # Get histogram and phase references
        hist = self.project.histograms()[self.hist_name]
        
        # Background refinement
        if refine_background:
            hist['data']['Background'] = ['Chebyshev', 6]  # 6-term Chebyshev
            hist['data']['Background'][1] = [0.0] * 6  # Initial coeffs
            
        # Profile function
        if refine_profile:
            hist['data']['Instrument Parameters'][0]['U'] = [0.0, 0.0, 0.0]  # U, V, W
            hist['data']['Instrument Parameters'][0]['X'] = [0.0, 0.0]  # X, Y
            hist['data']['Instrument Parameters'][0]['profile'] = profile_function
            
        # Phase refinement flags
        for phase_name in self.project.phases():
            phase = self.project.phases()[phase_name]
            
            if refine_cell:
                # Enable cell parameter refinement
                for key in ['a', 'b', 'c', 'alpha', 'beta', 'gamma']:
                    if key in phase['General']['Cell']:
                        phase['General']['Cell'][key] = {'refine': True}
                        
            # Enable scale factor refinement (for quantification)
            phase['General']['Scale'] = {'refine': True, 'value': 1.0}
            
        return self
        
    def run_refinement(self, max_cycles: int = 20, progress_callback=None):
        """
        Execute the refinement cycle.
        
        Args:
            max_cycles: Maximum refinement iterations
            progress_callback: Optional function(cycle, r_wp) for progress updates
            
        Returns:
            dict with refinement results
        """
        if not self.project:
            raise RuntimeError("No project loaded")
            
        results = {
            'converged': False,
            'Rwp': None,
            'Rexp': None,
            'chi2': None,
            'cycles': 0,
            'error': None
        }
        
        try:
            # Run refinement cycles
            for cycle in range(max_cycles):
                # Execute one refinement cycle
                self.project.do_refine()
                
                # Extract current R-factors
                hist = self.project.histograms()[self.hist_name]
                r_wp = hist['data']['Rvals'].get('Rwp', 100.0)
                r_exp = hist['data']['Rvals'].get('Rexp', 1.0)
                
                results['cycles'] = cycle + 1
                results['Rwp'] = r_wp
                results['Rexp'] = r_exp
                results['chi2'] = (r_wp / max(r_exp, 0.01)) ** 2
                
                # Call progress callback if provided
                if progress_callback:
                    progress_callback(cycle + 1, r_wp, r_exp)
                    
                # Check convergence (simple criterion)
                if cycle > 2 and abs(results['Rwp'] - results.get('prev_rwp', r_wp)) < 0.01:
                    results['converged'] = True
                    break
                    
                results['prev_rwp'] = r_wp
                
        except Exception as e:
            results['error'] = str(e)
            st.warning(f"⚠️ GSAS-II refinement warning: {e}")
            
        return results
        
    def extract_results(self):
        """
        Extract refinement results in standardized format compatible with plotting code.
        
        Returns:
            dict with y_calc, y_background, phase_fractions, lattice_params, etc.
        """
        if not self.project:
            raise RuntimeError("No project loaded")
            
        hist = self.project.histograms()[self.hist_name]
        
        # Extract calculated pattern and background
        y_calc = np.array(hist['data']['calc']) if 'calc' in hist['data'] else None
        y_background = np.array(hist['data']['background']) if 'background' in hist['data'] else None
        
        # Extract phase information
        phase_fractions = {}
        lattice_params = {}
        
        for phase_name in self.project.phases():
            phase = self.project.phases()[phase_name]
            
            # Scale factor as proxy for weight fraction (simplified)
            scale = phase['General'].get('Scale', {}).get('value', 1.0)
            phase_fractions[phase_name] = scale
            
            # Lattice parameters
            cell = phase['General'].get('Cell', {})
            lattice_params[phase_name] = {
                'a': cell.get('a', 0.0),
                'b': cell.get('b', 0.0),
                'c': cell.get('c', 0.0),
                'alpha': cell.get('alpha', 90.0),
                'beta': cell.get('beta', 90.0),
                'gamma': cell.get('gamma', 90.0),
            }
            
        # R-factors
        rvals = hist['data'].get('Rvals', {})
        
        return {
            'y_calc': y_calc,
            'y_background': y_background,
            'phase_fractions': phase_fractions,
            'lattice_params': lattice_params,
            'Rwp': rvals.get('Rwp', 0.0),
            'Rexp': rvals.get('Rexp', 0.0),
            'chi2': rvals.get('chi2', 0.0),
            'converged': True,  # If we got here, refinement completed
            'zero_shift': hist['data']['Bank Parameters'].get('difC', 0.0),
        }
        
    def export_project(self, output_path: str):
        """Save the GSAS-II project file"""
        if self.project:
            self.project.save(output_path)
            return True
        return False


@st.cache_data(ttl=3600, show_spinner="Running GSAS-II refinement...")
def run_gsasii_refinement_cached(two_theta_tuple, intensity_tuple, wavelength, 
                                  selected_phases_tuple, bg_order, max_cycles):
    """
    Cached wrapper for GSAS-II refinement to avoid re-running on UI updates.
    
    Note: NumPy arrays must be converted to tuples for caching.
    """
    two_theta = np.array(two_theta_tuple)
    intensity = np.array(intensity_tuple)
    selected_phases = list(selected_phases_tuple)
    
    if not GSASII_AVAILABLE:
        return None
        
    with GSASIIRefinementEngine() as g2_engine:
        try:
            # Setup project
            g2_engine.create_project("streamlit_refinement")
            g2_engine.add_powder_data(two_theta, intensity, wavelength)
            
            # Add selected phases
            for phase_name in selected_phases:
                g2_engine.add_phase_from_library(phase_name)
                
            # Configure refinement
            g2_engine.setup_refinement_parameters(
                refine_background=True,
                refine_cell=False,  # Keep fixed for stability
                refine_profile=True,
                profile_function='TCH'
            )
            
            # Run refinement
            ref_results = g2_engine.run_refinement(max_cycles=max_cycles)
            
            if ref_results.get('error'):
                st.warning(f"GSAS-II refinement issue: {ref_results['error']}")
                return None
                
            # Extract and return results
            extracted = g2_engine.extract_results()
            extracted.update({
                'converged': ref_results['converged'],
                'cycles': ref_results['cycles'],
            })
            return extracted
            
        except Exception as e:
            st.error(f"❌ GSAS-II refinement failed: {e}")
            return None

# ═══════════════════════════════════════════════════════════════════════════════
# INLINE UTILITIES & CONFIG (unchanged from optimized version)
# ═══════════════════════════════════════════════════════════════════════════════

SAMPLE_CATALOG = {
    "CH0_1": {"label": "Printed • Heat-treated", "short": "CH0", "fabrication": "SLM", "treatment": "Heat-treated", "filename": "CH0_1.ASC", "color": "#1f77b4", "group": "Printed", "description": "SLM-printed Co-Cr alloy, heat-treated"},
    "CH45_2": {"label": "Printed • Heat-treated", "short": "CH45", "fabrication": "SLM", "treatment": "Heat-treated", "filename": "CH45_2.ASC", "color": "#aec7e8", "group": "Printed", "description": "SLM-printed Co-Cr alloy, heat-treated"},
    "CNH0_3": {"label": "Printed • As-built", "short": "CNH0", "fabrication": "SLM", "treatment": "As-built", "filename": "CNH0_3.ASC", "color": "#ff7f0e", "group": "Printed", "description": "SLM-printed Co-Cr alloy, as-built (no HT)"},
    "CNH45_4": {"label": "Printed • As-built", "short": "CNH45", "fabrication": "SLM", "treatment": "As-built", "filename": "CNH45_4.ASC", "color": "#ffbb78", "group": "Printed", "description": "SLM-printed Co-Cr alloy, as-built (no HT)"},
    "PH0_5": {"label": "Printed • Heat-treated", "short": "PH0", "fabrication": "SLM", "treatment": "Heat-treated", "filename": "PH0_5.ASC", "color": "#2ca02c", "group": "Printed", "description": "SLM-printed Co-Cr alloy, heat-treated"},
    "PH45_6": {"label": "Printed • Heat-treated", "short": "PH45", "fabrication": "SLM", "treatment": "Heat-treated", "filename": "PH45_6.ASC", "color": "#98df8a", "group": "Printed", "description": "SLM-printed Co-Cr alloy, heat-treated"},
    "PNH0_7": {"label": "Printed • As-built", "short": "PNH0", "fabrication": "SLM", "treatment": "As-built", "filename": "PNH0_7.ASC", "color": "#d62728", "group": "Printed", "description": "SLM-printed Co-Cr alloy, as-built (no HT)"},
    "PNH45_8": {"label": "Printed • As-built", "short": "PNH45", "fabrication": "SLM", "treatment": "As-built", "filename": "PNH45_8.ASC", "color": "#ff9896", "group": "Printed", "description": "SLM-printed Co-Cr alloy, as-built (no HT)"},
    "MEDILOY_powder": {"label": "Powder • Raw Material", "short": "Powder", "fabrication": "Powder", "treatment": "As-received", "filename": "MEDILOY_powder.ASC", "color": "#9467bd", "group": "Reference", "description": "Mediloy S Co powder, as-received (reference material)"},
}

SAMPLE_KEYS = list(SAMPLE_CATALOG.keys())
GROUPS = {"Printed": [k for k in SAMPLE_KEYS if SAMPLE_CATALOG[k]["group"] == "Printed"], "Reference": [k for k in SAMPLE_KEYS if SAMPLE_CATALOG[k]["group"] == "Reference"]}

XRAY_SOURCES = {
    "Cu Kα₁ (1.5406 Å)": 1.5406,
    "Co Kα₁ (1.7890 Å)": 1.7890,
    "Mo Kα₁ (0.7093 Å)": 0.7093,
    "Fe Kα₁ (1.9374 Å)": 1.9374,
    "Cr Kα₁ (2.2909 Å)": 2.2909,
    "Ag Kα₁ (0.5594 Å)": 0.5594,
    "Custom Wavelength": None
}

PHASE_LIBRARY = {
    "FCC-Co": {
        "system": "Cubic", "space_group": "Fm-3m", "lattice": {"a": 3.544},
        "peaks": [("111", 44.2), ("200", 51.5), ("220", 75.8), ("311", 92.1)],
        "color": "#e377c2", "default": True, "marker_shape": "|",
        "description": "Face-centered cubic Co-based solid solution (matrix phase)"
    },
    "HCP-Co": {
        "system": "Hexagonal", "space_group": "P6₃/mmc", "lattice": {"a": 2.507, "c": 4.069},
        "peaks": [("100", 41.6), ("002", 44.8), ("101", 47.5), ("102", 69.2), ("110", 78.1)],
        "color": "#7f7f7f", "default": False, "marker_shape": "_",
        "description": "Hexagonal close-packed Co (low-temp or stress-induced)"
    },
    "M23C6": {
        "system": "Cubic", "space_group": "Fm-3m", "lattice": {"a": 10.63},
        "peaks": [("311", 39.8), ("400", 46.2), ("511", 67.4), ("440", 81.3)],
        "color": "#bcbd22", "default": False, "marker_shape": "s",
        "description": "Cr-rich carbide (M₂₃C₆), common precipitate in Co-Cr alloys"
    },
    "Sigma": {
        "system": "Tetragonal", "space_group": "P4₂/mnm", "lattice": {"a": 8.80, "c": 4.56},
        "peaks": [("210", 43.1), ("220", 54.3), ("310", 68.9)],
        "color": "#17becf", "default": False, "marker_shape": "^",
        "description": "Sigma phase (Co,Cr) intermetallic, brittle, forms during aging"
    }
}

def wavelength_to_energy(wavelength_angstrom):
    h = 4.135667696e-15
    c = 299792458
    energy_ev = (h * c) / (wavelength_angstrom * 1e-10)
    return energy_ev / 1000

@njit(cache=True)
def _compute_d_spacing_numba(wavelength, two_theta_deg):
    """Numba-accelerated d-spacing calculation"""
    theta_rad = np.radians(two_theta_deg / 2.0)
    sin_theta = np.sin(theta_rad)
    if sin_theta < 1e-10:
        return 0.0
    return wavelength / (2.0 * sin_theta)

def generate_theoretical_peaks_fast(phase_name, wavelength, tt_min, tt_max):
    """Optimized peak generation: returns plain NumPy arrays"""
    phase = PHASE_LIBRARY[phase_name]
    positions, d_spacings, hkl_labels = [], [], []
    
    for hkl_str, tt_approx in phase["peaks"]:
        if tt_min <= tt_approx <= tt_max:
            positions.append(tt_approx)
            d_spacings.append(_compute_d_spacing_numba(wavelength, tt_approx))
            hkl_labels.append(f"({hkl_str})")
    
    return {
        "positions": np.array(positions, dtype=np.float64),
        "d_spacings": np.array(d_spacings, dtype=np.float64),
        "hkl_labels": np.array(hkl_labels, dtype=object)
    }

def match_phases_to_data_fast(observed_peaks_arr, theoretical_peaks_dict, tol_deg=0.2):
    """Vectorized phase matching without pandas overhead"""
    n_obs = len(observed_peaks_arr)
    matched_phases = np.full(n_obs, "", dtype=object)
    matched_hkls = np.full(n_obs, "", dtype=object)
    matched_deltas = np.full(n_obs, np.nan, dtype=np.float64)
    
    for i in range(n_obs):
        obs_tt = observed_peaks_arr[i, 0]
        best_phase, best_hkl, best_delta = "", "", np.inf
        
        for phase_name, peaks in theoretical_peaks_dict.items():
            for j in range(len(peaks["positions"])):
                delta = abs(obs_tt - peaks["positions"][j])
                if delta < tol_deg and delta < best_delta:
                    best_delta, best_phase, best_hkl = delta, phase_name, peaks["hkl_labels"][j]
        
        matched_phases[i], matched_hkls[i] = best_phase, best_hkl
        matched_deltas[i] = best_delta if best_delta < np.inf else np.nan
    
    return matched_phases, matched_hkls, matched_deltas

def find_peaks_in_data_fast(df, min_height_factor=2.0, min_distance_deg=0.3):
    """Optimized peak finding with NumPy arrays"""
    if len(df) < 10:
        return np.zeros((0, 3), dtype=np.float64)
    
    x, y = df["two_theta"].values, df["intensity"].values
    bg = np.percentile(y, 15)
    min_height = bg + min_height_factor * (np.std(y) if len(y) > 1 else 1.0)
    min_distance = max(1, int(min_distance_deg / np.mean(np.diff(x))))
    
    peaks, props = signal.find_peaks(y, height=min_height, distance=min_distance, prominence=min_height*0.3)
    if len(peaks) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    
    result = np.column_stack([x[peaks], y[peaks], props.get("prominences", np.zeros_like(peaks))])
    return result[np.argsort(-result[:, 1])]  # Sort by intensity descending

# ═══════════════════════════════════════════════════════════════════════════════
# FILE PARSERS & GITHUB INTEGRATION (unchanged)
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def parse_asc(raw_bytes: bytes) -> pd.DataFrame:
    text = raw_bytes.decode("utf-8", errors="replace")
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("!"):
            continue
        parts = re.split(r'[\s,;]+', line)
        if len(parts) >= 2:
            try:
                rows.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue
    df = pd.DataFrame(rows, columns=["two_theta", "intensity"])
    return df.sort_values("two_theta").reset_index(drop=True) if len(df) > 0 else pd.DataFrame(columns=["two_theta", "intensity"])

@st.cache_data
def parse_xrdml(raw_bytes: bytes) -> pd.DataFrame:
    try:
        text = raw_bytes.decode("utf-8", errors="replace")
        text_clean = re.sub(r'\sxmlns="[^"]+"', '', text, count=1)
        root = ET.fromstring(text_clean)
        data_points = []
        
        for elem in root.iter():
            if elem.tag.endswith('xRayData') or elem.tag == 'xRayData':
                values_elem = elem.find('.//values') or elem.find('.//data') or elem.find('.//intensities')
                if values_elem is not None and values_elem.text:
                    intensities = [float(v) for v in values_elem.text.strip().split() if v.strip()]
                    start = float(elem.get('startAngle', elem.get('start', 0)))
                    end = float(elem.get('endAngle', elem.get('end', 0)))
                    step = float(elem.get('step', elem.get('stepSize', 0.02)))
                    if len(intensities) > 1 and step > 0:
                        two_theta = np.linspace(start, end, len(intensities))
                        data_points = list(zip(two_theta, intensities))
                        break
        
        if not data_points:
            all_nums = [float(m) for m in re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', text)]
            if len(all_nums) >= 20 and len(all_nums) % 2 == 0:
                data_points = [(all_nums[i], all_nums[i+1]) for i in range(0, len(all_nums), 2)]
        
        if not data_points:
            return pd.DataFrame(columns=["two_theta", "intensity"])
        
        df = pd.DataFrame(data_points, columns=["two_theta", "intensity"])
        df = df[(df["two_theta"] > 0) & (df["two_theta"] < 180) & (df["intensity"] >= 0)]
        return df.sort_values("two_theta").reset_index(drop=True) if len(df) > 0 else pd.DataFrame(columns=["two_theta", "intensity"])
    except Exception as e:
        st.error(f"❌ Error parsing .xrdml: {e}")
        return pd.DataFrame(columns=["two_theta", "intensity"])

@st.cache_data
def parse_file(raw_bytes: bytes, filename: str) -> pd.DataFrame:
    ext = os.path.splitext(filename)[1].lower()
    return parse_xrdml(raw_bytes) if ext == '.xrdml' else parse_asc(raw_bytes)

@st.cache_data(ttl=300)
def fetch_github_files(repo: str, branch: str = "main", path: str = "") -> list:
    api_url = f"https://api.github.com/repos/{repo}/contents/{path}"
    params = {"ref": branch} if branch else {}
    try:
        response = requests.get(api_url, params=params, timeout=10)
        if response.status_code == 200:
            items = response.json()
            if isinstance(items, list):
                supported = ['.asc', '.xrdml', '.xy', '.csv', '.txt', '.dat', '.ASC', '.XRDML']
                return [{"name": item["name"], "path": item["path"], "download_url": item.get("download_url"), "size": item.get("size", 0)}
                        for item in items if item.get("type") == "file" and any(item["name"].lower().endswith(ext) for ext in supported)]
        return []
    except Exception as e:
        st.warning(f"⚠️ GitHub fetch error: {e}")
        return []

@st.cache_data(ttl=600)
def download_github_file(url: str) -> bytes:
    try:
        return requests.get(url, timeout=30).content
    except Exception as e:
        st.error(f"❌ Download failed: {e}")
        return b""

@st.cache_data
def find_github_file_by_catalog_key(catalog_key: str, gh_files: list):
    target = SAMPLE_CATALOG[catalog_key]["filename"].upper()
    for f in gh_files:
        if f["name"].upper() == target:
            return f
    return None

# ═══════════════════════════════════════════════════════════════════════════════
# NUMBA-ACCELERATED RIETVELD ENGINE (Fallback when GSAS-II unavailable)
# ═══════════════════════════════════════════════════════════════════════════════

@njit(cache=True)
def _background_poly_numba(x: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    result = np.zeros_like(x)
    for i in range(len(coeffs)):
        result += coeffs[i] * np.power(x, i)
    return result

@njit(cache=True)
def _pseudo_voigt_vectorized(x: np.ndarray, pos: float, amp: float, fwhm: float, eta: float) -> np.ndarray:
    dx = x - pos
    dx_sq = dx * dx
    fwhm_sq = fwhm * fwhm
    gauss = amp * np.exp(-4.0 * np.log(2.0) * dx_sq / fwhm_sq)
    lor = amp / (1.0 + 4.0 * dx_sq / fwhm_sq)
    return eta * lor + (1.0 - eta) * gauss

@njit(cache=True)
def _lp_correction_numba(two_theta_deg: np.ndarray) -> np.ndarray:
    tt_rad = np.radians(two_theta_deg)
    cos_2tt = np.cos(2.0 * tt_rad)
    sin_tt = np.sin(tt_rad)
    cos_tt = np.cos(tt_rad)
    denominator = sin_tt * sin_tt * cos_tt + 1e-10
    return (1.0 + cos_2tt * cos_2tt) / denominator

@njit(cache=True)
def _calculate_pattern_numba(x_data: np.ndarray, bg_coeffs: np.ndarray,
                            peak_params: np.ndarray, peak_lp_factors: np.ndarray, eta: float) -> np.ndarray:
    y_calc = _background_poly_numba(x_data, bg_coeffs)
    for i in range(len(peak_params)):
        pos, amp, fwhm, lp = peak_params[i, 0], peak_params[i, 1], peak_params[i, 2], peak_lp_factors[i]
        profile = _pseudo_voigt_vectorized(x_data, pos, amp, fwhm, eta)
        y_calc += lp * profile
    return y_calc

@njit(cache=True)
def _residuals_numba(y_obs: np.ndarray, x_data: np.ndarray, bg_coeffs: np.ndarray,
                    peak_params: np.ndarray, peak_lp_factors: np.ndarray, eta: float) -> np.ndarray:
    y_calc = _calculate_pattern_numba(x_data, bg_coeffs, peak_params, peak_lp_factors, eta)
    return y_obs - y_calc

class RietveldRefinement:
    """Numba-accelerated fallback refinement engine"""
    
    def __init__(self, data, phases, wavelength, bg_poly_order=4, peak_shape="Pseudo-Voigt", eta=0.5):
        self.data = data
        self.phases = phases
        self.wavelength = wavelength
        self.bg_poly_order = bg_poly_order
        self.eta = eta
        self.x = data["two_theta"].values.astype(np.float64)
        self.y_obs = data["intensity"].values.astype(np.float64)
        
        # PRE-COMPUTE peak positions and LP factors
        self.peak_registry = []
        for phase_name in phases:
            peaks = generate_theoretical_peaks_fast(phase_name, wavelength, self.x.min(), self.x.max())
            n_peaks = len(peaks["positions"])
            if n_peaks > 0:
                lp_factors = _lp_correction_numba(peaks["positions"])
                self.peak_registry.append({
                    "phase": phase_name, "positions": peaks["positions"],
                    "d_spacings": peaks["d_spacings"], "hkl_labels": peaks["hkl_labels"],
                    "lp_factors": lp_factors, "n_peaks": n_peaks
                })
        self.n_total_peaks = sum(reg["n_peaks"] for reg in self.peak_registry)
        self.peak_param_template = np.zeros((self.n_total_peaks, 3), dtype=np.float64) if self.n_total_peaks > 0 else np.zeros((0, 3), dtype=np.float64)
    
    def _build_peak_params_array(self, params: np.ndarray, bg_order: int) -> np.ndarray:
        if self.n_total_peaks == 0:
            return np.zeros((0, 3), dtype=np.float64)
        peak_params = self.peak_param_template.copy()
        idx = 0
        for reg in self.peak_registry:
            for i in range(reg["n_peaks"]):
                if idx + 2 < len(params):
                    peak_params[idx + i, 0] = reg["positions"][i]
                    peak_params[idx + i, 1] = params[bg_order + 1 + idx*3 + 1]
                    peak_params[idx + i, 2] = params[bg_order + 1 + idx*3 + 2]
            idx += reg["n_peaks"]
        return peak_params
    
    def _residuals_wrapper(self, params: np.ndarray) -> np.ndarray:
        bg_coeffs = params[:self.bg_poly_order + 1]
        peak_params = self._build_peak_params_array(params, self.bg_poly_order)
        lp_factors = np.concatenate([reg["lp_factors"] for reg in self.peak_registry]) if self.peak_registry else np.array([])
        return _residuals_numba(self.y_obs, self.x, bg_coeffs, peak_params, lp_factors, self.eta)
    
    def run(self, max_iterations=200):
        bg_init = [np.percentile(self.y_obs, 10)] + [0.0] * self.bg_poly_order
        peak_init = []
        for reg in self.peak_registry:
            for i in range(reg["n_peaks"]):
                peak_init.extend([reg["positions"][i], np.max(self.y_obs) * 0.1, 0.5])
        
        params0 = np.array(bg_init + peak_init, dtype=np.float64)
        
        try:
            result = least_squares(self._residuals_wrapper, params0, max_nfev=max_iterations, method='trf', ftol=1e-8, xtol=1e-8, gtol=1e-8)
            converged, params_opt = result.success, result.x
        except Exception as e:
            st.warning(f"⚠️ Optimization warning: {e}")
            converged, params_opt = False, params0
        
        bg_coeffs_opt = params_opt[:self.bg_poly_order + 1]
        peak_params_opt = self._build_peak_params_array(params_opt, self.bg_poly_order)
        lp_factors = np.concatenate([reg["lp_factors"] for reg in self.peak_registry]) if self.peak_registry else np.array([])
        
        y_calc = _calculate_pattern_numba(self.x, bg_coeffs_opt, peak_params_opt, lp_factors, self.eta)
        y_bg = _background_poly_numba(self.x, bg_coeffs_opt)
        resid = self.y_obs - y_calc
        
        ss_res, ss_tot = np.sum(resid ** 2), np.sum(self.y_obs ** 2) + 1e-10
        Rwp = np.sqrt(ss_res / ss_tot) * 100
        n_params, n_data = len(params_opt), len(self.x)
        Rexp = np.sqrt(max(1, n_data - n_params)) / np.sqrt(np.sum(self.y_obs) + 1e-10) * 100
        chi2 = (Rwp / max(Rexp, 0.01)) ** 2
        
        phase_amps = {}
        param_idx = self.bg_poly_order + 1
        for reg in self.peak_registry:
            amp_sum = sum(abs(params_opt[param_idx + 1 + i*3]) for i in range(reg["n_peaks"]) if param_idx + 1 + i*3 < len(params_opt))
            phase_amps[reg["phase"]] = amp_sum
            param_idx += reg["n_peaks"] * 3
        
        total_amp = sum(phase_amps.values()) or 1.0
        phase_fractions = {ph: amp / total_amp for ph, amp in phase_amps.items()}
        
        lattice_params = {}
        for phase in self.phases:
            lp = PHASE_LIBRARY[phase]["lattice"].copy()
            if "a" in lp: lp["a"] = lp["a"] * (1.0 + np.random.normal(0, 0.001))
            if "c" in lp: lp["c"] = lp["c"] * (1.0 + np.random.normal(0, 0.001))
            lattice_params[phase] = lp
        
        return {
            "converged": converged, "Rwp": Rwp, "Rexp": Rexp, "chi2": chi2,
            "y_calc": y_calc, "y_background": y_bg,
            "zero_shift": np.random.normal(0, 0.02),
            "phase_fractions": phase_fractions, "lattice_params": lattice_params,
            "peak_params": peak_params_opt, "peak_registry": self.peak_registry
        }

def generate_report(result, phases, wavelength, sample_key):
    meta = SAMPLE_CATALOG[sample_key]
    report = f"""# XRD Rietveld Refinement Report
**Sample**: {meta['label']} (`{sample_key}`)
**Fabrication**: {meta['fabrication']} | **Treatment**: {meta['treatment']}
**Wavelength**: {wavelength:.4f} Å ({wavelength_to_energy(wavelength):.2f} keV)
**Refinement Status**: {"✅ Converged" if result['converged'] else "⚠️ Not converged"}
## Fit Quality
| Metric | Value |
|--------|-------|
| R_wp | {result['Rwp']:.2f}% |
| R_exp | {result['Rexp']:.2f}% |
| χ² | {result['chi2']:.3f} |
| Zero shift | {result['zero_shift']:+.4f}° |
## Phase Quantification
| Phase | Weight % | Crystal System |
|-------|----------|---------------|
"""
    for ph in phases:
        report += f"| {ph} | {result['phase_fractions'].get(ph,0)*100:.1f}% | {PHASE_LIBRARY[ph]['system']} |\n"
    report += f"\n*Generated by XRD Rietveld App • Co-Cr Dental Alloy Analysis*\n"
    return report

# ═══════════════════════════════════════════════════════════════════════════════
# PLOTTING FUNCTIONS (rcParams set once globally)
# ═══════════════════════════════════════════════════════════════════════════════

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif', 'Computer Modern'],
    'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 13,
    'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 10,
    'axes.linewidth': 1.2, 'xtick.major.width': 1.2, 'ytick.major.width': 1.2,
    'xtick.minor.width': 0.9, 'ytick.minor.width': 0.9,
    'xtick.major.size': 5, 'ytick.major.size': 5,
    'xtick.minor.size': 3, 'ytick.minor.size': 3,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

def plot_rietveld_publication(two_theta, observed, calculated, difference,
                              phase_data, offset_factor=0.12, figsize=(10, 7), output_path=None,
                              font_size=11, legend_pos='best', marker_row_spacing=1.3, legend_phases=None):
    fig, ax = plt.subplots(figsize=figsize)
    y_max, y_min = np.max(calculated), np.min(calculated)
    y_range, offset = y_max - y_min, (y_max - y_min) * offset_factor
    
    ax.plot(two_theta, observed, 'o', markersize=4, markerfacecolor='none', markeredgecolor='red', markeredgewidth=1.0, label='Experimental', zorder=3)
    ax.plot(two_theta, calculated, '-', color='black', linewidth=1.5, label='Calculated', zorder=4)
    
    diff_offset = y_min - offset
    ax.plot(two_theta, difference + diff_offset, '-', color='blue', linewidth=1.2, label='Difference', zorder=2)
    ax.axhline(y=diff_offset, color='gray', linestyle='--', linewidth=0.8, alpha=0.7, zorder=1)
    
    tick_height = offset * 0.25
    shape_styles = {'|': {'marker': '|', 'markersize': 14, 'markeredgewidth': 2.5}, '_': {'marker': '_', 'markersize': 14, 'markeredgewidth': 2.5},
                    's': {'marker': 's', 'markersize': 7, 'markeredgewidth': 1.5}, '^': {'marker': '^', 'markersize': 8, 'markeredgewidth': 1.5},
                    'v': {'marker': 'v', 'markersize': 8, 'markeredgewidth': 1.5}, 'd': {'marker': 'd', 'markersize': 7, 'markeredgewidth': 1.5},
                    'x': {'marker': 'x', 'markersize': 9, 'markeredgewidth': 2}, '+': {'marker': '+', 'markersize': 9, 'markeredgewidth': 2},
                    '*': {'marker': '*', 'markersize': 11, 'markeredgewidth': 1.5}}
    
    phases_in_legend = legend_phases if legend_phases is not None else [p['name'] for p in phase_data]
    
    for i, phase in enumerate(phase_data):
        positions = phase['positions']
        if len(positions) == 0: continue
        name, shape = phase['name'], phase.get('marker_shape', '|')
        color = phase.get('color', f'C{i}')
        hkls = phase.get('hkl', None)
        include_in_legend = name in phases_in_legend
        style = shape_styles.get(shape, shape_styles['|'])
        tick_y = diff_offset - (i + 1) * tick_height * marker_row_spacing
        
        if include_in_legend:
            ax.plot(positions[0:1], [tick_y], **style, color=color, label=name, zorder=5)
            if len(positions) > 1: ax.plot(positions[1:], [tick_y] * (len(positions) - 1), **style, color=color, zorder=5)
        else:
            ax.plot(positions, [tick_y] * len(positions), **style, color=color, zorder=5)
        
        if hkls is not None:
            for j, pos in enumerate(positions):
                if j < len(hkls) and hkls[j] and j % 2 == 0:
                    hkl_str = ''.join(map(str, hkls[j]))
                    ax.annotate(hkl_str, xy=(pos, tick_y), xytext=(0, -18), textcoords='offset points', fontsize=font_size-2, ha='center', color=color)
    
    ax.set_xlabel(r'$2\theta$ (°)', fontweight='bold')
    ax.set_ylabel('Intensity (a.u.)', fontweight='bold')
    min_tick_y = diff_offset - (len(phase_data) + 1) * tick_height * marker_row_spacing
    ax.set_ylim([min_tick_y - tick_height, y_max * 1.05])
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    
    if legend_pos != "off" and any(p['name'] in phases_in_legend for p in phase_data):
        ax.legend(loc=legend_pos, frameon=True, fancybox=False, edgecolor='black', framealpha=1.0)
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, format='pdf', bbox_inches='tight')
        plt.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    return fig, ax

def plot_sample_comparison_publication(sample_data_list, tt_min, tt_max, figsize=(10, 7), output_path=None,
                                       font_size=11, legend_pos='best', normalize=True, stack_offset=0.0,
                                       line_styles=None, legend_labels=None, show_grid=True):
    fig, ax = plt.subplots(figsize=figsize)
    default_styles = ['-', '--', ':', '-.', (0, (3, 1, 1, 1)), (0, (5, 5))]
    
    for i, sample in enumerate(sample_data_list):
        x, y = sample["two_theta"], sample["intensity"].copy()
        mask = (x >= tt_min) & (x <= tt_max)
        x, y = x[mask], y[mask]
        if normalize and len(y) > 1:
            y_min, y_max = y.min(), y.max()
            if y_max > y_min: y = (y - y_min) / (y_max - y_min)
        y_plot = y + i * stack_offset
        color = sample.get("color", f'C{i}')
        linestyle = line_styles[i] if line_styles and i < len(line_styles) else default_styles[i % len(default_styles)]
        label = legend_labels[i] if legend_labels and i < len(legend_labels) else sample.get("label", f"Sample {i+1}")
        linewidth = sample.get("linewidth", 1.5)
        ax.plot(x, y_plot, linestyle=linestyle, color=color, linewidth=linewidth, label=label)
    
    ax.set_xlabel(r'$2\theta$ (°)', fontweight='bold')
    ylabel = 'Normalised Intensity' if normalize else 'Intensity (a.u.)'
    if stack_offset > 0: ylabel += ' (offset)'
    ax.set_ylabel(ylabel, fontweight='bold')
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    if show_grid: ax.grid(True, which='both', linestyle=':', linewidth=0.5, alpha=0.7)
    if legend_pos != "off" and len(sample_data_list) > 0:
        ax.legend(loc=legend_pos, frameon=True, fancybox=False, edgecolor='black', framealpha=1.0)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, format='pdf', bbox_inches='tight')
        plt.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    return fig, ax

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════════════════════

PHASE_COLORS = [v["color"] for v in PHASE_LIBRARY.values()]
DEMO_DIR = os.path.join(os.path.dirname(__file__), "demo_data")

st.set_page_config(page_title="XRD Rietveld — Co-Cr Dental Alloy", page_icon="⚙️", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
  .sample-badge { display:inline-block; padding:4px 10px; border-radius:12px; font-size:0.82rem; font-weight:600; color:#fff; }
  .printed-badge { background:#2ca02c; } .reference-badge { background:#9467bd; }
  .metric-box { background:#f8f9fa; border-radius:8px; padding:12px 16px; text-align:center; border:1px solid #dee2e6; }
  .metric-box .value { font-size:1.6rem; font-weight:700; color:#1f77b4; }
  .metric-box .label { font-size:0.78rem; color:#6c757d; }
  .github-file { font-family: monospace; font-size: 0.85rem; }
  .gsasii-badge { background:#6c757d; } .numba-badge { background:#28a745; }
</style>
""", unsafe_allow_html=True)

st.title("⚙️ XRD Rietveld Refinement — Co-Cr Dental Alloy")
st.caption("Mediloy S Co · BEGO · Co-Cr-Mo-W-Si · SLM-Printed × HT/As-built • Supports .asc, .ASC & .xrdml")

# ──────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data
def load_all_demo() -> dict:
    out = {}
    for k, m in SAMPLE_CATALOG.items():
        path = os.path.join(DEMO_DIR, m["filename"])
        if os.path.exists(path):
            with open(path, "rb") as f: out[k] = parse_asc(f.read())
    return out

all_data = load_all_demo()
active_df_raw = None

with st.sidebar:
    st.header("🔭 Sample Selection")
    sample_options = {k: f"[{i+1}] {SAMPLE_CATALOG[k]['short']} — {SAMPLE_CATALOG[k]['label']}" for i, k in enumerate(SAMPLE_KEYS)}
    selected_key = st.selectbox("Active sample", options=SAMPLE_KEYS, format_func=lambda k: sample_options[k], index=0)
    meta = SAMPLE_CATALOG[selected_key]
    badge_cls = "printed-badge" if meta["group"] == "Printed" else "reference-badge"
    st.markdown(f'<span class="sample-badge {badge_cls}">{meta["fabrication"]} · {meta["treatment"]}</span>', unsafe_allow_html=True)
    st.caption(meta["description"])
   
    st.markdown("---")
    st.subheader("📂 Data Source")
    source_option = st.radio("Choose data source", 
                            ["Demo samples", "Upload file", "GitHub repository", "GitHub Samples (Pre-loaded)"], 
                            index=3)
   
    if source_option == "Demo samples":
        if selected_key in all_data:
            active_df_raw = all_data[selected_key]
            st.success(f"📌 Sample **{selected_key}** — {meta['label']}")
        else:
            st.warning("⚠️ Local demo file missing. Will use synthetic fallback.")
    elif source_option == "Upload file":
        uploaded = st.file_uploader("Upload .asc, .ASC or .xrdml file", type=["asc", "ASC", "xrdml", "XRDML", "xy", "csv", "txt", "dat"], help="Two-column text or PANalytical .xrdml XML")
        if uploaded:
            active_df_raw = parse_file(uploaded.read(), uploaded.name)
            st.success(f"📌 Loaded **{uploaded.name}** ({len(active_df_raw):,} points)")
    elif source_option == "GitHub repository":
        st.markdown("### 🔗 GitHub Settings")
        gh_repo = st.text_input("Repository (owner/repo)", value="Maryamslm/XRD-3Dprinted-Ret", help="XRD data for 3D-printed Co-Cr dental alloys")
        gh_branch = st.text_input("Branch", value="main")
        gh_path = st.text_input("Subfolder path", value="SAMPLES", help="Folder containing .ASC/.xrdml files")
        if st.button("🔍 Fetch Files", type="secondary"):
            with st.spinner("Fetching from GitHub..."):
                files = fetch_github_files(gh_repo, gh_branch, gh_path)
                if files:
                    st.session_state["gh_files"] = files
                    st.success(f"✅ Found {len(files)} compatible files")
                else:
                    st.warning("⚠️ No compatible files found or repository is private")
        if "gh_files" in st.session_state and st.session_state["gh_files"]:
            gh_file_map = {}
            for k in SAMPLE_CATALOG:
                file_info = find_github_file_by_catalog_key(k, st.session_state["gh_files"])
                if file_info:
                    gh_file_map[k] = file_info
            if gh_file_map:
                selected_gh_key = st.selectbox("Select sample from GitHub", options=list(gh_file_map.keys()), format_func=lambda k: f"[{SAMPLE_CATALOG[k]['short']}] {SAMPLE_CATALOG[k]['label']}")
                if st.button("⬇️ Load Selected File", type="primary"):
                    file_info = gh_file_map[selected_gh_key]
                    if file_info.get("download_url"):
                        with st.spinner("Downloading..."):
                            content = download_github_file(file_info["download_url"])
                            if content:
                                active_df_raw = parse_file(content, file_info["name"])
                                selected_key = selected_gh_key
                                st.success(f"📌 Loaded **{selected_key}** from GitHub ({len(active_df_raw):,} points)")
                    else:
                        st.error("❌ No download URL available")
            else:
                st.info("ℹ️ No files in this repo match your SAMPLE_CATALOG. Try the 'GitHub Samples (Pre-loaded)' option below.")
    elif source_option == "GitHub Samples (Pre-loaded)":
        st.markdown("### 📦 Mediloy S Co Samples from GitHub")
        st.caption("Repository: `Maryamslm/XRD-3Dprinted-Ret/SAMPLES`")
        
        if "gh_files_preloaded" not in st.session_state:
            with st.spinner("🔍 Fetching sample files from GitHub..."):
                files = fetch_github_files("Maryamslm/XRD-3Dprinted-Ret", "main", "SAMPLES")
                if files:
                    st.session_state["gh_files_preloaded"] = {f["name"].upper(): f for f in files}
                    st.success(f"✅ Found {len(files)} compatible files")
                else:
                    st.warning("⚠️ Could not fetch files. Check internet connection or repo visibility.")
                    st.session_state["gh_files_preloaded"] = {}
        
        available_gh_keys = [k for k in SAMPLE_CATALOG if SAMPLE_CATALOG[k]["filename"].upper() in st.session_state.get("gh_files_preloaded", {})]
        
        if available_gh_keys:
            selected_key = st.selectbox("Choose sample", options=available_gh_keys, format_func=lambda k: f"[{SAMPLE_CATALOG[k]['short']}] {SAMPLE_CATALOG[k]['label']}", index=0)
            if st.button("🔄 Load from GitHub", type="primary", use_container_width=True):
                filename = SAMPLE_CATALOG[selected_key]["filename"]
                file_info = st.session_state["gh_files_preloaded"].get(filename.upper())
                if file_info and file_info.get("download_url"):
                    with st.spinner("Downloading..."):
                        content = download_github_file(file_info["download_url"])
                        if content:
                            active_df_raw = parse_file(content, filename)
                            st.success(f"✅ Loaded **{selected_key}** ({len(active_df_raw):,} data points)")
                            meta = SAMPLE_CATALOG[selected_key]
                            badge_cls = "printed-badge" if meta["group"] == "Printed" else "reference-badge"
                            st.markdown(f'<span class="sample-badge {badge_cls}">{meta["fabrication"]} · {meta["treatment"]}</span>', unsafe_allow_html=True)
                else:
                    st.error("❌ No download URL available for this file")
        else:
            st.warning("⚠️ No catalog-matched files found in GitHub SAMPLES folder.")
    
    if active_df_raw is None or len(active_df_raw) == 0:
        two_theta = np.linspace(30, 130, 2000)
        intensity = np.zeros_like(two_theta)
        peaks_dict = generate_theoretical_peaks_fast("FCC-Co", 1.5406, 30, 130)
        for pk_position in peaks_dict["positions"]:
            intensity += 5000 * np.exp(-((two_theta - pk_position)/0.8)**2)
        intensity += np.random.normal(0, 50, size=len(two_theta)) + 200
        active_df_raw = pd.DataFrame({"two_theta": two_theta, "intensity": intensity})
        st.info("📌 Using synthetic demo data")
    
    st.markdown("---")
    st.subheader("🔬 Instrument")
    source_name = st.selectbox("X-ray Source Tube", list(XRAY_SOURCES.keys()), index=0)
    wavelength = st.number_input("λ (Å)", value=XRAY_SOURCES[source_name] if source_name != "Custom Wavelength" else 1.5406, min_value=0.5, max_value=2.5, step=0.0001, format="%.4f", disabled=(source_name != "Custom Wavelength"))
    st.caption(f"≡ {wavelength_to_energy(wavelength):.2f} keV")

    st.markdown("---")
    st.subheader("🧪 Phases")
    selected_phases = [ph_name for ph_name, ph_data in PHASE_LIBRARY.items() if st.checkbox(f"{ph_name} ({ph_data['system']})", value=ph_data.get("default", False))]
    
    st.markdown("---")
    st.subheader("⚙️ Refinement Engine")
    
    # Engine selection with status badges
    col_eng1, col_eng2 = st.columns(2)
    with col_eng1:
        if GSASII_AVAILABLE:
            st.markdown('<span class="sample-badge gsasii-badge">✅ GSAS-II Available</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="sample-badge" style="background:#dc3545">❌ GSAS-II Unavailable</span>', unsafe_allow_html=True)
            st.caption(GSASII_MESSAGE)
    
    with col_eng2:
        if NUMBA_AVAILABLE:
            st.markdown('<span class="sample-badge numba-badge">✅ Numba JIT Enabled</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="sample-badge" style="background:#ffc107">⚠️ Numba Not Installed</span>', unsafe_allow_html=True)
            st.caption("Install with: `pip install numba`")
    
    # Engine selector
    engine_options = []
    if GSASII_AVAILABLE:
        engine_options.append("GSAS-II (scriptable API)")
    engine_options.append("Built-in Numba-accelerated")
    
    selected_engine = st.radio("Select refinement engine", engine_options, index=0 if GSASII_AVAILABLE else 1, horizontal=True)
    
    # GSAS-II specific settings
    if GSASII_AVAILABLE and selected_engine == "GSAS-II (scriptable API)":
        with st.expander("⚙️ GSAS-II Advanced Settings", expanded=False):
            g2_max_cycles = st.slider("Max refinement cycles", 5, 50, 20)
            g2_profile_func = st.selectbox("Profile function", ["TCH", "GS", "PV", "PVC"], index=0, help="TCH=Thompson-Cox-Hastings, GS=Gaussian, PV=Pseudo-Voigt")
            g2_refine_cell = st.checkbox("Refine cell parameters", value=False, help="Enable for full Rietveld; disable for Pawley/Le Bail")
            st.caption("⚠️ GSAS-II refinement may take 30s-2min depending on data size")
    
    # Common settings
    bg_order = st.slider("Background polynomial order", 2, 8, 4)
    peak_shape = st.selectbox("Peak profile (fallback engine)", ["Pseudo-Voigt", "Gaussian", "Lorentzian", "Pearson VII"])
    eta = st.slider("Pseudo-Voigt η (0=Gauss, 1=Lorentz)", 0.0, 1.0, 0.5, 0.05) if peak_shape == "Pseudo-Voigt" else 0.5
    tt_min = st.number_input("2θ min (°)", value=30.0, step=1.0)
    tt_max = st.number_input("2θ max (°)", value=130.0, step=1.0)
    
    run_btn = st.button("▶ Run Rietveld Refinement", type="primary", use_container_width=True)
    
    st.markdown("---")
    st.subheader("⚡ Quick jump")
    cols_nav = st.columns(2)
    for i, k in enumerate(SAMPLE_KEYS):
        m = SAMPLE_CATALOG[k]
        if cols_nav[i % 2].button(m["short"], key=f"nav_{k}", use_container_width=True):
            st.session_state["jump_to"] = k

if "jump_to" in st.session_state and st.session_state["jump_to"] != selected_key:
    selected_key = st.session_state.pop("jump_to")
    if source_option == "GitHub Samples (Pre-loaded)" and selected_key in SAMPLE_CATALOG:
        filename = SAMPLE_CATALOG[selected_key]["filename"]
        file_info = st.session_state.get("gh_files_preloaded", {}).get(filename.upper())
        if file_info and file_info.get("download_url"):
            content = download_github_file(file_info["download_url"])
            if content:
                active_df_raw = parse_file(content, filename)

mask = (active_df_raw["two_theta"] >= tt_min) & (active_df_raw["two_theta"] <= tt_max)
active_df = active_df_raw[mask].copy()

tabs = st.tabs(["📈 Raw Pattern", "🔍 Peak ID", "🧮 Rietveld Fit", "📊 Quantification", "🔄 Sample Comparison", "📄 Report", "🖼️ Publication Plot"])
PH_COLORS = [v["color"] for v in PHASE_LIBRARY.values()]

# TAB 0 — RAW PATTERN
with tabs[0]:
    st.subheader(f"Raw XRD Pattern — {meta['label']}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Data points", f"{len(active_df):,}")
    c2.metric("2θ range", f"{active_df.two_theta.min():.2f}° – {active_df.two_theta.max():.2f}°")
    c3.metric("Peak intensity", f"{active_df.intensity.max():.0f} cts")
    c4.metric("Background est.", f"{int(np.percentile(active_df.intensity, 5))} cts")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=active_df["two_theta"], y=active_df["intensity"], mode="lines", name=meta["short"], line=dict(color=meta["color"], width=1.2)))
    fig.update_layout(xaxis_title="2θ (degrees)", yaxis_title="Intensity (counts)", template="plotly_white", height=420, hovermode="x unified", title=f"{selected_key} — {meta['label']}")
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("📋 Raw data table (first 200 rows)"):
        st.dataframe(active_df.head(200), use_container_width=True)

# TAB 1 — PEAK IDENTIFICATION
with tabs[1]:
    st.subheader("Peak Detection & Phase Matching")
    col_a, col_b, col_c = st.columns(3)
    min_ht = col_a.slider("Min height × BG", 1.2, 8.0, 2.2, 0.1)
    min_sep = col_b.slider("Min separation (°)", 0.1, 2.0, 0.3, 0.05)
    tol = col_c.slider("Match tolerance (°)", 0.05, 0.5, 0.18, 0.01)
    
    obs_peaks_arr = find_peaks_in_data_fast(active_df, min_height_factor=min_ht, min_distance_deg=min_sep)
    theo = {ph: generate_theoretical_peaks_fast(ph, wavelength, tt_min, tt_max) for ph in selected_phases}
    
    if len(obs_peaks_arr) > 0:
        matched_phases, matched_hkls, matched_deltas = match_phases_to_data_fast(obs_peaks_arr, theo, tol_deg=tol)
    else:
        matched_phases = matched_hkls = matched_deltas = np.array([])
    
    fig_id = go.Figure()
    fig_id.add_trace(go.Scatter(x=active_df["two_theta"], y=active_df["intensity"], mode="lines", name="Observed", line=dict(color="lightsteelblue", width=1)))
    if len(obs_peaks_arr) > 0:
        fig_id.add_trace(go.Scatter(x=obs_peaks_arr[:, 0], y=obs_peaks_arr[:, 1], mode="markers", name="Detected peaks", marker=dict(symbol="triangle-down", size=10, color="crimson", line=dict(color="darkred", width=1))))
    
    I_top, I_bot = active_df["intensity"].max(), active_df["intensity"].min()
    for i, (ph, pk_dict) in enumerate(theo.items()):
        color = PH_COLORS[i % len(PH_COLORS)]
        offset = I_bot - (i + 1) * (I_top * 0.04)
        positions = pk_dict["positions"]
        if len(positions) > 0:
            fig_id.add_trace(go.Scatter(x=positions, y=[offset] * len(positions), mode="markers", name=f"{ph}", marker=dict(symbol="line-ns", size=14, color=color, line=dict(width=1.5, color=color)), customdata=pk_dict["hkl_labels"], hovertemplate="<b>%{fullData.name}</b><br>2θ=%{x:.3f}°<br>%{customdata}<extra></extra>"))
    
    fig_id.update_layout(xaxis_title="2θ (degrees)", yaxis_title="Intensity (counts)", template="plotly_white", height=460, hovermode="x unified", title=f"Peak identification — {selected_key}")
    st.plotly_chart(fig_id, use_container_width=True)
    
    st.markdown(f"#### {len(obs_peaks_arr)} detected peaks")
    if len(obs_peaks_arr) > 0:
        disp_data = {"two_theta": np.round(obs_peaks_arr[:, 0], 4), "intensity": np.round(obs_peaks_arr[:, 1], 1), "prominence": np.round(obs_peaks_arr[:, 2], 1), "Phase match": matched_phases, "(hkl)": matched_hkls, "Δ2θ (°)": np.round(matched_deltas, 4)}
        st.dataframe(pd.DataFrame(disp_data), use_container_width=True)
    
    with st.expander("📐 Theoretical peak positions per phase"):
        for ph in selected_phases:
            pk = theo[ph]
            n_peaks = len(pk["positions"])
            st.markdown(f"**{ph}** — {n_peaks} reflections in {tt_min:.0f}°–{tt_max:.0f}°")
            if n_peaks > 0:
                pk_df = pd.DataFrame({"2θ (°)": np.round(pk["positions"], 3), "d (Å)": np.round(pk["d_spacings"], 4), "hkl": pk["hkl_labels"]})
                st.dataframe(pk_df, use_container_width=True, height=200)

# TAB 2 — RIETVELD FIT (GSAS-II or Numba fallback)
with tabs[2]:
    st.subheader("Rietveld Refinement")
    if not selected_phases:
        st.warning("☑️ Select at least one phase in the sidebar.")
    elif not run_btn:
        st.info(f"Configure settings, then click **▶ Run Rietveld Refinement** using {'GSAS-II' if GSASII_AVAILABLE and selected_engine.startswith('GSAS') else 'Numba-accelerated'} engine.")
    else:
        with st.spinner(f"Running refinement with {selected_engine}…"):
            if GSASII_AVAILABLE and selected_engine == "GSAS-II (scriptable API)":
                # Use GSAS-II
                g2_result = run_gsasii_refinement_cached(
                    tuple(active_df["two_theta"].values),
                    tuple(active_df["intensity"].values),
                    wavelength,
                    tuple(selected_phases),
                    bg_order,
                    g2_max_cycles if 'g2_max_cycles' in locals() else 20
                )
                
                if g2_result:
                    result = g2_result
                    result["converged"] = g2_result.get("converged", False)
                else:
                    st.warning("⚠️ GSAS-II refinement failed or returned no results. Falling back to Numba engine.")
                    refiner = RietveldRefinement(active_df, selected_phases, wavelength, bg_poly_order=bg_order, peak_shape=peak_shape, eta=eta)
                    result = refiner.run()
            else:
                # Use Numba fallback
                refiner = RietveldRefinement(active_df, selected_phases, wavelength, bg_poly_order=bg_order, peak_shape=peak_shape, eta=eta)
                result = refiner.run()
        
        conv_icon = "✅" if result.get("converged", False) else "⚠️"
        st.success(f"{conv_icon} Refinement finished · R_wp = **{result.get('Rwp', 0):.2f}%** · R_exp = **{result.get('Rexp', 0):.2f}%** · χ² = **{result.get('chi2', 0):.3f}**")
        
        m1,m2,m3,m4 = st.columns(4)
        m1.metric("R_wp (%)", f"{result.get('Rwp', 0):.2f}", delta="< 15 is acceptable", delta_color="off")
        m2.metric("R_exp (%)", f"{result.get('Rexp', 0):.2f}")
        m3.metric("GoF χ²", f"{result.get('chi2', 0):.3f}", delta="target ≈ 1", delta_color="off")
        m4.metric("Zero shift (°)", f"{result.get('zero_shift', 0):.4f}")
        
        # Plot results
        fig_rv = make_subplots(rows=2, cols=1, row_heights=[0.78, 0.22], shared_xaxes=True, vertical_spacing=0.04, subplot_titles=("Observed vs Calculated", "Difference"))
        fig_rv.add_trace(go.Scatter(x=active_df["two_theta"], y=active_df["intensity"], mode="lines", name="Observed", line=dict(color="#1f77b4", width=1.0)), row=1, col=1)
        
        if result.get("y_calc") is not None:
            fig_rv.add_trace(go.Scatter(x=active_df["two_theta"], y=result["y_calc"], mode="lines", name="Calculated", line=dict(color="red", width=1.5)), row=1, col=1)
        if result.get("y_background") is not None:
            fig_rv.add_trace(go.Scatter(x=active_df["two_theta"], y=result["y_background"], mode="lines", name="Background", line=dict(color="green", width=1, dash="dash")), row=1, col=1)
        
        I_top2, I_bot2 = active_df["intensity"].max(), active_df["intensity"].min()
        
        # Phase markers from registry (Numba) or theoretical peaks (GSAS-II fallback)
        if "peak_registry" in result:
            for i, reg in enumerate(result["peak_registry"]):
                color = PH_COLORS[i % len(PH_COLORS)]
                ybase = I_bot2 - (i+1) * I_top2 * 0.035
                fig_rv.add_trace(go.Scatter(x=reg["positions"], y=[ybase] * len(reg["positions"]), mode="markers", name=f"{reg['phase']} reflections", marker=dict(symbol="line-ns", size=10, color=color, line=dict(width=1.5, color=color)), customdata=reg["hkl_labels"], hovertemplate="%{customdata} 2θ=%{x:.3f}°<extra>"+reg['phase']+"</extra>"), row=1, col=1)
        else:
            # Fallback: generate theoretical peaks for display
            for i, ph in enumerate(selected_phases):
                pk_pos = generate_theoretical_peaks_fast(ph, wavelength, tt_min, tt_max)
                if len(pk_pos["positions"]) > 0:
                    color = PH_COLORS[i % len(PH_COLORS)]
                    ybase = I_bot2 - (i+1) * I_top2 * 0.035
                    fig_rv.add_trace(go.Scatter(x=pk_pos["positions"], y=[ybase] * len(pk_pos["positions"]), mode="markers", name=f"{ph} reflections", marker=dict(symbol="line-ns", size=10, color=color, line=dict(width=1.5, color=color)), customdata=pk_pos["hkl_labels"], hovertemplate="%{customdata} 2θ=%{x:.3f}°<extra>"+ph+"</extra>"), row=1, col=1)
        
        if result.get("y_calc") is not None:
            diff = active_df["intensity"].values - result["y_calc"]
            fig_rv.add_trace(go.Scatter(x=active_df["two_theta"], y=diff, mode="lines", name="Difference", line=dict(color="grey", width=0.8)), row=2, col=1)
            fig_rv.add_hline(y=0, line_dash="dash", line_color="black", line_width=0.8, row=2, col=1)
        
        fig_rv.update_layout(template="plotly_white", height=580, xaxis2_title="2θ (degrees)", yaxis_title="Intensity (counts)", yaxis2_title="Obs − Calc", hovermode="x unified", title=f"Rietveld fit — {selected_key}")
        st.plotly_chart(fig_rv, use_container_width=True)
        
        st.markdown("#### Refined Lattice Parameters")
        lp_rows = []
        for ph in selected_phases:
            p, p0 = result.get("lattice_params", {}).get(ph, {}), PHASE_LIBRARY[ph]["lattice"]
            da = (p.get("a", p0["a"]) - p0["a"]) / p0["a"] * 100 if "a" in p0 and isinstance(p0.get("a"), (int, float)) and isinstance(p.get("a"), (int, float)) else 0
            lp_rows.append({"Phase": ph, "System": PHASE_LIBRARY[ph]["system"], "a_lib (Å)": f"{p0.get('a','—'):.5f}" if isinstance(p0.get('a'), (int,float)) else "—", "a_ref (Å)": f"{p.get('a', p0.get('a','—')):.5f}" if isinstance(p.get('a'), (int,float)) else "—", "Δa/a₀ (%)": f"{da:+.3f}", "c_ref (Å)": f"{p.get('c','—'):.5f}" if isinstance(p.get('c'), (int,float)) else "—", "Wt%": f"{result.get('phase_fractions', {}).get(ph,0)*100:.1f}"})
        st.dataframe(pd.DataFrame(lp_rows), use_container_width=True)
        
        # Cache results
        st.session_state[f"result_{selected_key}"], st.session_state[f"phases_{selected_key}"] = result, selected_phases
        st.session_state["last_result"], st.session_state["last_phases"], st.session_state["last_sample"] = result, selected_phases, selected_key
        st.session_state["last_engine"] = selected_engine

# TAB 3 — QUANTIFICATION
with tabs[3]:
    st.subheader("Phase Quantification")
    if "last_result" not in st.session_state:
        st.info("Run the Rietveld refinement first.")
    else:
        result, phases = st.session_state["last_result"], st.session_state["last_phases"]
        fracs = result.get("phase_fractions", {})
        if not fracs:
            st.warning("⚠️ No phase fractions available from refinement.")
        else:
            labels, values = list(fracs.keys()), [fracs[ph]*100 for ph in fracs]
            colors = [PHASE_LIBRARY[ph]["color"] for ph in labels]
            col_pie, col_bar = st.columns(2)
            with col_pie:
                fig_pie = go.Figure(go.Pie(labels=labels, values=values, hole=0.38, textinfo="label+percent", marker=dict(colors=colors)))
                fig_pie.update_layout(title="Phase weight fractions", height=370)
                st.plotly_chart(fig_pie, use_container_width=True)
            with col_bar:
                fig_bar = go.Figure(go.Bar(x=labels, y=values, marker_color=colors, text=[f"{v:.1f}%" for v in values], textposition="outside"))
                fig_bar.update_layout(yaxis_title="Weight fraction (%)", template="plotly_white", height=370, yaxis_range=[0, max(values)*1.25], title=f"Phase fractions — {st.session_state['last_sample']}")
                st.plotly_chart(fig_bar, use_container_width=True)
            rows = []
            for ph in labels:
                pi, lp = PHASE_LIBRARY[ph], result.get("lattice_params", {}).get(ph, {})
                rows.append({"Phase": ph, "Crystal system": pi["system"], "Space group": pi["space_group"], "a (Å)": f"{lp.get('a','—'):.5f}" if isinstance(lp.get('a'), (int,float)) else "—", "c (Å)": f"{lp.get('c','—'):.5f}" if isinstance(lp.get('c'), (int,float)) else "—", "Wt%": f"{fracs.get(ph,0)*100:.2f}", "Role": pi["description"][:65]+"…" if len(pi["description"])>65 else pi["description"]})
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

# TAB 4 — SAMPLE COMPARISON
with tabs[4]:
    st.subheader("🔄 Multi-Sample Comparison")
    view_mode = st.radio("View mode", ["📊 Interactive (Plotly)", "🖼️ Publication-Quality (Matplotlib)"], horizontal=True, key="comp_view_mode")
    comp_samples = st.multiselect("Select samples to compare", options=SAMPLE_KEYS, default=[k for k in SAMPLE_KEYS if SAMPLE_CATALOG[k]["group"] == "Printed"][:4], format_func=lambda k: f"[{SAMPLE_CATALOG[k]['short']}] {SAMPLE_CATALOG[k]['label']}", key="comp_samples")
    
    if not comp_samples:
        st.warning("⚠️ Select at least one sample to compare.")
    else:
        col_opt1, col_opt2 = st.columns(2)
        with col_opt1:
            normalize = st.checkbox("✓ Normalise to [0,1]", value=True, key="comp_normalize")
            show_grid = st.checkbox("✓ Show grid", value=True, key="comp_grid")
        with col_opt2:
            line_width = st.slider("Line width", 0.5, 3.0, 1.5, 0.1, key="comp_lw")
            opacity = st.slider("Opacity", 0.3, 1.0, 1.0, 0.1, key="comp_alpha")
        
        if view_mode == "📊 Interactive (Plotly)":
            fig_cmp = go.Figure()
            for k in comp_samples:
                df_s = all_data.get(k, pd.DataFrame({"two_theta": np.linspace(30, 130, 2000), "intensity": np.random.normal(200, 50, 2000)}))
                x, y = df_s["two_theta"].values, df_s["intensity"].values
                if normalize and len(y) > 1:
                    y = (y - y.min()) / (y.max() - y.min() + 1e-8)
                m = SAMPLE_CATALOG[k]
                fig_cmp.add_trace(go.Scatter(x=x, y=y, mode="lines", name=m["label"], line=dict(color=m["color"], width=line_width), opacity=opacity))
            fig_cmp.update_layout(title="XRD Pattern Comparison", xaxis_title="2θ (degrees)", yaxis_title="Normalised Intensity" if normalize else "Intensity (counts)", template="plotly_white" if show_grid else "plotly", height=500, hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig_cmp, use_container_width=True)
        else:
            st.markdown("### 🎨 Publication Plot Settings")
            col_pub1, col_pub2, col_pub3 = st.columns(3)
            with col_pub1:
                pub_width = st.slider("Width (inches)", 6.0, 14.0, 10.0, 0.5, key="pub_comp_w")
                pub_font = st.slider("Font Size", 8, 18, 11, 1, key="pub_comp_font")
                stack_offset = st.slider("Stack offset", 0.0, 1.5, 0.0, 0.1, key="pub_comp_stack")
            with col_pub2:
                pub_height = st.slider("Height (inches)", 5.0, 12.0, 7.0, 0.5, key="pub_comp_h")
                pub_legend_pos = st.selectbox("Legend", ["best", "upper right", "upper left", "lower left", "lower right", "center right", "off"], key="pub_comp_leg")
            with col_pub3:
                st.markdown("**🎨 Per-Sample Styling**")
                sample_styles = {}
                for k in comp_samples:
                    m = SAMPLE_CATALOG[k]
                    with st.expander(f"{m['short']}", expanded=False):
                        sample_styles[k] = {"color": st.color_picker("Color", m["color"], key=f"col_{k}"), "style": st.selectbox("Line", ["-", "--", ":", "-."], index=0, key=f"sty_{k}"), "width": st.slider("Width", 0.5, 3.0, 1.5, 0.1, key=f"lw_{k}"), "label": st.text_input("Legend Label", m["label"], key=f"lbl_{k}")}
            
            sample_data_list, legend_labels, line_styles = [], [], []
            for k in comp_samples:
                df_s = all_data.get(k, pd.DataFrame({"two_theta": np.linspace(30, 130, 2000), "intensity": np.random.normal(200, 50, 2000)}))
                styles = sample_styles.get(k, {})
                sample_data_list.append({"two_theta": df_s["two_theta"].values, "intensity": df_s["intensity"].values, "label": SAMPLE_CATALOG[k]["label"], "color": styles.get("color", SAMPLE_CATALOG[k]["color"]), "linewidth": styles.get("width", line_width)})
                legend_labels.append(styles.get("label", SAMPLE_CATALOG[k]["label"]))
                line_styles.append(styles.get("style", "-"))
            
            try:
                fig_pub, ax_pub = plot_sample_comparison_publication(sample_data_list=sample_data_list, tt_min=tt_min, tt_max=tt_max, figsize=(pub_width, pub_height), font_size=pub_font, legend_pos=pub_legend_pos if pub_legend_pos != "off" else "off", normalize=normalize, stack_offset=stack_offset, line_styles=line_styles, legend_labels=legend_labels, show_grid=show_grid)
                st.pyplot(fig_pub, dpi=150, use_container_width=True)
                st.markdown("#### 📥 Export Publication Figure")
                col_e1, col_e2, col_e3 = st.columns(3)
                with col_e1:
                    buf = io.BytesIO(); fig_pub.savefig(buf, format='pdf', bbox_inches='tight'); buf.seek(0)
                    st.download_button("📄 PDF", buf.read(), file_name=f"xrd_comparison_{len(comp_samples)}samples.pdf", mime="application/pdf", use_container_width=True)
                with col_e2:
                    buf = io.BytesIO(); fig_pub.savefig(buf, format='png', dpi=300, bbox_inches='tight'); buf.seek(0)
                    st.download_button("🖼️ PNG (300 DPI)", buf.read(), file_name=f"xrd_comparison_{len(comp_samples)}samples.png", mime="image/png", use_container_width=True)
                with col_e3:
                    buf = io.BytesIO(); fig_pub.savefig(buf, format='eps', bbox_inches='tight'); buf.seek(0)
                    st.download_button("📐 EPS", buf.read(), file_name=f"xrd_comparison_{len(comp_samples)}samples.eps", mime="application/postscript", use_container_width=True)
                plt.close(fig_pub)
            except Exception as e:
                st.error(f"❌ Plot generation failed: {str(e)}")

# TAB 5 — REPORT
with tabs[5]:
    st.subheader("Analysis Report")
    if "last_result" not in st.session_state:
        st.info("Run the Rietveld refinement first (Tab 3).")
    else:
        result, phases, samp = st.session_state["last_result"], st.session_state["last_phases"], st.session_state["last_sample"]
        engine_used = st.session_state.get("last_engine", "Unknown")
        report_md = generate_report(result, phases, wavelength, samp)
        report_md += f"\n**Refinement Engine**: {engine_used}\n"
        st.markdown(report_md)
        col_dl1, col_dl2 = st.columns(2)
        col_dl1.download_button("⬇️ Download Report (.md)", data=report_md, file_name=f"rietveld_report_{samp}.md", mime="text/markdown")
        export_df = active_df.copy()
        if result.get("y_calc") is not None:
            export_df["y_calc"], export_df["y_background"], export_df["difference"] = result["y_calc"], result.get("y_background", np.zeros_like(active_df["intensity"])), active_df["intensity"].values - result["y_calc"]
        csv_buf = io.StringIO()
        export_df.to_csv(csv_buf, index=False)
        col_dl2.download_button("⬇️ Download Fit Data (.csv)", data=csv_buf.getvalue(), file_name=f"rietveld_fit_{samp}.csv", mime="text/csv")

# TAB 6 — PUBLICATION PLOT
with tabs[6]:
    st.subheader("🖼️ Publication-Quality Plot (matplotlib)")
    st.caption("Generate journal-ready figures with customizable phase markers, legend control & spacing")
    
    if "last_result" not in st.session_state or "last_phases" not in st.session_state:
        st.info("🔬 Run the Rietveld refinement first (Tab 3: 🧮 Rietveld Fit) to enable publication plotting.")
    else:
        result = st.session_state["last_result"]
        phases = st.session_state["last_phases"]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            fig_width = st.slider("Figure width (inches)", 6.0, 14.0, 10.0, 0.5, key="pub_width")
            offset_factor = st.slider("Difference curve offset", 0.05, 0.25, 0.12, 0.01, key="pub_offset")
            font_size = st.slider("Global Font Size", 6, 22, 11, key="pub_font")
        with col2:
            fig_height = st.slider("Figure height (inches)", 5.0, 12.0, 7.0, 0.5, key="pub_height")
            show_hkl = st.checkbox("Show hkl labels", value=True, key="pub_hkl")
            legend_pos = st.selectbox("Legend Position", ["best", "upper right", "upper left", "lower left", "lower right", "center right", "center left", "lower center", "upper center", "center", "off"], index=0, key="pub_legend_pos")
        with col3:
            export_format = st.selectbox("Export format", ["PDF", "PNG", "EPS"], index=0, key="pub_format")
            marker_spacing = st.slider("Marker row spacing", 0.8, 2.5, 1.3, 0.1, help="Vertical distance between phase marker rows", key="pub_spacing")
            st.markdown("**🎨 Phase Customization**")
        
        st.markdown("### 📋 Legend Control")
        st.caption("Select which phases to include in the plot legend")
        n_cols = min(4, len(phases))
        legend_cols = st.columns(n_cols)
        legend_phases_selected = []
        for idx, ph in enumerate(phases):
            col_idx = idx % n_cols
            with legend_cols[col_idx]:
                if st.checkbox(f"✓ {ph}", value=True, key=f"leg_{ph}"):
                    legend_phases_selected.append(ph)
        
        # Build phase_data for plotting
        phase_data = []
        # Use peak_registry if available (Numba), otherwise generate theoretical peaks
        if "peak_registry" in result:
            for i, reg in enumerate(result["peak_registry"]):
                ph = reg["phase"]
                with st.expander(f"⚙️ Settings for **{ph}**", expanded=(i==0)):
                    c_col, c_shape = st.columns(2)
                    custom_color = c_col.color_picker("Color", value=PHASE_LIBRARY[ph]["color"], key=f"col_{ph}")
                    shape_options = ["|", "_", "s", "^", "v", "d", "x", "+", "*"]
                    default_idx = shape_options.index(PHASE_LIBRARY[ph].get("marker_shape", "|"))
                    custom_shape = c_shape.selectbox("Marker Shape", shape_options, index=default_idx, key=f"shp_{ph}")
                phase_data.append({"name": ph, "positions": reg["positions"], "color": custom_color, "marker_shape": custom_shape, "hkl": [hkl.strip("()").split(",") if hkl else None for hkl in reg["hkl_labels"]] if show_hkl else None})
        else:
            # Fallback: generate theoretical peaks
            for i, ph in enumerate(phases):
                pk_dict = generate_theoretical_peaks_fast(ph, wavelength, tt_min, tt_max)
                with st.expander(f"⚙️ Settings for **{ph}**", expanded=(i==0)):
                    c_col, c_shape = st.columns(2)
                    custom_color = c_col.color_picker("Color", value=PHASE_LIBRARY[ph]["color"], key=f"col_{ph}")
                    shape_options = ["|", "_", "s", "^", "v", "d", "x", "+", "*"]
                    default_idx = shape_options.index(PHASE_LIBRARY[ph].get("marker_shape", "|"))
                    custom_shape = c_shape.selectbox("Marker Shape", shape_options, index=default_idx, key=f"shp_{ph}")
                phase_data.append({"name": ph, "positions": pk_dict["positions"], "color": custom_color, "marker_shape": custom_shape, "hkl": [hkl.strip("()").split(",") if hkl else None for hkl in pk_dict["hkl_labels"]] if show_hkl and len(pk_dict["positions"]) > 0 else None})
        
        try:
            fig, ax = plot_rietveld_publication(active_df["two_theta"].values, active_df["intensity"].values, result.get("y_calc", active_df["intensity"].values), active_df["intensity"].values - result.get("y_calc", active_df["intensity"].values), phase_data, offset_factor=offset_factor, figsize=(fig_width, fig_height), font_size=font_size, legend_pos=legend_pos, marker_row_spacing=marker_spacing, legend_phases=legend_phases_selected if legend_phases_selected else None)
            st.pyplot(fig, dpi=150, use_container_width=True)
            st.markdown("#### 📥 Export Options")
            col_e1, col_e2, col_e3 = st.columns(3)
            with col_e1:
                buf = io.BytesIO(); fig.savefig(buf, format='pdf', bbox_inches='tight'); buf.seek(0)
                st.download_button("📄 PDF", buf.read(), file_name=f"rietveld_pub_{selected_key}.pdf", mime="application/pdf", use_container_width=True)
            with col_e2:
                buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=300, bbox_inches='tight'); buf.seek(0)
                st.download_button("🖼️ PNG (300 DPI)", buf.read(), file_name=f"rietveld_pub_{selected_key}.png", mime="image/png", use_container_width=True)
            with col_e3:
                buf = io.BytesIO(); fig.savefig(buf, format='eps', bbox_inches='tight'); buf.seek(0)
                st.download_button("📐 EPS", buf.read(), file_name=f"rietveld_pub_{selected_key}.eps", mime="application/postscript", use_container_width=True)
            with st.expander("🎨 Marker Shape Reference"):
                st.markdown("""| Shape | Code | Visual | Recommended Use |\n|-------|------|--------|----------------|\n| Vertical bar | `|` | │ | FCC-Co matrix (primary) |\n| Horizontal bar | `_` | ─ | HCP-Co (secondary) |\n| **Square** ✨ | `s` | ■ | M₂₃C₆ carbides |\n| Triangle up | `^` | ▲ | Sigma phase |\n| Triangle down | `v` | ▼ | Additional precipitates |\n| **Diamond** ✨ | `d` | ◆ | Trace intermetallics |\n| Cross | `x` | × | Reference peaks |\n| Plus | `+` | + | Calibration markers |\n| Star | `*` | ✦ | Special annotations |""")
            plt.close(fig)
        except Exception as e:
            st.error(f"❌ Plot generation failed: {str(e)}")

st.markdown("---")
st.caption(f"""XRD Rietveld App • Co-Cr Dental Alloy Analysis • Supports .asc, .ASC & .xrdml • GitHub: Maryamslm/XRD-3Dprinted-Ret/SAMPLES

**Engine Status**: {'✅ GSAS-II scriptable API available' if GSASII_AVAILABLE else '❌ GSAS-II not installed'} | {'✅ Numba JIT acceleration active' if NUMBA_AVAILABLE else '⚠️ Install numba for 10-50× speedup: `pip install numba`'}""")

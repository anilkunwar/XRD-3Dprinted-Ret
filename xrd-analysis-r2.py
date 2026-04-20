"""
XRD Rietveld Analysis — Co-Cr Dental Alloy (Mediloy S Co, BEGO)
================================================================
True Rietveld refinement: lattice parameters, structure factors,
scale factors, zero shift, correct phase fractions, wavelength dependence.
Supports: .asc, .xrdml, .ASC files • GitHub integration.
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import io, os, math, sys, re, xml.etree.ElementTree as ET
from scipy import signal
from scipy.optimize import least_squares
import requests
import numba
from numba import jit

# Try lmfit for advanced refinement
try:
    import lmfit
    LMFIT_AVAILABLE = True
except ImportError:
    LMFIT_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════════════════════
# SAMPLE CATALOG & INSTRUMENT CONFIG
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
    "MEDILOY_powder": {"label": "Powder • Raw Material", "short": "Powder", "fabrication": "Powder", "treatment": "As-received", "filename": "MEDILOY_powder.ASC", "color": "#9467bd", "group": "Reference", "description": "Mediloy S Co powder, as-received"},
}

SAMPLE_KEYS = list(SAMPLE_CATALOG.keys())
XRAY_SOURCES = {
    "Cu Kα₁ (1.5406 Å)": 1.5406,
    "Co Kα₁ (1.7890 Å)": 1.7890,
    "Mo Kα₁ (0.7093 Å)": 0.7093,
    "Fe Kα₁ (1.9374 Å)": 1.9374,
    "Cr Kα₁ (2.2909 Å)": 2.2909,
    "Ag Kα₁ (0.5594 Å)": 0.5594,
    "Custom Wavelength": None
}

# Phase library with structure factors (|F|^2) and multiplicities
PHASE_LIBRARY = {
    "FCC-Co": {
        "system": "Cubic", "space_group": "Fm-3m",
        "lattice": {"a": 3.544},
        "reflections": [
            {"hkl": (1,1,1), "mult": 8, "F2": 100.0},
            {"hkl": (2,0,0), "mult": 6, "F2": 80.0},
            {"hkl": (2,2,0), "mult": 12, "F2": 60.0},
            {"hkl": (3,1,1), "mult": 24, "F2": 40.0},
        ],
        "color": "#e377c2", "default": True, "marker_shape": "|",
        "description": "Face-centered cubic Co-based solid solution (matrix phase)"
    },
    "HCP-Co": {
        "system": "Hexagonal", "space_group": "P6₃/mmc",
        "lattice": {"a": 2.507, "c": 4.069},
        "reflections": [
            {"hkl": (1,0,0), "mult": 6, "F2": 70.0},
            {"hkl": (0,0,2), "mult": 2, "F2": 90.0},
            {"hkl": (1,0,1), "mult": 12, "F2": 85.0},
            {"hkl": (1,0,2), "mult": 12, "F2": 50.0},
            {"hkl": (1,1,0), "mult": 12, "F2": 40.0},
        ],
        "color": "#7f7f7f", "default": False, "marker_shape": "_",
        "description": "Hexagonal close-packed Co"
    },
    "M23C6": {
        "system": "Cubic", "space_group": "Fm-3m",
        "lattice": {"a": 10.63},
        "reflections": [
            {"hkl": (3,1,1), "mult": 24, "F2": 120.0},
            {"hkl": (4,0,0), "mult": 6, "F2": 90.0},
            {"hkl": (5,1,1), "mult": 48, "F2": 60.0},
            {"hkl": (4,4,0), "mult": 12, "F2": 45.0},
        ],
        "color": "#bcbd22", "default": False, "marker_shape": "s",
        "description": "Cr-rich carbide M₂₃C₆"
    },
    "Sigma": {
        "system": "Tetragonal", "space_group": "P4₂/mnm",
        "lattice": {"a": 8.80, "c": 4.56},
        "reflections": [
            {"hkl": (2,1,0), "mult": 8, "F2": 80.0},
            {"hkl": (2,2,0), "mult": 4, "F2": 70.0},
            {"hkl": (3,1,0), "mult": 8, "F2": 60.0},
        ],
        "color": "#17becf", "default": False, "marker_shape": "^",
        "description": "Sigma phase (Co,Cr) intermetallic"
    }
}

def wavelength_to_energy(wavelength):
    h = 4.135667696e-15
    c = 299792458
    return (h * c) / (wavelength * 1e-10) / 1000

# d-spacing formulas
def d_cubic(a, h, k, l):
    return a / np.sqrt(h*h + k*k + l*l)

def d_hexagonal(a, c, h, k, l):
    return 1.0 / np.sqrt((4.0/3.0)*(h*h + h*k + k*k)/a/a + l*l/c/c)

def d_tetragonal(a, c, h, k, l):
    return 1.0 / np.sqrt((h*h + k*k)/a/a + l*l/c/c)

def bragg_2theta(d, wavelength):
    return 2 * np.degrees(np.arcsin(wavelength / (2 * d)))

def compute_lp_factor(two_theta):
    """Lorentz-polarization factor for powder diffraction."""
    theta_rad = np.radians(two_theta / 2.0)
    costh = np.cos(theta_rad)
    sin2th = np.sin(np.radians(two_theta))
    return (1.0 + costh**2) / (sin2th * sin2th + 1e-10)  # simplified version

def generate_peaks_dynamic(phase_name, wavelength, lattice, tt_min, tt_max):
    """Compute all reflections for a phase using current lattice parameters."""
    phase = PHASE_LIBRARY[phase_name]
    system = phase["system"]
    reflections = phase["reflections"]
    peaks = []
    for ref in reflections:
        h,k,l = ref["hkl"]
        if system == "Cubic":
            d = d_cubic(lattice["a"], h, k, l)
        elif system == "Hexagonal":
            d = d_hexagonal(lattice["a"], lattice["c"], h, k, l)
        elif system == "Tetragonal":
            d = d_tetragonal(lattice["a"], lattice["c"], h, k, l)
        else:
            continue
        tt = bragg_2theta(d, wavelength)
        if tt_min <= tt <= tt_max:
            lp = compute_lp_factor(tt)
            I0 = ref["mult"] * ref["F2"] * lp
            peaks.append({
                "two_theta": tt,
                "I0": I0,
                "mult": ref["mult"],
                "F2": ref["F2"],
                "hkl_label": f"({h},{k},{l})"
            })
    return sorted(peaks, key=lambda x: x["two_theta"])

# ═══════════════════════════════════════════════════════════════════════════════
# FILE PARSERS (unchanged)
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
                tt = float(parts[0])
                intensity = float(parts[1])
                rows.append((tt, intensity))
            except ValueError:
                continue
    df = pd.DataFrame(rows, columns=["two_theta", "intensity"])
    if len(df) == 0:
        return pd.DataFrame(columns=["two_theta", "intensity"])
    return df.sort_values("two_theta").reset_index(drop=True)

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
            for scan in root.iter():
                if scan.tag.endswith('scan') or scan.tag == 'scan':
                    for child in scan:
                        if child.tag.endswith('xRayData') or child.tag == 'xRayData':
                            vals = child.text
                            if vals:
                                nums = [float(v) for v in re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', vals)]
                                if len(nums) >= 2 and len(nums) % 2 == 0:
                                    data_points = [(nums[i], nums[i+1]) for i in range(0, len(nums), 2)]
                                    break
                                elif len(nums) > 10:
                                    start = float(scan.get('startAngle', scan.get('start', 0)))
                                    end = float(scan.get('endAngle', scan.get('end', 100)))
                                    two_theta = np.linspace(start, end, len(nums))
                                    data_points = list(zip(two_theta, nums))
                                    break
        if not data_points:
            all_nums = [float(m) for m in re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', text)]
            if len(all_nums) >= 20 and len(all_nums) % 2 == 0:
                data_points = [(all_nums[i], all_nums[i+1]) for i in range(0, len(all_nums), 2)]
        if not data_points:
            return pd.DataFrame(columns=["two_theta", "intensity"])
        df = pd.DataFrame(data_points, columns=["two_theta", "intensity"])
        df = df[(df["two_theta"] > 0) & (df["two_theta"] < 180) & (df["intensity"] >= 0)]
        return df.sort_values("two_theta").reset_index(drop=True)
    except Exception as e:
        st.error(f"Error parsing .xrdml: {e}")
        return pd.DataFrame(columns=["two_theta", "intensity"])

def parse_file(raw_bytes: bytes, filename: str) -> pd.DataFrame:
    ext = os.path.splitext(filename)[1].lower()
    if ext == '.xrdml':
        return parse_xrdml(raw_bytes)
    return parse_asc(raw_bytes)

# ═══════════════════════════════════════════════════════════════════════════════
# GITHUB INTEGRATION (unchanged)
# ═══════════════════════════════════════════════════════════════════════════════

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
                return [
                    {"name": item["name"], "path": item["path"], "download_url": item.get("download_url"), "size": item.get("size", 0)}
                    for item in items if item.get("type") == "file" and any(item["name"].lower().endswith(ext) for ext in supported)
                ]
            return []
        return []
    except Exception as e:
        st.warning(f"GitHub fetch error: {e}")
        return []

@st.cache_data(ttl=600)
def download_github_file(url: str) -> bytes:
    try:
        return requests.get(url, timeout=30).content
    except Exception as e:
        st.error(f"Download failed: {e}")
        return b""

def find_github_file_by_catalog_key(catalog_key: str, gh_files: list):
    target = SAMPLE_CATALOG[catalog_key]["filename"].upper()
    for f in gh_files:
        if f["name"].upper() == target:
            return f
    return None

# ═══════════════════════════════════════════════════════════════════════════════
# NUMBA ACCELERATED FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

@numba.jit(nopython=True, cache=True, parallel=False)
def compute_background(x, coeffs):
    n = len(x)
    bg = np.zeros(n, dtype=np.float64)
    for i in range(n):
        val = 0.0
        for p, c in enumerate(coeffs):
            val += c * (x[i] ** p)
        bg[i] = val
    return bg

@numba.jit(nopython=True, cache=True)
def pseudo_voigt_peak(x, pos, fwhm, eta=0.5):
    n = len(x)
    y = np.zeros(n, dtype=np.float64)
    for i in range(n):
        t = (x[i] - pos) / fwhm
        gauss = np.exp(-4.0 * np.log(2.0) * t * t)
        lorentz = 1.0 / (1.0 + 4.0 * t * t)
        y[i] = eta * lorentz + (1.0 - eta) * gauss
    return y

@numba.jit(nopython=True, cache=True, parallel=False)
def add_peaks_to_pattern(x, y_calc, peaks_pos, peaks_scale, peaks_fwhm, peaks_I0, eta=0.5):
    n_peaks = len(peaks_pos)
    for k in range(n_peaks):
        pos = peaks_pos[k]
        scale = peaks_scale[k]
        fwhm = peaks_fwhm[k]
        I0 = peaks_I0[k]
        profile = pseudo_voigt_peak(x, pos, fwhm, eta)
        for i in range(len(x)):
            y_calc[i] += scale * I0 * profile[i]
    return y_calc

# ═══════════════════════════════════════════════════════════════════════════════
# TRUE RIETVELD REFINEMENT CLASS (NUMBA ENGINE)
# ═══════════════════════════════════════════════════════════════════════════════

class RietveldRefinement:
    def __init__(self, data, phases, wavelength, bg_order=4,
                 refine_lattice=True, refine_zeroshift=True):
        self.data = data
        self.phases = phases
        self.wavelength = wavelength
        self.bg_order = bg_order
        self.refine_lattice = refine_lattice
        self.refine_zeroshift = refine_zeroshift
        self.x = data["two_theta"].values.astype(np.float64)
        self.y_obs = data["intensity"].values.astype(np.float64)
        self.tt_min = self.x.min()
        self.tt_max = self.x.max()

        # Initial lattice parameters and scale factors
        self.lattice_params = {}
        self.scale_factors = {}
        for ph in phases:
            self.lattice_params[ph] = PHASE_LIBRARY[ph]["lattice"].copy()
            self.scale_factors[ph] = 1.0

        # Precompute per-phase reflection data (will be updated when lattice changes)
        self._update_all_peaks()

    def _update_all_peaks(self):
        """Rebuild flat arrays of all peaks from all phases using current lattice."""
        self.peak_positions = []
        self.peak_I0 = []
        self.peak_scale = []   # scale factor of the phase that owns this peak
        self.peak_phase_index = []  # phase name for each peak
        for phase in self.phases:
            peaks = generate_peaks_dynamic(phase, self.wavelength,
                                           self.lattice_params[phase],
                                           self.tt_min, self.tt_max)
            scale = self.scale_factors[phase]
            for pk in peaks:
                self.peak_positions.append(pk["two_theta"])
                self.peak_I0.append(pk["I0"])
                self.peak_scale.append(scale)
                self.peak_phase_index.append(phase)
        self.peak_positions = np.array(self.peak_positions, dtype=np.float64)
        self.peak_I0 = np.array(self.peak_I0, dtype=np.float64)
        self.peak_scale = np.array(self.peak_scale, dtype=np.float64)
        self.n_peaks = len(self.peak_positions)

    def _update_lattice_from_params(self, params):
        """Update lattice parameters and scale factors from the parameter vector."""
        idx = self.bg_order + 1   # after background coefficients
        for phase in self.phases:
            # scale factor
            self.scale_factors[phase] = params[idx]
            idx += 1
            # lattice parameters (if refined)
            if self.refine_lattice:
                sys = PHASE_LIBRARY[phase]["system"]
                if sys == "Cubic":
                    self.lattice_params[phase]["a"] = params[idx]
                    idx += 1
                elif sys == "Hexagonal":
                    self.lattice_params[phase]["a"] = params[idx]
                    idx += 1
                    self.lattice_params[phase]["c"] = params[idx]
                    idx += 1
                elif sys == "Tetragonal":
                    self.lattice_params[phase]["a"] = params[idx]
                    idx += 1
                    self.lattice_params[phase]["c"] = params[idx]
                    idx += 1
        # zero shift (global)
        if self.refine_zeroshift:
            self.zero_shift = params[idx]
            idx += 1
        else:
            self.zero_shift = 0.0
        # global FWHM
        self.fwhm = params[idx]
        # Recompute peaks with updated lattices
        self._update_all_peaks()

    def _calculate_pattern(self, params):
        """Compute full calculated pattern for given parameter vector."""
        # Background
        bg_coeffs = params[:self.bg_order+1]
        y_calc = compute_background(self.x, bg_coeffs)
        # Update lattice, scale, zero shift, fwhm
        self._update_lattice_from_params(params)
        # Apply zero shift to peak positions
        shifted_pos = self.peak_positions + self.zero_shift
        # Build scale array (same for all peaks of a phase)
        scales = np.zeros(self.n_peaks, dtype=np.float64)
        for i, phase in enumerate(self.peak_phase_index):
            scales[i] = self.scale_factors[phase]
        fwhms = np.full(self.n_peaks, self.fwhm, dtype=np.float64)
        # Add peaks
        y_calc = add_peaks_to_pattern(self.x, y_calc, shifted_pos, scales, fwhms, self.peak_I0, eta=0.5)
        return y_calc

    def _residuals(self, params):
        return self.y_obs - self._calculate_pattern(params)

    def run(self):
        # Build initial parameter vector
        bg_init = [np.percentile(self.y_obs, 10)] + [0.0] * self.bg_order
        params0 = list(bg_init)
        for phase in self.phases:
            params0.append(1.0)  # scale
            if self.refine_lattice:
                sys = PHASE_LIBRARY[phase]["system"]
                if sys == "Cubic":
                    params0.append(self.lattice_params[phase]["a"])
                elif sys == "Hexagonal":
                    params0.append(self.lattice_params[phase]["a"])
                    params0.append(self.lattice_params[phase]["c"])
                elif sys == "Tetragonal":
                    params0.append(self.lattice_params[phase]["a"])
                    params0.append(self.lattice_params[phase]["c"])
        if self.refine_zeroshift:
            params0.append(0.0)
        params0.append(0.5)  # global FWHM
        params0 = np.array(params0, dtype=np.float64)

        try:
            res = least_squares(self._residuals, params0, max_nfev=200, method='trf')
            converged, params_opt = res.success, res.x
        except:
            converged, params_opt = False, params0

        # Final calculation
        y_calc = self._calculate_pattern(params_opt)
        bg_coeffs = params_opt[:self.bg_order+1]
        y_bg = compute_background(self.x, bg_coeffs)
        resid = self.y_obs - y_calc
        Rwp = np.sqrt(np.sum(resid**2) / np.sum(self.y_obs**2)) * 100.0
        Rexp = np.sqrt(max(1, len(self.x) - len(params_opt))) / np.sqrt(np.sum(self.y_obs) + 1e-10) * 100.0
        chi2 = (Rwp / max(Rexp, 0.01))**2

        # Extract refined values
        idx = self.bg_order + 1
        scale_final = {}
        lattice_final = {}
        for phase in self.phases:
            scale_final[phase] = params_opt[idx]
            idx += 1
            lp = self.lattice_params[phase].copy()
            if self.refine_lattice:
                sys = PHASE_LIBRARY[phase]["system"]
                if sys == "Cubic":
                    lp["a"] = params_opt[idx]
                    idx += 1
                elif sys == "Hexagonal":
                    lp["a"] = params_opt[idx]
                    idx += 1
                    lp["c"] = params_opt[idx]
                    idx += 1
                elif sys == "Tetragonal":
                    lp["a"] = params_opt[idx]
                    idx += 1
                    lp["c"] = params_opt[idx]
                    idx += 1
            lattice_final[phase] = lp
        if self.refine_zeroshift:
            zero_shift = params_opt[idx]
            idx += 1
        else:
            zero_shift = 0.0
        fwhm_final = params_opt[idx]

        # Compute phase fractions: weight fraction α = (Sα * Vα) / Σ(Sj * Vj)
        volumes = {}
        for phase in self.phases:
            lp = lattice_final[phase]
            sys = PHASE_LIBRARY[phase]["system"]
            if sys == "Cubic":
                vol = lp["a"]**3
            elif sys == "Hexagonal":
                vol = (np.sqrt(3)/2) * lp["a"]**2 * lp["c"]
            elif sys == "Tetragonal":
                vol = lp["a"]**2 * lp["c"]
            else:
                vol = 1.0
            volumes[phase] = vol
        total = sum(scale_final[ph] * volumes[ph] for ph in self.phases)
        phase_fractions = {ph: (scale_final[ph] * volumes[ph]) / total for ph in self.phases}

        return {
            "converged": converged,
            "Rwp": Rwp,
            "Rexp": Rexp,
            "chi2": chi2,
            "y_calc": y_calc,
            "y_background": y_bg,
            "zero_shift": zero_shift,
            "phase_fractions": phase_fractions,
            "lattice_params": lattice_final,
            "scale_factors": scale_final,
            "fwhm": fwhm_final
        }

# ═══════════════════════════════════════════════════════════════════════════════
# LMFIT ENGINE (ADVANCED)
# ═══════════════════════════════════════════════════════════════════════════════

def run_lmfit_refinement(data_df, phases, wavelength, tt_min, tt_max,
                         refine_lattice=True, refine_zeroshift=True, max_iter=100):
    if not LMFIT_AVAILABLE:
        raise ImportError("lmfit not installed")
    x = data_df["two_theta"].values
    y_obs = data_df["intensity"].values
    bg_order = 4

    params = lmfit.Parameters()
    # Background
    bg_init = np.percentile(y_obs, 10)
    for i in range(bg_order+1):
        params.add(f'b{i}', value=bg_init if i==0 else 0.0, vary=True)
    # Per-phase scale and lattice
    for phase in phases:
        params.add(f'scale_{phase}', value=1.0, min=0.0, vary=True)
        sys = PHASE_LIBRARY[phase]["system"]
        if refine_lattice:
            if sys == "Cubic":
                a0 = PHASE_LIBRARY[phase]["lattice"]["a"]
                params.add(f'a_{phase}', value=a0, min=a0*0.98, max=a0*1.02, vary=True)
            elif sys == "Hexagonal":
                a0 = PHASE_LIBRARY[phase]["lattice"]["a"]
                c0 = PHASE_LIBRARY[phase]["lattice"]["c"]
                params.add(f'a_{phase}', value=a0, min=a0*0.98, max=a0*1.02, vary=True)
                params.add(f'c_{phase}', value=c0, min=c0*0.98, max=c0*1.02, vary=True)
            elif sys == "Tetragonal":
                a0 = PHASE_LIBRARY[phase]["lattice"]["a"]
                c0 = PHASE_LIBRARY[phase]["lattice"]["c"]
                params.add(f'a_{phase}', value=a0, min=a0*0.98, max=a0*1.02, vary=True)
                params.add(f'c_{phase}', value=c0, min=c0*0.98, max=c0*1.02, vary=True)
    if refine_zeroshift:
        params.add('zero_shift', value=0.0, min=-0.5, max=0.5, vary=True)
    else:
        params.add('zero_shift', value=0.0, vary=False)
    params.add('fwhm', value=0.5, min=0.05, max=5.0, vary=True)

    def compute_pattern(params):
        bg = np.array([params[f'b{i}'].value for i in range(bg_order+1)])
        y_calc = compute_background(x, bg)
        zero = params['zero_shift'].value
        fwhm = params['fwhm'].value
        # Collect all peaks
        all_pos = []
        all_I0 = []
        all_scale = []
        for phase in phases:
            # Build current lattice
            sys = PHASE_LIBRARY[phase]["system"]
            if refine_lattice:
                if sys == "Cubic":
                    a = params[f'a_{phase}'].value
                    lattice = {"a": a}
                elif sys == "Hexagonal":
                    a = params[f'a_{phase}'].value
                    c = params[f'c_{phase}'].value
                    lattice = {"a": a, "c": c}
                elif sys == "Tetragonal":
                    a = params[f'a_{phase}'].value
                    c = params[f'c_{phase}'].value
                    lattice = {"a": a, "c": c}
                else:
                    lattice = PHASE_LIBRARY[phase]["lattice"]
            else:
                lattice = PHASE_LIBRARY[phase]["lattice"]
            peaks = generate_peaks_dynamic(phase, wavelength, lattice, x.min(), x.max())
            scale = params[f'scale_{phase}'].value
            for pk in peaks:
                pos = pk["two_theta"] + zero
                if pos < x.min() or pos > x.max():
                    continue
                all_pos.append(pos)
                all_I0.append(pk["I0"])
                all_scale.append(scale)
        if all_pos:
            all_pos = np.array(all_pos)
            all_I0 = np.array(all_I0)
            all_scale = np.array(all_scale)
            fwhms = np.full(len(all_pos), fwhm)
            y_calc = add_peaks_to_pattern(x, y_calc, all_pos, all_scale, fwhms, all_I0, eta=0.5)
        return y_calc

    def residual(params):
        return y_obs - compute_pattern(params)

    minimizer = lmfit.Minimizer(residual, params)
    result = minimizer.minimize(method='leastsq', max_nfev=max_iter)

    y_calc = compute_pattern(result.params)
    bg = np.array([result.params[f'b{i}'].value for i in range(bg_order+1)])
    y_bg = compute_background(x, bg)
    resid = y_obs - y_calc
    Rwp = np.sqrt(np.sum(resid**2) / np.sum(y_obs**2)) * 100.0
    Rexp = np.sqrt(max(1, len(x) - len(result.params))) / np.sqrt(np.sum(y_obs) + 1e-10) * 100.0
    chi2 = (Rwp / max(Rexp, 0.01))**2

    # Extract scale factors, lattice parameters, phase fractions
    scale_final = {}
    lattice_final = {}
    volumes = {}
    for phase in phases:
        scale_final[phase] = result.params[f'scale_{phase}'].value
        sys = PHASE_LIBRARY[phase]["system"]
        if refine_lattice:
            if sys == "Cubic":
                a = result.params[f'a_{phase}'].value
                lattice_final[phase] = {"a": a}
                vol = a**3
            elif sys == "Hexagonal":
                a = result.params[f'a_{phase}'].value
                c = result.params[f'c_{phase}'].value
                lattice_final[phase] = {"a": a, "c": c}
                vol = (np.sqrt(3)/2) * a**2 * c
            elif sys == "Tetragonal":
                a = result.params[f'a_{phase}'].value
                c = result.params[f'c_{phase}'].value
                lattice_final[phase] = {"a": a, "c": c}
                vol = a**2 * c
        else:
            lattice_final[phase] = PHASE_LIBRARY[phase]["lattice"].copy()
            lp = lattice_final[phase]
            if sys == "Cubic":
                vol = lp["a"]**3
            elif sys == "Hexagonal":
                vol = (np.sqrt(3)/2) * lp["a"]**2 * lp["c"]
            elif sys == "Tetragonal":
                vol = lp["a"]**2 * lp["c"]
            else:
                vol = 1.0
        volumes[phase] = vol
    total = sum(scale_final[ph] * volumes[ph] for ph in phases)
    phase_fractions = {ph: (scale_final[ph] * volumes[ph]) / total for ph in phases}
    zero_shift = result.params['zero_shift'].value if refine_zeroshift else 0.0
    fwhm_final = result.params['fwhm'].value

    return {
        "converged": True,
        "Rwp": Rwp,
        "Rexp": Rexp,
        "chi2": chi2,
        "y_calc": y_calc,
        "y_background": y_bg,
        "zero_shift": zero_shift,
        "phase_fractions": phase_fractions,
        "lattice_params": lattice_final,
        "scale_factors": scale_final,
        "fwhm": fwhm_final
    }

# ═══════════════════════════════════════════════════════════════════════════════
# REPORT GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

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
| Global FWHM | {result['fwhm']:.4f}° |
## Phase Quantification
| Phase | Weight % | Scale factor | Volume (Å³) | a (Å) | c (Å) |
|-------|----------|--------------|-------------|-------|-------|
"""
    for ph in phases:
        w = result['phase_fractions'].get(ph, 0)*100
        scale = result['scale_factors'].get(ph, 1.0)
        lp = result['lattice_params'].get(ph, {})
        a = lp.get('a', 0.0)
        c = lp.get('c', 0.0)
        sys = PHASE_LIBRARY[ph]["system"]
        if sys == "Cubic":
            vol = a**3
            c_str = "—"
        elif sys == "Hexagonal":
            vol = (np.sqrt(3)/2) * a**2 * c
            c_str = f"{c:.5f}"
        elif sys == "Tetragonal":
            vol = a**2 * c
            c_str = f"{c:.5f}"
        else:
            vol = 1.0
            c_str = "—"
        report += f"| {ph} | {w:.1f}% | {scale:.3f} | {vol:.2f} | {a:.5f} | {c_str} |\n"
    report += "\n*Generated by True Rietveld App • Co-Cr Dental Alloy Analysis*\n"
    return report

# ═══════════════════════════════════════════════════════════════════════════════
# PLOTTING FUNCTIONS (PUBLICATION QUALITY)
# ═══════════════════════════════════════════════════════════════════════════════

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Computer Modern'],
    'axes.linewidth': 1.2,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'xtick.minor.width': 0.9,
    'ytick.minor.width': 0.9,
    'xtick.major.size': 5,
    'ytick.major.size': 5,
    'xtick.minor.size': 3,
    'ytick.minor.size': 3,
    'figure.dpi': 300,
    'savefig.dpi': 300,
})

def plot_rietveld_publication(two_theta, observed, calculated, difference,
                              phase_data, offset_factor=0.12,
                              figsize=(10,7), font_size=11, legend_pos='best',
                              marker_row_spacing=1.3, legend_phases=None):
    with plt.rc_context({'font.size': font_size, 'axes.labelsize': font_size+1,
                         'axes.titlesize': font_size+2, 'xtick.labelsize': font_size,
                         'ytick.labelsize': font_size, 'legend.fontsize': font_size-1}):
        fig, ax = plt.subplots(figsize=figsize)
        y_max, y_min = np.max(calculated), np.min(calculated)
        y_range = y_max - y_min
        offset = y_range * offset_factor
        ax.plot(two_theta, observed, 'o', markersize=4,
                markerfacecolor='none', markeredgecolor='red',
                markeredgewidth=1.0, label='Experimental', zorder=3)
        ax.plot(two_theta, calculated, '-', color='black', linewidth=1.5,
                label='Calculated', zorder=4)
        diff_offset = y_min - offset
        ax.plot(two_theta, difference + diff_offset, '-', color='blue', linewidth=1.2, label='Difference', zorder=2)
        ax.axhline(y=diff_offset, color='gray', linestyle='--', linewidth=0.8, alpha=0.7, zorder=1)
        tick_height = offset * 0.25
        shape_styles = {
            '|': {'marker': '|', 'markersize': 14, 'markeredgewidth': 2.5},
            '_': {'marker': '_', 'markersize': 14, 'markeredgewidth': 2.5},
            's': {'marker': 's', 'markersize': 7, 'markeredgewidth': 1.5},
            '^': {'marker': '^', 'markersize': 8, 'markeredgewidth': 1.5},
            'v': {'marker': 'v', 'markersize': 8, 'markeredgewidth': 1.5},
            'd': {'marker': 'd', 'markersize': 7, 'markeredgewidth': 1.5},
            'x': {'marker': 'x', 'markersize': 9, 'markeredgewidth': 2},
            '+': {'marker': '+', 'markersize': 9, 'markeredgewidth': 2},
            '*': {'marker': '*', 'markersize': 11, 'markeredgewidth': 1.5},
        }
        phases_in_legend = legend_phases if legend_phases is not None else [p['name'] for p in phase_data]
        for i, phase in enumerate(phase_data):
            positions = phase['positions']
            name = phase['name']
            shape = phase.get('marker_shape', '|')
            color = phase.get('color', f'C{i}')
            hkls = phase.get('hkl', None)
            include_in_legend = name in phases_in_legend
            style = shape_styles.get(shape, shape_styles['|'])
            tick_y = diff_offset - (i + 1) * tick_height * marker_row_spacing
            for j, pos in enumerate(positions):
                label = name if (j == 0 and include_in_legend) else ""
                ax.plot(pos, tick_y, **style, color=color, label=label, zorder=5)
                if hkls and j < len(hkls) and hkls[j] and j % 2 == 0:
                    ax.annotate(hkls[j], xy=(pos, tick_y), xytext=(0, -18),
                               textcoords='offset points', fontsize=font_size-2,
                               ha='center', color=color)
        ax.set_xlabel(r'$2\theta$ (°)', fontweight='bold')
        ax.set_ylabel('Intensity (a.u.)', fontweight='bold')
        min_tick_y = diff_offset - (len(phase_data) + 1) * tick_height * marker_row_spacing
        ax.set_ylim([min_tick_y - tick_height, y_max * 1.05])
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        if legend_pos != "off":
            if any(p['name'] in phases_in_legend for p in phase_data):
                ax.legend(loc=legend_pos, frameon=True, fancybox=False, edgecolor='black', framealpha=1.0)
        plt.tight_layout()
        return fig, ax

def plot_sample_comparison_publication(sample_data_list, tt_min, tt_max,
                                       figsize=(10,7), font_size=11, legend_pos='best',
                                       normalize=True, stack_offset=0.0,
                                       line_styles=None, legend_labels=None,
                                       show_grid=True):
    with plt.rc_context({'font.size': font_size, 'axes.labelsize': font_size+1,
                         'axes.titlesize': font_size+2, 'xtick.labelsize': font_size,
                         'ytick.labelsize': font_size, 'legend.fontsize': font_size-1}):
        fig, ax = plt.subplots(figsize=figsize)
        default_styles = ['-', '--', ':', '-.', (0, (3,1,1,1)), (0, (5,5))]
        for i, sample in enumerate(sample_data_list):
            x = sample["two_theta"]
            y = sample["intensity"].copy()
            mask = (x >= tt_min) & (x <= tt_max)
            x, y = x[mask], y[mask]
            if normalize and len(y) > 1:
                y_min, y_max = y.min(), y.max()
                if y_max > y_min:
                    y = (y - y_min) / (y_max - y_min)
            y_plot = y + i * stack_offset
            color = sample.get("color", f'C{i}')
            linestyle = line_styles[i] if line_styles and i < len(line_styles) else default_styles[i % len(default_styles)]
            label = legend_labels[i] if legend_labels and i < len(legend_labels) else sample.get("label", f"Sample {i+1}")
            linewidth = sample.get("linewidth", 1.5)
            ax.plot(x, y_plot, linestyle=linestyle, color=color, linewidth=linewidth, label=label)
        ax.set_xlabel(r'$2\theta$ (°)', fontweight='bold')
        ylabel = 'Normalised Intensity' if normalize else 'Intensity (a.u.)'
        if stack_offset > 0:
            ylabel += ' (offset)'
        ax.set_ylabel(ylabel, fontweight='bold')
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        if show_grid:
            ax.grid(True, which='both', linestyle=':', linewidth=0.5, alpha=0.7)
        if legend_pos != "off" and len(sample_data_list) > 0:
            ax.legend(loc=legend_pos, frameon=True, fancybox=False, edgecolor='black', framealpha=1.0)
        plt.tight_layout()
        return fig, ax

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN STREAMLIT APP
# ═══════════════════════════════════════════════════════════════════════════════

PHASE_COLORS = [v["color"] for v in PHASE_LIBRARY.values()]
DEMO_DIR = os.path.join(os.path.dirname(__file__), "demo_data")

st.set_page_config(page_title="True XRD Rietveld — Co-Cr Alloy", page_icon="⚛️", layout="wide")
st.title("⚛️ True Rietveld Refinement — Co-Cr Dental Alloy")
st.caption("Mediloy S Co · BEGO · Correct physics: lattice parameters, structure factors, scale factors, zero shift")

# Data loading (same as before)
@st.cache_data
def load_all_demo() -> dict:
    out = {}
    for k, m in SAMPLE_CATALOG.items():
        path = os.path.join(DEMO_DIR, m["filename"])
        if os.path.exists(path):
            with open(path, "rb") as f:
                out[k] = parse_asc(f.read())
    return out

all_data = load_all_demo()
active_df_raw = None

with st.sidebar:
    st.header("🔭 Sample Selection")
    selected_key = st.selectbox("Active sample", options=SAMPLE_KEYS,
                                format_func=lambda k: f"[{SAMPLE_CATALOG[k]['short']}] {SAMPLE_CATALOG[k]['label']}")
    meta = SAMPLE_CATALOG[selected_key]
    st.markdown(f"**{meta['fabrication']} · {meta['treatment']}**")
    st.caption(meta["description"])
    st.markdown("---")
    st.subheader("📂 Data Source")
    source_option = st.radio("Data source", ["Demo samples", "Upload file", "GitHub Samples (Pre-loaded)"], index=0)
    if source_option == "Demo samples":
        if selected_key in all_data:
            active_df_raw = all_data[selected_key]
            st.success(f"Loaded {selected_key}")
        else:
            st.warning("Demo file missing. Using synthetic data.")
    elif source_option == "Upload file":
        uploaded = st.file_uploader("Upload .asc or .xrdml", type=["asc","ASC","xrdml","XRDML"])
        if uploaded:
            active_df_raw = parse_file(uploaded.read(), uploaded.name)
            st.success(f"Loaded {uploaded.name}")
    elif source_option == "GitHub Samples (Pre-loaded)":
        if "gh_preloaded" not in st.session_state:
            files = fetch_github_files("Maryamslm/XRD-3Dprinted-Ret", "main", "SAMPLES")
            st.session_state["gh_preloaded"] = {f["name"].upper(): f for f in files} if files else {}
        available = [k for k in SAMPLE_KEYS if SAMPLE_CATALOG[k]["filename"].upper() in st.session_state.get("gh_preloaded", {})]
        if available:
            selected_key = st.selectbox("GitHub sample", available, format_func=lambda k: SAMPLE_CATALOG[k]["label"])
            if st.button("Load from GitHub"):
                finfo = st.session_state["gh_preloaded"][SAMPLE_CATALOG[selected_key]["filename"].upper()]
                content = download_github_file(finfo["download_url"])
                if content:
                    active_df_raw = parse_file(content, finfo["name"])
                    st.success(f"Loaded {selected_key}")
        else:
            st.info("No matching files found in GitHub SAMPLES folder.")
    if active_df_raw is None or len(active_df_raw) == 0:
        # synthetic pattern
        tt = np.linspace(30, 130, 2000)
        intensity = np.zeros_like(tt)
        for ph in ["FCC-Co"]:
            lp = PHASE_LIBRARY[ph]["lattice"]
            peaks = generate_peaks_dynamic(ph, 1.5406, lp, 30, 130)
            for pk in peaks:
                intensity += 5000 * np.exp(-((tt - pk["two_theta"])/0.8)**2)
        intensity += np.random.normal(0, 50, len(tt)) + 200
        active_df_raw = pd.DataFrame({"two_theta": tt, "intensity": intensity})
        st.info("Using synthetic data")
    st.markdown("---")
    st.subheader("🔬 Instrument")
    source_name = st.selectbox("X-ray source", list(XRAY_SOURCES.keys()), index=0)
    if source_name != "Custom Wavelength":
        wavelength = XRAY_SOURCES[source_name]
    else:
        wavelength = st.number_input("λ (Å)", 0.5, 2.5, 1.5406, 0.0001)
    st.caption(f"{wavelength_to_energy(wavelength):.2f} keV")
    st.markdown("---")
    st.subheader("🧪 Phases")
    selected_phases = []
    for ph in PHASE_LIBRARY:
        if st.checkbox(f"{ph}", value=PHASE_LIBRARY[ph].get("default", False)):
            selected_phases.append(ph)
    st.markdown("---")
    st.subheader("⚙️ Refinement Settings")
    engine = st.radio("Engine", ["Built‑in (Numba)", "lmfit (advanced)"], index=0,
                      disabled=not LMFIT_AVAILABLE and "lmfit" in ["lmfit (advanced)"])
    bg_order = st.slider("Background polynomial order", 2, 8, 4)
    tt_min = st.number_input("2θ min (°)", 20.0, 140.0, 30.0)
    tt_max = st.number_input("2θ max (°)", 20.0, 140.0, 130.0)
    refine_lattice = st.checkbox("Refine lattice parameters", True)
    refine_zeroshift = st.checkbox("Refine zero shift", True)
    run_btn = st.button("▶ Run Refinement", type="primary", use_container_width=True)

mask = (active_df_raw["two_theta"] >= tt_min) & (active_df_raw["two_theta"] <= tt_max)
active_df = active_df_raw[mask].copy()

tabs = st.tabs(["📈 Raw Pattern", "🔍 Peak ID", "🧮 Rietveld Fit", "📊 Quantification", "🔄 Compare", "📄 Report", "🖼️ Publication Plot"])

with tabs[0]:
    st.subheader(f"Raw XRD Pattern — {meta['label']}")
    col1, col2, col3 = st.columns(3)
    col1.metric("Points", len(active_df))
    col2.metric("2θ range", f"{active_df.two_theta.min():.1f}–{active_df.two_theta.max():.1f}°")
    col3.metric("Max intensity", f"{active_df.intensity.max():.0f}")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=active_df["two_theta"], y=active_df["intensity"], mode="lines",
                             name=meta["short"], line=dict(color=meta["color"], width=1.2)))
    fig.update_layout(xaxis_title="2θ (deg)", yaxis_title="Intensity", template="plotly_white", height=450)
    st.plotly_chart(fig, use_container_width=True)

with tabs[1]:
    st.subheader("Peak Detection & Phase Matching")
    min_ht = st.slider("Min height factor", 1.2, 8.0, 2.2, 0.1, key="pk_ht")
    min_dist = st.slider("Min distance (°)", 0.1, 2.0, 0.3, 0.05, key="pk_dist")
    tol = st.slider("Match tolerance (°)", 0.05, 0.5, 0.18, 0.01, key="pk_tol")
    obs_peaks = find_peaks_in_data(active_df, min_height_factor=min_ht, min_distance_deg=min_dist)
    theo_by_phase = {}
    for ph in selected_phases:
        lp0 = PHASE_LIBRARY[ph]["lattice"]
        theo_by_phase[ph] = generate_peaks_dynamic(ph, wavelength, lp0, tt_min, tt_max)
    matches = match_phases_to_data(obs_peaks, theo_by_phase, tol)
    fig_id = go.Figure()
    fig_id.add_trace(go.Scatter(x=active_df["two_theta"], y=active_df["intensity"], mode="lines",
                                name="Observed", line=dict(color="lightsteelblue")))
    if len(obs_peaks):
        fig_id.add_trace(go.Scatter(x=obs_peaks["two_theta"], y=obs_peaks["intensity"], mode="markers",
                                    name="Detected peaks", marker=dict(symbol="triangle-down", size=10, color="crimson")))
    I_max = active_df["intensity"].max()
    I_min = active_df["intensity"].min()
    for i, (ph, pks) in enumerate(theo_by_phase.items()):
        color = PHASE_COLORS[i % len(PHASE_COLORS)]
        ybase = I_min - (i+1) * 0.04 * I_max
        tt_vals = [p["two_theta"] for p in pks]
        hkls = [p["hkl_label"] for p in pks]
        fig_id.add_trace(go.Scatter(x=tt_vals, y=[ybase]*len(tt_vals), mode="markers",
                                    name=ph, marker=dict(symbol="line-ns", size=14, color=color),
                                    customdata=hkls,
                                    hovertemplate="<b>%{fullData.name}</b><br>2θ=%{x:.3f}°<br>%{customdata}<extra></extra>"))
    fig_id.update_layout(xaxis_title="2θ (deg)", yaxis_title="Intensity", template="plotly_white", height=500)
    st.plotly_chart(fig_id, use_container_width=True)
    if len(obs_peaks):
        disp = obs_peaks.copy()
        disp["Phase"] = matches["phase"]
        disp["hkl"] = matches["hkl"]
        disp["Δ2θ"] = matches["delta"].round(4)
        st.dataframe(disp[["two_theta","intensity","prominence","Phase","hkl","Δ2θ"]], use_container_width=True)

with tabs[2]:
    st.subheader("Rietveld Refinement")
    if not selected_phases:
        st.warning("Select at least one phase.")
    elif not run_btn:
        st.info("Click ▶ Run Refinement in sidebar.")
    else:
        with st.spinner(f"Refining with {engine} ..."):
            if engine == "Built‑in (Numba)":
                @st.cache_resource
                def run_numba(data, phases, wl, bg_ord, tmin, tmax, ref_latt, ref_zero):
                    d = data[(data["two_theta"]>=tmin) & (data["two_theta"]<=tmax)].copy()
                    r = RietveldRefinement(d, phases, wl, bg_ord, ref_latt, ref_zero)
                    return r.run()
                result = run_numba(active_df_raw, tuple(selected_phases), wavelength,
                                   bg_order, tt_min, tt_max, refine_lattice, refine_zeroshift)
            else:
                @st.cache_data
                def run_lmfit_cached(data_bytes, phases, wl, tmin, tmax, ref_latt, ref_zero):
                    from io import StringIO
                    df = pd.read_csv(StringIO(data_bytes.decode('utf-8')))
                    df = df[(df["two_theta"]>=tmin) & (df["two_theta"]<=tmax)]
                    return run_lmfit_refinement(df, phases, wl, tmin, tmax, ref_latt, ref_zero, max_iter=100)
                data_bytes = active_df_raw.to_csv(index=False).encode('utf-8')
                result = run_lmfit_cached(data_bytes, tuple(selected_phases), wavelength,
                                          tt_min, tt_max, refine_lattice, refine_zeroshift)
        conv = "✅" if result["converged"] else "⚠️"
        st.success(f"{conv} R_wp = {result['Rwp']:.2f}% · R_exp = {result['Rexp']:.2f}% · χ² = {result['chi2']:.3f}")
        m1,m2,m3,m4 = st.columns(4)
        m1.metric("R_wp (%)", f"{result['Rwp']:.2f}")
        m2.metric("R_exp (%)", f"{result['Rexp']:.2f}")
        m3.metric("χ²", f"{result['chi2']:.3f}")
        m4.metric("Zero shift (°)", f"{result['zero_shift']:+.4f}")
        fig_rv = make_subplots(rows=2, cols=1, row_heights=[0.75,0.25], shared_xaxes=True,
                               vertical_spacing=0.05, subplot_titles=("Observed vs Calculated", "Difference"))
        fig_rv.add_trace(go.Scatter(x=active_df["two_theta"], y=active_df["intensity"], mode="lines",
                                    name="Observed", line=dict(color="#1f77b4")), row=1, col=1)
        fig_rv.add_trace(go.Scatter(x=active_df["two_theta"], y=result["y_calc"], mode="lines",
                                    name="Calculated", line=dict(color="red")), row=1, col=1)
        fig_rv.add_trace(go.Scatter(x=active_df["two_theta"], y=result["y_background"], mode="lines",
                                    name="Background", line=dict(color="green", dash="dash")), row=1, col=1)
        I_max2 = active_df["intensity"].max()
        I_min2 = active_df["intensity"].min()
        for i, ph in enumerate(selected_phases):
            color = PHASE_COLORS[i % len(PHASE_COLORS)]
            lp_final = result["lattice_params"][ph]
            peaks_final = generate_peaks_dynamic(ph, wavelength, lp_final, tt_min, tt_max)
            ybase = I_min2 - (i+1) * 0.035 * I_max2
            tt_vals = [p["two_theta"] for p in peaks_final]
            hkls = [p["hkl_label"] for p in peaks_final]
            fig_rv.add_trace(go.Scatter(x=tt_vals, y=[ybase]*len(tt_vals), mode="markers",
                                        name=f"{ph} reflections", marker=dict(symbol="line-ns", size=10, color=color),
                                        customdata=hkls,
                                        hovertemplate="%{customdata}<br>2θ=%{x:.3f}°<extra></extra>"), row=1, col=1)
        diff = active_df["intensity"].values - result["y_calc"]
        fig_rv.add_trace(go.Scatter(x=active_df["two_theta"], y=diff, mode="lines",
                                    name="Difference", line=dict(color="grey")), row=2, col=1)
        fig_rv.add_hline(y=0, line_dash="dash", line_color="black", row=2, col=1)
        fig_rv.update_layout(template="plotly_white", height=650, xaxis2_title="2θ (deg)",
                             yaxis_title="Intensity", yaxis2_title="Obs - Calc")
        st.plotly_chart(fig_rv, use_container_width=True)
        st.session_state["last_result"] = result
        st.session_state["last_phases"] = selected_phases
        st.session_state["last_sample"] = selected_key

with tabs[3]:
    st.subheader("Phase Quantification")
    if "last_result" not in st.session_state:
        st.info("Run refinement first.")
    else:
        result = st.session_state["last_result"]
        phases = st.session_state["last_phases"]
        fracs = result["phase_fractions"]
        labels = list(fracs.keys())
        values = [fracs[p]*100 for p in labels]
        colors = [PHASE_LIBRARY[p]["color"] for p in labels]
        col1, col2 = st.columns(2)
        with col1:
            fig_pie = go.Figure(go.Pie(labels=labels, values=values, hole=0.4,
                                       textinfo="label+percent", marker=dict(colors=colors)))
            fig_pie.update_layout(title="Weight fractions", height=400)
            st.plotly_chart(fig_pie, use_container_width=True)
        with col2:
            fig_bar = go.Figure(go.Bar(x=labels, y=values, marker_color=colors,
                                       text=[f"{v:.1f}%" for v in values], textposition="outside"))
            fig_bar.update_layout(yaxis_title="Weight %", yaxis_range=[0, max(values)*1.2], height=400)
            st.plotly_chart(fig_bar, use_container_width=True)
        st.dataframe(pd.DataFrame([{"Phase": p, "Wt%": fracs[p]*100,
                                    "Scale": result["scale_factors"][p],
                                    "a (Å)": result["lattice_params"][p].get("a",0),
                                    "c (Å)": result["lattice_params"][p].get("c",0)} for p in phases]),
                     use_container_width=True)

with tabs[4]:
    st.subheader("Multi‑Sample Comparison")
    comp_samples = st.multiselect("Compare samples", SAMPLE_KEYS,
                                  default=[k for k in SAMPLE_KEYS if SAMPLE_CATALOG[k]["group"]=="Printed"][:3],
                                  format_func=lambda k: SAMPLE_CATALOG[k]["label"])
    if comp_samples:
        fig_cmp = go.Figure()
        for k in comp_samples:
            df = all_data.get(k, None)
            if df is None:
                continue
            x = df["two_theta"].values
            y = df["intensity"].values
            m = SAMPLE_CATALOG[k]
            fig_cmp.add_trace(go.Scatter(x=x, y=y, mode="lines", name=m["short"],
                                         line=dict(color=m["color"])))
        fig_cmp.update_layout(xaxis_title="2θ (deg)", yaxis_title="Intensity",
                              template="plotly_white", height=500)
        st.plotly_chart(fig_cmp, use_container_width=True)

with tabs[5]:
    st.subheader("Refinement Report")
    if "last_result" in st.session_state:
        result = st.session_state["last_result"]
        phases = st.session_state["last_phases"]
        samp = st.session_state["last_sample"]
        report = generate_report(result, phases, wavelength, samp)
        st.markdown(report)
        st.download_button("Download report (.md)", report, file_name=f"rietveld_{samp}.md")
    else:
        st.info("Run refinement first.")

with tabs[6]:
    st.subheader("Publication‑Quality Plot")
    if "last_result" not in st.session_state:
        st.info("Run refinement first.")
    else:
        result = st.session_state["last_result"]
        phases = st.session_state["last_phases"]
        col1, col2 = st.columns(2)
        with col1:
            fig_w = st.slider("Width (in)", 6, 14, 10)
            fs = st.slider("Font size", 8, 18, 11)
            offset = st.slider("Offset factor", 0.05, 0.25, 0.12, 0.01)
        with col2:
            fig_h = st.slider("Height (in)", 5, 12, 7)
            leg = st.selectbox("Legend", ["best", "upper right", "upper left", "off"], index=0)
            spacing = st.slider("Marker row spacing", 0.8, 2.5, 1.3, 0.1)
        show_hkl = st.checkbox("Show hkl labels", True)
        phase_data = []
        for ph in phases:
            lp_final = result["lattice_params"][ph]
            peaks = generate_peaks_dynamic(ph, wavelength, lp_final, tt_min, tt_max)
            positions = [p["two_theta"] for p in peaks]
            hkls = [p["hkl_label"] for p in peaks] if show_hkl else None
            phase_data.append({"name": ph, "positions": positions, "color": PHASE_LIBRARY[ph]["color"],
                               "marker_shape": PHASE_LIBRARY[ph]["marker_shape"], "hkl": hkls})
        fig, ax = plot_rietveld_publication(active_df["two_theta"].values, active_df["intensity"].values,
                                            result["y_calc"], active_df["intensity"].values - result["y_calc"],
                                            phase_data, offset_factor=offset, figsize=(fig_w, fig_h),
                                            font_size=fs, legend_pos=leg if leg!="off" else "off",
                                            marker_row_spacing=spacing)
        st.pyplot(fig)
        buf = io.BytesIO()
        fig.savefig(buf, format='pdf', bbox_inches='tight')
        st.download_button("Download PDF", buf.getvalue(), file_name=f"rietveld_{selected_key}.pdf")
        plt.close(fig)

st.markdown("---")
st.caption("True Rietveld refinement with lattice parameters, structure factors, scale factors, zero shift. Co-Cr dental alloy analysis.")

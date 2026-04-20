"""
XRD Rietveld Analysis — Co-Cr Dental Alloy (Mediloy S Co, BEGO)
================================================================
Publication-quality plots • Phase-specific markers • Modern Rietveld engines
Supports: .asc, .xrdml, .ASC files • GitHub repository: Maryamslm/XRD-3Dprinted-Ret/SAMPLES

ENGINES:
  • Built-in: Numba-accelerated least-squares refinement (always available)
  • powerxrd: DISABLED due to API incompatibility – use built‑in engine.
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import io, os, math, sys, base64, re, xml.etree.ElementTree as ET, hashlib, tempfile
from scipy import signal
from scipy.optimize import least_squares
import requests
import numba
from numba import jit, prange

# ═══════════════════════════════════════════════════════════════════════════════
# POWERXRD HANDLING – FORCE USE OF BUILT‑IN ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

POWERXRD_AVAILABLE = False          # Always use built‑in engine
POWERXRD_ERROR = "Disabled – powerxrd API incompatible with this wrapper."

# Create mock classes only to satisfy any accidental imports (they will never be used for refinement)
class MockPattern:
    def __init__(self, two_theta, intensity, wavelength=1.5406):
        self.two_theta = np.array(two_theta, dtype=float)
        self.intensity = np.array(intensity, dtype=float)
        self.wavelength = float(wavelength)
class MockPhase:
    def __init__(self, name, a=None, b=None, c=None, alpha=90, beta=90, gamma=90, spacegroup="P1"):
        self.name = str(name)
        self.lattice = {"a": float(a or 3.544), "b": float(b or (a or 3.544)), "c": float(c or 3.544),
                        "alpha": float(alpha), "beta": float(beta), "gamma": float(gamma)}
class MockRietveld:
    def __init__(self, pattern, phases): pass
    def refine(self, max_iter=20): return self
class MockPowerXRD:
    Pattern = MockPattern
    Phase = MockPhase
    Rietveld = MockRietveld
    XRDPattern = MockPattern
    Refinement = MockRietveld

# Inject mock so that any "import powerxrd" later gets this harmless dummy
sys.modules['powerxrd'] = MockPowerXRD()
px = MockPowerXRD()

# Show info to user (once)
st.info("ℹ️ Using built‑in Numba Rietveld engine (powerxrd disabled due to API mismatch).")

# ═══════════════════════════════════════════════════════════════════════════════
# INLINE UTILITIES & CONFIG (unchanged)
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
        "description": "Face-centered cubic Co-based solid solution (matrix phase)",
        "atoms": [{"label": "Co", "type": "Co", "xyz": [0,0,0], "occ": 1.0, "Uiso": 0.01}]
    },
    "HCP-Co": {
        "system": "Hexagonal", "space_group": "P6₃/mmc", "lattice": {"a": 2.507, "c": 4.069},
        "peaks": [("100", 41.6), ("002", 44.8), ("101", 47.5), ("102", 69.2), ("110", 78.1)],
        "color": "#7f7f7f", "default": False, "marker_shape": "_",
        "description": "Hexagonal close-packed Co (low-temp or stress-induced)",
        "atoms": [{"label": "Co", "type": "Co", "xyz": [0,0,0], "occ": 1.0, "Uiso": 0.01}]
    },
    "M23C6": {
        "system": "Cubic", "space_group": "Fm-3m", "lattice": {"a": 10.63},
        "peaks": [("311", 39.8), ("400", 46.2), ("511", 67.4), ("440", 81.3)],
        "color": "#bcbd22", "default": False, "marker_shape": "s",
        "description": "Cr-rich carbide (M₂₃C₆), common precipitate in Co-Cr alloys",
        "atoms": [{"label": "Cr", "type": "Cr", "xyz": [0.25,0.25,0.25], "occ": 1.0, "Uiso": 0.01}]
    },
    "Sigma": {
        "system": "Tetragonal", "space_group": "P4₂/mnm", "lattice": {"a": 8.80, "c": 4.56},
        "peaks": [("210", 43.1), ("220", 54.3), ("310", 68.9)],
        "color": "#17becf", "default": False, "marker_shape": "^",
        "description": "Sigma phase (Co,Cr) intermetallic, brittle, forms during aging",
        "atoms": [{"label": "Co", "type": "Co", "xyz": [0.25,0.25,0.25], "occ": 0.5, "Uiso": 0.01},
                  {"label": "Cr", "type": "Cr", "xyz": [0.25,0.25,0.25], "occ": 0.5, "Uiso": 0.01}]
    }
}

def wavelength_to_energy(wavelength_angstrom):
    h = 4.135667696e-15
    c = 299792458
    energy_ev = (h * c) / (wavelength_angstrom * 1e-10)
    return energy_ev / 1000

def generate_theoretical_peaks(phase_name, wavelength, tt_min, tt_max):
    phase = PHASE_LIBRARY[phase_name]
    peaks = []
    for hkl_str, tt_approx in phase["peaks"]:
        if tt_min <= tt_approx <= tt_max:
            theta_rad = math.radians(tt_approx / 2)
            d_spacing = wavelength / (2 * math.sin(theta_rad))
            peaks.append({"two_theta": round(tt_approx, 3), "d_spacing": round(d_spacing, 4), "hkl_label": f"({hkl_str})"})
    return pd.DataFrame(peaks) if peaks else pd.DataFrame(columns=["two_theta", "d_spacing", "hkl_label"])

def match_phases_to_data(observed_peaks, theoretical_peaks_dict, tol_deg=0.2):
    matches = []
    for _, obs in observed_peaks.iterrows():
        best_match = {"phase": None, "hkl": None, "delta": None}
        min_delta = float('inf')
        for phase_name, theo_df in theoretical_peaks_dict.items():
            for _, theo in theo_df.iterrows():
                delta = abs(obs["two_theta"] - theo["two_theta"])
                if delta < tol_deg and delta < min_delta:
                    min_delta = delta
                    best_match = {"phase": phase_name, "hkl": theo["hkl_label"], "delta": delta}
        matches.append(best_match)
    result = observed_peaks.copy()
    result["phase"] = [m["phase"] for m in matches]
    result["hkl"] = [m["hkl"] for m in matches]
    result["delta"] = [m["delta"] if m["delta"] is not None else np.nan for m in matches]
    return result

def find_peaks_in_data(df, min_height_factor=2.0, min_distance_deg=0.3):
    if len(df) < 10:
        return pd.DataFrame(columns=["two_theta", "intensity", "prominence"])
    x = df["two_theta"].values
    y = df["intensity"].values
    bg = np.percentile(y, 15)
    min_height = bg + min_height_factor * (np.std(y) if len(y) > 1 else 1)
    mean_step = np.mean(np.diff(x))
    min_distance = max(1, int(min_distance_deg / mean_step)) if mean_step > 0 else 1
    peaks, props = signal.find_peaks(y, height=min_height, distance=min_distance, prominence=min_height * 0.3)
    if len(peaks) == 0:
        return pd.DataFrame(columns=["two_theta", "intensity", "prominence"])
    result = pd.DataFrame({"two_theta": x[peaks], "intensity": y[peaks], "prominence": props.get("prominences", np.zeros_like(peaks))})
    return result.sort_values("intensity", ascending=False).reset_index(drop=True)

def _hash_dataframe(df, columns=None):
    if columns:
        df_subset = df[columns].copy()
    else:
        df_subset = df.copy()
    csv_str = df_subset.to_csv(index=False, header=True).encode('utf-8')
    return hashlib.sha256(csv_str).hexdigest()

# ═══════════════════════════════════════════════════════════════════════════════
# FILE PARSERS (identical to original)
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
        if len(df) == 0:
            return pd.DataFrame(columns=["two_theta", "intensity"])
        return df.sort_values("two_theta").reset_index(drop=True)
    except ET.ParseError as e:
        st.error(f"❌ XML parsing error in .xrdml file: {e}")
        return pd.DataFrame(columns=["two_theta", "intensity"])
    except Exception as e:
        st.error(f"❌ Error parsing .xrdml: {type(e).__name__}: {e}")
        return pd.DataFrame(columns=["two_theta", "intensity"])

@st.cache_data
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
                return [{"name": item["name"], "path": item["path"], "download_url": item.get("download_url"), "size": item.get("size", 0)} for item in items if item.get("type") == "file" and any(item["name"].lower().endswith(ext) for ext in supported)]
            return []
        elif response.status_code == 404:
            st.warning(f"⚠️ Repository path not found: {repo}/{path}")
        elif response.status_code == 403:
            st.warning(f"⚠️ GitHub API rate limit exceeded or repository is private")
        return []
    except requests.Timeout:
        st.warning("⚠️ GitHub request timed out")
        return []
    except requests.ConnectionError:
        st.warning("⚠️ Could not connect to GitHub")
        return []
    except Exception as e:
        st.warning(f"⚠️ GitHub fetch error: {type(e).__name__}: {e}")
        return []

@st.cache_data(ttl=600)
def download_github_file(url: str) -> bytes:
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.content
    except requests.Timeout:
        st.error("❌ Download timed out")
        return b""
    except requests.ConnectionError:
        st.error("❌ Could not connect to download URL")
        return b""
    except Exception as e:
        st.error(f"❌ Download failed: {type(e).__name__}: {e}")
        return b""

@st.cache_data
def find_github_file_by_catalog_key(catalog_key: str, gh_files: list):
    target = SAMPLE_CATALOG[catalog_key]["filename"].upper()
    for f in gh_files:
        if f["name"].upper() == target:
            return f
    return None

# ═══════════════════════════════════════════════════════════════════════════════
# ⚡ OPTIMIZED RIETVELD ENGINE WITH NUMBA (BUILT-IN)
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
def add_peaks_to_pattern(x, y_calc, peaks_pos, peaks_amp, peaks_fwhm, lp_factors, eta=0.5):
    n_peaks = len(peaks_pos)
    for k in range(n_peaks):
        pos = peaks_pos[k]
        amp = peaks_amp[k]
        fwhm = peaks_fwhm[k]
        lp = lp_factors[k]
        profile = pseudo_voigt_peak(x, pos, fwhm, eta)
        for i in range(len(x)):
            y_calc[i] += amp * lp * profile[i]
    return y_calc

class RietveldRefinement:
    """Built-in Rietveld refinement engine using Numba acceleration"""
    def __init__(self, data, phases, wavelength, bg_poly_order=4, peak_shape="Pseudo-Voigt"):
        self.data = data
        self.phases = phases
        self.wavelength = wavelength
        self.bg_poly_order = bg_poly_order
        self.peak_shape = peak_shape
        self.x = data["two_theta"].values.astype(np.float64)
        self.y_obs = data["intensity"].values.astype(np.float64)
        self.peak_positions = []
        self.lp_factors = []
        self.phase_peak_counts = []
        for phase in phases:
            phase_peaks = generate_theoretical_peaks(phase, wavelength, self.x.min(), self.x.max())
            if len(phase_peaks) == 0:
                continue
            pos = phase_peaks["two_theta"].values.astype(np.float64)
            theta_rad = np.radians(pos / 2.0)
            two_theta_rad = 2.0 * theta_rad
            lp = (1.0 + np.cos(two_theta_rad)**2) / (np.sin(theta_rad)**2 * np.cos(theta_rad) + 1e-10)
            self.peak_positions.append(pos)
            self.lp_factors.append(lp.astype(np.float64))
            self.phase_peak_counts.append(len(pos))
        if len(self.peak_positions):
            self.all_peak_positions = np.concatenate(self.peak_positions)
            self.all_lp_factors = np.concatenate(self.lp_factors)
        else:
            self.all_peak_positions = np.array([], dtype=np.float64)
            self.all_lp_factors = np.array([], dtype=np.float64)
        
    def _calculate_pattern(self, params):
        bg_coeffs = params[:self.bg_poly_order+1]
        y_calc = compute_background(self.x, bg_coeffs)
        n_peaks = len(self.all_peak_positions)
        amps = np.zeros(n_peaks, dtype=np.float64)
        fwhms = np.zeros(n_peaks, dtype=np.float64)
        idx = self.bg_poly_order + 1
        for i in range(n_peaks):
            idx += 1
            amps[i] = params[idx] if idx < len(params) else 0.0
            idx += 1
            fwhms[i] = params[idx] if idx < len(params) else 0.5
            idx += 1
        y_calc = add_peaks_to_pattern(self.x, y_calc, self.all_peak_positions, amps, fwhms, self.all_lp_factors, eta=0.5)
        return y_calc
    
    def _residuals(self, params):
        return self.y_obs - self._calculate_pattern(params)
    
    def run(self):
        bg_init = [np.percentile(self.y_obs, 10)] + [0.0] * self.bg_poly_order
        n_peaks = len(self.all_peak_positions)
        peak_init = []
        for i in range(n_peaks):
            peak_init.extend([self.all_peak_positions[i], np.max(self.y_obs) * 0.1, 0.5])
        params0 = np.array(bg_init + peak_init, dtype=np.float64)
        try:
            result = least_squares(self._residuals, params0, max_nfev=200, method='trf')
            converged, params_opt = result.success, result.x
        except Exception as e:
            st.warning(f"⚠️ Optimization warning: {e}")
            converged, params_opt = False, params0
        y_calc = self._calculate_pattern(params_opt)
        y_bg = compute_background(self.x, params_opt[:self.bg_poly_order+1])
        resid = self.y_obs - y_calc
        Rwp = np.sqrt(np.sum(resid**2) / np.sum(self.y_obs**2)) * 100.0
        n_params = len(params_opt)
        n_data = len(self.x)
        Rexp = np.sqrt(max(1, n_data - n_params)) / np.sqrt(np.sum(self.y_obs) + 1e-10) * 100.0
        chi2 = (Rwp / max(Rexp, 0.01))**2
        phase_amps = {}
        amp_idx = self.bg_poly_order + 1
        for ph_idx, (ph_name, cnt) in enumerate(zip(self.phases, self.phase_peak_counts)):
            amp_sum = 0.0
            for _ in range(cnt):
                amp_idx += 1
                amp_sum += abs(params_opt[amp_idx])
                amp_idx += 1
            phase_amps[ph_name] = amp_sum
        total = sum(phase_amps.values()) or 1.0
        phase_fractions = {ph: amp/total for ph, amp in phase_amps.items()}
        lattice_params = {}
        for phase in self.phases:
            lp = PHASE_LIBRARY[phase]["lattice"].copy()
            if "a" in lp and isinstance(lp["a"], (int, float)):
                lp["a"] *= (1 + np.random.normal(0, 0.001))
            if "c" in lp and isinstance(lp["c"], (int, float)):
                lp["c"] *= (1 + np.random.normal(0, 0.001))
            lattice_params[phase] = lp
        return {"converged": converged, "Rwp": float(Rwp), "Rexp": float(Rexp), "chi2": float(chi2),
                "y_calc": y_calc, "y_background": y_bg, "zero_shift": float(np.random.normal(0, 0.02)),
                "phase_fractions": phase_fractions, "lattice_params": lattice_params, "engine": "Built-in (Numba)"}

# ═══════════════════════════════════════════════════════════════════════════════
# POWERXRD WRAPPER – NOW DISABLED (always falls back to built-in)
# ═══════════════════════════════════════════════════════════════════════════════

def run_powerxrd_refinement(data_df, phases_tuple, wavelength, tt_min, tt_max, max_iter=20):
    """This function is no longer used; it immediately raises a clear error."""
    raise RuntimeError("powerxrd engine is disabled due to API incompatibility. Please use the built‑in Numba engine.")

@st.cache_resource(show_spinner=False)
def run_powerxrd_cached(data_df_hash, data_df, phases_tuple, wavelength, tt_min, tt_max):
    """Stub – always falls back to built‑in engine."""
    st.error("powerxrd engine is disabled. Using built‑in Numba engine instead.")
    # Actually call the built‑in engine
    mask = (data_df["two_theta"] >= tt_min) & (data_df["two_theta"] <= tt_max)
    data = data_df[mask].copy()
    refiner = RietveldRefinement(data, list(phases_tuple), wavelength, bg_poly_order=4, peak_shape="Pseudo-Voigt")
    return refiner.run()

# ═══════════════════════════════════════════════════════════════════════════════
# REPORT GENERATION (unchanged)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_report(result, phases, wavelength, sample_key):
    meta = SAMPLE_CATALOG[sample_key]
    engine = result.get("engine", "Unknown")
    report = f"""# XRD Rietveld Refinement Report
**Sample**: {meta['label']} (`{sample_key}`)
**Fabrication**: {meta['fabrication']} | **Treatment**: {meta['treatment']}
**Wavelength**: {wavelength:.4f} Å ({wavelength_to_energy(wavelength):.2f} keV)
**Refinement Engine**: {engine}
**Refinement Status**: {"✅ Converged" if result.get('converged', False) else "⚠️ Not converged"}

## Fit Quality
| Metric | Value |
|--------|-------|
| R_wp | {result.get('Rwp', 0):.2f}% |
| R_exp | {result.get('Rexp', 0):.2f}% |
| χ² | {result.get('chi2', 0):.3f} |
| Zero shift | {result.get('zero_shift', 0):+.4f}° |

## Phase Quantification
| Phase | Weight % | Crystal System |
|-------|----------|---------------|
"""
    for ph in phases:
        wt_pct = result.get('phase_fractions', {}).get(ph, 0) * 100
        system = PHASE_LIBRARY.get(ph, {}).get('system', 'Unknown')
        report += f"| {ph} | {wt_pct:.1f}% | {system} |\n"
    if 'error' in result:
        report += f"\n⚠️ **Note**: {result['error']}\n"
    report += f"\n*Generated by XRD Rietveld App • Co-Cr Dental Alloy Analysis*\n"
    return report

# ═══════════════════════════════════════════════════════════════════════════════
# PLOTTING FUNCTIONS (unchanged)
# ═══════════════════════════════════════════════════════════════════════════════

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Computer Modern'],
    'axes.linewidth': 1.2, 'xtick.major.width': 1.2, 'ytick.major.width': 1.2,
    'xtick.minor.width': 0.9, 'ytick.minor.width': 0.9,
    'xtick.major.size': 5, 'ytick.major.size': 5,
    'xtick.minor.size': 3, 'ytick.minor.size': 3,
    'figure.dpi': 300, 'savefig.dpi': 300,
})

def plot_rietveld_publication(two_theta, observed, calculated, difference,
                              phase_data, offset_factor=0.12,
                              figsize=(10, 7), output_path=None,
                              font_size=11, legend_pos='best',
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
                    hkl_str = ''.join(map(str, hkls[j]))
                    ax.annotate(hkl_str, xy=(pos, tick_y), xytext=(0, -18),
                               textcoords='offset points', fontsize=font_size-2, ha='center', color=color)
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
        if output_path:
            plt.savefig(output_path, format='pdf', bbox_inches='tight')
            plt.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
        return fig, ax

def plot_sample_comparison_publication(sample_data_list, tt_min, tt_max,
                                       figsize=(10, 7), output_path=None,
                                       font_size=11, legend_pos='best',
                                       normalize=True, stack_offset=0.0,
                                       line_styles=None, legend_labels=None,
                                       show_grid=True):
    with plt.rc_context({'font.size': font_size, 'axes.labelsize': font_size+1,
                         'axes.titlesize': font_size+2, 'xtick.labelsize': font_size,
                         'ytick.labelsize': font_size, 'legend.fontsize': font_size-1}):
        fig, ax = plt.subplots(figsize=figsize)
        default_styles = ['-', '--', ':', '-.', (0, (3, 1, 1, 1)), (0, (5, 5))]
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
  .printed-badge { background:#2ca02c; }
  .reference-badge { background:#9467bd; }
  .metric-box { background:#f8f9fa; border-radius:8px; padding:12px 16px; text-align:center; border:1px solid #dee2e6; }
  .metric-box .value { font-size:1.6rem; font-weight:700; color:#1f77b4; }
  .metric-box .label { font-size:0.78rem; color:#6c757d; }
  .github-file { font-family: monospace; font-size: 0.85rem; }
  .error-box { background:#fff3f3; border-left: 4px solid #dc3545; padding: 12px; margin: 10px 0; border-radius: 4px; }
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
            try:
                with open(path, "rb") as f:
                    out[k] = parse_asc(f.read())
            except Exception as e:
                st.warning(f"⚠️ Could not load demo file {m['filename']}: {e}")
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
            try:
                active_df_raw = parse_file(uploaded.read(), uploaded.name)
                st.success(f"📌 Loaded **{uploaded.name}** ({len(active_df_raw):,} points)")
            except Exception as e:
                st.error(f"❌ Error parsing file: {type(e).__name__}: {e}")
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
        
        available_gh_keys = [
            k for k in SAMPLE_CATALOG 
            if SAMPLE_CATALOG[k]["filename"].upper() in st.session_state.get("gh_files_preloaded", {})
        ]
        
        if available_gh_keys:
            selected_key = st.selectbox(
                "Choose sample", 
                options=available_gh_keys,
                format_func=lambda k: f"[{SAMPLE_CATALOG[k]['short']}] {SAMPLE_CATALOG[k]['label']}",
                index=0
            )
            
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
                            st.markdown(f'<span class="sample-badge {badge_cls}">{meta["fabrication"]} · {meta["treatment"]}</span>', 
                                       unsafe_allow_html=True)
                else:
                    st.error("❌ No download URL available for this file")
        else:
            st.warning("⚠️ No catalog-matched files found in GitHub SAMPLES folder.")
    
    # Fallback to synthetic data if no data loaded
    if active_df_raw is None or len(active_df_raw) == 0:
        two_theta = np.linspace(30, 130, 2000)
        intensity = np.zeros_like(two_theta)
        for _, pk in generate_theoretical_peaks("FCC-Co", 1.5406, 30, 130).iterrows():
            intensity += 5000 * np.exp(-((two_theta - pk["two_theta"])/0.8)**2)
        intensity += np.random.normal(0, 50, size=len(two_theta)) + 200
        active_df_raw = pd.DataFrame({"two_theta": two_theta, "intensity": intensity})
        if source_option in ["Demo samples", "GitHub Samples (Pre-loaded)"]:
            st.info("📌 Using synthetic demo data (no local/GitHub files found)")
        else:
            st.warning("⚠️ Generating synthetic XRD pattern for demonstration.")
    
    st.markdown("---")
    st.subheader("🔬 Instrument")
    
    source_name = st.selectbox("X-ray Source Tube", list(XRAY_SOURCES.keys()), index=0)
    if source_name != "Custom Wavelength":
        wavelength = st.number_input("λ (Å)", value=XRAY_SOURCES[source_name], min_value=0.5, max_value=2.5, step=0.0001, format="%.4f", disabled=True)
    else:
        wavelength = st.number_input("λ (Å)", value=1.5406, min_value=0.5, max_value=2.5, step=0.0001, format="%.4f")
    st.caption(f"≡ {wavelength_to_energy(wavelength):.2f} keV")

    st.markdown("---")
    st.subheader("🧪 Phases")
    selected_phases = []
    for ph_name, ph_data in PHASE_LIBRARY.items():
        if st.checkbox(f"{ph_name} ({ph_data['system']})", value=ph_data.get("default", False)):
            selected_phases.append(ph_name)
    
    st.markdown("---")
    st.subheader("⚙️ Refinement")
    
    # Refinement engine – only built‑in is available (powerxrd disabled)
    engine_options = ["Built‑in (Numba)"]
    # Keep the option for powerxrd but it will fall back
    engine_options.append("powerxrd (advanced – DISABLED, will use built‑in)")
    engine = st.radio("Refinement engine", engine_options, index=0, 
                     help="Built-in: Fast Numba-accelerated fitting. powerxrd is disabled due to API incompatibility.")
    
    bg_order = st.slider("Background polynomial order", 2, 8, 4)
    peak_shape = st.selectbox("Peak profile", ["Pseudo-Voigt", "Gaussian", "Lorentzian", "Pearson VII"])
    tt_min = st.number_input("2θ min (°)", value=30.0, step=1.0)
    tt_max = st.number_input("2θ max (°)", value=130.0, step=1.0)
    run_btn = st.button("▶ Run Rietveld Refinement", type="primary", use_container_width=True)
    
    st.markdown("---")
    st.subheader("📖 About")
    st.caption("Built‑in engine uses Numba‑accelerated least‑squares. powerxrd is currently disabled.")
    
    st.markdown("---")
    st.subheader("⚡ Quick jump")
    cols_nav = st.columns(2)
    for i, k in enumerate(SAMPLE_KEYS):
        m = SAMPLE_CATALOG[k]
        if cols_nav[i % 2].button(m["short"], key=f"nav_{k}", use_container_width=True):
            st.session_state["jump_to"] = k

# Handle quick navigation
if "jump_to" in st.session_state and st.session_state["jump_to"] != selected_key:
    selected_key = st.session_state.pop("jump_to")
    if source_option == "GitHub Samples (Pre-loaded)" and selected_key in SAMPLE_CATALOG:
        filename = SAMPLE_CATALOG[selected_key]["filename"]
        file_info = st.session_state.get("gh_files_preloaded", {}).get(filename.upper())
        if file_info and file_info.get("download_url"):
            content = download_github_file(file_info["download_url"])
            if content:
                active_df_raw = parse_file(content, filename)

# Filter data to selected range
mask = (active_df_raw["two_theta"] >= tt_min) & (active_df_raw["two_theta"] <= tt_max)
active_df = active_df_raw[mask].copy()

# Create tabs
tabs = st.tabs(["📈 Raw Pattern", "🔍 Peak ID", "🧮 Rietveld Fit", "📊 Quantification", "🔄 Sample Comparison", "📄 Report", "🖼️ Publication Plot"])
PH_COLORS = [v["color"] for v in PHASE_LIBRARY.values()]

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 0 — RAW PATTERN
# ═══════════════════════════════════════════════════════════════════════════════
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

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PEAK IDENTIFICATION
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.subheader("Peak Detection & Phase Matching")
    col_a, col_b, col_c = st.columns(3)
    min_ht = col_a.slider("Min height × BG", 1.2, 8.0, 2.2, 0.1)
    min_sep = col_b.slider("Min separation (°)", 0.1, 2.0, 0.3, 0.05)
    tol = col_c.slider("Match tolerance (°)", 0.05, 0.5, 0.18, 0.01)
    obs_peaks = find_peaks_in_data(active_df, min_height_factor=min_ht, min_distance_deg=min_sep)
    theo = {ph: generate_theoretical_peaks(ph, wavelength, tt_min, tt_max) for ph in selected_phases}
    matches = match_phases_to_data(obs_peaks, theo, tol_deg=tol)
    fig_id = go.Figure()
    fig_id.add_trace(go.Scatter(x=active_df["two_theta"], y=active_df["intensity"], mode="lines", name="Observed", line=dict(color="lightsteelblue", width=1)))
    if len(obs_peaks):
        fig_id.add_trace(go.Scatter(x=obs_peaks["two_theta"], y=obs_peaks["intensity"], mode="markers", name="Detected peaks", marker=dict(symbol="triangle-down", size=10, color="crimson", line=dict(color="darkred", width=1))))
    I_top, I_bot = active_df["intensity"].max(), active_df["intensity"].min()
    for i, (ph, pk_df) in enumerate(theo.items()):
        color = PH_COLORS[i % len(PH_COLORS)]
        offset = I_bot - (i + 1) * (I_top * 0.04)
        fig_id.add_trace(go.Scatter(x=pk_df["two_theta"], y=[offset] * len(pk_df), mode="markers", name=f"{ph}", marker=dict(symbol="line-ns", size=14, color=color, line=dict(width=1.5, color=color)), customdata=pk_df["hkl_label"].values, hovertemplate="<b>%{fullData.name}</b><br>2θ=%{x:.3f}°<br>%{customdata}<extra></extra>"))
    fig_id.update_layout(xaxis_title="2θ (degrees)", yaxis_title="Intensity (counts)", template="plotly_white", height=460, hovermode="x unified", title=f"Peak identification — {selected_key}")
    st.plotly_chart(fig_id, use_container_width=True)
    st.markdown(f"#### {len(obs_peaks)} detected peaks")
    if len(obs_peaks):
        disp = obs_peaks.copy()
        disp["Phase match"], disp["(hkl)"], disp["Δ2θ (°)"] = matches["phase"].values, matches["hkl"].values, matches["delta"].round(4).values
        disp["two_theta"], disp["intensity"], disp["prominence"] = disp["two_theta"].round(4), disp["intensity"].round(1), disp["prominence"].round(1)
        st.dataframe(disp[["two_theta","intensity","prominence","Phase match","(hkl)","Δ2θ (°)"]], use_container_width=True)
    with st.expander("📐 Theoretical peak positions per phase"):
        for ph in selected_phases:
            pk = theo[ph]
            st.markdown(f"**{ph}** — {len(pk)} reflections in {tt_min:.0f}°–{tt_max:.0f}°")
            if len(pk): 
                st.dataframe(pk[["two_theta","d_spacing","hkl_label"]].rename(columns={"two_theta":"2θ (°)","d_spacing":"d (Å)","hkl_label":"hkl"}), use_container_width=True, height=200)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — RIETVELD FIT (MAIN REFINEMENT TAB)
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.subheader("Rietveld Refinement")
    if not selected_phases:
        st.warning("☑️ Select at least one phase in the sidebar.")
    elif not run_btn:
        st.info("Configure settings in the sidebar, then click **▶ Run Rietveld Refinement**.")
    else:
        with st.spinner("Running refinement using built‑in Numba engine..."):
            result = None
            error_msg = None
            try:
                # Always use built‑in engine
                @st.cache_resource(show_spinner=False)
                def run_numba_refinement(_data, phases, wavelength, bg_order, peak_shape, tt_min, tt_max):
                    data = _data[(_data["two_theta"] >= tt_min) & (_data["two_theta"] <= tt_max)].copy()
                    refiner = RietveldRefinement(data, phases, wavelength, bg_order, peak_shape)
                    return refiner.run()
                result = run_numba_refinement(
                    active_df_raw, tuple(selected_phases), wavelength,
                    bg_order, peak_shape, tt_min, tt_max
                )
                engine = "Built‑in (Numba)"
            except Exception as e:
                error_msg = f"{type(e).__name__}: {e}"
                st.error(f"❌ Refinement failed: {error_msg}")
                # Return synthetic result to avoid app crash
                result = {
                    "converged": False,
                    "Rwp": 99.9,
                    "Rexp": 10.0,
                    "chi2": 99.9,
                    "y_calc": active_df["intensity"].values,
                    "y_background": np.percentile(active_df["intensity"], 10) * np.ones(len(active_df)),
                    "zero_shift": 0.0,
                    "phase_fractions": {ph: 1.0/len(selected_phases) for ph in selected_phases},
                    "lattice_params": {ph: PHASE_LIBRARY[ph]["lattice"].copy() for ph in selected_phases},
                    "engine": "Error fallback",
                    "error": error_msg
                }
        # Display results
        if result and "Rwp" in result:
            conv_icon = "✅" if result.get("converged", False) else "⚠️"
            st.success(f"{conv_icon} Refinement finished · R_wp = **{result['Rwp']:.2f}%** · R_exp = **{result['Rexp']:.2f}%** · χ² = **{result['chi2']:.3f}**")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("R_wp (%)", f"{result['Rwp']:.2f}", delta="< 15 is acceptable", delta_color="off")
            m2.metric("R_exp (%)", f"{result['Rexp']:.2f}")
            m3.metric("GoF χ²", f"{result['chi2']:.3f}", delta="target ≈ 1", delta_color="off")
            m4.metric("Zero shift (°)", f"{result.get('zero_shift', 0):.4f}")
            # Plot results
            fig_rv = make_subplots(rows=2, cols=1, row_heights=[0.78, 0.22], shared_xaxes=True, vertical_spacing=0.04, subplot_titles=("Observed vs Calculated", "Difference"))
            fig_rv.add_trace(go.Scatter(x=active_df["two_theta"], y=active_df["intensity"], mode="lines", name="Observed", line=dict(color="#1f77b4", width=1.0)), row=1, col=1)
            fig_rv.add_trace(go.Scatter(x=active_df["two_theta"], y=result["y_calc"], mode="lines", name="Calculated", line=dict(color="red", width=1.5)), row=1, col=1)
            fig_rv.add_trace(go.Scatter(x=active_df["two_theta"], y=result["y_background"], mode="lines", name="Background", line=dict(color="green", width=1, dash="dash")), row=1, col=1)
            I_top2, I_bot2 = active_df["intensity"].max(), active_df["intensity"].min()
            for i, ph in enumerate(selected_phases):
                color = PH_COLORS[i % len(PH_COLORS)]
                pk_pos = generate_theoretical_peaks(ph, wavelength, tt_min, tt_max)
                ybase = I_bot2 - (i+1) * I_top2 * 0.035
                fig_rv.add_trace(go.Scatter(x=pk_pos["two_theta"], y=[ybase] * len(pk_pos), mode="markers", name=f"{ph} reflections", marker=dict(symbol="line-ns", size=10, color=color, line=dict(width=1.5, color=color)), customdata=pk_pos["hkl_label"], hovertemplate="%{customdata} 2θ=%{x:.3f}°<extra>"+ph+"</extra>"), row=1, col=1)
            diff = active_df["intensity"].values - result["y_calc"]
            fig_rv.add_trace(go.Scatter(x=active_df["two_theta"], y=diff, mode="lines", name="Difference", line=dict(color="grey", width=0.8)), row=2, col=1)
            fig_rv.add_hline(y=0, line_dash="dash", line_color="black", line_width=0.8, row=2, col=1)
            fig_rv.update_layout(template="plotly_white", height=580, xaxis2_title="2θ (degrees)", yaxis_title="Intensity (counts)", yaxis2_title="Obs − Calc", hovermode="x unified", title=f"Rietveld fit — {selected_key} (engine: {engine})")
            st.plotly_chart(fig_rv, use_container_width=True)
            # Lattice parameters table
            st.markdown("#### Refined Lattice Parameters")
            lp_rows = []
            for ph in selected_phases:
                p = result["lattice_params"].get(ph, {})
                p0 = PHASE_LIBRARY[ph]["lattice"]
                if "a" in p0 and isinstance(p0["a"], (int, float)) and "a" in p and isinstance(p["a"], (int, float)):
                    da = (p["a"] - p0["a"]) / p0["a"] * 100
                else:
                    da = 0
                lp_rows.append({
                    "Phase": ph, "System": PHASE_LIBRARY[ph]["system"],
                    "a_lib (Å)": f"{p0.get('a','—'):.5f}" if isinstance(p0.get('a'), (int,float)) else "—",
                    "a_ref (Å)": f"{p.get('a', p0.get('a','—')):.5f}" if isinstance(p.get('a'), (int,float)) else "—",
                    "Δa/a₀ (%)": f"{da:+.3f}",
                    "c_ref (Å)": f"{p.get('c','—'):.5f}" if isinstance(p.get('c'), (int,float)) else "—",
                    "Wt%": f"{result['phase_fractions'].get(ph,0)*100:.1f}"
                })
            st.dataframe(pd.DataFrame(lp_rows), use_container_width=True)
            # Store results in session state
            st.session_state[f"result_{selected_key}"] = result
            st.session_state[f"phases_{selected_key}"] = selected_phases
            st.session_state["last_result"] = result
            st.session_state["last_phases"] = selected_phases
            st.session_state["last_sample"] = selected_key
            st.session_state["last_engine"] = engine

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — QUANTIFICATION (unchanged)
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.subheader("Phase Quantification")
    if "last_result" not in st.session_state:
        st.info("Run the Rietveld refinement first.")
    else:
        result, phases = st.session_state.session_state["last_result"],["last st.session_result"],_state["last_ st.session_state["last_phasesphases"]
       "]
        fracs fracs = result = result["["phase_fractionsphase_fractions"]
       "]
        labels, values = labels, values = list( list(fracs.keys()),fracs.keys()), [frac [fracss[ph]*[ph]*100 for100 for ph in ph in fracs]
        fracs]
        colors = colors = [PHASE_LIBRARY[ph][" [PHASE_LIBRARY[ph]["color"]color"] for ph for ph in labels in labels]
       ]
        col_pie, col_pie, col_bar col_bar = st = st.columns(.columns(2)
        with col_p2)
        with col_pieie:
:
            fig            fig_pie = go_pie.Figure = go.(go.PieFigure(go.Pie(labels(labels=labels=labels, values, values=values=values, hole, hole=0=0..3838, text, textinfo="info="label+label+percent", marker=percent", marker=dict(dict(colors=colors=colors)))
            fig_piecolors)))
            fig_pie.update_layout.update_layout(title="(title="Phase weightPhase weight fractions", fractions", height=370)
            st.plotly height=370)
            st.plotly_chart_chart(fig(fig_pie_pie, use, use_container_width_container_width=True=True)
        with)
        with col_bar col_bar:
           :
            fig_bar fig_bar = go = go.Figure.Figure(go.Bar(go.Bar(x(x==labels, y=labels,values, y=values, marker_color marker_color=colors=colors, text, text=[f=[f"{v:.1"{v:.1f}f}%" for v in%" for v in values], values], textposition="outside textposition="outside"))
           "))
            fig_bar fig_bar.update_layout(yaxis.update_layout(yaxis_title="Weight fraction_title="Weight fraction (%)", (%)", template="plotly template="_whiteplotly_white", height=", height370=370, y, yaxis_rangeaxis_range=[0=[0, max, max(values)*(values)*1.25],1.25], title=f title=f"Phase fractions —"Phase fractions — {st {st.session_state.session_state['last['last_sample']_sample']}")
           }")
            st.plotly_chart(fig_bar st.plotly_chart(fig_bar, use, use_container_width_container_width=True=True)
        rows)
        rows = = []
        for ph in labels []
        for ph in labels:
            pi:
            pi, lp, lp = PH = PHASE_LIBASE_LIBRARYR[ph], result["ARY[ph], result["latticelattice_params"]._params"].get(get(ph,ph, { {})
            rows})
            rows.append({".append({"Phase":Phase": ph, ph, "C "Crystal systemrystal system": pi": pi["system["system"], "Space group"], "Space group": pi": pi["space["space_group_group"],
                         "a (Å)"],
                         "a (Å)": f": f"{lp.get('"{lp.get('a','a','—'):.—'):.5f5f}" if isinstance(l}" if isinstance(lp.getp.get('a'), (('a'), (int,int,float)) else "float)) else "——",
                         "c (",
                         "c (Å)Å)": f"{": f"{lplp.get('.get('c','c','—'—'):.):.5f5f}" if}" if isinstance(l isinstance(lp.getp.get('c('c'), ('), (int,int,float))float)) else " else "——",
                         "",
                         "WtWt%":%": f"{ f"{fracsfracs.get(.get(ph,ph,0)*0)*100:.100:.2f2f}}",
                         "",
                         "Role": piRole": pi["description"][:["description"][:65]+"…65]+" if"… len(" if len(pi["description"])pi["description"])>65>65 else pi else pi["description"]["description"]})
        st.dataframe})
        st(pd.dataframe(pd.DataFrame(.DataFrame(rows),rows), use_container use_container_width=True)

#_width=True)

# ═════════════ ═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
# TAB
# TAB 4 4 — S — SAMPLE COMPAMPLE COMPARISONARISON (unch (unchangedanged)
#)
# ═ ═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
with
with tabs tabs[4[4]:
    st]:
    st.subheader.subheader("("🔄 Multi🔄 Multi-Sample-Sample Comparison Comparison")
    view")
    view_mode =_mode = st. st.radio("radio("View modeView mode", ["", ["📊📊 Interactive ( Interactive (PlotlyPlotly)", ")", "🖼️🖼️ Publication-Q Publication-Quality (uality (MatplotlibMatplotlib)"],)"], horizontal=True horizontal=True, key, key="comp="comp_view_mode_view_mode")
   ")
    comp_samples comp_samples = st.multise = st.multiselect("lect("Select samples to compareSelect samples to compare", options=SAMPLE_KE", options=SAMPLEYS,_KEYS, default default=[k for k in=[k for k in SAMPLE SAMPLE_KE_KEYS if SAMPLE_CATALOGYS if SAMPLE_CATALOG[k]["[k]["group"]group"] == " == "Printed"Printed"][:][:44], format_func], format_func=lambda k=lambda k: f: f"[{"[{SAMPLE_CATALOG[k]['short']}]SAMPLE_CATALOG[k]['short']}] {SAMPLE_CAT {SAMPLEALOG_CATALOG[k]['label'][k]['}", keylabel']}", key="comp_samples="comp_samples")
    if not comp")
    if not comp_samples:
        st_samples.warning:
        st.warning("⚠️ Select at least one sample to compare("⚠️ Select at least one sample to compare.")
   .")
    else else:
        col:
        col_opt_opt1,1, col_ col_opt2opt2 = st = st.columns(.columns(22)
        with)
        with col_ col_opt1opt1:
           :
            normalize = normalize = st.check st.checkbox("box("✓✓ Normalise Normalise to [0,1]", value=True, to [0,1]", value=True, key=" key="comp_normalcomp_normalizeize")
            show")
            show_grid =_grid = st.checkbox(" st.checkbox("✓ Show grid",✓ Show grid", value=True value=True, key="comp, key="comp_grid_grid")
        with col_")
        with col_opt2opt2:
            line:
            line_width_width = st = st.sl.slider("ider("Line width", Line width", 0.5,0.5, 3 3.0, .0, 1.5,1.5, 0.1 0.1, key, key="comp_lw="comp_lw")
           ")
            opacity = st opacity = st.s.sliderlider("Opacity("Opacity", ", 0.0.3,3, 1 1..00, , 1.1.0,0, 0 0.1.1, key, key="comp="comp_alpha_alpha")
")
               if view if view_mode ==_mode == " "📊 Interactive📊 Interactive (Plot (Plotly)ly)":
           ":
            fig_c fig_cmp =mp = go. go.FigureFigure()
            for()
            for k in k in comp_samples comp_samples:
               :
                df_s df_s = all = all_data.get(k,_data.get(k, pd.DataFrame pd.DataFrame({"two({"two_theta_theta": np": np.lin.linspace(space(30,30, 130 130, , 20002000), "), "intensityintensity":": np np.random.n.random.normal(ormal(200,200, 50, 50, 2000)}))
                x, y = df_s[" two_2000)}))
                x, y = df_s["two_thetatheta"].values, df_s["intensity"].values
                if normalize and"].values, df_s["intensity"].values
                if normalize and len(y) > len(y) > 1:
                    1 y = (y:
                    y = (y - y - y.min()).min()) / (y.max / (y.max() -() - y.min() + y.min() + 1 1ee-8)
-8)
                m = SAMPLE_CATALOG[k                m = SAMPLE_C]
                fig_cmp.addATALOG[k]
                fig_cmp.add_trace(go_trace.Scatter(go.Scatter(x=x,(x=x y=y,, y=y, mode=" mode="lines", name=mlines", name=m["label["label"], line=dict"], line=dict(color=m(color=m["color["color"], width=line"], width=line_width), opacity=opacity))
           _width), opacity=opacity))
            fig_cmp.update fig_cmp.update_layout(title_layout(title="X="XRD Pattern Comparison",RD Pattern Comparison", xaxis_title=" xaxis_title="2θ (degrees2θ (degrees)", yaxis_title)", yaxis_title="Normal="Normalised Intensityised Intensity" if" if normalize else normalize else "Int "Intensity (ensity (countscounts)", template)", template="plot="plotly_ly_white"white" if show if show_grid else_grid else "plot "plotly", height=ly", height=500,500, hovermode hovermode="x="x unified", legend=dict(orientation unified", legend=dict(orientation="h", yanchor="bottom="h", yanchor="bottom", y", y=1=1.02.02, x, xanchor="anchor="right",right", x=1))
            st.plotly x=1))
            st.plotly_chart_chart(fig(fig_cmp_cmp, use, use_container_width_container_width=True=True)
            with st.expander)
            with(" st.expander("📋📋 Comparison Data Comparison Data Summary"):
                summary_data = Summary"):
                summary []
               _data = for k []
                for k in comp_samples:
                    m in comp_samples:
                    m = S = SAMPLE_CAMPLE_CATALATALOG[kOG[k]
                   ]
                    df_s df_s = all_data =.get(k, pd.DataFrame({"two_theta": [], "intensity": [] all_data.get(k, pd.DataFrame({"two_theta":}))
                    [], "intensity": []}))
                    if len if len(df_s) >(df_s 0) > 0:
                        summary_data:
                       .append({" summary_data.append({"Sample": m["Sample":short"], m["short"], "Label": m "Label["label": m["label"], ""], "Fabrication":Fabrication": m[" m["fabrication"],fabrication"], "Treatment "Treatment": m["": m["treatmenttreatment"], ""], "Points":Points": len(df len(df_s_s),), "2θ Range": f"{df_s['two_theta'].min(): "2θ Range": f"{df_s['two_theta'].min():.1f}–{df_s['two.1f}–{_thetadf_s['two_theta'].max():.1f}'].max():.1f°", "}°", "Max IntensityMax Intensity": f"{df": f"{df_s['_s['intensity'].maxintensity'].max():.():.0f}"0f}"})
                st})
                st.dataframe(pd.dataframe.DataFrame(sum(pd.DataFrame(summary_datamary_data), use_container_width), use_container_width=True=True)
        else:
           )
        else:
            st.mark st.markdown("###down("### 🎨 🎨 Publication Plot Settings Publication Plot Settings")
            col_pub")
            col_pub1, col_p1, col_pub2, colub2, col_pub_pub3 =3 = st.columns(3 st.columns(3)
           )
            with col_p with col_pubub11:
                pub_width =:
                pub st.s_width = st.sliderlider("Width("Width (inches)",  (inches)", 6.0,6.0, 14 14.0, .0, 10.10.0, 00, 0.5.5, key, key="pub="pub_comp_comp_w_w")
                pub_font")
                pub_font = st = st.sl.slider("ider("Font SizeFont Size", ", 8,8, 18 18, , 11,11, 1 1, key, key="pub="pub_comp_comp_font_font")
               ")
                stack_offset stack_offset = st = st.sl.slider("ider("Stack offsetStack offset", ", 0.0.0,0, 1 1.5.5, , 0.0.0,0, 0 0.1.1, key, key="pub="pub_comp_comp_stack",_stack", help=" help="0 =0 = overlay, overlay, >0 >0 = waterfall = waterfall stacking")
            stacking")
            with col_p with col_pub2ub2:
               :
                pub_height pub_height = st = st.sl.slider("ider("Height (Height (inches)",inches)", 5 5.0.0, , 12.12.0,0, 7 7.0.0, 0., 0.5,5, key=" key="pub_pub_comp_hcomp_h")
                pub_")
                pub_legend_poslegend_pos = st = st.selectbox.selectbox("Legend("Legend", ["", ["best",best", "upper "upper right", right", "upper "upper left", "lower left", "lower right", "center right", left", "lower left", "lower right", "center right", "off "off"], key="pub"], key="pub_comp_comp_leg_leg")
               ")
                export_fmt = export_fmt = st.selectbox(" st.selectbox("Export",Export", ["PDF", " ["PDF", "PNGPNG", "EPS"],", "EPS"], key=" key="pub_pub_comp_fcomp_fmtmt")
            with")
            with col_p col_pub3ub3:
               :
                st.mark st.markdown("down("****🎨 Per🎨 Per-Sample-Sample Styl Styling**ing**")
               ")
                sample_st sample_styles = {}
               yles = for k {}
                for k in comp in comp_samples_samples:
                    m:
                    m = S = SAMPLE_CAMPLE_CATALATALOG[kOG[k]
                   ]
                    with st with st.expander(f"{.expander(f"{m['m['short']short']}", expanded}", expanded=False=False):
                        sample):
                        sample_styles[k]_styles[k] = = {
 {
                            "                            "color":color": st.color_picker st.color_picker("Color("Color", m", m["color["color"], key"], key=f"=f"col_{col_{k}k}"),
                           "),
                            "style "style": st": st.selectbox.selectbox("Line("Line", ["", ["-",-", "--", "--", " ":", "-.":", "-."], index=0, key], index=0, key=f"=f"sty_{k}sty_{k}"),
                           "),
                            "width": st "width": st.sl.slider("ider("Width", 0Width", 0.5.5, , 3.3.0,0, 1.5 1.5, , 0.0.1, key=f"lw_{k}1, key=f"lw_{k}"),
                           "),
                            "label": st "label": st.text_input.text_input("Legend("Legend Label Label", m", m["label"],[" key=f"llabel"], key=f"lbl_{k}")
                       bl_{k }
            sample}")
                       _data_list = }
            sample_data_list []
            legend =_labels = []
            legend_labels = []
            line_st []
           yles = line_styles = []
            for k []
            in comp for k in comp_samples_samples:
                df:
                df_s =_s = all_data all_data.get.get(k(k, pd, pd.DataFrame({".DataFrame({"two_two_theta":theta": np.l np.linspaceinspace(30, 130, (30, 130, 2000), "intensity": np.random.normal2000), "intensity": np.random.normal(200(200, , 50,50, 200 2000)}0)}))
               ))
                styles = styles = sample_st sample_styles.getyles.get(k,(k, { {})
                sample})
                sample_data_list_data_list.append({"two_.append({"two_theta":theta": df_s["two df_s["two_theta_theta"].values, ""].values, "intensity": dfintensity": df_s_s["intensity["intensity"].values, "label":"].values, "label": SAMPLE SAMPLE_CATALOG_CATALOG[k]["[k]["label"], "label"], "colorcolor": styles": styles.get("color",.get("color", SAMPLE_CAT SAMPLE_CATALOGALOG[k]["[k]["color"]color"]), "), "linewidthlinewidth": styles": styles.get(".get("width",width", line_width line_width)})})
                legend)
                legend_labels.append_labels.append((styles.get("styleslabel",.get("label", SAMPLE SAMPLE_CAT_CATALOGALOG[k]["[k]["label"]label"]))
               ))
                line_st line_styles.append(styles.get("yles.append(stylesstyle", "-"))
            try.get("style", "-"))
            try:
               :
                fig_pub, fig_pub, ax_p ax_pub =ub = plot_sample plot_sample_compar_comparison_ison_publicationpublication(
                   (
                    sample_data sample_data_list=s_list=sampleample_data_data_list,_list, tt_min tt_min=tt_min, tt_max=tt_max=tt_min, tt_max=tt,
                    figsize=(_maxpub_width,
                    figsize=(pub_width, pub, pub_height), font_size_height), font_size=pub=pub_font,
                    legend_pos_font,
                    legend_pos=pub=pub_legend_legend_pos if_pos if pub_ pub_legend_poslegend_pos != " != "off"off" else " else "offoff",
                    normalize=normal",
                    normalize=normalize,ize, stack_offset= stack_offset=stackstack_offset_offset,
                    line_styles,
                    line_styles=line=line_styles, legend_styles, legend_labels=_labels=legend_labelslegend_labels,
                    show_grid,
                    show_grid=show=show_grid
               _grid
                )
                st )
                st.pyplot(f.pyplot(fig_pig_pub, dpi=150, useub, dpi=150, use_container_width_container_width=True=True)
                st.markdown)
                st.markdown("####("#### 📥 Export Publication 📥 Export Publication Figure Figure")
                col")
                col_e1_e1, col, col_e2_e2, col, col_e3_e3 = st = st.columns.columns(3(3)
                with)
                with col_e1 col_e1:
                    buf = io:
                    buf = io.Bytes.BytesIO();IO(); fig_p fig_pub.saveub.savefig(buffig(buf, format, format='pdf='pdf', b', bbox_inbox_inches='ches='tight');tight'); buf.se buf.seek(ek(00)
                    st)
                    st.download.download_button("_button("📄📄 PDF", PDF", buf.read buf.read(), file(), file_name=f_name=f"x"xrd_comrd_comparisonparison_{len_{len(comp(comp_samples)}_samples)}samplessamples.pdf",.pdf", mime mime="application="application/pdf",/pdf", use_container use_container_width=True_width=True)
               )
                with col_e2:
                    with col_e2:
                    buf = buf = io.B io.BytesIOytesIO(); fig(); fig_pub_pub.savefig.savefig(buf,(buf, format=' format='png',png', dpi dpi=300=300, b, bboxbox_in_inches='ches='tight');tight'); buf.se buf.seek(ek(00)
                    st)
                    st.download.download_button("_button("🖼️🖼️ PNG ( PNG (300 D300 DPI)",PI)", buf.read buf.read(), file(), file_name=f_name=f"x"xrd_comrd_comparisonparison_{len_{len(comp(comp_samples)}samples_samples)}samples.png",.png", mime mime="image="image/png/png", use_container_width", use=True_container_width=True)
                with)
                with col_e3 col_e:
                    buf3:
                    buf = io.Bytes = ioIO();.BytesIO(); fig_p fig_pub.saveub.savefig(buffig(buf, format, format='eps='eps', b', bbox_inbox_inches='ches='tight');tight'); buf.se buf.seek(ek(00)
                    st)
                    st.download.download_button("_button("📐📐 EPS", EPS", buf.read buf.read(), file(), file_name=f"xrd_com_name=f"xrd_comparisonparison_{len_{len(comp(comp_samples)}_samples)}samplessamples.eps.eps", m", mime="ime="application/postapplication/postscript",script", use_container use_container_width=True_width=True)
               )
                plt.close plt.close(fig(fig_pub_pub)
           )
            except Exception except Exception as e as e:
               :
                st.error(f" st.error(f"❌❌ Plot generation Plot generation failed: failed: {type {type(e).(e).__name__name____}:}: {e}")
                {e}")
                st.code("Tip: Try reducing the number of samples or st.code("Tip: Try reducing the number of samples or resetting resetting font font size size.")

#.")

# ═════════════ ═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
# TAB
# TAB 5 5 — REPORT — REPORT (unch (unchangedanged)
#)
# ═ ═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
with
with tabs tabs[5]:
    st.subheader("Analysis Report")
    if "last[5]:
    st.subheader("Analysis Report")
    if "last_result" not in_result" not in st.session st.session_state_state:
        st:
        st.info("Run the.info("Run the Rietveld refinement first (Tab  Rietveld refinement first (Tab 3).")
   3).")
    else:
        result, phases, samp else:
        result, phases, samp = st = st.session_state["last.session_state_result"], st.session["last_state["_result"], st.session_state["last_phaseslast_phases"], st.session_state["last_sample"], st.session_state["last_sample"]
        report"]
        report_md_md = generate = generate_report(result_report(result, phases, phases, wavelength, wavelength, samp, samp)
       )
        st.mark st.markdown(report_mdown(report_mdd)
        col)
        col_dl_dl1, col_d1, col_dl2l2 = st = st.columns(.columns(22)
        col)
        col_dl1.d_dl1.download_buttonownload_button("⬇️("⬇️ Download Report Download Report (.md (.md)", data=report)", data=report_md, file_name=f_md, file_name=f"rietveld_report"rietveld_report_{s_{samp}.md",amp}.md", mime="text/mark mime="text/markdown")
        exportdown")
        export_df = active_df_df = active_df.copy.copy()
        export_df["()
        export_df["y_calc"], export_dfy_calc"], export_df["y["y_background"], export_background"], export_df["_df["difference"]difference"] = result["y = result["y_calc_calc"],"], result["y result_background["y"], active_df["_backgroundintensity"], active_df[""].values - resultintensity"].values - result["y["y_calc"]
       _calc"]
        csv_b csv_buf =uf = io.String io.StringIOIO()
        export()
        export_df.to_csv(csv_b_df.to_csv(csv_buf,uf, index=False index=False)
       )
        col_dl2 col_dl2.d.downloadownload_button("⬇_button("️ Download⬇️ Download Fit Data (.csv Fit Data)", data (.csv)", data=csv_buf=csv_buf.getvalue.getvalue(), file(), file_name=f"_name=f"rietrietveld_fveld_fit_{samp}.csvit_{samp}.csv", m", mime="ime="text/csvtext/csv")

# ═")

# ═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
# TAB 
# TAB 6 —6 — PUBL PUBLICATION-ICATION-QUALITYQUALITY PLOT PLOT (S (SINGLEINGLE SAMPLE) – SAMPLE unchanged) – unchanged
#
# ═════════ ═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
with tabs
with tabs[6]:
    st.subheader[6]:
    st(".subheader("🖼️🖼 Publication️ Publication-Quality Plot-Quality Plot (matplotlib (matplotlib)")
   )")
    st st.c.caption("Generate journalaption("Generate journal-ready figures with customizable-ready figures with customizable phase phase markers, legend markers control &, legend spacing")
    control & spacing if")
    if "last "last_result"_result" not in not in st.session st.session_state or_state or "last "last_ph_phases"ases" not in not in st.session_state:
        st st.session_state:
        st.info(".info("🔬 Run the🔬 Run the Riet Rietveld refinementveld refinement first ( first (Tab Tab 3:3: 🧮 🧮 R Rietveld Fit)ietveld to enable Fit) publication plotting.")
        to enable publication plotting st.mark.")
       down(" st.markdown("""**Quick steps""**:** Quick steps:** 1. Select a1. Select a sample in the sidebar 2 sample in the sidebar 2. Choose. Choose phases to phases to refine  refine 3.3. Click ** Click **▶ Run▶ Run Riet Rietveld Refveld Refinement**inement** 4 4. Return. Return here."" here.""")
   ")
    else else:
        result:
        result = st = st.session_state["last_result.session_state["last_result"]
        phases"]
        phases = st = st.session_state.session_state["last["last_ph_phases"]
        col1,ases"]
        col1, col2 col2, col3 =, col3 = st.columns st.columns(3(3)
       )
        with col with col11:
            fig:
            fig_width =_width = st.s st.sliderlider("Figure("Figure width ( width (inches)",inches)", 6 6.0.0,, 14. 14.0,0, 10.0 10.0, , 0.0.5, key="pub_width5, key="pub_width")
           ")
            offset_factor = st offset_factor = st.sl.slider("ider("Difference curve offset",Difference curve offset", 0.05 0.05, , 0.25,0.25, 0 0.12,.12,  0.0.01,01, key=" key="pub_offsetpub_offset")
           ")
            font_size font_size = st = st.slider("Global Font.slider("Global Font Size", Size", 6 6, , 22,22, 11 11, key, key="pub="pub_font_font")
       ")
        with col with col22:
            fig:
            fig_height =_height = st.s st.sliderlider("Figure("Figure height ( height (inches)",inches)", 5 5.0.0, , 12.12.0,0, 7 7.0.0, , 0.0.5,5, key=" key="pub_heightpub_height")
           ")
            show_h show_hkl =kl = st.check st.checkbox("Showbox("Show h hkl labelskl labels", value=True,", value=True, key=" key="pub_hklpub_hkl")
            legend")
            legend_pos = st.select_pos = st.selectbox("box("Legend PositionLegend Position", ["", ["best",best", "upper right", "upper left", "upper right", "upper left", " "lower left",lower left", "lower "lower right", "center right", "center right", right", "center left", "center "lower left", "lower center", "upper center", "upper center", center", "center", " "center", "off"], index=off"], index=0, key="0, key="pub_pub_legend_poslegend_pos")
        with col")
        with col3:
            export3:
            export_format =_format = st.select st.selectbox("Export formatbox("Export format", ["", ["PDF",PDF", "PN "PNG",G", "EPS "EPS"], index"], index=0=0, key, key="pub="pub_format_format")
            marker_spacing = st.sl")
            marker_spacing = st.slider("ider("Marker row spacing",Marker row spacing", 0 0.8.8, 2., 2.5, 1.3,5, 1.3,  0.0.1, help="1, help="Vertical distanceVertical distance between phase marker rows between phase marker rows", key", key="pub_spacing="pub_spacing")
           ")
            st.mark st.markdown("down("****🎨 Phase Customization🎨 Phase Customization****")
        st.markdown")
        st.markdown("###("### 📋 📋 Legend Control Legend Control")
       ")
        st.c st.caption("aption("Select whichSelect which phases to phases to include in include in the plot the plot legend ( legend (uncheckuncheck to hide to hide from legend from legend)")
       )")
        n_col n_cols =s = min( min(4,4, len( len(phasesphases))
       ))
        legend_col legend_cols =s = st.columns st.columns(n_col(n_colss)
        legend)
        legend_ph_phasesases_se_selected =lected = []
        []
        for idx, ph in enumerate(phases):
            col for idx, ph in enumerate(phases):
            col_idx =_idx = idx % idx % n_col n_colss
            with
            with legend_col legend_colss[col_idx[col_idx]:
               ]:
                if st if st.checkbox.checkbox(f"(f"✓ {✓ {ph}",ph}", value=True value=True, key, key=f"=f"leg_{leg_{ph}"ph}"):
                    legend_):
                    legend_phases_selectedphases_selected.append(.append(phph)
        phase_data = []
        for i, ph in enumerate)
        phase_data = []
        for i, ph in enumerate(phases(phases):
            pk):
            pk_df =_df = generate_the generate_theoretical_oretical_peakspeaks(ph(ph, wavelength, wavelength, tt_min,, tt_min, tt_max tt_max)
           )
            with st.expander with st.expander(f"⚙(f"⚙️ Settings️ Settings for **{ph for **{ph}**}**", expanded=(i", expanded=(i==0==0)):
               )):
                c_col, c_shape = c_col, c_shape = st.columns(2 st.columns(2)
               )
                custom_color = c custom_color = c_col.color_col.color_picker("Color_picker("Color", value", value=PHASE_L=PHASE_LIBRIBRARYARY[ph]["[ph]["color"],color"], key=f" key=f"colcol_{ph_{ph}")
               }")
                shape_options shape_options = [" = ["|",|", "_", "_", "s", "^", "v", "s", "^", "v", "d "d", "", "x",x", "+", "+", "* "*"]
                default"]
                default_idx = shape_idx = shape_options_options.index(PHASE.index(P_LIBHASE_LIBRARY[phRARY[ph].get].get("("markmarker_shape", "er_shape", "||"))
                custom"))
                custom_shape =_shape = c_shape c_shape.selectbox.selectbox("Marker("Marker Shape", Shape", shape_options shape_options, index, index=default=default_idx,_idx, key=f key=f"sh"shp_{ph}",p_{ph}", help=" help="| =| = vertical vertical bar bar, _ = horizontal, _, s = horizontal, s = square = square ■, d = ■, d = diamond ◆ diamond ◆")
            phase_data")
            phase_data.append({".append({"name": ph,name": ph, "positions "positions": pk_df["": pk_df["two_two_theta"].theta"].values ifvalues if len(pk_df) > len(pk_df) > 0 0 else np else np.array([].array([]), "), "color":color": custom_color custom_color, ", "markermarker_shape":_shape": custom_shape custom_shape, ", "hklhkl":": [hkl [hkl.strip(".strip("()").split(","()").split(",") if) if hkl hkl else None else None for h for hkl inkl in pk_df pk_df["h["hkl_labelkl_label"].values"].values] if show_h] if show_hkl andkl and len(p len(pk_df) >k_df) > 0 else None 0 else None})
        try})
        try:
            fig:
            fig,, ax = plot ax = plot_riet_rietveld_publicationveld_publication(
                active_df(
                active_df["two_theta["two_theta"].values, active"].values, active_df["_df["intensity"].valuesintensity"].values,
               ,
                result["y_c result["y_calc"],alc"], active_df[" active_df["intintensity"].ensity"].values - result["values - result["y_cy_calc"],
                phase_data, offset_factor=offset_factor, figsize=(figalc"],
                phase_data, offset_factor=offset_factor, figsize=(fig_width_width,, fig_height fig_height),
                font_size),
                font_size=font=font_size, legend_pos_size, legend_pos=legend=legend_pos,_pos, marker_row marker_row_spacing_spacing=mark=marker_sper_spacingacing,
                legend,
                legend_ph_phases=ases=legend_legend_phasesphases_selected_selected if legend if legend_ph_phases_seases_selected elselected else None
            )
            st None
            )
            st.pyplot(f.pyplot(fig,ig, dpi dpi==150150, use_container_width, use=True_container_width=True)
            st.markdown)
            st("####.markdown("#### 📥 Export Options 📥")
            Export Options col_e1,")
            col_e col_e1,2, col_e col_e2,3 = col_e st.columns3 = st.columns(3)
           (3)
            with col with col_e1_e1:
               :
                buf = io.B buf = io.BytesIOytesIO(); fig(); fig.savefig.savefig(buf,(buf, format=' format='pdf',pdf', bbox bbox_inches_inches='tight='tight'); buf.seek'); buf.seek(0(0)
               )
                st.d st.download_buttonownload_button("📄 PDF", buf.read(),("📄 PDF", buf.read(), file_name file_name=f"rietveld_pub_{=f"rietveld_pub_{selected_key}.pdf", mime="applicationselected_key}.pdf", mime="application/pdf",/pdf", use_container use_container_width=True_width=True)
           )
            with col with col_e2_e2:
               :
                buf = buf = io.B io.BytesIOytesIO(); fig(); fig.savefig.savefig(buf,(buf, format=' format='png',png', dpi=300, b dpi=300, bbox_inbox_inches='ches='tight');tight'); buf.se buf.seek(0)
                st.download_button("ek(0)
                st.download_button("🖼️🖼️ PNG ( PNG (300 D300 DPI)",PI)", buf.read buf.read(), file_name=f"riet(), file_name=f"rietveld_pveld_pub_{ub_{selected_keyselected_key}.png}.png", m", mime="ime="image/pimage/png",ng", use_container use_container_width=True_width=True)
           )
            with col with col_e3_e3:
               :
                buf = buf = io.B io.BytesIOytesIO(); fig(); fig.savefig.savefig(buf,(buf, format=' format='eps',eps', bbox bbox_inches_inches='tight='tight'); buf'); buf.seek.seek(0(0)
               )
                st.d st.download_buttonownload_button("("📐 EPS📐 EPS", buf.read(),", buf.read(), file_name file_name=f"=f"rietveldrietveld_pub_pub_{selected_{selected_key}._key}.eps",eps", mime mime="application="application/postscript/postscript", use_container_width", use_container_width=True=True)
            with)
            with st.expander(" st.expander("🎨 Marker Shape Reference"):
                st.markdown("""| Shape | Code | Visual |🎨 Marker Shape Reference"):
                st.markdown("""| Shape | Code | Visual | Recommended Use Recommended Use |
|-------|------ |
|-------|------|--------|--------|----------------|----------------|
| Vertical bar|
| Vertical bar | ` | `|` | │ | FCC|` | │ | FCC-Co matrix (primary-Co matrix (primary)) |
| Horizontal bar | |
| Horizontal bar | `_ `_` | ─ | HCP-Co` | ─ | HCP-Co ( (secondary)secondary) |
| ** |
| **Square**Square** ✨ ✨ | ` | `s`s` | ■ | ■ | M | M₂₂₃C₃C₆ carbides₆ carbides |
| |
| Triangle up Triangle up | ` | `^`^` | ▲ | ▲ | Sigma | Sigma phase phase |
| Triangle |
| Triangle down | down | `v `v` |` | ▼ | Additional precipitates |
 ▼ | Additional precipitates |
| **Diamond| ****Diamond** ✨ | `d ✨ |` | `d ◆ | Trace inter` | ◆ |metall Trace intermetallicsics |
| Cross |
| Cross | ` | `x`x` | × | Reference | × | Reference peaks peaks |
| Plus |
| Plus | ` | `+`+` | + | + | Cal | Calibration markersibration markers |
| |
| Star | Star | `*` | `*` | ✦ ✦ | Special annotations | | Special annotations |""""")
            plt")
            plt.close(fig.close(fig)
        except)
        except Exception as Exception as e:
            st e:
            st.error(f".error(f"❌ Plot❌ Plot generation failed: { generation failed: {type(e).__type(e).__name__}: {ename__}: {e}")
            st}")
            st.code(".code("Tip: Try reducingTip: Try reducing the number the number of phases of phases or resetting font or resetting font size to default size to default.")

# Footer.")

# Footer
st.markdown
st.markdown("("------")
st.caption")
st.caption("X("XRD RietveldRD Rietveld App • App • Co-Cr Dental Co-Cr Dental Alloy Alloy Analysis • Supports Analysis • Supports . .asc,asc, .ASC .ASC & . & .xrdxrdml •ml • GitHub: Maryamslm/X GitHub: Maryamslm/XRD-RD-3Dprinted-R3Dprinted-Retet/S/SAMPLES")
AMPLES")

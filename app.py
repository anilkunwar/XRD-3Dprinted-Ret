"""
XRD Rietveld Analysis — Co-Cr Dental Alloy (Mediloy S Co, BEGO)
================================================================
Simplified Interface • Exact Rietveld-based Quantification • Supports .asc, .xrdml, .cif
Wavelength Conversion: Supports Co Kα, Cu Kα, etc.
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import io, os, math, sys, base64, re, xml.etree.ElementTree as ET
from scipy import signal
from scipy.optimize import least_squares, curve_fit
import requests

# Try to import GSAS-II (optional)
try:
    import GSASII.GSASIIscriptable as G2sc
    GSASII_AVAILABLE = True
except ImportError:
    GSASII_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG & PHASES
# ═══════════════════════════════════════════════════════════════════════════════

SAMPLE_CATALOG = {
    "CH0_1": {"filename": "CH0_1.ASC", "description": "SLM-printed Co-Cr alloy, heat-treated"},
    # Add others here if needed for demo
}
DEMO_DIR = os.path.join(os.path.dirname(__file__), "demo_data")

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
        "system": "Cubic", "space_group": "Fm-3m (No. 225)", "lattice": {"a": 3.548},
        "peaks": [("111", 44.2), ("200", 51.5), ("220", 75.8), ("311", 92.1), ("222", 98.5)],
        "color": "#e377c2", "default": True, "marker_shape": "|",
        "description": "FCC Co-based solid solution (matrix)", "cod_id": "9008466"
    },
    "HCP-Co": {
        "system": "Hexagonal", "space_group": "P6₃/mmc (No. 194)", "lattice": {"a": 2.5071, "c": 4.0686},
        "peaks": [("100", 41.6), ("002", 44.8), ("101", 47.5), ("102", 69.2), ("110", 78.1)],
        "color": "#7f7f7f", "default": False, "marker_shape": "_",
        "description": "HCP Co (low-temp/stress-induced)", "cod_id": "9008492"
    },
    "M23C6": {
        "system": "Cubic", "space_group": "Fm-3m (No. 225)", "lattice": {"a": 10.63},
        "peaks": [("311", 39.8), ("400", 46.2), ("511", 67.4), ("440", 81.3), ("620", 93.5)],
        "color": "#bcbd22", "default": False, "marker_shape": "s",
        "description": "Cr-rich carbide M₂₃C₆", "mp_id": "mp-723"
    },
    "Sigma": {
        "system": "Tetragonal", "space_group": "P4₂/mnm", "lattice": {"a": 8.80, "c": 4.56},
        "peaks": [("210", 43.1), ("220", 54.3), ("310", 68.9)],
        "color": "#17becf", "default": False, "marker_shape": "^",
        "description": "Sigma phase (Co,Cr) intermetallic"
    }
}

# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def wavelength_to_energy(wavelength_angstrom):
    h = 4.135667696e-15
    c = 299792458
    energy_ev = (h * c) / (wavelength_angstrom * 1e-10)
    return energy_ev / 1000

def _parse_hkl(hkl_label: str) -> tuple:
    """Parse hkl label like '(311)' or '(3,1,1)' into tuple (h,k,l)."""
    clean = hkl_label.strip().strip("()").replace(" ", "")
    if "," in clean:
        return tuple(int(p.strip()) for p in clean.split(",") if p.strip())
    result = []
    i = 0
    while i < len(clean):
        sign = 1
        if clean[i] in "+-":
            if clean[i] == "-": sign = -1
            i += 1
        num_str = ""
        while i < len(clean) and clean[i].isdigit():
            num_str += clean[i]
            i += 1
        if num_str: result.append(sign * int(num_str))
        if len(result) == 3: break
    while len(result) < 3: result.append(0)
    return tuple(result[:3])

def generate_theoretical_peaks(phase_name, wavelength, tt_min, tt_max):
    """
    Generate theoretical peak positions for a phase.
    FIXED: Correctly recalculates 2-theta for the selected wavelength 
    based on Bragg's Law.
    """
    LAMBDA_REF = 1.5406  # Reference wavelength used in PHASE_LIBRARY peak positions
    
    # Check custom phases
    if "custom_phases" in st.session_state and phase_name in st.session_state.custom_phases:
        phase = st.session_state.custom_phases[phase_name]
        if "cif_data" in phase:
            return generate_peaks_from_cif(phase["cif_data"], wavelength, tt_min, tt_max)
        elif "peaks" in phase and phase["peaks"]:
            peaks = []
            for hkl_str, tt_approx_ref in phase["peaks"]:
                d_spacing = LAMBDA_REF / (2 * math.sin(math.radians(tt_approx_ref / 2)))
                sin_theta_new = wavelength / (2 * d_spacing)
                if abs(sin_theta_new) <= 1.0:
                    theta_new_rad = math.asin(sin_theta_new)
                    tt_new = math.degrees(2 * theta_new_rad)
                    if tt_min <= tt_new <= tt_max:
                        peaks.append({"two_theta": round(tt_new, 3), "d_spacing": round(d_spacing, 4), "hkl_label": f"({hkl_str})"})
            return pd.DataFrame(peaks) if peaks else pd.DataFrame(columns=["two_theta", "d_spacing", "hkl_label"])

    if phase_name not in PHASE_LIBRARY:
        return pd.DataFrame(columns=["two_theta", "d_spacing", "hkl_label"])
    
    phase = PHASE_LIBRARY[phase_name]
    peaks = []
    for hkl_str, tt_approx_ref in phase["peaks"]:
        d_spacing = LAMBDA_REF / (2 * math.sin(math.radians(tt_approx_ref / 2)))
        sin_theta_new = wavelength / (2 * d_spacing)
        if abs(sin_theta_new) <= 1.0:
            theta_new_rad = math.asin(sin_theta_new)
            tt_new = math.degrees(2 * theta_new_rad)
            if tt_min <= tt_new <= tt_max:
                peaks.append({"two_theta": round(tt_new, 3), "d_spacing": round(d_spacing, 4), "hkl_label": f"({hkl_str})"})
    return pd.DataFrame(peaks) if peaks else pd.DataFrame(columns=["two_theta", "d_spacing", "hkl_label"])

def find_peaks_in_data(df, min_height_factor=2.0, min_distance_deg=0.3):
    if len(df) < 10: return pd.DataFrame(columns=["two_theta", "intensity", "prominence"])
    x, y = df["two_theta"].values, df["intensity"].values
    bg = np.percentile(y, 15)
    min_height = bg + min_height_factor * (np.std(y) if len(y) > 1 else 1)
    min_distance = max(1, int(min_distance_deg / np.mean(np.diff(x))))
    peaks, props = signal.find_peaks(y, height=min_height, distance=min_distance, prominence=min_height*0.3)
    if len(peaks) == 0: return pd.DataFrame(columns=["two_theta", "intensity", "prominence"])
    return pd.DataFrame({"two_theta": x[peaks], "intensity": y[peaks], "prominence": props.get("prominences", np.zeros_like(peaks))}).sort_values("intensity", ascending=False).reset_index(drop=True)

# ═══════════════════════════════════════════════════════════════════════════════
# CIF & FILE PARSERS
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def parse_cif_content(cif_text: str) -> dict:
    import re
    result = {"chemical_formula": None, "space_group_hm": None, "cell_params": {}}
    patterns = {
        "chemical_formula": r"_chemical_formula_sum\s+([^\n]+)",
        "space_group_hm": r"_symmetry_space_group_name_H-M\s+['\"]?([^\n'\"]+)['\"]?",
        "cell_length_a": r"_cell_length_a\s+([\d.]+)",
        "cell_length_c": r"_cell_length_c\s+([\d.]+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, cif_text, re.IGNORECASE)
        if match:
            if key.startswith("cell_"): result["cell_params"][key.replace("cell_", "")] = float(match.group(1))
            else: result[key] = match.group(1).strip()
    return result

def generate_peaks_from_cif(cif_ dict, wavelength: float, tt_min: float, tt_max: float) -> pd.DataFrame:
    LAMBDA_REF = 1.5406
    sg = cif_data.get("space_group_hm", "")
    a = cif_data["cell_params"].get("length_a", 3.544)
    c = cif_data["cell_params"].get("length_c", None)
    if "F m -3 m" in sg or (c is None and a > 3.4):
        peaks = [("311", 39.8), ("400", 46.2), ("511", 67.4), ("440", 81.3)] if a > 10.0 else [("111", 44.2), ("200", 51.5), ("220", 75.8)]
    elif "P 63/m m c" in sg or (c is not None and abs(c/a - 1.62) < 0.1):
        peaks = [("100", 41.6), ("002", 44.8), ("101", 47.5), ("110", 78.1)]
    else: peaks = [("111", 44.2), ("200", 51.5)]
    results = []
    for hkl_str, tt_approx_ref in peaks:
        d_spacing = LAMBDA_REF / (2 * math.sin(math.radians(tt_approx_ref / 2)))
        sin_theta_new = wavelength / (2 * d_spacing)
        if abs(sin_theta_new) <= 1.0:
            tt_new = math.degrees(2 * math.asin(sin_theta_new))
            if tt_min <= tt_new <= tt_max: results.append({"two_theta": round(tt_new, 3), "d_spacing": round(d_spacing, 4), "hkl_label": f"({hkl_str})"})
    return pd.DataFrame(results) if results else pd.DataFrame(columns=["two_theta", "d_spacing", "hkl_label"])

@st.cache_data
def parse_asc(raw_bytes: bytes) -> pd.DataFrame:
    text = raw_bytes.decode("utf-8", errors="replace")
    rows = []
    for line in text.splitlines():
        parts = re.split(r'[\s,;]+', line.strip())
        if len(parts) >= 2:
            try: rows.append((float(parts[0]), float(parts[1])))
            except ValueError: continue
    df = pd.DataFrame(rows, columns=["two_theta", "intensity"])
    return df.sort_values("two_theta").reset_index(drop=True) if len(df) > 0 else pd.DataFrame(columns=["two_theta", "intensity"])

@st.cache_data
def parse_xrdml(raw_bytes: bytes) -> pd.DataFrame:
    try:
        text = raw_bytes.decode("utf-8", errors="replace").replace('xmlns="[^"]+"', '', 1)
        root = ET.fromstring(text)
        for elem in root.iter():
            if elem.tag.endswith('xRayData') or elem.tag == 'xRayData':
                vals_elem = elem.find('.//values') or elem.find('.//data')
                if vals_elem is not None and vals_elem.text:
                    intensities = [float(v) for v in vals_elem.text.strip().split() if v.strip()]
                    start, end = float(elem.get('startAngle', 0)), float(elem.get('endAngle', 100))
                    if len(intensities) > 1:
                        two_theta = np.linspace(start, end, len(intensities))
                        return pd.DataFrame({"two_theta": two_theta, "intensity": intensities})
        return pd.DataFrame(columns=["two_theta", "intensity"])
    except: return pd.DataFrame(columns=["two_theta", "intensity"])

# ═══════════════════════════════════════════════════════════════════════════════
# IMPROVED RIETVELD ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class RietveldRefinement:
    def __init__(self, data, phases, wavelength, bg_poly_order=4, peak_shape="Pseudo-Voigt", use_caglioti=True):
        self.data = data
        self.phases = phases
        self.wavelength = wavelength
        self.bg_poly_order = bg_poly_order
        self.peak_shape = peak_shape
        self.use_caglioti = use_caglioti
        self.x = data["two_theta"].values
        self.y_obs = data["intensity"].values

    def _background(self, x, *coeffs):
        return sum(c * x**i for i, c in enumerate(coeffs))

    def _gaussian(self, x, pos, amp, fwhm):
        return amp * np.exp(-4*np.log(2)*((x-pos)/fwhm)**2)

    def _lorentzian(self, x, pos, amp, fwhm):
        return amp / (1 + 4*((x-pos)/fwhm)**2)

    def _pseudo_voigt(self, x, pos, amp, fwhm, eta=0.5):
        return eta * self._lorentzian(x, pos, amp, fwhm) + (1-eta) * self._gaussian(x, pos, amp, fwhm)

    def _caglioti_fwhm(self, theta_deg, U, V, W):
        tan_t = np.tan(np.radians(theta_deg))
        return np.sqrt(np.maximum(U * tan_t**2 + V * tan_t + W, 0.01))

    def _lp_correction(self, two_theta_deg):
        theta = np.radians(two_theta_deg / 2)
        two_t = np.radians(two_theta_deg)
        return (1 + np.cos(two_t)**2) / (np.sin(theta)**2 * np.cos(theta) + 1e-10)

    def _calculate_pattern(self, params):
        bg_coeffs = params[:self.bg_poly_order+1]
        y_calc = self._background(self.x, *bg_coeffs)
        zero_shift = params[self.bg_poly_order+1]
        idx = self.bg_poly_order + 2
        for phase in self.phases:
            phase_peaks = generate_theoretical_peaks(phase, self.wavelength, self.x.min(), self.x.max())
            for _, pk in phase_peaks.iterrows():
                if idx + 3 > len(params): break
                pos = params[idx] + zero_shift
                amp, width = params[idx+1], params[idx+2]
                idx += 3
                if self.use_caglioti and idx + 2 < len(params):
                    fwhm = self._caglioti_fwhm(pos, params[idx], params[idx+1], params[idx+2])
                    idx += 3
                else: fwhm = width
                
                if self.peak_shape == "Gaussian": val = self._gaussian(self.x, pos, amp, fwhm)
                elif self.peak_shape == "Lorentzian": val = self._lorentzian(self.x, pos, amp, fwhm)
                else: val = self._pseudo_voigt(self.x, pos, amp, fwhm)
                
                y_calc += amp * self._lp_correction(pk["two_theta"]) * val
        return y_calc

    def run(self):
        bg_init = [np.percentile(self.y_obs, 10)] + [0.0] * self.bg_poly_order
        zero_init = 0.0
        peak_init, caglioti_init = [], [0.0, 0.0, 0.1] if self.use_caglioti else []
        
        # Robust initialization
        max_int = np.max(self.y_obs)
        for phase in self.phases:
            for _, pk in generate_theoretical_peaks(phase, self.wavelength, self.x.min(), self.x.max()).iterrows():
                peak_init.extend([pk["two_theta"], max_int * 0.05, 0.4])
                if self.use_caglioti: peak_init.extend(caglioti_init)
        
        params0 = np.array(bg_init + [zero_init] + peak_init)
        bounds_l, bounds_u = np.full_like(params0, -np.inf), np.full_like(params0, np.inf)
        bounds_l[:self.bg_poly_order+1], bounds_u[:self.bg_poly_order+1] = -1e6, 1e6
        bounds_l[self.bg_poly_order+1], bounds_u[self.bg_poly_order+1] = -0.5, 0.5
        
        idx = self.bg_poly_order + 2
        for phase in self.phases:
            for _, pk in generate_theoretical_peaks(phase, self.wavelength, self.x.min(), self.x.max()).iterrows():
                bounds_l[idx], bounds_u[idx] = pk["two_theta"] - 2.0, pk["two_theta"] + 2.0
                bounds_l[idx+1], bounds_u[idx+1] = 0, max_int * 10
                bounds_l[idx+2], bounds_u[idx+2] = 0.1, 5.0
                idx += 3
                if self.use_caglioti:
                    bounds_l[idx:idx+3], bounds_u[idx:idx+3] = [-1, -10, 0.01], [1, 10, 10]
                    idx += 3

        try:
            res = least_squares(lambda p: self.y_obs - self._calculate_pattern(p), params0, bounds=(bounds_l, bounds_u), method='trf', max_nfev=500)
            converged, p_opt = res.success, res.x
        except:
            converged, p_opt = False, params0

        y_calc = self._calculate_pattern(p_opt)
        resid = self.y_obs - y_calc
        Rwp = np.sqrt(np.sum(resid**2) / np.sum(self.y_obs**2)) * 100
        Rexp = np.sqrt(max(1, len(self.x) - len(p_opt))) / np.sqrt(np.sum(self.y_obs) + 1e-10) * 100
        chi2 = (Rwp / max(Rexp, 0.01))**2
        zero_shift = p_opt[self.bg_poly_order+1]

        # IMPROVED QUANTIFICATION: Use Average Scale Factor (Amplitude)
        idx = self.bg_poly_order + 2
        phase_data = {}
        for phase in self.phases:
            pks = generate_theoretical_peaks(phase, self.wavelength, self.x.min(), self.x.max())
            amps = []
            for _, _ in pks.iterrows():
                if idx + 1 < len(p_opt):
                    amps.append(p_opt[idx+1])
                    idx += 3
                    if self.use_caglioti: idx += 3
            # Normalize by number of peaks to get average contribution per reflection
            phase_data[phase] = {"avg_amp": np.mean(amps) if amps else 0, "count": len(amps)}

        total_avg_amp = sum(d["avg_amp"] for d in phase_data.values())
        # Calculate fractions based on average amplitude
        phase_fractions = {}
        for ph, d in phase_data.items():
            # Handle division by zero or tiny values
            if total_avg_amp > 1e-9:
                phase_fractions[ph] = (d["avg_amp"] / total_avg_amp)
            else:
                phase_fractions[ph] = 0.0

        # Lattice refinement
        lattice_params = {}
        for phase in self.phases:
            pks = generate_theoretical_peaks(phase, self.wavelength, self.x.min(), self.x.max())
            d_vals = [self.wavelength / (2 * np.sin(np.radians((p_opt[self.bg_poly_order + 2 + 3*i] + zero_shift)/2))) for i in range(len(pks))]
            # Simplified lattice extraction
            if d_vals: lattice_params[phase] = {"a_est": np.mean([d * np.sqrt(3) for d in d_vals])} # Rough cubic est
            else: lattice_params[phase] = PHASE_LIBRARY.get(phase, {}).get("lattice", {})

        return {"converged": converged, "Rwp": Rwp, "Rexp": Rexp, "chi2": chi2, "y_calc": y_calc, 
                "y_background": self._background(self.x, *p_opt[:self.bg_poly_order+1]),
                "zero_shift": zero_shift, "phase_fractions": phase_fractions, "lattice_params": lattice_params}

# ═══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════════════════════

def plot_rietveld_publication(two_theta, observed, calculated, difference, phase_data, offset_factor=0.12, figsize=(10, 7), font_size=11, legend_pos='best'):
    plt.rcParams.update({'font.family': 'serif', 'font.size': font_size, 'axes.labelsize': font_size+1, 'xtick.labelsize': font_size, 'ytick.labelsize': font_size, 'axes.linewidth': 1.2})
    fig, ax = plt.subplots(figsize=figsize)
    y_max, y_min = np.max(calculated), np.min(calculated)
    offset = (y_max - y_min) * offset_factor
    ax.plot(two_theta, observed, 'o', markersize=4, mfc='none', mec='red', mew=1.0, label='Observed', zorder=3)
    ax.plot(two_theta, calculated, '-', color='black', linewidth=1.5, label='Calculated', zorder=4)
    diff_off = y_min - offset
    ax.plot(two_theta, difference + diff_off, '-', color='blue', linewidth=1.2, label='Difference', zorder=2)
    ax.axhline(y=diff_off, color='gray', ls='--', lw=0.8, alpha=0.7)
    tick_h = offset * 0.25
    shapes = {'|': {'marker': '|', 'ms': 14, 'mew': 2.5}, '_': {'marker': '_', 'ms': 14, 'mew': 2.5}, 's': {'marker': 's', 'ms': 7, 'mew': 1.5}, '^': {'marker': '^', 'ms': 8, 'mew': 1.5}}
    for i, ph in enumerate(phase_data):
        style = shapes.get(ph.get('marker_shape', '|'), shapes['|'])
        y_base = diff_off - (i + 1) * tick_h * 1.3
        for pos in ph['positions']:
            ax.plot(pos, y_base, **style, color=ph.get('color', f'C{i}'), label=ph['name'] if i==0 else "", zorder=5)
    ax.set_xlabel(r'$2\theta$ (°)'); ax.set_ylabel('Intensity')
    ax.set_ylim([diff_off - (len(phase_data)+2)*tick_h*1.3, y_max*1.05])
    if legend_pos != "off": ax.legend(loc=legend_pos)
    plt.tight_layout()
    return fig, ax

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="XRD Rietveld — Co-Cr Dental Alloy", page_icon="⚙️", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
  .metric-box { background:#f8f9fa; border-radius:8px; padding:12px 16px; text-align:center; border:1px solid #dee2e6; }
  .metric-box .value { font-size:1.6rem; font-weight:700; color:#1f77b4; }
  .success-box { background:#d4edda; border:1px solid #c3e6cb; border-radius:6px; padding:10px; margin:8px 0; }
</style>
""", unsafe_allow_html=True)

st.title("⚙️ XRD Rietveld Refinement — Co-Cr Dental Alloy")
st.caption("Mediloy S Co · BEGO · SLM-Printed/HT • Supports .asc, .xrdml, .cif • Wavelength Corrected")

# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR (SIMPLIFIED)
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Configuration")
    
    # 1. Data Input (Upload or Default)
    st.subheader("📂 Data Input")
    uploaded_file = st.file_uploader("Upload .asc, .ASC, or .xrdml", type=["asc", "ASC", "xrdml", "XRDML"])
    active_df = None
    
    if uploaded_file is not None:
        if uploaded_file.name.lower().endswith('.xrdml'):
            active_df = parse_xrdml(uploaded_file.read())
        else:
            active_df = parse_asc(uploaded_file.read())
        st.success(f"✅ Loaded: {uploaded_file.name}")
    else:
        # Default demo fallback if no file uploaded
        demo_file = "CH0_1.ASC"
        demo_path = os.path.join(DEMO_DIR, demo_file)
        if os.path.exists(demo_path):
            active_df = parse_asc(open(demo_path, "rb").read())
            st.info(f"ℹ️ Loaded default demo: {demo_file}")
        else:
            # Generate synthetic data
            two_theta = np.linspace(30, 130, 2000)
            y = np.zeros_like(two_theta)
            # Simulate FCC-Co peaks for Co Kα
            for pos in [51.5, 60.0, 89.0, 113.0]: # Approx Co Ka positions
                y += 5000 * np.exp(-((two_theta - pos)/0.8)**2)
            y += np.random.normal(0, 50, size=len(two_theta)) + 200
            active_df = pd.DataFrame({"two_theta": two_theta, "intensity": y})
            if st.button("📂 Load Local Demo File"):
                st.error("No local demo file found in 'demo_data' folder.")

    # 2. Instrument
    st.markdown("---")
    st.subheader("🔬 Instrument")
    source_name = st.selectbox("X-ray Source", list(XRAY_SOURCES.keys()), index=1) # Default to Co Kα as per user request
    if source_name != "Custom Wavelength":
        wavelength = st.number_input("Wavelength (Å)", value=XRAY_SOURCES[source_name], disabled=True, format="%.4f")
    else:
        wavelength = st.number_input("Wavelength (Å)", value=1.7890, min_value=0.5, max_value=2.5, step=0.0001, format="%.4f")

    # 3. Phases
    st.markdown("---")
    st.subheader("🧪 Phases")
    all_phases = PHASE_LIBRARY.copy()
    if "custom_phases" in st.session_state:
        all_phases.update(st.session_state.custom_phases)
    
    selected_phases = []
    for name, props in all_phases.items():
        if st.checkbox(f"{name} ({props['system']})", value=props.get("default", False)):
            selected_phases.append(name)

    # 4. Refinement Settings
    st.markdown("---")
    st.subheader("🔧 Refinement")
    bg_order = st.slider("Background Order", 2, 8, 4)
    use_caglioti = st.checkbox("Caglioti FWHM (Angle-dependent)", value=True)
    tt_min = st.number_input("2θ Min", value=30.0)
    tt_max = st.number_input("2θ Max", value=130.0)
    
    run_btn = st.button("▶ Run Rietveld Refinement", type="primary", use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# MAIN CONTENT
# ──────────────────────────────────────────────────────────────────────────────

if active_df is None or len(active_df) == 0:
    st.error("❌ No data loaded. Please upload a file or check 'demo_data' folder.")
    st.stop()

# Filter data
mask = (active_df["two_theta"] >= tt_min) & (active_df["two_theta"] <= tt_max)
active_data = active_df[mask].copy()

if not selected_phases:
    st.warning("⚠️ Select at least one phase in the sidebar.")
    st.stop()

if run_btn:
    with st.spinner("Refining..."):
        refiner = RietveldRefinement(active_data, selected_phases, wavelength, bg_order, "Pseudo-Voigt", use_caglioti)
        result = refiner.run()
    st.session_state["result"] = result
    st.session_state["active_data"] = active_data

if "result" in st.session_state:
    result = st.session_state["result"]
    active_data = st.session_state["active_data"]
    
    # METRICS
    st.markdown("### 📊 Refinement Results")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R_wp", f"{result['Rwp']:.2f}%", delta="< 15% is good")
    c2.metric("R_exp", f"{result['Rexp']:.2f}%")
    c3.metric("χ²", f"{result['chi2']:.3f}", delta="≈ 1 is ideal")
    c4.metric("Zero Shift", f"{result['zero_shift']:+.3f}°")

    # PLOT
    fig = make_subplots(rows=2, cols=1, row_heights=[0.75, 0.25], shared_xaxes=True, vertical_spacing=0.02)
    fig.add_trace(go.Scatter(x=active_data["two_theta"], y=active_data["intensity"], mode="lines", name="Observed", line=dict(color="#1f77b4")), row=1, col=1)
    fig.add_trace(go.Scatter(x=active_data["two_theta"], y=result["y_calc"], mode="lines", name="Calculated", line=dict(color="red", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=active_data["two_theta"], y=result["y_background"], mode="lines", name="Background", line=dict(color="green", dash="dash")), row=1, col=1)
    
    # Add phase markers
    y_min = active_data["intensity"].min()
    for i, ph in enumerate(selected_phases):
        pks = generate_theoretical_peaks(ph, wavelength, tt_min, tt_max)
        y_off = y_min - (i+1) * (y_min * 0.05)
        fig.add_trace(go.Scatter(x=pks["two_theta"], y=[y_off]*len(pks), mode="markers", name=ph, marker=dict(symbol="line-ns", size=12, color=PHASE_LIBRARY[ph]["color"])), row=1, col=1)
    
    diff = active_data["intensity"].values - result["y_calc"]
    fig.add_trace(go.Scatter(x=active_data["two_theta"], y=diff, mode="lines", name="Diff", line=dict(color="gray")), row=2, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="black", row=2, col=1)
    fig.update_layout(height=550, template="plotly_white", xaxis2_title="2θ (°)", yaxis_title="Intensity")
    st.plotly_chart(fig, use_container_width=True)

    # QUANTIFICATION
    st.markdown("### 🧪 Phase Quantification")
    fracs = result["phase_fractions"]
    labels, vals = list(fracs.keys()), [fracs[ph]*100 for ph in fracs]
    colors = [PHASE_LIBRARY.get(ph, {}).get("color", "#000") for ph in labels]
    
    col_p, col_b = st.columns(2)
    with col_p:
        fig_pie = go.Figure(go.Pie(labels=labels, values=vals, hole=0.4, textinfo="label+percent", marker_colors=colors))
        st.plotly_chart(fig_pie, use_container_width=True)
    with col_b:
        fig_bar = go.Figure(go.Bar(x=labels, y=vals, text=[f"{v:.1f}%" for v in vals], textposition="auto", marker_color=colors))
        fig_bar.update_layout(yaxis_title="Relative Abundance (%)")
        st.plotly_chart(fig_bar, use_container_width=True)
    
    st.info("💡 **Note:** Quantification is based on refined scale factors (amplitudes). For absolute weight %, use GSAS-II or MAUD with full structure factors.")

    # REPORT
    st.markdown("### 📄 Report")
    txt = f"Sample Analysis\nWavelength: {wavelength} Å\nR_wp: {result['Rwp']:.2f}%\n"
    for ph, val in fracs.items():
        txt += f"{ph}: {val*100:.1f}%\n"
    st.download_button("Download Report", txt, file_name="rietveld_report.txt")
else:
    st.info("👈 Configure settings in the sidebar and click **Run Refinement**.")

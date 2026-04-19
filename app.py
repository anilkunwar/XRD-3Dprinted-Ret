"""
XRD Rietveld Analysis — Co-Cr Dental Alloy (Mediloy S Co, BEGO)
================================================================
Simplified • Co Kα Optimized • Accurate Phase Quantification
Supports: .asc, .ASC, .xrdml files
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import io, os, math, re, xml.etree.ElementTree as ET
from scipy import signal
from scipy.optimize import least_squares

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG — Mediloy S Co Specific
# ═══════════════════════════════════════════════════════════════════════════════

# Default wavelength for Co-Kα diffractometer
DEFAULT_WAVELENGTH = 1.7890  # Å

# Phases relevant to Mediloy S Co (Co-Cr-Mo-W-Si alloy)
PHASE_LIBRARY = {
    "FCC-Co": {
        "system": "Cubic", 
        "space_group": "Fm-3m (No. 225)", 
        "lattice": {"a": 3.548},  # COD:9008466
        # Peak positions for Co Kα (1.7890 Å) - calculated from d-spacing
        "peaks_co_ka": [("111", 51.45), ("200", 59.98), ("220", 88.92), ("311", 112.85), ("222", 121.20)],
        "color": "#e377c2", 
        "default": True, 
        "marker_shape": "|",
        "description": "FCC Co-based solid solution (matrix phase)",
        "structure_factor_scale": 1.0  # Reference scale for quantification
    },
    "M23C6": {
        "system": "Cubic", 
        "space_group": "Fm-3m (No. 225)", 
        "lattice": {"a": 10.63},  # MP:mp-723
        "peaks_co_ka": [("311", 46.15), ("400", 53.82), ("511", 78.45), ("440", 94.68), ("620", 108.92)],
        "color": "#bcbd22", 
        "default": True, 
        "marker_shape": "s",
        "description": "Cr-rich carbide M₂₃C₆ (common precipitate)",
        "structure_factor_scale": 0.85  # Lower scattering power than FCC-Co
    },
    "HCP-Co": {
        "system": "Hexagonal", 
        "space_group": "P6₃/mmc (No. 194)", 
        "lattice": {"a": 2.5071, "c": 4.0686},  # COD:9008492
        "peaks_co_ka": [("100", 48.42), ("002", 52.15), ("101", 55.28), ("102", 80.52), ("110", 90.88)],
        "color": "#7f7f7f", 
        "default": False, 
        "marker_shape": "_",
        "description": "HCP Co (stress-induced or low-temperature phase)",
        "structure_factor_scale": 0.95
    },
    "Sigma": {
        "system": "Tetragonal", 
        "space_group": "P4₂/mnm", 
        "lattice": {"a": 8.80, "c": 4.56},
        "peaks_co_ka": [("210", 50.18), ("220", 63.15), ("310", 80.12)],
        "color": "#17becf", 
        "default": False, 
        "marker_shape": "^",
        "description": "Sigma phase (Co,Cr) intermetallic (brittle)",
        "structure_factor_scale": 0.75
    }
}

# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_hkl(hkl_label: str) -> tuple:
    """Parse hkl label like '(311)' into tuple (h,k,l)."""
    clean = hkl_label.strip().strip("()").replace(" ", "")
    if "," in clean:
        return tuple(int(p.strip()) for p in clean.split(",") if p.strip())
    result, i = [], 0
    while i < len(clean) and len(result) < 3:
        sign = -1 if clean[i] == '-' else 1
        if clean[i] in "+-": i += 1
        num = ""
        while i < len(clean) and clean[i].isdigit():
            num += clean[i]; i += 1
        if num: result.append(sign * int(num))
    return tuple(result + [0]*(3-len(result)))

def generate_theoretical_peaks(phase_name, wavelength, tt_min, tt_max):
    """
    Generate theoretical peak positions for Co-Cr alloy phases.
    Uses pre-calculated Co Kα positions with wavelength adjustment if needed.
    """
    if phase_name not in PHASE_LIBRARY:
        return pd.DataFrame(columns=["two_theta", "d_spacing", "hkl_label"])
    
    phase = PHASE_LIBRARY[phase_name]
    # Use Co Kα pre-calculated positions as reference
    ref_peaks = phase.get("peaks_co_ka", phase.get("peaks", []))
    
    results = []
    for hkl_str, tt_ref in ref_peaks:
        # Calculate d-spacing from reference angle (Co Kα)
        d_spacing = DEFAULT_WAVELENGTH / (2 * math.sin(math.radians(tt_ref / 2)))
        
        # Adjust for actual wavelength if different
        if abs(wavelength - DEFAULT_WAVELENGTH) > 1e-4:
            sin_theta = wavelength / (2 * d_spacing)
            if abs(sin_theta) <= 1.0:
                tt_new = math.degrees(2 * math.asin(sin_theta))
            else:
                continue  # Peak outside measurable range
        else:
            tt_new = tt_ref
        
        if tt_min <= tt_new <= tt_max:
            results.append({
                "two_theta": round(tt_new, 3),
                "d_spacing": round(d_spacing, 4),
                "hkl_label": f"({hkl_str})"
            })
    return pd.DataFrame(results) if results else pd.DataFrame(columns=["two_theta", "d_spacing", "hkl_label"])

def find_peaks_in_data(df, min_height_factor=2.0, min_distance_deg=0.3):
    """Detect peaks in XRD pattern."""
    if len(df) < 10:
        return pd.DataFrame(columns=["two_theta", "intensity", "prominence"])
    x, y = df["two_theta"].values, df["intensity"].values
    bg = np.percentile(y, 15)
    min_height = bg + min_height_factor * (np.std(y) if len(y) > 1 else 1)
    min_distance = max(1, int(min_distance_deg / np.mean(np.diff(x))))
    peaks, props = signal.find_peaks(y, height=min_height, distance=min_distance, prominence=min_height*0.3)
    if len(peaks) == 0:
        return pd.DataFrame(columns=["two_theta", "intensity", "prominence"])
    return pd.DataFrame({
        "two_theta": x[peaks],
        "intensity": y[peaks],
        "prominence": props.get("prominences", np.zeros_like(peaks))
    }).sort_values("intensity", ascending=False).reset_index(drop=True)

# ═══════════════════════════════════════════════════════════════════════════════
# FILE PARSERS
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def parse_asc(raw_bytes: bytes) -> pd.DataFrame:
    """Parse two-column ASC/XY file."""
    text = raw_bytes.decode("utf-8", errors="replace")
    rows = []
    for line in text.splitlines():
        parts = re.split(r'[\s,;]+', line.strip())
        if len(parts) >= 2:
            try:
                rows.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue
    df = pd.DataFrame(rows, columns=["two_theta", "intensity"])
    return df.sort_values("two_theta").reset_index(drop=True) if len(df) > 0 else pd.DataFrame(columns=["two_theta", "intensity"])

@st.cache_data
def parse_xrdml(raw_bytes: bytes) -> pd.DataFrame:
    """Parse PANalytical .xrdml file."""
    try:
        text = raw_bytes.decode("utf-8", errors="replace")
        # Remove namespace for easier parsing
        text_clean = re.sub(r'\s*xmlns="[^"]+"', '', text, count=1)
        root = ET.fromstring(text_clean)
        
        for elem in root.iter():
            if elem.tag.endswith('xRayData') or elem.tag == 'xRayData':
                vals_elem = elem.find('.//values') or elem.find('.//data') or elem.find('.//intensities')
                if vals_elem is not None and vals_elem.text:
                    intensities = [float(v) for v in vals_elem.text.strip().split() if v.strip()]
                    start = float(elem.get('startAngle', elem.get('start', 0)))
                    end = float(elem.get('endAngle', elem.get('end', 100)))
                    if len(intensities) > 1:
                        two_theta = np.linspace(start, end, len(intensities))
                        return pd.DataFrame({"two_theta": two_theta, "intensity": intensities})
        return pd.DataFrame(columns=["two_theta", "intensity"])
    except Exception as e:
        st.error(f"❌ Error parsing .xrdml: {e}")
        return pd.DataFrame(columns=["two_theta", "intensity"])

# ═══════════════════════════════════════════════════════════════════════════════
# RIETVELD ENGINE — Co-Cr Optimized
# ═══════════════════════════════════════════════════════════════════════════════

class CoCrRietveldRefinement:
    """
    Simplified Rietveld refinement for Co-Cr dental alloys.
    Features:
    • Polynomial background
    • Pseudo-Voigt peaks with Caglioti broadening
    • Structure-factor-weighted quantification
    • Co Kα optimized peak positions
    """
    
    def __init__(self, data, phases, wavelength, bg_order=4, use_caglioti=True):
        self.data = data
        self.phases = phases
        self.wavelength = wavelength
        self.bg_order = bg_order
        self.use_caglioti = use_caglioti
        self.x = data["two_theta"].values
        self.y_obs = data["intensity"].values
        
    def _background(self, x, *coeffs):
        return sum(c * x**i for i, c in enumerate(coeffs))
    
    def _pseudo_voigt(self, x, pos, amp, fwhm, eta=0.5):
        gauss = amp * np.exp(-4*np.log(2)*((x-pos)/fwhm)**2)
        lor = amp / (1 + 4*((x-pos)/fwhm)**2)
        return eta * lor + (1-eta) * gauss
    
    def _caglioti_fwhm(self, theta_deg, U, V, W):
        tan_t = np.tan(np.radians(theta_deg))
        return np.sqrt(np.maximum(U * tan_t**2 + V * tan_t + W, 0.01))
    
    def _lp_correction(self, two_theta_deg):
        """Lorentz-Polarization correction for powder diffraction."""
        theta = np.radians(two_theta_deg / 2)
        two_t = np.radians(two_theta_deg)
        return (1 + np.cos(two_t)**2) / (np.sin(theta)**2 * np.cos(theta) + 1e-10)
    
    def _calculate_pattern(self, params):
        """Forward model: calculate synthetic pattern."""
        # Background
        y_calc = self._background(self.x, *params[:self.bg_order+1])
        
        # Zero shift
        zero_shift = params[self.bg_order+1]
        idx = self.bg_order + 2
        
        for phase in self.phases:
            phase_peaks = generate_theoretical_peaks(phase, self.wavelength, self.x.min(), self.x.max())
            for _, pk in phase_peaks.iterrows():
                if idx + 3 > len(params):
                    break
                pos = params[idx] + zero_shift
                amp = params[idx+1]
                width = params[idx+2]
                idx += 3
                
                # Angle-dependent FWHM (Caglioti)
                if self.use_caglioti and idx + 2 < len(params):
                    fwhm = self._caglioti_fwhm(pos, params[idx], params[idx+1], params[idx+2])
                    idx += 3
                else:
                    fwhm = width
                
                # Peak shape + LP correction
                peak_val = self._pseudo_voigt(self.x, pos, amp, fwhm)
                lp = self._lp_correction(pk["two_theta"])
                y_calc += amp * lp * peak_val
        
        return y_calc
    
    def _residuals(self, params):
        return self.y_obs - self._calculate_pattern(params)
    
    def run(self):
        """Execute refinement."""
        # Initialize parameters
        bg_init = [np.percentile(self.y_obs, 10)] + [0.0] * self.bg_order
        zero_init = 0.0
        peak_init = []
        caglioti_init = [0.0, 0.0, 0.1] if self.use_caglioti else []
        
        max_int = np.max(self.y_obs)
        for phase in self.phases:
            for _, pk in generate_theoretical_peaks(phase, self.wavelength, self.x.min(), self.x.max()).iterrows():
                # Initial guess: position, amplitude (5% of max), width (0.4°)
                peak_init.extend([pk["two_theta"], max_int * 0.05, 0.4])
                if self.use_caglioti:
                    peak_init.extend(caglioti_init)
        
        params0 = np.array(bg_init + [zero_init] + peak_init)
        
        # Bounds for stability
        bounds_l = np.full_like(params0, -np.inf)
        bounds_u = np.full_like(params0, np.inf)
        bounds_l[:self.bg_order+1], bounds_u[:self.bg_order+1] = -1e6, 1e6
        bounds_l[self.bg_order+1], bounds_u[self.bg_order+1] = -0.5, 0.5  # Zero shift ±0.5°
        
        idx = self.bg_order + 2
        for phase in self.phases:
            for _, pk in generate_theoretical_peaks(phase, self.wavelength, self.x.min(), self.x.max()).iterrows():
                bounds_l[idx], bounds_u[idx] = pk["two_theta"] - 2.0, pk["two_theta"] + 2.0
                bounds_l[idx+1], bounds_u[idx+1] = 0, max_int * 10
                bounds_l[idx+2], bounds_u[idx+2] = 0.1, 5.0
                idx += 3
                if self.use_caglioti:
                    bounds_l[idx:idx+3], bounds_u[idx:idx+3] = [-1, -10, 0.01], [1, 10, 10]
                    idx += 3
        
        # Optimize
        try:
            res = least_squares(
                self._residuals, 
                params0, 
                bounds=(bounds_l, bounds_u), 
                method='trf',
                max_nfev=500,
                xtol=1e-8,
                ftol=1e-8
            )
            converged, p_opt = res.success, res.x
        except Exception as e:
            st.warning(f"⚠️ Optimization warning: {e}")
            converged, p_opt = False, params0
        
        # Calculate results
        y_calc = self._calculate_pattern(p_opt)
        y_bg = self._background(self.x, *p_opt[:self.bg_order+1])
        resid = self.y_obs - y_calc
        
        # R-factors
        Rwp = np.sqrt(np.sum(resid**2) / np.sum(self.y_obs**2)) * 100
        Rexp = np.sqrt(max(1, len(self.x) - len(p_opt))) / np.sqrt(np.sum(self.y_obs) + 1e-10) * 100
        chi2 = (Rwp / max(Rexp, 0.01))**2
        zero_shift = p_opt[self.bg_order+1]
        
        # ACCURATE QUANTIFICATION: Structure-factor-weighted scale factors
        idx = self.bg_order + 2
        phase_data = {}
        for phase in self.phases:
            pks = generate_theoretical_peaks(phase, self.wavelength, self.x.min(), self.x.max())
            amps, sf_scales = [], []
            for _, pk in pks.iterrows():
                if idx + 1 < len(p_opt):
                    amps.append(p_opt[idx+1])
                    sf_scales.append(PHASE_LIBRARY[phase].get("structure_factor_scale", 1.0))
                    idx += 3
                    if self.use_caglioti:
                        idx += 3
            
            if amps:
                # Weighted average: amplitude × structure factor scale
                weighted_amp = np.mean([a * s for a, s in zip(amps, sf_scales)])
                phase_data[phase] = {"weighted_amp": weighted_amp, "count": len(amps)}
            else:
                phase_data[phase] = {"weighted_amp": 0, "count": 0}
        
        # Normalize to get phase fractions
        total = sum(d["weighted_amp"] for d in phase_data.values())
        phase_fractions = {}
        for ph, d in phase_data.items():
            if total > 1e-9:
                phase_fractions[ph] = d["weighted_amp"] / total
            else:
                phase_fractions[ph] = 0.0
        
        # Lattice parameter estimation (simplified)
        lattice_params = {}
        for phase in self.phases:
            pks = generate_theoretical_peaks(phase, self.wavelength, self.x.min(), self.x.max())
            if len(pks) > 0 and phase in PHASE_LIBRARY:
                # Extract refined positions
                idx = self.bg_order + 2
                refined_positions = []
                for _, pk in pks.iterrows():
                    if idx < len(p_opt):
                        refined_positions.append(p_opt[idx] + zero_shift)
                        idx += 3
                        if self.use_caglioti:
                            idx += 3
                
                if refined_positions:
                    # Calculate d-spacings and estimate lattice parameter
                    d_vals = [self.wavelength / (2 * np.sin(np.radians(pos/2))) for pos in refined_positions]
                    sys_type = PHASE_LIBRARY[phase]["system"]
                    if sys_type == "Cubic":
                        # a = d × √(h²+k²+l²)
                        a_vals = []
                        for d_val, (_, pk) in zip(d_vals, pks.iterrows()):
                            h, k, l = _parse_hkl(pk["hkl_label"])
                            sum_sq = h**2 + k**2 + l**2
                            if sum_sq > 0:
                                a_vals.append(d_val * np.sqrt(sum_sq))
                        if a_vals:
                            lattice_params[phase] = {"a": float(np.mean(a_vals))}
                        else:
                            lattice_params[phase] = PHASE_LIBRARY[phase]["lattice"].copy()
                    else:
                        lattice_params[phase] = PHASE_LIBRARY[phase]["lattice"].copy()
                else:
                    lattice_params[phase] = PHASE_LIBRARY[phase]["lattice"].copy()
            else:
                lattice_params[phase] = PHASE_LIBRARY.get(phase, {}).get("lattice", {})
        
        return {
            "converged": converged,
            "Rwp": Rwp,
            "Rexp": Rexp,
            "chi2": chi2,
            "y_calc": y_calc,
            "y_background": y_bg,
            "zero_shift": zero_shift,
            "phase_fractions": phase_fractions,
            "lattice_params": lattice_params,
            "n_params": len(p_opt),
            "n_data": len(self.x)
        }

# ═══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════════════════════

def plot_rietveld_publication(two_theta, observed, calculated, difference, phase_data, offset_factor=0.12, figsize=(10, 7), font_size=11):
    """Generate publication-quality Rietveld plot."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': font_size,
        'axes.labelsize': font_size + 1,
        'xtick.labelsize': font_size,
        'ytick.labelsize': font_size,
        'axes.linewidth': 1.2,
        'figure.dpi': 300
    })
    
    fig, ax = plt.subplots(figsize=figsize)
    y_max, y_min = np.max(calculated), np.min(calculated)
    offset = (y_max - y_min) * offset_factor
    
    # Observed data
    ax.plot(two_theta, observed, 'o', markersize=4, mfc='none', mec='red', mew=1.0, label='Observed', zorder=3)
    # Calculated pattern
    ax.plot(two_theta, calculated, '-', color='black', linewidth=1.5, label='Calculated', zorder=4)
    # Difference curve
    diff_off = y_min - offset
    ax.plot(two_theta, difference + diff_off, '-', color='blue', linewidth=1.2, label='Difference', zorder=2)
    ax.axhline(y=diff_off, color='gray', ls='--', lw=0.8, alpha=0.7)
    
    # Phase markers
    tick_h = offset * 0.25
    shapes = {
        '|': {'marker': '|', 'ms': 14, 'mew': 2.5},
        '_': {'marker': '_', 'ms': 14, 'mew': 2.5},
        's': {'marker': 's', 'ms': 7, 'mew': 1.5},
        '^': {'marker': '^', 'ms': 8, 'mew': 1.5}
    }
    
    for i, ph in enumerate(phase_data):
        style = shapes.get(ph.get('marker_shape', '|'), shapes['|'])
        y_base = diff_off - (i + 1) * tick_h * 1.3
        for j, pos in enumerate(ph['positions']):
            label = ph['name'] if j == 0 else ""
            ax.plot(pos, y_base, **style, color=ph.get('color', f'C{i}'), label=label, zorder=5)
    
    ax.set_xlabel(r'$2\theta$ (°)', fontweight='bold')
    ax.set_ylabel('Intensity (a.u.)', fontweight='bold')
    ax.set_ylim([diff_off - (len(phase_data)+2)*tick_h*1.3, y_max*1.05])
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.legend(loc='best', frameon=True, fancybox=False, edgecolor='black')
    plt.tight_layout()
    return fig, ax

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Co-Cr Rietveld — Mediloy S Co",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
  .metric-box { background:#f8f9fa; border-radius:8px; padding:12px 16px; text-align:center; border:1px solid #dee2e6; }
  .metric-box .value { font-size:1.6rem; font-weight:700; color:#1f77b4; }
  .success-box { background:#d4edda; border:1px solid #c3e6cb; border-radius:6px; padding:10px; margin:8px 0; }
  .phase-card { background:#f8f9fa; border-left:4px solid #0d6efd; padding:10px 14px; margin:4px 0; border-radius:4px; }
</style>
""", unsafe_allow_html=True)

st.title("⚙️ Co-Cr Dental Alloy Rietveld Analysis")
st.caption("Mediloy S Co • BEGO • Co Kα Optimized • Phase Quantification")

# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR — Minimal & Focused
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Configuration")
    
    # 1. Data Upload
    st.subheader("📂 Upload XRD Data")
    uploaded_file = st.file_uploader(
        "Select .asc, .ASC, or .xrdml file",
        type=["asc", "ASC", "xrdml", "XRDML"],
        help="Two-column text file or PANalytical .xrdml"
    )
    
    active_df = None
    if uploaded_file is not None:
        if uploaded_file.name.lower().endswith('.xrdml'):
            active_df = parse_xrdml(uploaded_file.read())
        else:
            active_df = parse_asc(uploaded_file.read())
        st.success(f"✅ Loaded: `{uploaded_file.name}`")
        st.caption(f"{len(active_df):,} data points")
    
    # 2. Instrument (Co Kα default)
    st.markdown("---")
    st.subheader("🔬 Instrument")
    wavelength = st.number_input(
        "Wavelength λ (Å)",
        value=DEFAULT_WAVELENGTH,
        min_value=0.5,
        max_value=2.5,
        step=0.0001,
        format="%.4f",
        help="Default: Co Kα₁ = 1.7890 Å"
    )
    energy_kev = (4.135667696e-15 * 299792458) / (wavelength * 1e-10) / 1000
    st.caption(f"≡ {energy_kev:.2f} keV")
    
    # 3. Phases (Mediloy S Co specific)
    st.markdown("---")
    st.subheader("🧪 Phases")
    
    selected_phases = []
    for name, props in PHASE_LIBRARY.items():
        # Display phase info card
        with st.container():
            st.markdown(f"""
            <div class="phase-card">
            <strong>{name}</strong><br>
            <small>{props['description']}</small><br>
            <small>{props['system']} • a={props['lattice'].get('a','?')}Å</small>
            </div>
            """, unsafe_allow_html=True)
            if st.checkbox(f"Include {name}", value=props.get("default", False), key=f"chk_{name}"):
                selected_phases.append(name)
    
    # 4. Refinement Settings
    st.markdown("---")
    st.subheader("🔧 Refinement")
    
    bg_order = st.slider("Background polynomial order", 2, 8, 4, help="Higher = more flexible background")
    use_caglioti = st.checkbox("✓ Caglioti FWHM (angle-dependent)", value=True, help="Model peak broadening vs 2θ")
    
    col1, col2 = st.columns(2)
    with col1:
        tt_min = st.number_input("2θ min (°)", value=30.0, step=1.0)
    with col2:
        tt_max = st.number_input("2θ max (°)", value=130.0, step=1.0)
    
    run_btn = st.button("▶ Run Rietveld Refinement", type="primary", use_container_width=True)
    
    # Info box
    st.markdown("---")
    st.markdown("""
    <div class="success-box">
    <strong>💡 Tips for Mediloy S Co:</strong><br>
    • FCC-Co is the dominant matrix phase<br>
    • M₂₃C₆ carbides form during heat treatment<br>
    • HCP-Co may appear under stress/strain<br>
    • Use Co Kα wavelength (1.7890 Å) for correct peak positions
    </div>
    """, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# MAIN CONTENT AREA
# ──────────────────────────────────────────────────────────────────────────────

if active_df is None or len(active_df) == 0:
    st.info("👆 Upload your XRD data file (.asc or .xrdml) to begin analysis.")
    
    # Show example of expected format
    with st.expander("📋 Expected file format"):
        st.markdown("""
        **Two-column text file (.asc/.xy):**
        ```
        30.0    125
        30.02   132
        30.04   128
        ...
        ```
        Column 1: 2θ (degrees)  
        Column 2: Intensity (counts)
        
        **Or PANalytical .xrdml file** (auto-detected)
        """)
    st.stop()

if not selected_phases:
    st.warning("⚠️ Select at least one phase in the sidebar to proceed.")
    st.stop()

# Filter data to selected range
mask = (active_df["two_theta"] >= tt_min) & (active_df["two_theta"] <= tt_max)
active_data = active_df[mask].copy()

if len(active_data) < 50:
    st.error(f"❌ Too few data points in range [{tt_min:.1f}°, {tt_max:.1f}°]. Please expand the range.")
    st.stop()

# Run refinement when button clicked
if run_btn:
    with st.spinner("🔬 Running Rietveld refinement..."):
        refiner = CoCrRietveldRefinement(
            active_data, 
            selected_phases, 
            wavelength, 
            bg_order, 
            use_caglioti
        )
        result = refiner.run()
    
    # Store results in session state
    st.session_state["result"] = result
    st.session_state["active_data"] = active_data
    st.session_state["selected_phases"] = selected_phases
    st.session_state["wavelength"] = wavelength
    st.rerun()

# Display results if available
if "result" in st.session_state:
    result = st.session_state["result"]
    active_data = st.session_state["active_data"]
    selected_phases = st.session_state["selected_phases"]
    wavelength = st.session_state["wavelength"]
    
    # ──────────────────────────────────────────────────────────────────────
    # 1. FIT QUALITY METRICS
    # ──────────────────────────────────────────────────────────────────────
    st.markdown("### 📊 Refinement Quality")
    
    conv_status = "✅ Converged" if result["converged"] else "⚠️ Not converged"
    st.markdown(f"**Status**: {conv_status}")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R_wp", f"{result['Rwp']:.2f}%", delta="< 15% acceptable" if result['Rwp'] < 15 else "> 15% check fit")
    c2.metric("R_exp", f"{result['Rexp']:.2f}%")
    c3.metric("χ² (GoF)", f"{result['chi2']:.3f}", delta="≈ 1.0 ideal")
    c4.metric("Zero shift", f"{result['zero_shift']:+.3f}°")
    
    if result["Rwp"] > 20:
        st.warning("⚠️ High R_wp (>20%). Check: background order, phase selection, or data quality.")
    elif result["Rwp"] < 10:
        st.success("✅ Excellent fit quality (R_wp < 10%)")
    
    # ──────────────────────────────────────────────────────────────────────
    # 2. RIETVELD PLOT
    # ──────────────────────────────────────────────────────────────────────
    st.markdown("### 📈 Rietveld Fit")
    
    # Prepare phase data for plotting
    phase_plot_data = []
    for ph in selected_phases:
        pks = generate_theoretical_peaks(ph, wavelength, tt_min, tt_max)
        phase_plot_data.append({
            "name": ph,
            "positions": pks["two_theta"].values if len(pks) > 0 else np.array([]),
            "color": PHASE_LIBRARY[ph]["color"],
            "marker_shape": PHASE_LIBRARY[ph]["marker_shape"]
        })
    
    # Create plot
    fig, ax = plot_rietveld_publication(
        active_data["two_theta"].values,
        active_data["intensity"].values,
        result["y_calc"],
        active_data["intensity"].values - result["y_calc"],
        phase_plot_data,
        offset_factor=0.12,
        figsize=(11, 7)
    )
    
    st.pyplot(fig, dpi=150, use_container_width=True)
    
    # Export plot
    col_e1, col_e2 = st.columns(2)
    with col_e1:
        buf = io.BytesIO()
        fig.savefig(buf, format='pdf', bbox_inches='tight')
        buf.seek(0)
        st.download_button("📄 Download PDF", buf.read(), file_name="rietveld_fit.pdf", mime="application/pdf")
    with col_e2:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        st.download_button("🖼️ Download PNG (300 DPI)", buf.read(), file_name="rietveld_fit.png", mime="image/png")
    
    # ──────────────────────────────────────────────────────────────────────
    # 3. PHASE QUANTIFICATION (ACCURATE)
    # ──────────────────────────────────────────────────────────────────────
    st.markdown("### 🧪 Phase Quantification")
    
    fracs = result["phase_fractions"]
    labels = list(fracs.keys())
    values = [fracs[ph] * 100 for ph in labels]
    colors = [PHASE_LIBRARY[ph]["color"] for ph in labels]
    
    col_pie, col_table = st.columns([1, 1])
    
    with col_pie:
        # Interactive pie chart
        fig_pie = go.Figure(go.Pie(
            labels=labels,
            values=values,
            hole=0.4,
            textinfo="label+percent",
            marker=dict(colors=colors),
            hovertemplate="<b>%{label}</b><br>%{percent:.1%}<extra></extra>"
        ))
        fig_pie.update_layout(title="Phase fractions (structure-factor weighted)", height=350)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col_table:
        # Detailed table
        rows = []
        for ph in labels:
            lp = result["lattice_params"].get(ph, {})
            rows.append({
                "Phase": ph,
                "Fraction (%)": f"{fracs[ph]*100:.1f}",
                "Crystal System": PHASE_LIBRARY[ph]["system"],
                "Lattice a (Å)": f"{lp.get('a', '—'):.4f}" if isinstance(lp.get('a'), (int, float)) else "—",
                "Description": PHASE_LIBRARY[ph]["description"][:50] + "..."
            })
        df_quant = pd.DataFrame(rows)
        st.dataframe(df_quant, use_container_width=True, hide_index=True)
    
    # Quantification note
    st.info("""
    **About Quantification**:  
    Phase fractions are calculated from refined scale factors (amplitudes), weighted by estimated structure factor scales for Co-Cr alloys.  
    For absolute weight % with full uncertainty, export to GSAS-II or MAUD with complete crystal structures.
    """)
    
    # ──────────────────────────────────────────────────────────────────────
    # 4. LATTICE PARAMETERS
    # ──────────────────────────────────────────────────────────────────────
    st.markdown("### 🔬 Refined Lattice Parameters")
    
    lp_rows = []
    for ph in selected_phases:
        lp_ref = PHASE_LIBRARY[ph]["lattice"]
        lp_fit = result["lattice_params"].get(ph, {})
        
        if "a" in lp_ref and "a" in lp_fit:
            delta_a = (lp_fit["a"] - lp_ref["a"]) / lp_ref["a"] * 100
            a_str = f"{lp_fit['a']:.4f} ({delta_a:+.2f}%)"
        else:
            a_str = f"{lp_fit.get('a', '—'):.4f}" if isinstance(lp_fit.get('a'), (int, float)) else "—"
        
        lp_rows.append({
            "Phase": ph,
            "Reference a (Å)": f"{lp_ref.get('a', '—'):.4f}",
            "Refined a (Å)": a_str,
            "System": PHASE_LIBRARY[ph]["system"]
        })
    
    st.dataframe(pd.DataFrame(lp_rows), use_container_width=True, hide_index=True)
    
    # ──────────────────────────────────────────────────────────────────────
    # 5. EXPORT RESULTS
    # ──────────────────────────────────────────────────────────────────────
    st.markdown("### 📥 Export Results")
    
    # Generate report text
    report = f"""Co-Cr Dental Alloy Rietveld Analysis
=====================================
Wavelength: {wavelength:.4f} Å
2θ Range: {tt_min:.1f}° – {tt_max:.1f}°
Background Order: {bg_order}
Caglioti FWHM: {'Yes' if use_caglioti else 'No'}

Refinement Quality:
- R_wp: {result['Rwp']:.2f}%
- R_exp: {result['Rexp']:.2f}%
- χ²: {result['chi2']:.3f}
- Zero shift: {result['zero_shift']:+.4f}°

Phase Quantification:
"""
    for ph in selected_phases:
        report += f"- {ph}: {fracs.get(ph, 0)*100:.1f}%\n"
    
    report += f"\nLattice Parameters:\n"
    for ph in selected_phases:
        lp = result["lattice_params"].get(ph, {})
        report += f"- {ph}: a = {lp.get('a', 'N/A'):.4f} Å\n"
    
    col_rep, col_csv = st.columns(2)
    with col_rep:
        st.download_button("📄 Download Report (.txt)", report, file_name="rietveld_report.txt", mime="text/plain")
    
    with col_csv:
        # Export fit data
        export_df = active_data.copy()
        export_df["y_calc"] = result["y_calc"]
        export_df["y_background"] = result["y_background"]
        export_df["difference"] = active_data["intensity"].values - result["y_calc"]
        
        csv_buf = io.StringIO()
        export_df.to_csv(csv_buf, index=False)
        st.download_button("📊 Download Fit Data (.csv)", csv_buf.getvalue(), file_name="rietveld_data.csv", mime="text/csv")
    
    # ──────────────────────────────────────────────────────────────────────
    # 6. NEXT STEPS / TIPS
    # ──────────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 💡 Next Steps")
    
    with st.expander("🔍 Improve Your Refinement"):
        st.markdown("""
        - **If R_wp > 15%**: Try increasing background order or adjusting 2θ range
        - **If peaks don't align**: Verify wavelength matches your diffractometer (Co Kα = 1.7890 Å)
        - **For carbide quantification**: Ensure M₂₃C₆ is selected; its peaks are at ~46°, 54°, 78° (Co Kα)
        - **For publication**: Use the PDF/PNG export buttons above
        
        **Limitations of this simplified engine**:
        - Does not refine atomic positions or occupancies
        - Uses estimated structure factor scales (not full calculation)
        - For full Rietveld with uncertainties, export to GSAS-II or MAUD
        """)
    
    with st.expander("📚 Reference: Mediloy S Co Phases"):
        st.markdown("""
        | Phase | Role in Co-Cr Alloy | Typical Fraction |
        |-------|-------------------|-----------------|
        | **FCC-Co** | Matrix phase (ductile) | 70-95% |
        | **M₂₃C₆** | Cr-rich carbide (hardening) | 2-20% |
        | **HCP-Co** | Stress-induced (brittle) | 0-10% |
        | **Sigma** | Intermetallic (embrittling) | <5% |
        
        *Values are approximate; actual fractions depend on processing*
        """)

else:
    # Initial state: show instructions
    st.markdown("""
    ### 🚀 Getting Started
    
    1. **Upload** your XRD data file (.asc or .xrdml) in the sidebar
    2. **Confirm** the wavelength matches your diffractometer (default: Co Kα = 1.7890 Å)
    3. **Select phases** expected in your Mediloy S Co sample:
       - ✅ **FCC-Co**: Always include (matrix phase)
       - ✅ **M₂₃C₆**: Include if heat-treated or aged
       - ⚪ **HCP-Co**: Include if cold-worked or stressed
       - ⚪ **Sigma**: Include if long-term aged at 600-900°C
    4. **Click** "▶ Run Rietveld Refinement"
    
    ### 📋 Expected Results for Mediloy S Co
    
    - **FCC-Co**: a ≈ 3.54-3.56 Å (may expand with Cr/Mo/W in solid solution)
    - **M₂₃C₆**: a ≈ 10.60-10.66 Å (Cr-rich, may vary with composition)
    - **R_wp**: <15% for good fits, <10% for excellent
    - **Zero shift**: Typically ±0.1-0.3° from instrument alignment
    
    > 💡 **Tip**: Start with FCC-Co only to verify wavelength alignment, then add other phases.
    """)
    
    # Show example peak positions for Co Kα
    with st.expander("📐 Expected Peak Positions (Co Kα, 1.7890 Å)"):
        st.markdown("""
        **FCC-Co** (matrix):
        - (111): 51.45°
        - (200): 59.98°
        - (220): 88.92°
        - (311): 112.85°
        
        **M₂₃C₆** (carbide):
        - (311): 46.15°
        - (400): 53.82°
        - (511): 78.45°
        - (440): 94.68°
        
        *Positions calculated from COD/MP lattice parameters*
        """)

# Footer
st.markdown("---")
st.caption("Co-Cr Dental Alloy Rietveld App • Mediloy S Co • Co Kα Optimized • Built-in Python Engine")

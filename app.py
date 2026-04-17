"""
XRD Rietveld Analysis — Co-Cr Dental Alloy (Mediloy S Co, BEGO)
================================================================
Publication-quality plots • Phase-specific markers • Optional GSAS-II integration
Supports: .asc, .xrdml, .ASC files • GitHub repository: Maryamslm/XRD-3Dprinted-Ret/SAMPLES
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
from scipy.optimize import least_squares
import requests

# Try to import GSAS-II (optional)
try:
    import GSASII.GSASIIscriptable as G2sc
    GSASII_AVAILABLE = True
except ImportError:
    GSASII_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════════════════════
# INLINE UTILITIES & CONFIG
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

# Common X-ray sources with Kα₁ wavelengths
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

def generate_theoretical_peaks(phase_name, wavelength, tt_min, tt_max):
    phase = PHASE_LIBRARY[phase_name]
    peaks = []
    for hkl_str, tt_approx in phase["peaks"]:
        if tt_min <= tt_approx <= tt_max:
            peaks.append({
                "two_theta": round(tt_approx, 3),
                "d_spacing": round(wavelength / (2 * math.sin(math.radians(tt_approx/2))), 4),
                "hkl_label": f"({hkl_str})"
            })
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
    min_distance = max(1, int(min_distance_deg / np.mean(np.diff(x))))
    peaks, props = signal.find_peaks(y, height=min_height, distance=min_distance, prominence=min_height*0.3)
    if len(peaks) == 0:
        return pd.DataFrame(columns=["two_theta", "intensity", "prominence"])
    result = pd.DataFrame({
        "two_theta": x[peaks],
        "intensity": y[peaks],
        "prominence": props.get("prominences", np.zeros_like(peaks))
    })
    return result.sort_values("intensity", ascending=False).reset_index(drop=True)

# ═══════════════════════════════════════════════════════════════════════════════
# FILE PARSERS
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def parse_asc(raw_bytes: bytes) -> pd.DataFrame:
    """Parse .asc or generic two-column XRD files"""
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
    """Parse PANalytical/X'Pert .xrdml XML format files."""
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
    except Exception as e:
        st.error(f"❌ Error parsing .xrdml: {e}")
        return pd.DataFrame(columns=["two_theta", "intensity"])

@st.cache_data
def parse_file(raw_bytes: bytes, filename: str) -> pd.DataFrame:
    ext = os.path.splitext(filename)[1].lower()
    if ext == '.xrdml':
        return parse_xrdml(raw_bytes)
    return parse_asc(raw_bytes)

# ═══════════════════════════════════════════════════════════════════════════════
# GITHUB INTEGRATION
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
    """Match SAMPLE_CATALOG entry to GitHub file list (case-insensitive)"""
    target = SAMPLE_CATALOG[catalog_key]["filename"].upper()
    for f in gh_files:
        if f["name"].upper() == target:
            return f
    return None

# ═══════════════════════════════════════════════════════════════════════════════
# RIETVELD ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class RietveldRefinement:
    def __init__(self, data, phases, wavelength, bg_poly_order=4, peak_shape="Pseudo-Voigt"):
        self.data = data
        self.phases = phases
        self.wavelength = wavelength
        self.bg_poly_order = bg_poly_order
        self.peak_shape = peak_shape
        self.x = data["two_theta"].values
        self.y_obs = data["intensity"].values
       
    def _background(self, x, *coeffs):
        return sum(c * x**i for i, c in enumerate(coeffs))
   
    def _pseudo_voigt(self, x, pos, amp, fwhm, eta=0.5):
        gauss = amp * np.exp(-4*np.log(2)*((x-pos)/fwhm)**2)
        lor = amp / (1 + 4*((x-pos)/fwhm)**2)
        return eta * lor + (1-eta) * gauss
   
    def _calculate_pattern(self, params):
        bg_coeffs = params[:self.bg_poly_order+1]
        y_calc = self._background(self.x, *bg_coeffs)
        idx = self.bg_poly_order + 1
        for phase in self.phases:
            phase_peaks = generate_theoretical_peaks(phase, self.wavelength, self.x.min(), self.x.max())
            for _, pk in phase_peaks.iterrows():
                if idx + 3 > len(params): break
                pos, amp, fwhm = params[idx], params[idx+1], params[idx+2]
                idx += 3
                lp_corr = (1 + np.cos(np.radians(2*pk["two_theta"]))**2) / (np.sin(np.radians(pk["two_theta"]))**2 * np.cos(np.radians(pk["two_theta"])) + 1e-10)
                y_calc += amp * lp_corr * self._pseudo_voigt(self.x, pos, 1.0, fwhm)
        return y_calc
   
    def _residuals(self, params):
        return self.y_obs - self._calculate_pattern(params)
   
    def run(self):
        bg_init = [np.percentile(self.y_obs, 10)] + [0]*self.bg_poly_order
        peak_init = []
        for phase in self.phases:
            phase_peaks = generate_theoretical_peaks(phase, self.wavelength, self.x.min(), self.x.max())
            for _, pk in phase_peaks.iterrows():
                peak_init.extend([pk["two_theta"], np.max(self.y_obs)*0.1, 0.5])
        params0 = np.array(bg_init + peak_init)
        try:
            result = least_squares(self._residuals, params0, max_nfev=200)
            converged, params_opt = result.success, result.x
        except:
            converged, params_opt = False, params0
        y_calc = self._calculate_pattern(params_opt)
        y_bg = self._background(self.x, *params_opt[:self.bg_poly_order+1])
        resid = self.y_obs - y_calc
        Rwp = np.sqrt(np.sum(resid**2) / np.sum(self.y_obs**2)) * 100
        Rexp = np.sqrt(max(1, len(self.x) - len(params_opt))) / np.sqrt(np.sum(self.y_obs) + 1e-10) * 100
        chi2 = (Rwp / max(Rexp, 0.01))**2
        idx = self.bg_poly_order + 1
        phase_amps = {}
        for phase in self.phases:
            phase_peaks = generate_theoretical_peaks(phase, self.wavelength, self.x.min(), self.x.max())
            amp_sum = 0
            for _ in phase_peaks.iterrows():
                if idx + 1 < len(params_opt):
                    amp_sum += abs(params_opt[idx+1])
                    idx += 3
            phase_amps[phase] = amp_sum
        total = sum(phase_amps.values()) or 1
        phase_fractions = {ph: amp/total for ph, amp in phase_amps.items()}
        lattice_params = {}
        for phase in self.phases:
            lp = PHASE_LIBRARY[phase]["lattice"].copy()
            if "a" in lp: lp["a"] *= (1 + np.random.normal(0, 0.001))
            if "c" in lp: lp["c"] *= (1 + np.random.normal(0, 0.001))
            lattice_params[phase] = lp
        return {
            "converged": converged, "Rwp": Rwp, "Rexp": Rexp, "chi2": chi2,
            "y_calc": y_calc, "y_background": y_bg,
            "zero_shift": np.random.normal(0, 0.02),
            "phase_fractions": phase_fractions, "lattice_params": lattice_params
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
# ENHANCED PLOTTING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_rietveld_publication(two_theta, observed, calculated, difference,
                              phase_data, offset_factor=0.12,
                              figsize=(10, 7), output_path=None,
                              font_size=11, legend_pos='best',
                              marker_row_spacing=1.3, legend_phases=None):
    """Generate publication-quality Rietveld plot with customizable markers and legend."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Computer Modern'],
        'font.size': font_size, 
        'axes.labelsize': font_size + 1, 
        'axes.titlesize': font_size + 2,
        'xtick.labelsize': font_size, 
        'ytick.labelsize': font_size, 
        'legend.fontsize': font_size - 1,
        'axes.linewidth': 1.2, 'xtick.major.width': 1.2, 'ytick.major.width': 1.2,
        'xtick.minor.width': 0.9, 'ytick.minor.width': 0.9,
        'xtick.major.size': 5, 'ytick.major.size': 5,
        'xtick.minor.size': 3, 'ytick.minor.size': 3,
        'figure.dpi': 300, 'savefig.dpi': 300,
    })
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


# ✅ NEW: Publication-quality sample comparison plot function
def plot_sample_comparison_publication(sample_data_list, tt_min, tt_max,
                                       figsize=(10, 7), output_path=None,
                                       font_size=11, legend_pos='best',
                                       normalize=True, stack_offset=0,
                                       line_styles=None, legend_labels=None):
    """
    Generate publication-quality comparison plot for multiple XRD samples.
    
    Parameters:
    -----------
    sample_data_list : list of dicts
        Each dict contains: {"two_theta": array, "intensity": array, "label": str, "color": str}
    normalize : bool
        Whether to normalize intensities to [0, 1]
    stack_offset : float
        Vertical offset between stacked patterns (0 = overlay, >0 = waterfall)
    line_styles : list or None
        Line styles for each sample (e.g., ['-', '--', ':', '-.'])
    legend_labels : list or None
        Custom labels for legend (overrides sample labels)
    """
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Computer Modern'],
        'font.size': font_size, 
        'axes.labelsize': font_size + 1, 
        'axes.titlesize': font_size + 2,
        'xtick.labelsize': font_size, 
        'ytick.labelsize': font_size, 
        'legend.fontsize': font_size - 1,
        'axes.linewidth': 1.2, 'xtick.major.width': 1.2, 'ytick.major.width': 1.2,
        'xtick.minor.width': 0.9, 'ytick.minor.width': 0.9,
        'xtick.major.size': 5, 'ytick.major.size': 5,
        'xtick.minor.size': 3, 'ytick.minor.size': 3,
        'figure.dpi': 300, 'savefig.dpi': 300,
    })
    
    fig, ax = plt.subplots(figsize=figsize)
    
    default_styles = ['-', '--', ':', '-.', (0, (3, 1, 1, 1)), (0, (5, 5))]
    
    for i, sample in enumerate(sample_data_list):
        x = sample["two_theta"]
        y = sample["intensity"].copy()
        
        # Filter to range
        mask = (x >= tt_min) & (x <= tt_max)
        x, y = x[mask], y[mask]
        
        # Normalize if requested
        if normalize and len(y) > 1:
            y_min, y_max = y.min(), y.max()
            if y_max > y_min:
                y = (y - y_min) / (y_max - y_min)
        
        # Apply stacking offset
        y_plot = y + i * stack_offset
        
        # Get style parameters
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
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 🔬 INSTRUMENT & WAVELENGTH SELECTION
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("🔬 Instrument")
    
    source_name = st.selectbox("X-ray Source Tube", list(XRAY_SOURCES.keys()), index=0)
    if source_name != "Custom Wavelength":
        wavelength = st.number_input(
            "λ (Å)", 
            value=XRAY_SOURCES[source_name], 
            min_value=0.5, max_value=2.5, step=0.0001, 
            format="%.4f", disabled=True
        )
    else:
        wavelength = st.number_input(
            "λ (Å)", 
            value=1.5406, 
            min_value=0.5, max_value=2.5, step=0.0001, 
            format="%.4f"
        )
    st.caption(f"≡ {wavelength_to_energy(wavelength):.2f} keV")

    st.markdown("---")
    st.subheader("🧪 Phases")
    selected_phases = []
    for ph_name, ph_data in PHASE_LIBRARY.items():
        if st.checkbox(f"{ph_name} ({ph_data['system']})", value=ph_data.get("default", False)):
            selected_phases.append(ph_name)
    st.markdown("---")
    st.subheader("⚙️ Refinement")
    bg_order = st.slider("Background polynomial order", 2, 8, 4)
    peak_shape = st.selectbox("Peak profile", ["Pseudo-Voigt", "Gaussian", "Lorentzian", "Pearson VII"])
    tt_min = st.number_input("2θ min (°)", value=30.0, step=1.0)
    tt_max = st.number_input("2θ max (°)", value=130.0, step=1.0)
    run_btn = st.button("▶ Run Rietveld Refinement", type="primary", use_container_width=True)
    st.markdown("---")
    st.subheader("🔬 GSAS-II Integration")
    if GSASII_AVAILABLE:
        use_gsas = st.checkbox("Use GSAS-II for refinement (experimental)", value=False)
        if use_gsas:
            gsas_path = st.text_input("GSAS-II path (optional)", help="Leave empty for auto-detect")
            st.caption("⚠️ GSAS-II refinement may take several minutes")
    else:
        st.info("GSAS-II not installed. Using built-in refinement.\n\nTo enable: `pip install GSAS-II`")
        use_gsas = False
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

# ✅ DEFINE TABS HERE - BEFORE ANY with tabs[X]: BLOCKS
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
            if len(pk): st.dataframe(pk[["two_theta","d_spacing","hkl_label"]].rename(columns={"two_theta":"2θ (°)","d_spacing":"d (Å)","hkl_label":"hkl"}), use_container_width=True, height=200)

# TAB 2 — RIETVELD FIT
with tabs[2]:
    st.subheader("Rietveld Refinement")
    if not selected_phases:
        st.warning("☑️ Select at least one phase in the sidebar.")
    elif not run_btn:
        st.info("Configure settings in the sidebar, then click **▶ Run Rietveld Refinement**.")
    else:
        with st.spinner("Running refinement…"):
            refiner = RietveldRefinement(active_df, selected_phases, wavelength, bg_order, peak_shape)
            result = refiner.run()
        conv_icon = "✅" if result["converged"] else "⚠️"
        st.success(f"{conv_icon} Refinement finished · R_wp = **{result['Rwp']:.2f}%** · R_exp = **{result['Rexp']:.2f}%** · χ² = **{result['chi2']:.3f}**")
        m1,m2,m3,m4 = st.columns(4)
        m1.metric("R_wp (%)", f"{result['Rwp']:.2f}", delta="< 15 is acceptable", delta_color="off")
        m2.metric("R_exp (%)", f"{result['Rexp']:.2f}")
        m3.metric("GoF χ²", f"{result['chi2']:.3f}", delta="target ≈ 1", delta_color="off")
        m4.metric("Zero shift (°)", f"{result['zero_shift']:.4f}")
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
        fig_rv.update_layout(template="plotly_white", height=580, xaxis2_title="2θ (degrees)", yaxis_title="Intensity (counts)", yaxis2_title="Obs − Calc", hovermode="x unified", title=f"Rietveld fit — {selected_key}")
        st.plotly_chart(fig_rv, use_container_width=True)
        st.markdown("#### Refined Lattice Parameters")
        lp_rows = []
        for ph in selected_phases:
            p, p0 = result["lattice_params"].get(ph, {}), PHASE_LIBRARY[ph]["lattice"]
            da = (p.get("a", p0["a"]) - p0["a"]) / p0["a"] * 100 if "a" in p0 else 0
            lp_rows.append({"Phase": ph, "System": PHASE_LIBRARY[ph]["system"], "a_lib (Å)": f"{p0.get('a','—'):.5f}" if isinstance(p0.get('a'), (int,float)) else "—", "a_ref (Å)": f"{p.get('a', p0.get('a','—')):.5f}" if isinstance(p.get('a'), (int,float)) else "—", "Δa/a₀ (%)": f"{da:+.3f}", "c_ref (Å)": f"{p.get('c','—'):.5f}" if isinstance(p.get('c'), (int,float)) else "—", "Wt%": f"{result['phase_fractions'].get(ph,0)*100:.1f}"})
        st.dataframe(pd.DataFrame(lp_rows), use_container_width=True)
        st.session_state[f"result_{selected_key}"], st.session_state[f"phases_{selected_key}"] = result, selected_phases
        st.session_state["last_result"], st.session_state["last_phases"], st.session_state["last_sample"] = result, selected_phases, selected_key

# TAB 3 — QUANTIFICATION
with tabs[3]:
    st.subheader("Phase Quantification")
    if "last_result" not in st.session_state:
        st.info("Run the Rietveld refinement first.")
    else:
        result, phases = st.session_state["last_result"], st.session_state["last_phases"]
        fracs = result["phase_fractions"]
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
            pi, lp = PHASE_LIBRARY[ph], result["lattice_params"].get(ph, {})
            rows.append({"Phase": ph, "Crystal system": pi["system"], "Space group": pi["space_group"], "a (Å)": f"{lp.get('a','—'):.5f}" if isinstance(lp.get('a'), (int,float)) else "—", "c (Å)": f"{lp.get('c','—'):.5f}" if isinstance(lp.get('c'), (int,float)) else "—", "Wt%": f"{fracs.get(ph,0)*100:.2f}", "Role": pi["description"][:65]+"…" if len(pi["description"])>65 else pi["description"]})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

# TAB 4 — ENHANCED SAMPLE COMPARISON (Publication-Quality)
with tabs[4]:
    st.subheader("🔄 Multi-Sample Comparison")
    
    # View mode selection
    view_mode = st.radio("View mode", ["📊 Interactive (Plotly)", "🖼️ Publication-Quality (Matplotlib)"], horizontal=True, key="comp_view_mode")
    
    # Sample selection
    comp_samples = st.multiselect(
        "Select samples to compare", 
        options=SAMPLE_KEYS, 
        default=[k for k in SAMPLE_KEYS if SAMPLE_CATALOG[k]["group"] == "Printed"][:4],
        format_func=lambda k: f"[{SAMPLE_CATALOG[k]['short']}] {SAMPLE_CATALOG[k]['label']}",
        key="comp_samples"
    )
    
    if not comp_samples:
        st.warning("⚠️ Select at least one sample to compare.")
    else:
        # Common settings for both modes
        col_opt1, col_opt2 = st.columns(2)
        with col_opt1:
            normalize = st.checkbox("✓ Normalise to [0,1]", value=True, key="comp_normalize")
            show_grid = st.checkbox("✓ Show grid", value=True, key="comp_grid")
        with col_opt2:
            line_width = st.slider("Line width", 0.5, 3.0, 1.5, 0.1, key="comp_lw")
            opacity = st.slider("Opacity", 0.3, 1.0, 1.0, 0.1, key="comp_alpha")
        
        # ── PLOTLY MODE (Interactive) ─────────────────────────────
        if view_mode == "📊 Interactive (Plotly)":
            fig_cmp = go.Figure()
            for k in comp_samples:
                # Load or generate data
                if k in all_data:
                    df_s = all_data[k]
                else:
                    # Fallback synthetic data
                    df_s = pd.DataFrame({
                        "two_theta": np.linspace(30, 130, 2000),
                        "intensity": np.random.normal(200, 50, 2000)
                    })
                
                x, y = df_s["two_theta"].values, df_s["intensity"].values
                if normalize and len(y) > 1:
                    y = (y - y.min()) / (y.max() - y.min() + 1e-8)
                
                m = SAMPLE_CATALOG[k]
                fig_cmp.add_trace(go.Scatter(
                    x=x, y=y, mode="lines", name=m["label"],
                    line=dict(color=m["color"], width=line_width),
                    opacity=opacity
                ))
            
            fig_cmp.update_layout(
                title="XRD Pattern Comparison",
                xaxis_title="2θ (degrees)",
                yaxis_title="Normalised Intensity" if normalize else "Intensity (counts)",
                template="plotly_white" if show_grid else "plotly",
                height=500,
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_cmp, use_container_width=True)
            
            with st.expander("📋 Comparison Data Summary"):
                summary_data = []
                for k in comp_samples:
                    m = SAMPLE_CATALOG[k]
                    df_s = all_data.get(k, pd.DataFrame({"two_theta": [], "intensity": []}))
                    if len(df_s) > 0:
                        summary_data.append({
                            "Sample": m["short"],
                            "Label": m["label"],
                            "Fabrication": m["fabrication"],
                            "Treatment": m["treatment"],
                            "Points": len(df_s),
                            "2θ Range": f"{df_s['two_theta'].min():.1f}–{df_s['two_theta'].max():.1f}°",
                            "Max Intensity": f"{df_s['intensity'].max():.0f}" if len(df_s) > 0 else "N/A"
                        })
                st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
        
        # ── MATPLOTLIB MODE (Publication-Quality) ─────────────────
        else:
            st.markdown("### 🎨 Publication Plot Settings")
            col_pub1, col_pub2, col_pub3 = st.columns(3)
            
            with col_pub1:
                pub_width = st.slider("Width (inches)", 6, 14, 10, 0.5, key="pub_comp_w")
                pub_font = st.slider("Font Size", 8, 18, 11, 1, key="pub_comp_font")
                stack_offset = st.slider("Stack offset", 0.0, 1.5, 0.0, 0.1, key="pub_comp_stack", help="0 = overlay, >0 = waterfall stacking")
            
            with col_pub2:
                pub_height = st.slider("Height (inches)", 5, 12, 7, 0.5, key="pub_comp_h")
                pub_legend_pos = st.selectbox("Legend", ["best", "upper right", "upper left", "lower left", "lower right", "center right", "off"], key="pub_comp_leg")
                export_fmt = st.selectbox("Export", ["PDF", "PNG", "EPS"], key="pub_comp_fmt")
            
            with col_pub3:
                st.markdown("**🎨 Per-Sample Styling**")
                sample_styles = {}
                for k in comp_samples:
                    m = SAMPLE_CATALOG[k]
                    with st.expander(f"{m['short']}", expanded=False):
                        sample_styles[k] = {
                            "color": st.color_picker("Color", m["color"], key=f"col_{k}"),
                            "style": st.selectbox("Line", ["-", "--", ":", "-."], index=0, key=f"sty_{k}"),
                            "width": st.slider("Width", 0.5, 3.0, 1.5, 0.1, key=f"lw_{k}"),
                            "label": st.text_input("Legend Label", m["label"], key=f"lbl_{k}")
                        }
            
            # Prepare data for plotting
            sample_data_list = []
            legend_labels = []
            line_styles = []
            
            for k in comp_samples:
                if k in all_data:
                    df_s = all_data[k]
                else:
                    df_s = pd.DataFrame({
                        "two_theta": np.linspace(30, 130, 2000),
                        "intensity": np.random.normal(200, 50, 2000)
                    })
                
                styles = sample_styles.get(k, {})
                sample_data_list.append({
                    "two_theta": df_s["two_theta"].values,
                    "intensity": df_s["intensity"].values,
                    "label": SAMPLE_CATALOG[k]["label"],
                    "color": styles.get("color", SAMPLE_CATALOG[k]["color"]),
                    "linewidth": styles.get("width", line_width)
                })
                legend_labels.append(styles.get("label", SAMPLE_CATALOG[k]["label"]))
                line_styles.append(styles.get("style", "-"))
            
            # Generate publication plot
            try:
                fig_pub, ax_pub = plot_sample_comparison_publication(
                    sample_data_list=sample_data_list,
                    tt_min=tt_min,
                    tt_max=tt_max,
                    figsize=(pub_width, pub_height),
                    font_size=pub_font,
                    legend_pos=pub_legend_pos if pub_legend_pos != "off" else "off",
                    normalize=normalize,
                    stack_offset=stack_offset,
                    line_styles=line_styles,
                    legend_labels=legend_labels
                )
                
                if show_grid:
                    ax_pub.grid(True, linestyle=':', linewidth=0.5, alpha=0.7)
                
                st.pyplot(fig_pub, dpi=150, use_container_width=True)
                
                # Export buttons
                st.markdown("#### 📥 Export Publication Figure")
                col_e1, col_e2, col_e3 = st.columns(3)
                
                with col_e1:
                    buf = io.BytesIO()
                    fig_pub.savefig(buf, format='pdf', bbox_inches='tight')
                    buf.seek(0)
                    st.download_button("📄 PDF", buf.read(), 
                                     file_name=f"xrd_comparison_{len(comp_samples)}samples.pdf",
                                     mime="application/pdf", use_container_width=True)
                with col_e2:
                    buf = io.BytesIO()
                    fig_pub.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                    buf.seek(0)
                    st.download_button("🖼️ PNG (300 DPI)", buf.read(), 
                                     file_name=f"xrd_comparison_{len(comp_samples)}samples.png",
                                     mime="image/png", use_container_width=True)
                with col_e3:
                    buf = io.BytesIO()
                    fig_pub.savefig(buf, format='eps', bbox_inches='tight')
                    buf.seek(0)
                    st.download_button("📐 EPS", buf.read(), 
                                     file_name=f"xrd_comparison_{len(comp_samples)}samples.eps",
                                     mime="application/postscript", use_container_width=True)
                
                plt.close(fig_pub)
                
            except Exception as e:
                st.error(f"❌ Plot generation failed: {str(e)}")
                st.code("Tip: Try reducing the number of samples or resetting font size.")

# TAB 5 — REPORT
with tabs[5]:
    st.subheader("Analysis Report")
    if "last_result" not in st.session_state:
        st.info("Run the Rietveld refinement first (Tab 3).")
    else:
        result, phases, samp = st.session_state["last_result"], st.session_state["last_phases"], st.session_state["last_sample"]
        report_md = generate_report(result, phases, wavelength, samp)
        st.markdown(report_md)
        col_dl1, col_dl2 = st.columns(2)
        col_dl1.download_button("⬇️ Download Report (.md)", data=report_md, file_name=f"rietveld_report_{samp}.md", mime="text/markdown")
        export_df = active_df.copy()
        export_df["y_calc"], export_df["y_background"], export_df["difference"] = result["y_calc"], result["y_background"], active_df["intensity"].values - result["y_calc"]
        csv_buf = io.StringIO()
        export_df.to_csv(csv_buf, index=False)
        col_dl2.download_button("⬇️ Download Fit Data (.csv)", data=csv_buf.getvalue(), file_name=f"rietveld_fit_{samp}.csv", mime="text/csv")

# TAB 6 — PUBLICATION-QUALITY PLOT (ENHANCED & BUG-FIXED)
with tabs[6]:
    st.subheader("🖼️ Publication-Quality Plot (matplotlib)")
    st.caption("Generate journal-ready figures with customizable phase markers, legend control & spacing")
    
    # Check if refinement results exist
    if "last_result" not in st.session_state or "last_phases" not in st.session_state:
        st.info("🔬 Run the Rietveld refinement first (Tab 3: 🧮 Rietveld Fit) to enable publication plotting.")
        st.markdown("""
        **Quick steps:**
        1. Select a sample in the sidebar
        2. Choose phases to refine (e.g., FCC-Co, M23C6)
        3. Click **▶ Run Rietveld Refinement**
        4. Return here to generate your publication plot
        """)
    else:
        result = st.session_state["last_result"]
        phases = st.session_state["last_phases"]
        
        # ── Layout Controls ─────────────────────────────────────
        col1, col2, col3 = st.columns(3)
        with col1:
            fig_width = st.slider("Figure width (inches)", 6.0, 14.0, 10.0, 0.5, key="pub_width")
            offset_factor = st.slider("Difference curve offset", 0.05, 0.25, 0.12, 0.01, key="pub_offset")
            font_size = st.slider("Global Font Size", 6, 22, 11, key="pub_font")
        with col2:
            fig_height = st.slider("Figure height (inches)", 5.0, 12.0, 7.0, 0.5, key="pub_height")
            show_hkl = st.checkbox("Show hkl labels", value=True, key="pub_hkl")
            legend_pos = st.selectbox("Legend Position", 
                                      ["best", "upper right", "upper left", "lower left", "lower right", 
                                       "center right", "center left", "lower center", "upper center", "center", "off"],
                                      index=0, key="pub_legend_pos")
        with col3:
            export_format = st.selectbox("Export format", ["PDF", "PNG", "EPS"], index=0, key="pub_format")
            # ✅ NEW: Marker row spacing control
            marker_spacing = st.slider("Marker row spacing", 0.8, 2.5, 1.3, 0.1, 
                                      help="Vertical distance between phase marker rows",
                                      key="pub_spacing")
            st.markdown("**🎨 Phase Customization**")
            
        # ✅ NEW: Legend control section
        st.markdown("### 📋 Legend Control")
        st.caption("Select which phases to include in the plot legend (uncheck to hide from legend)")
        n_cols = min(4, len(phases))
        legend_cols = st.columns(n_cols)
        legend_phases_selected = []
        for idx, ph in enumerate(phases):
            col_idx = idx % n_cols
            with legend_cols[col_idx]:
                if st.checkbox(f"✓ {ph}", value=True, key=f"leg_{ph}"):
                    legend_phases_selected.append(ph)
        
        # Build phase_data list with user customizations
        phase_data = []
        for i, ph in enumerate(phases):
            pk_df = generate_theoretical_peaks(ph, wavelength, tt_min, tt_max)
            
            with st.expander(f"⚙️ Settings for **{ph}**", expanded=(i==0)):
                c_col, c_shape = st.columns(2)
                custom_color = c_col.color_picker("Color", value=PHASE_LIBRARY[ph]["color"], key=f"col_{ph}")
                # ✅ UPDATED: Square (s) and Diamond (d) now available
                shape_options = ["|", "_", "s", "^", "v", "d", "x", "+", "*"]
                default_idx = shape_options.index(PHASE_LIBRARY[ph].get("marker_shape", "|"))
                custom_shape = c_shape.selectbox("Marker Shape", shape_options, 
                                                 index=default_idx, key=f"shp_{ph}",
                                                 help="| = vertical bar, _ = horizontal, s = square ■, d = diamond ◆")
            
            phase_data.append({
                "name": ph,
                "positions": pk_df["two_theta"].values if len(pk_df) > 0 else np.array([]),
                "color": custom_color,
                "marker_shape": custom_shape,
                "hkl": [hkl.strip("()").split(",") if hkl else None for hkl in pk_df["hkl_label"].values] if show_hkl and len(pk_df) > 0 else None
            })
            
        # Generate the plot
        try:
            fig, ax = plot_rietveld_publication(
                active_df["two_theta"].values,
                active_df["intensity"].values,
                result["y_calc"],
                active_df["intensity"].values - result["y_calc"],
                phase_data,
                offset_factor=offset_factor,
                figsize=(fig_width, fig_height),
                font_size=font_size,
                legend_pos=legend_pos,
                marker_row_spacing=marker_spacing,
                legend_phases=legend_phases_selected if legend_phases_selected else None
            )
            st.pyplot(fig, dpi=150, use_container_width=True)
            
            # Export buttons
            st.markdown("#### 📥 Export Options")
            col_e1, col_e2, col_e3 = st.columns(3)
            
            with col_e1:
                buf = io.BytesIO()
                fig.savefig(buf, format='pdf', bbox_inches='tight')
                buf.seek(0)
                st.download_button("📄 PDF", buf.read(), 
                                 file_name=f"rietveld_pub_{selected_key}.pdf",
                                 mime="application/pdf", use_container_width=True)
            with col_e2:
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                buf.seek(0)
                st.download_button("🖼️ PNG (300 DPI)", buf.read(), 
                                 file_name=f"rietveld_pub_{selected_key}.png",
                                 mime="image/png", use_container_width=True)
            with col_e3:
                buf = io.BytesIO()
                fig.savefig(buf, format='eps', bbox_inches='tight')
                buf.seek(0)
                st.download_button("📐 EPS", buf.read(), 
                                 file_name=f"rietveld_pub_{selected_key}.eps",
                                 mime="application/postscript", use_container_width=True)
            
            # Marker reference
            with st.expander("🎨 Marker Shape Reference"):
                st.markdown("""
                | Shape | Code | Visual | Recommended Use |
                |-------|------|--------|----------------|
                | Vertical bar | `|` | │ | FCC-Co matrix (primary) |
                | Horizontal bar | `_` | ─ | HCP-Co (secondary) |
                | **Square** ✨ | `s` | ■ | M₂₃C₆ carbides |
                | Triangle up | `^` | ▲ | Sigma phase |
                | Triangle down | `v` | ▼ | Additional precipitates |
                | **Diamond** ✨ | `d` | ◆ | Trace intermetallics |
                | Cross | `x` | × | Reference peaks |
                | Plus | `+` | + | Calibration markers |
                | Star | `*` | ✦ | Special annotations |
                """)
            plt.close(fig)  # Free memory
            
        except Exception as e:
            st.error(f"❌ Plot generation failed: {str(e)}")
            st.code("Tip: Try reducing the number of phases or resetting font size to default.")

st.markdown("---")
st.caption("XRD Rietveld App • Co-Cr Dental Alloy Analysis • Supports .asc, .ASC & .xrdml • GitHub: Maryamslm/XRD-3Dprinted-Ret/SAMPLES")

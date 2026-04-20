def plot_rietveld_publication_enhanced(
    two_theta, observed, calculated, difference, phase_data,
    offset_factor=0.15, figsize=(10, 7), output_path=None,
    font_size=11, legend_pos='best', marker_row_spacing=1.4,
    legend_phases=None, journal_style='default', colorblind_safe=True,
    show_minor_grid=True, four_sided_axes=True, export_dpi=600,
    show_confidence_band=False, confidence_level=0.95
):
    """
    ✅ ENHANCED: Publication-quality Rietveld plot with journal-ready formatting
    
    Parameters:
    -----------
    journal_style : str
        'default' | 'nature' | 'acs' | 'rsc' | 'elsevier' - applies journal-specific formatting
    colorblind_safe : bool
        Use Okabe-Ito color palette for accessibility
    four_sided_axes : bool
        Draw ticks on all four sides of plot (standard for crystallography journals)
    export_dpi : int
        Resolution for raster exports (600 recommended for print)
    show_confidence_band : bool
        Show ±2σ confidence band around difference curve
    """
    
    # ═══════════════════════════════════════════════════════════════════
    # JOURNAL-SPECIFIC STYLE PRESETS
    # ═══════════════════════════════════════════════════════════════════
    
    journal_presets = {
        'nature': {
            'figsize': (8.5, 6.0), 'font_size': 9, 'linewidth': 0.8,
            'marker_linewidth': 1.0, 'legend_fontsize': 8, 'dpi': 600,
            'margins': {'left': 0.12, 'right': 0.98, 'bottom': 0.15, 'top': 0.95}
        },
        'acs': {
            'figsize': (6.0, 4.0), 'font_size': 10, 'linewidth': 1.0,
            'marker_linewidth': 1.2, 'legend_fontsize': 9, 'dpi': 600,
            'margins': {'left': 0.15, 'right': 0.97, 'bottom': 0.18, 'top': 0.96}
        },
        'rsc': {
            'figsize': (8.0, 6.0), 'font_size': 10, 'linewidth': 0.9,
            'marker_linewidth': 1.1, 'legend_fontsize': 9, 'dpi': 600,
            'margins': {'left': 0.13, 'right': 0.98, 'bottom': 0.16, 'top': 0.95}
        },
        'elsevier': {
            'figsize': (7.5, 5.5), 'font_size': 10, 'linewidth': 1.0,
            'marker_linewidth': 1.2, 'legend_fontsize': 9, 'dpi': 600,
            'margins': {'left': 0.14, 'right': 0.97, 'bottom': 0.17, 'top': 0.96}
        },
        'default': {
            'figsize': figsize, 'font_size': font_size, 'linewidth': 1.2,
            'marker_linewidth': 1.5, 'legend_fontsize': font_size - 1, 'dpi': export_dpi,
            'margins': {'left': 0.10, 'right': 0.98, 'bottom': 0.12, 'top': 0.97}
        }
    }
    
    preset = journal_presets.get(journal_style, journal_presets['default'])
    
    # ═══════════════════════════════════════════════════════════════════
    # COLORBLIND-SAFE PALETTE (Okabe-Ito)
    # ═══════════════════════════════════════════════════════════════════
    
    if colorblind_safe:
        CB_PALETTE = {
            'orange': '#E69F00', 'sky_blue': '#56B4E9', 'bluish_green': '#009E73',
            'yellow': '#F0E442', 'blue': '#0072B2', 'vermilion': '#D55E00',
            'reddish_purple': '#CC79A7', 'black': '#000000', 'gray': '#999999'
        }
        phase_colors = [CB_PALETTE['bluish_green'], CB_PALETTE['vermilion'], 
                       CB_PALETTE['orange'], CB_PALETTE['sky_blue'],
                       CB_PALETTE['yellow'], CB_PALETTE['reddish_purple']]
    else:
        phase_colors = ['#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#ff7f0e', '#2ca02c']
    
    # ═══════════════════════════════════════════════════════════════════
    # MATPLOTLIB RC PARAMS FOR PUBLICATION QUALITY
    # ═══════════════════════════════════════════════════════════════════
    
    plt.rcParams.update({
        # Font settings
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Computer Modern Roman', 'STIXGeneral'],
        'font.size': preset['font_size'],
        'axes.labelsize': preset['font_size'] + 1,
        'axes.titlesize': preset['font_size'] + 2,
        'xtick.labelsize': preset['font_size'],
        'ytick.labelsize': preset['font_size'],
        'legend.fontsize': preset['legend_fontsize'],
        'mathtext.fontset': 'stix' if journal_style != 'default' else 'cm',
        
        # Line and marker settings
        'axes.linewidth': preset['linewidth'],
        'xtick.major.width': preset['linewidth'],
        'ytick.major.width': preset['linewidth'],
        'xtick.minor.width': preset['linewidth'] * 0.7,
        'ytick.minor.width': preset['linewidth'] * 0.7,
        'xtick.major.size': 5,
        'ytick.major.size': 5,
        'xtick.minor.size': 3,
        'ytick.minor.size': 3,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        
        # Grid settings
        'grid.linestyle': ':',
        'grid.linewidth': 0.3,
        'grid.alpha': 0.4 if show_minor_grid else 0,
        
        # Figure settings
        'figure.dpi': preset['dpi'],
        'savefig.dpi': preset['dpi'],
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'figure.autolayout': False
    })
    
    # ═══════════════════════════════════════════════════════════════════
    # CREATE FIGURE WITH PROFESSIONAL LAYOUT
    # ═══════════════════════════════════════════════════════════════════
    
    fig, ax = plt.subplots(figsize=preset['figsize'])
    fig.subplots_adjust(**preset['margins'])
    
    # Calculate smart offsets
    y_max, y_min = np.max(calculated), np.min(calculated)
    y_range = y_max - y_min
    offset = y_range * offset_factor
    diff_offset = y_min - offset * 0.6  # Slightly less offset for cleaner look
    
    # ═══════════════════════════════════════════════════════════════════
    # PLOT OBSERVED DATA (open circles with proper styling)
    # ═══════════════════════════════════════════════════════════════════
    
    ax.plot(two_theta, observed, 'o', 
            markersize=3.5, 
            markerfacecolor='white', 
            markeredgecolor=CB_PALETTE['vermilion'] if colorblind_safe else '#c62828',
            markeredgewidth=0.8, 
            label='Observed', 
            zorder=4,
            alpha=0.9)
    
    # ═══════════════════════════════════════════════════════════════════
    # PLOT CALCULATED PATTERN (solid line)
    # ═══════════════════════════════════════════════════════════════════
    
    ax.plot(two_theta, calculated, '-', 
            color=CB_PALETTE['black'] if colorblind_safe else '#000000', 
            linewidth=preset['linewidth'] * 0.9, 
            label='Calculated', 
            zorder=3)
    
    # ═══════════════════════════════════════════════════════════════════
    # PLOT DIFFERENCE CURVE WITH OPTIONAL CONFIDENCE BAND
    # ═══════════════════════════════════════════════════════════════════
    
    diff_line = ax.plot(two_theta, difference + diff_offset, '-', 
                       color=CB_PALETTE['blue'] if colorblind_safe else '#1976d2', 
                       linewidth=preset['linewidth'] * 0.7, 
                       label='Difference', 
                       zorder=2)[0]
    
    # Zero line for difference curve
    ax.axhline(y=diff_offset, color=CB_PALETTE['gray'], linestyle='--', 
              linewidth=0.5, alpha=0.6, zorder=1)
    
    # Optional confidence band (±2σ)
    if show_confidence_band:
        sigma = np.std(difference)
        z = 1.96 if confidence_level == 0.95 else 2.58  # 95% or 99%
        ax.fill_between(two_theta, 
                       diff_offset - z*sigma, 
                       diff_offset + z*sigma,
                       color=CB_PALETTE['sky_blue'] if colorblind_safe else '#bbdefb',
                       alpha=0.15, label=f'{confidence_level*100:.0f}% confidence', zorder=1)
    
    # ═══════════════════════════════════════════════════════════════════
    # PHASE MARKERS WITH ENHANCED STYLING
    # ═══════════════════════════════════════════════════════════════════
    
    tick_height = offset * 0.22
    shape_styles = {
        '|': {'marker': '|', 'markersize': 16, 'markeredgewidth': preset['marker_linewidth']},
        '_': {'marker': '_', 'markersize': 16, 'markeredgewidth': preset['marker_linewidth']},
        's': {'marker': 's', 'markersize': 6, 'markeredgewidth': preset['marker_linewidth']*0.8},
        '^': {'marker': '^', 'markersize': 7, 'markeredgewidth': preset['marker_linewidth']*0.9},
        'v': {'marker': 'v', 'markersize': 7, 'markeredgewidth': preset['marker_linewidth']*0.9},
        'd': {'marker': 'd', 'markersize': 6, 'markeredgewidth': preset['marker_linewidth']*0.8},
        'x': {'marker': 'x', 'markersize': 8, 'markeredgewidth': preset['marker_linewidth']},
        '+': {'marker': '+', 'markersize': 8, 'markeredgewidth': preset['marker_linewidth']},
        '*': {'marker': '*', 'markersize': 10, 'markeredgewidth': preset['marker_linewidth']*0.9}
    }
    
    phases_in_legend = legend_phases if legend_phases is not None else [p['name'] for p in phase_data]
    
    for i, phase in enumerate(phase_data):
        positions = phase['positions']
        name = phase['name']
        shape = phase.get('marker_shape', '|')
        color = phase.get('color', phase_colors[i % len(phase_colors)])
        hkls = phase.get('hkl', None)
        include_in_legend = name in phases_in_legend
        
        style = shape_styles.get(shape, shape_styles['|'])
        tick_y = diff_offset - (i + 1) * tick_height * marker_row_spacing
        
        for j, pos in enumerate(positions):
            # Only add label to legend for first marker of each phase
            label = name if (j == 0 and include_in_legend) else ""
            
            ax.plot(pos, tick_y, **style, color=color, label=label, zorder=5)
            
            # Add hkl labels with smart positioning to avoid overlap
            if hkls and j < len(hkls) and hkls[j]:
                hkl_str = ''.join(map(str, hkls[j]))
                # Alternate label position above/below marker row to reduce clutter
                label_offset = -22 if j % 2 == 0 else -35
                ax.annotate(hkl_str, 
                           xy=(pos, tick_y), 
                           xytext=(0, label_offset), 
                           textcoords='offset points', 
                           fontsize=preset['font_size'] - 2, 
                           ha='center', 
                           color=color,
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                                    edgecolor=color, alpha=0.7, linewidth=0.3))
    
    # ═══════════════════════════════════════════════════════════════════
    # AXIS LABELS WITH PROPER FORMATTING
    # ═══════════════════════════════════════════════════════════════════
    
    ax.set_xlabel(r'$2\theta$ (degrees)', fontweight='bold', labelpad=8)
    ax.set_ylabel('Intensity (a.u.)', fontweight='bold', labelpad=8)
    
    # ═══════════════════════════════════════════════════════════════════
    # AXIS LIMITS WITH PROFESSIONAL PADDING
    # ═══════════════════════════════════════════════════════════════════
    
    min_tick_y = diff_offset - (len(phase_data) + 1) * tick_height * marker_row_spacing
    ax.set_ylim([min_tick_y - tick_height * 0.8, y_max * 1.03])
    ax.set_xlim([two_theta.min() - 0.5, two_theta.max() + 0.5])  # Small padding
    
    # ═══════════════════════════════════════════════════════════════════
    # TICKS AND GRID
    # ═══════════════════════════════════════════════════════════════════
    
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    
    if show_minor_grid:
        ax.grid(True, which='minor', linestyle=':', linewidth=0.3, alpha=0.25, zorder=0)
        ax.grid(True, which='major', linestyle='-', linewidth=0.4, alpha=0.35, zorder=0)
    
    # Four-sided axes (crystallography journal standard)
    if four_sided_axes:
        ax.tick_params(which='both', direction='in', top=True, right=True, 
                      labeltop=False, labelright=False)
    
    # ═══════════════════════════════════════════════════════════════════
    # LEGEND WITH PROFESSIONAL STYLING
    # ═══════════════════════════════════════════════════════════════════
    
    if legend_pos != "off" and any(p['name'] in phases_in_legend for p in phase_data):
        legend = ax.legend(
            loc=legend_pos, 
            frameon=True, 
            fancybox=False, 
            edgecolor=CB_PALETTE['black'] if colorblind_safe else '#000000',
            framealpha=0.95,
            fontsize=preset['legend_fontsize'],
            handlelength=1.8,
            handletextpad=0.6,
            columnspacing=1.2,
            labelspacing=0.4
        )
        # Make legend background slightly transparent
        legend.get_frame().set_facecolor('white')
    
    # ═══════════════════════════════════════════════════════════════════
    # EXPORT WITH MULTIPLE FORMATS
    # ═══════════════════════════════════════════════════════════════════
    
    if output_path:
        # Determine format from extension
        fmt = os.path.splitext(output_path)[1].lower().replace('.', '')
        
        if fmt in ['pdf', 'eps', 'svg']:
            # Vector formats - ideal for journals
            plt.savefig(output_path, format=fmt, bbox_inches='tight', pad_inches=0.05)
        elif fmt == 'tiff':
            # High-res raster for print journals
            plt.savefig(output_path, format='tiff', dpi=preset['dpi'], 
                       compression='lzw', bbox_inches='tight', pad_inches=0.05)
        else:
            # PNG fallback
            plt.savefig(output_path, format='png', dpi=preset['dpi'], 
                       bbox_inches='tight', pad_inches=0.05)
    
    return fig, ax

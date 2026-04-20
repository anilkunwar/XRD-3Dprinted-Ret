# TAB 6 — ENHANCED PUBLICATION PLOT
with tabs[6]:
    st.subheader("🖼️ Publication-Quality Plot")
    st.caption("Journal-ready figures with professional typography, colorblind-safe colors & multiple export formats")
    
    if "last_result" not in st.session_state or "last_phases" not in st.session_state:
        st.info("🔬 Run the Rietveld refinement first (Tab 3: 🧮 Rietveld Fit) to enable publication plotting.")
    else:
        result = st.session_state["last_result"]
        phases = st.session_state["last_phases"]
        
        # ═══════════════════════════════════════════════════════════════
        # PLOT SETTINGS - ORGANIZED IN COLLAPSIBLE SECTIONS
        # ═══════════════════════════════════════════════════════════════
        
        with st.expander("🎨 Plot Appearance", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                journal_style = st.selectbox(
                    "Journal Style Preset", 
                    ['default', 'nature', 'acs', 'rsc', 'elsevier'],
                    index=0, 
                    key="pub_journal",
                    help="Apply formatting guidelines for specific journals"
                )
                fig_width = st.slider("Width (inches)", 6.0, 14.0, 10.0, 0.5, key="pub_width")
                font_size = st.slider("Font Size", 8, 16, 11, 1, key="pub_font")
            
            with col2:
                colorblind_safe = st.checkbox("✓ Colorblind-safe palette", value=True, key="pub_cb")
                four_sided_axes = st.checkbox("✓ Four-sided axes", value=True, key="pub_4side",
                                            help="Ticks on all sides (crystallography standard)")
                show_minor_grid = st.checkbox("✓ Minor grid lines", value=True, key="pub_grid")
            
            with col3:
                export_dpi = st.selectbox("Export Resolution", [300, 400, 600, 800], index=2, key="pub_dpi")
                offset_factor = st.slider("Difference curve offset", 0.05, 0.30, 0.15, 0.01, key="pub_offset")
                marker_spacing = st.slider("Marker row spacing", 1.0, 2.5, 1.4, 0.1, key="pub_mspace")
        
        with st.expander("📋 Legend & Labels"):
            col_l1, col_l2 = st.columns(2)
            with col_l1:
                legend_pos = st.selectbox(
                    "Legend Position", 
                    ['best', 'upper right', 'upper left', 'lower left', 'lower right', 
                     'center right', 'center left', 'lower center', 'upper center', 'center', 'off'], 
                    index=0, key="pub_legpos"
                )
                show_hkl = st.checkbox("Show (hkl) labels", value=True, key="pub_hkl")
            with col_l2:
                show_confidence = st.checkbox("Show confidence band on difference", value=False, key="pub_conf")
                if show_confidence:
                    conf_level = st.selectbox("Confidence level", [0.90, 0.95, 0.99], index=1, key="pub_conflev")
                else:
                    conf_level = 0.95
            
            # Legend phase selection
            st.markdown("**Select phases to include in legend:**")
            n_cols = min(4, len(phases))
            legend_cols = st.columns(n_cols)
            legend_phases_selected = []
            for idx, ph in enumerate(phases):
                with legend_cols[idx % n_cols]:
                    if st.checkbox(f"✓ {ph}", value=True, key=f"leg_{ph}"):
                        legend_phases_selected.append(ph)
        
        with st.expander("⚙️ Per-Phase Marker Styling"):
            phase_data = []
            for i, ph in enumerate(phases):
                pk_df = generate_theoretical_peaks(ph, wavelength, tt_min, tt_max)
                
                with st.container():
                    st.markdown(f"**{ph}**")
                    c_col, c_shape, c_hkl = st.columns([2, 2, 3])
                    
                    # Color picker with preview
                    default_color = PHASE_LIBRARY.get(ph, {}).get("color", phase_colors[i % len(phase_colors)])
                    custom_color = c_col.color_picker("Color", value=default_color, key=f"col_{ph}")
                    c_col.markdown(f"<div style='width:20px;height:20px;background:{custom_color};border:1px solid #ccc;border-radius:3px'></div>", unsafe_allow_html=True)
                    
                    # Marker shape
                    shape_options = ["|", "_", "s", "^", "v", "d", "x", "+", "*"]
                    default_shape = PHASE_LIBRARY.get(ph, {}).get("marker_shape", "|")
                    default_idx = shape_options.index(default_shape) if default_shape in shape_options else 0
                    custom_shape = c_shape.selectbox("Marker", shape_options, index=default_idx, key=f"shp_{ph}")
                    
                    # hkl display toggle
                    show_this_hkl = c_hkl.checkbox(f"Show hkl", value=show_hkl, key=f"hkl_{ph}")
                
                # Build phase data for plotting
                hkl_data = None
                if show_this_hkl and len(pk_df) > 0:
                    hkl_data = [hkl.strip("()").split(",") if hkl else None for hkl in pk_df["hkl_label"].values]
                
                phase_data.append({
                    "name": ph, 
                    "positions": pk_df["two_theta"].values if len(pk_df) > 0 else np.array([]), 
                    "color": custom_color, 
                    "marker_shape": custom_shape, 
                    "hkl": hkl_data
                })
        
        # ═══════════════════════════════════════════════════════════════
        # GENERATE AND DISPLAY PLOT
        # ═══════════════════════════════════════════════════════════════
        
        try:
            with st.spinner("Generating publication plot..."):
                fig, ax = plot_rietveld_publication_enhanced(
                    two_theta=active_df["two_theta"].values,
                    observed=active_df["intensity"].values,
                    calculated=result["y_calc"],
                    difference=active_df["intensity"].values - result["y_calc"],
                    phase_data=phase_data,
                    offset_factor=offset_factor,
                    figsize=(fig_width, fig_width * 0.7),  # Maintain ~1.43 aspect ratio
                    font_size=font_size,
                    legend_pos=legend_pos if legend_pos != "off" else "off",
                    marker_row_spacing=marker_spacing,
                    legend_phases=legend_phases_selected if legend_phases_selected else None,
                    journal_style=journal_style,
                    colorblind_safe=colorblind_safe,
                    show_minor_grid=show_minor_grid,
                    four_sided_axes=four_sided_axes,
                    export_dpi=export_dpi,
                    show_confidence_band=show_confidence,
                    confidence_level=conf_level if show_confidence else 0.95
                )
                
                # Display in Streamlit
                st.pyplot(fig, dpi=min(150, export_dpi), use_container_width=True)  # Lower DPI for web display
                
                # ═══════════════════════════════════════════════════════════════
                # EXPORT OPTIONS - MULTIPLE FORMATS
                # ═══════════════════════════════════════════════════════════════
                
                st.markdown("#### 📥 Export Publication Figure")
                st.caption("All exports use professional formatting. Vector formats (PDF/EPS/SVG) recommended for journals.")
                
                col_e1, col_e2, col_e3, col_e4 = st.columns(4)
                
                filename_base = f"rietveld_pub_{selected_key}_{journal_style}" if journal_style != 'default' else f"rietveld_pub_{selected_key}"
                
                with col_e1:
                    # PDF (vector, universal)
                    buf = io.BytesIO()
                    fig.savefig(buf, format='pdf', bbox_inches='tight', pad_inches=0.05)
                    buf.seek(0)
                    st.download_button(
                        "📄 PDF (vector)", 
                        buf.read(), 
                        file_name=f"{filename_base}.pdf", 
                        mime="application/pdf", 
                        use_container_width=True
                    )
                
                with col_e2:
                    # EPS (vector, traditional journals)
                    buf = io.BytesIO()
                    fig.savefig(buf, format='eps', bbox_inches='tight', pad_inches=0.05)
                    buf.seek(0)
                    st.download_button(
                        "📐 EPS (vector)", 
                        buf.read(), 
                        file_name=f"{filename_base}.eps", 
                        mime="application/postscript", 
                        use_container_width=True
                    )
                
                with col_e3:
                    # TIFF 600 DPI (print journals)
                    buf = io.BytesIO()
                    fig.savefig(buf, format='tiff', dpi=export_dpi, compression='lzw', bbox_inches='tight', pad_inches=0.05)
                    buf.seek(0)
                    st.download_button(
                        "🖼️ TIFF 600 DPI", 
                        buf.read(), 
                        file_name=f"{filename_base}.tiff", 
                        mime="image/tiff", 
                        use_container_width=True
                    )
                
                with col_e4:
                    # PNG for presentations
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png', dpi=export_dpi, bbox_inches='tight', pad_inches=0.05)
                    buf.seek(0)
                    st.download_button(
                        "🖥️ PNG (web)", 
                        buf.read(), 
                        file_name=f"{filename_base}.png", 
                        mime="image/png", 
                        use_container_width=True
                    )
                
                plt.close(fig)
                
                # ═══════════════════════════════════════════════════════════════
                # JOURNAL SUBMISSION TIPS
                # ═══════════════════════════════════════════════════════════════
                
                with st.expander("📚 Journal Submission Guidelines"):
                    if journal_style == 'nature':
                        st.markdown("""
                        **Nature Portfolio Guidelines:**
                        - ✅ Use PDF or EPS (vector) for line art
                        - ✅ Minimum 600 DPI for raster images
                        - ✅ Font: Arial or Helvetica, 8-10 pt for labels
                        - ✅ Color mode: RGB for online, CMYK for print
                        - ✅ Figure width: 8.5 cm (single column) or 17 cm (double)
                        """)
                    elif journal_style == 'acs':
                        st.markdown("""
                        **ACS Guidelines:**
                        - ✅ Preferred: TIFF, EPS, or PDF at 600+ DPI
                        - ✅ Font: Times New Roman or Arial, 8-12 pt
                        - ✅ Line width: ≥0.5 pt for clarity
                        - ✅ Color: RGB; ensure readability in grayscale
                        - ✅ Max file size: 10 MB per figure
                        """)
                    elif journal_style == 'rsc':
                        st.markdown("""
                        **RSC Guidelines:**
                        - ✅ Submit as TIFF, EPS, or PDF at 600 DPI minimum
                        - ✅ Font: Arial or Helvetica preferred
                        - ✅ Ensure all text is legible at 100% zoom
                        - ✅ Color figures: provide grayscale version if required
                        - ✅ Include scale bars or axis labels with units
                        """)
                    else:
                        st.markdown("""
                        **General Best Practices:**
                        - ✅ Vector formats (PDF/EPS/SVG) preserve quality at any scale
                        - ✅ 600 DPI minimum for raster images in print journals
                        - ✅ Use sans-serif fonts for labels in presentations, serif for publications
                        - ✅ Ensure colorblind accessibility (tested with Okabe-Ito palette)
                        - ✅ Include figure caption with experimental details in manuscript
                        """)
                    
                    st.info("💡 Tip: Always check the target journal's "
                           "[Author Guidelines](https://www.nature.com/nature/for-authors/formatting-guide) "
                           "before final submission.")
                
        except Exception as e:
            st.error(f"❌ Plot generation failed: {str(e)}")
            st.code("Traceback:\n" + traceback.format_exc())
            st.warning("Try reducing font size, number of phases, or switching to 'default' journal style.")

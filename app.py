    elif source_option == "GitHub Samples (Pre-loaded)":
        st.markdown("### 📦 Mediloy S Co Samples from GitHub")
        st.caption("Repository: `Maryamslm/XRD-3Dprinted-Ret/SAMPLES`")
        
        # Fetch files once and cache
        if "gh_files_preloaded" not in st.session_state:
            with st.spinner("🔍 Fetching sample files from GitHub..."):
                files = fetch_github_files("Maryamslm/XRD-3Dprinted-Ret", "main", "SAMPLES")
                if files:
                    # ✅ FIXED: Removed 'if f["type"] == "file"' to prevent KeyError
                    st.session_state["gh_files_preloaded"] = {f["name"].upper(): f for f in files}
                    st.success(f"✅ Found {len(files)} compatible files")
                else:
                    st.warning("⚠️ Could not fetch files. Check internet connection or repo visibility.")
                    st.session_state["gh_files_preloaded"] = {}
        
        # Filter catalog keys that exist in the pre-loaded files
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

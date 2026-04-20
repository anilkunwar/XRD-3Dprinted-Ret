if run_btn:
    if not selected_phases:
        st.error("❌ Please select at least one phase in the sidebar!")
        st.stop()
    
    if len(active_df) == 0:
        st.error("❌ No data loaded! Please load XRD data first.")
        st.stop()
    
    with st.spinner("Running refinement…"):
        try:
            refiner = RietveldRefinement(
                active_df, 
                selected_phases, 
                wavelength, 
                bg_order, 
                peak_shape, 
                use_caglioti=use_caglioti, 
                estimate_uncertainty=estimate_unc
            )
            result = refiner.run()
            st.success("✅ Refinement completed!")
            
            # Store results
            st.session_state[f"result_{selected_key}"] = result
            st.session_state[f"phases_{selected_key}"] = selected_phases
            st.session_state["last_result"] = result
            st.session_state["last_phases"] = selected_phases
            st.session_state["last_sample"] = selected_key
            
        except Exception as e:
            st.error(f"❌ Refinement failed: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

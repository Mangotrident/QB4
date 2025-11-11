"""
Streamlit Dashboard for Quantum Bioenergetic Modeling Platform
================================================================

Interactive web interface for visualizing and analyzing quantum
bioenergetic modeling results.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys

# Add parent directory to path to import main module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Project_1_Quantum_Bioenergetic_Modeling import (
    run_full_pipeline,
    ProjectConfig,
    QuantumLifeEngine,
    ETCNetworkBuilder,
    QMNet,
    generate_realistic_tcga_data,
    plot_enaqt_curve,
    plot_cohort_comparison,
    cohort_comparison,
    compute_gsva_oxphos
)

# Page configuration
st.set_page_config(
    page_title="Quantum Bioenergetic Modeling",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #2563eb;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2563eb;
    }
    .stButton>button {
        width: 100%;
        background-color: #2563eb;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("⚛️ Quantum Bioenergetic Modeling")
st.sidebar.markdown("---")

# Navigation
page = st.sidebar.selectbox(
    "Navigation",
    ["🏠 Home", "🔬 Run Analysis", "📊 View Results", "🔍 Single Sample Analysis", "📈 ENAQT Visualization"]
)

# Home Page
if page == "🏠 Home":
    st.markdown('<div class="main-header">Quantum Bioenergetic Modeling Platform</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ## Mission Statement
    
    Building a new foundation for preventive medicine, one that quantifies the physics of life itself. 
    This platform maps how mitochondrial energy flow breaks down at the quantum level long before 
    genes or symptoms reveal disease.
    
    ## Key Features
    
    - **Quantum Transport Simulation**: Models ENAQT (Environment-Assisted Quantum Transport) in mitochondrial ETC
    - **Physics-Based Biomarkers**: Converts omics data to quantifiable metrics (ETE, τc, QLS)
    - **Early Disease Detection**: Identifies mitochondrial dysfunction before traditional biomarkers
    - **Drug Perturbation Analysis**: Validates model with known ETC inhibitors
    
    ## Platform Architecture
    
    1. **Phase 0**: Data acquisition and hypothesis framing
    2. **Phase 1**: Quantum Life Engine (QLE) - Lindblad master equation solver
    3. **Phase 2**: ETC Network Builder - Maps biology to quantum topology
    4. **Phase 3**: Q-MNet - Converts omics to physics biomarkers
    5. **Phase 4**: Statistical analysis and baseline comparison
    6. **Phase 5**: Perturbation validation
    7. **Phase 6**: Visualization and export
    
    ## Quick Start
    
    Navigate to **"Run Analysis"** to process a cohort of samples and generate quantum bioenergetic metrics.
    """)
    
    # Display system status
    st.markdown("### System Status")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Output Directory", ProjectConfig.OUTPUT_DIR, "✓ Ready")
    
    with col2:
        output_exists = os.path.exists(ProjectConfig.OUTPUT_DIR)
        st.metric("Results Available", "Yes" if output_exists else "No", 
                 "✓" if output_exists else "⚠️")
    
    with col3:
        data_dir = os.path.join(ProjectConfig.OUTPUT_DIR, "data")
        if os.path.exists(data_dir):
            files = len([f for f in os.listdir(data_dir) if f.endswith('.csv')])
            st.metric("Data Files", files, "✓")
        else:
            st.metric("Data Files", 0, "⚠️")

# Run Analysis Page
elif page == "🔬 Run Analysis":
    st.title("🔬 Run Cohort Analysis")
    
    st.markdown("""
    Process a cohort of samples through the quantum bioenergetic modeling pipeline.
    This will generate physics-based biomarkers for each sample.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        n_samples = st.slider("Number of samples per group", 10, 100, 50, 10)
        verbose = st.checkbox("Verbose output", value=True)
    
    with col2:
        st.markdown("### Analysis Parameters")
        st.write(f"- Total samples: {n_samples * 2}")
        st.write(f"- Tumor samples: {n_samples}")
        st.write(f"- Normal samples: {n_samples}")
        st.write(f"- Estimated runtime: ~{n_samples * 2 * 2 // 60} minutes")
    
    if st.button("🚀 Run Full Pipeline", type="primary"):
        with st.spinner("Running quantum bioenergetic modeling pipeline..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Run pipeline
                results = run_full_pipeline(n_samples=n_samples, verbose=verbose)
                
                progress_bar.progress(100)
                status_text.success("✓ Analysis complete!")
                
                # Display summary
                st.success("### Analysis Results")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Samples", len(results['cohort_metrics']))
                
                with col2:
                    best_auc = results['stats_df']['AUC'].max()
                    st.metric("Best AUC", f"{best_auc:.3f}")
                
                with col3:
                    min_p = results['stats_df']['P_Value'].min()
                    st.metric("Min P-Value", f"{min_p:.2e}")
                
                with col4:
                    max_d = results['stats_df']['Cohens_D'].abs().max()
                    st.metric("Max Effect Size", f"{max_d:.3f}")
                
                # Store results in session state
                st.session_state['results'] = results
                st.session_state['analysis_complete'] = True
                
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
                st.exception(e)

# View Results Page
elif page == "📊 View Results":
    st.title("📊 View Analysis Results")
    
    data_dir = os.path.join(ProjectConfig.OUTPUT_DIR, "data")
    
    if not os.path.exists(data_dir):
        st.warning("No results found. Please run an analysis first.")
        if st.button("Go to Run Analysis"):
            st.rerun()
    else:
        # Load data
        try:
            cohort_metrics = pd.read_csv(os.path.join(data_dir, 'cohort_metrics.csv'))
            stats_df = pd.read_csv(os.path.join(data_dir, 'statistical_analysis.csv'))
            
            st.success("✓ Results loaded successfully")
            
            # Summary statistics
            st.markdown("### Summary Statistics")
            st.dataframe(stats_df, use_container_width=True)
            
            # Interactive visualizations
            st.markdown("### Interactive Visualizations")
            
            metric_choice = st.selectbox(
                "Select metric to visualize",
                ['ETE_peak', 'gamma_optimal', 'tau_c', 'QLS']
            )
            
            # Create interactive plot
            fig = px.violin(
                cohort_metrics,
                x='group',
                y=metric_choice,
                color='group',
                box=True,
                points='all',
                title=f'{metric_choice} Distribution by Group'
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Comparison table
            st.markdown("### Detailed Comparison")
            comparison_df = cohort_metrics.groupby('group')[['ETE_peak', 'gamma_optimal', 'tau_c', 'QLS']].agg(['mean', 'std'])
            st.dataframe(comparison_df, use_container_width=True)
            
            # Download button
            csv = cohort_metrics.to_csv(index=False)
            st.download_button(
                label="📥 Download Results CSV",
                data=csv,
                file_name="quantum_bioenergetic_metrics.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"Error loading results: {str(e)}")

# Single Sample Analysis
elif page == "🔍 Single Sample Analysis":
    st.title("🔍 Single Sample Analysis")
    
    st.markdown("""
    Analyze a single sample's quantum bioenergetic profile.
    """)
    
    # Generate sample data
    if 'sample_data' not in st.session_state:
        expr_df, pheno_df, etc_genes = generate_realistic_tcga_data(n_samples=10, seed=42)
        st.session_state['sample_data'] = (expr_df, pheno_df, etc_genes)
    
    expr_df, pheno_df, etc_genes = st.session_state['sample_data']
    
    sample_id = st.selectbox("Select sample", expr_df.columns.tolist())
    
    if st.button("Analyze Sample"):
        with st.spinner("Computing quantum metrics..."):
            etc_builder = ETCNetworkBuilder()
            qle = QuantumLifeEngine(n_sites=etc_builder.n_sites)
            qmnet = QMNet(qle, etc_builder)
            
            try:
                metrics = qmnet.process_sample(expr_df, sample_id)
                
                st.success("✓ Analysis complete")
                
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("ETE Peak", f"{metrics['ETE_peak']:.4f}")
                
                with col2:
                    st.metric("γ* Optimal", f"{metrics['gamma_optimal']:.4f}")
                
                with col3:
                    st.metric("τc", f"{metrics['tau_c']:.4f}")
                
                with col4:
                    st.metric("QLS", f"{metrics['QLS']:.4f}")
                
                # Expression profile
                st.markdown("### ETC Complex Expression Profile")
                expr_dict = etc_builder.extract_expression(expr_df, sample_id)
                expr_df_viz = pd.DataFrame({
                    'Complex': list(expr_dict.keys()),
                    'Expression': list(expr_dict.values())
                })
                
                fig = px.bar(expr_df_viz, x='Complex', y='Expression', 
                            title='ETC Complex Expression Levels')
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error analyzing sample: {str(e)}")

# ENAQT Visualization
elif page == "📈 ENAQT Visualization":
    st.title("📈 ENAQT Curve Visualization")
    
    st.markdown("""
    Visualize the Environment-Assisted Quantum Transport (ENAQT) phenomenon.
    The ENAQT curve shows how energy transfer efficiency peaks at an optimal dephasing rate.
    """)
    
    # Parameters
    col1, col2 = st.columns(2)
    
    with col1:
        n_sites = st.slider("Number of sites", 5, 10, 7)
        gamma_min = st.number_input("γ min", 0.001, 0.01, 0.001, 0.001)
        gamma_max = st.number_input("γ max", 0.05, 0.2, 0.1, 0.01)
        gamma_steps = st.slider("γ steps", 20, 100, 50, 10)
    
    with col2:
        coupling_strength = st.number_input("Coupling strength", 0.01, 0.1, 0.05, 0.01)
        sink_rate = st.number_input("Sink rate", 0.05, 0.2, 0.1, 0.01)
        t_max = st.number_input("Time max", 50, 200, 100, 10)
    
    if st.button("Generate ENAQT Curve"):
        with st.spinner("Computing ENAQT curve..."):
            # Setup
            qle = QuantumLifeEngine(n_sites=n_sites)
            site_energies = np.array([-0.1 * i for i in range(n_sites)])
            couplings = {(i, i+1): coupling_strength for i in range(n_sites-1)}
            gamma_range = np.linspace(gamma_min, gamma_max, gamma_steps)
            
            # Compute
            etes = qle.sweep_gamma(site_energies, couplings, gamma_range, 
                                  sink_site=n_sites-1, sink_rate=sink_rate, t_max=t_max)
            
            gamma_optimal = gamma_range[np.argmax(etes)]
            ete_peak = np.max(etes)
            
            # Plot
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=gamma_range,
                y=etes,
                mode='lines+markers',
                name='ETE',
                line=dict(color='#2563eb', width=3),
                marker=dict(size=6)
            ))
            
            fig.add_vline(
                x=gamma_optimal,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Optimal γ* = {gamma_optimal:.4f}"
            )
            
            fig.add_hline(
                y=ete_peak,
                line_dash="dot",
                line_color="green",
                annotation_text=f"Peak ETE = {ete_peak:.4f}"
            )
            
            fig.update_layout(
                title="ENAQT: Noise-Assisted Quantum Transport",
                xaxis_title="Dephasing Rate γ (a.u.)",
                yaxis_title="Energy Transfer Efficiency",
                height=600,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Peak ETE", f"{ete_peak:.4f}")
            with col2:
                st.metric("Optimal γ*", f"{gamma_optimal:.4f}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    Quantum Bioenergetic Modeling Platform v1.0.0<br>
    Built with Streamlit ⚛️
</div>
""", unsafe_allow_html=True)


# Quantum Bioenergetic Modeling Platform - File Structure

## Project Files Overview

### Core Implementation Files

#### `Project_1_Quantum_Bioenergetic_Modeling.py`
**Main pipeline implementation** - Contains all phases of the quantum bioenergetic modeling platform:

- **Phase 0**: Framing & Hypothesis (ProjectConfig class)
- **Phase 1**: Quantum Life Engine (QLE) - Quantum transport simulator
- **Phase 2**: Data Acquisition & ETC Network Builder
- **Phase 3**: Quantum-Metabolic Network (Q-MNet) - Omics to physics conversion
- **Phase 4**: Cohort & Benchmark Analysis
- **Phase 5**: Perturbation Validation
- **Phase 6**: Visualization & Export

**Key Classes:**
- `ProjectConfig`: Configuration and constants
- `QuantumLifeEngine`: Lindblad master equation solver
- `ETCNetworkBuilder`: Maps biology to quantum topology
- `QMNet`: Converts omics to physics biomarkers

**Main Function:**
- `run_full_pipeline()`: Executes complete analysis pipeline

**Usage:**
```bash
python Project_1_Quantum_Bioenergetic_Modeling.py
```

---

#### `Project_2_requirements.txt`
**Python dependencies** - All required packages with version specifications:

- numpy==1.23.5
- scipy>=1.10.0
- pandas>=2.0.0
- matplotlib>=3.7.0
- seaborn>=0.12.0
- scikit-learn>=1.3.0
- statsmodels>=0.14.0
- networkx>=3.1
- qutip==4.7.3 (critical for quantum simulations)
- streamlit>=1.28.0
- plotly>=5.17.0

**Installation:**
```bash
pip install -r Project_2_requirements.txt
```

---

#### `Project_3_Streamlit_App.py`
**Interactive web dashboard** - Streamlit-based visualization and analysis interface:

**Pages:**
1. **Home**: Platform overview and system status
2. **Run Analysis**: Execute full pipeline with configurable parameters
3. **View Results**: Interactive visualization of cohort metrics
4. **Single Sample Analysis**: Analyze individual samples
5. **ENAQT Visualization**: Interactive ENAQT curve exploration

**Features:**
- Real-time metric visualization
- Interactive Plotly charts
- Download results as CSV
- Parameter tuning interface

**Usage:**
```bash
streamlit run Project_3_Streamlit_App.py
```

---

#### `Project_4_README.md`
**Comprehensive documentation** - Complete user guide including:

- Mission statement and scientific background
- Installation instructions
- Quick start guide
- Platform architecture overview
- API usage examples
- Troubleshooting guide
- Citation information

---

#### `Project_5_gitignore.txt`
**Git ignore patterns** - Excludes generated files and outputs:

- Python cache files
- Virtual environments
- Output directories (`qemd_outputs/`)
- IDE files
- OS-specific files

**Note:** Rename to `.gitignore` when using with Git:
```bash
mv Project_5_gitignore.txt .gitignore
```

---

#### `Project_6_setup.sh`
**Automated setup script** - One-command installation:

- Checks Python version (requires 3.10+)
- Creates virtual environment
- Installs all dependencies
- Verifies installation

**Usage:**
```bash
bash Project_6_setup.sh
# or
chmod +x Project_6_setup.sh && ./Project_6_setup.sh
```

---

## Output Structure

After running the pipeline, the following directory structure is created:

```
qemd_outputs/
├── data/
│   ├── cohort_metrics.csv          # Individual sample metrics
│   ├── statistical_analysis.csv    # Group comparisons
│   └── cohort_with_baseline.csv    # Combined metrics
├── figures/
│   ├── enaqt_curve.png             # ENAQT visualization
│   └── cohort_comparison.png       # Group comparison plots
└── final_report.txt                # Summary report
```

## Workflow

### 1. Initial Setup
```bash
# Option A: Use setup script
bash Project_6_setup.sh
source venv/bin/activate

# Option B: Manual setup
pip install -r Project_2_requirements.txt
```

### 2. Run Analysis
```bash
# Command line
python Project_1_Quantum_Bioenergetic_Modeling.py

# Or via Streamlit
streamlit run Project_3_Streamlit_App.py
```

### 3. View Results
- Check `qemd_outputs/data/` for CSV files
- Check `qemd_outputs/figures/` for visualizations
- Read `qemd_outputs/final_report.txt` for summary

## Key Metrics Generated

1. **ETE_peak**: Energy Transfer Efficiency at optimal dephasing
2. **gamma_optimal**: Optimal dephasing rate (γ*)
3. **tau_c**: Coherence time
4. **QLS**: Quantum Life Score (composite metric)

## Dependencies by File

### Project_1_Quantum_Bioenergetic_Modeling.py
- numpy, scipy, pandas
- matplotlib, seaborn
- scikit-learn, statsmodels
- qutip (critical)

### Project_3_Streamlit_App.py
- All dependencies from Project_1
- streamlit
- plotly

## Version Compatibility

- **Python**: 3.10+
- **QuTiP**: 4.7.3 (exact version required)
- **NumPy**: 1.23.5 (exact version for compatibility)

## Notes

- The main pipeline file is self-contained and can be run independently
- Streamlit app imports from the main module
- All outputs are saved to `qemd_outputs/` directory
- Generated data is synthetic (TCGA-BRCA-like) for demonstration


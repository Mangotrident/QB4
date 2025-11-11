#!/bin/bash

# Setup script for Quantum Bioenergetic Modeling Platform

echo "=========================================="
echo "Quantum Bioenergetic Modeling Platform"
echo "Setup Script"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python $python_version"

# Check if Python 3.10+
if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)"; then
    echo "ERROR: Python 3.10 or higher is required"
    exit 1
fi

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo ""
echo "Installing dependencies..."
pip install -r Project_2_requirements.txt

# Verify installation
echo ""
echo "Verifying installation..."
python3 -c "import qutip; print('✓ QuTiP version:', qutip.__version__)"
python3 -c "import streamlit; print('✓ Streamlit installed')"
python3 -c "import numpy, pandas, matplotlib, seaborn, scipy, sklearn; print('✓ All dependencies installed')"

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "To activate the virtual environment:"
echo "  source venv/bin/activate"
echo ""
echo "To run the main pipeline:"
echo "  python Project_1_Quantum_Bioenergetic_Modeling.py"
echo ""
echo "To launch Streamlit dashboard:"
echo "  streamlit run Project_3_Streamlit_App.py"
echo ""


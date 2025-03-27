#!/bin/bash

# Earth's Atmospheric Characterization - Environment Setup Script
# This script sets up the Python virtual environment and installs all dependencies

# Exit on error
set -e

echo "Setting up Earth's Atmospheric Characterization environment..."

# Create directory structure
echo "Creating directory structure..."
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/temporal
mkdir -p data/metrics
mkdir -p docs/api
mkdir -p docs/reports
mkdir -p docs/figures
mkdir -p notebooks
mkdir -p models

# Create Python virtual environment
echo "Creating Python virtual environment..."
python -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Set up pre-commit hooks if git is available
if command -v git &> /dev/null; then
    echo "Setting up git pre-commit hooks..."
    if [ -f .git/hooks/pre-commit ]; then
        echo "Pre-commit hook already exists, skipping..."
    else
        cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
# Pre-commit hook to run tests and linting

# Run tests
echo "Running tests..."
pytest -xvs tests/

# Run linting
echo "Running linting..."
flake8 src/

# If everything passes, allow the commit
exit 0
EOF
        chmod +x .git/hooks/pre-commit
    fi
fi

# Create initial configuration file
echo "Creating initial configuration file..."
cat > config.yaml << 'EOF'
# Earth's Atmospheric Characterization Configuration

# Data paths
data:
  raw_dir: "data/raw"
  processed_dir: "data/processed"
  temporal_dir: "data/temporal"
  metrics_dir: "data/metrics"

# API credentials
# Replace with your actual credentials
api:
  nasa:
    username: "your_nasa_username"
    password: "your_nasa_password"
  copernicus:
    username: "your_copernicus_username"
    password: "your_copernicus_password"

# Processing parameters
processing:
  target_resolution: 30.0  # meters
  normalization_method: "min-max"
  time_step: "1D"  # 1 day

# Validation parameters
validation:
  cloud_detection:
    reference_dataset: "MODIS_COSP_L2"
  aerosol_analysis:
    reference_dataset: "AERONET_L2"
    wavelength: 550.0
EOF

echo "Environment setup complete!"
echo "To activate the environment, run: source venv/bin/activate"
echo "To download initial datasets, run: python scripts/download_data.py"
echo ""
echo "Don't forget to update your API credentials in config.yaml"
# Earth's Atmospheric Characterization - Quick Start Guide

This guide will help you quickly set up and start using the Earth's Atmospheric Characterization framework for analyzing atmospheric data for Solar Power Satellite (SPS) development.

## Prerequisites

- Python 3.8 or higher
- Git
- 10+ GB of free disk space (for datasets)
- NASA Earthdata account (for MODIS and CALIPSO data)
- Copernicus account (for Sentinel-5P data)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Earths-Atmospheric-Characterization.git
cd Earths-Atmospheric-Characterization
```

2. Run the setup script to create the environment and install dependencies:
```bash
# Make the script executable
chmod +x scripts/setup_environment.sh

# Run the setup script
./scripts/setup_environment.sh
```

3. Activate the virtual environment:
```bash
source venv/bin/activate
```

4. Update the API credentials in `config.yaml` with your NASA Earthdata and Copernicus credentials.

## Downloading Data

1. Download sample datasets:
```bash
python scripts/download_data.py --start-date 2023-01-01 --end-date 2023-01-05
```

This will download a small subset of data for testing. You can adjust the date range as needed.

## Processing Data

1. Standardize the downloaded data:
```bash
python -c "from src.data_preprocessing.standardize import DataStandardizer; standardizer = DataStandardizer(); standardizer.standardize_directory(Path('data/raw'))"
```

2. Resample the data to a common resolution:
```bash
python -c "from src.data_preprocessing.resampling import SpatialResampler; resampler = SpatialResampler(); resampler.resample_directory(Path('data/processed'))"
```

3. Normalize the data:
```bash
python -c "from src.data_preprocessing.normalization import DataNormalizer; normalizer = DataNormalizer(); normalizer.normalize_directory(Path('data/processed'))"
```

## Running Analyses

### Cloud Detection

```python
from src.cloud_detection.modis import ModisCloudDetector
import xarray as xr

# Load a processed MODIS dataset
dataset = xr.open_dataset('data/processed/MOD06_L2/some_file.nc')

# Create a cloud detector
detector = ModisCloudDetector()

# Detect clouds
cloud_mask = detector.detect_from_dataset(dataset)

# Save the cloud mask
cloud_mask.to_netcdf('data/processed/cloud_mask.nc')
```

### Aerosol Analysis

```python
from src.aerosol_analysis.optical_depth import Sentinel5POpticalDepthEstimator
import xarray as xr

# Load a processed Sentinel-5P dataset
dataset = xr.open_dataset('data/processed/S5P_L2_AER_AI/some_file.nc')

# Create an optical depth estimator
estimator = Sentinel5POpticalDepthEstimator(wavelength=550.0)

# Estimate optical depth
aod = estimator.estimate_from_dataset(dataset)

# Save the optical depth
aod.to_netcdf('data/processed/aerosol_optical_depth.nc')
```

### Turbulence Modeling

```python
from src.turbulence_modeling.lidar_processing import CalipsoLidarProcessor
import xarray as xr

# Load a processed CALIPSO dataset
dataset = xr.open_dataset('data/processed/CAL_LID_L2_Cloud_Profile/some_file.nc')

# Create a lidar processor
processor = CalipsoLidarProcessor()

# Process the data
processed_data = processor.process_dataset(dataset)

# Save the processed data
processed_data.to_netcdf('data/processed/turbulence_data.nc')
```

### Wavelength Optimization

```python
from src.wavelength_optimizer.transmission import SimpleTransmissionSimulator
import numpy as np
import matplotlib.pyplot as plt

# Create a transmission simulator
simulator = SimpleTransmissionSimulator()

# Define wavelength range
wavelengths = np.linspace(300, 2500, 1000)

# Simulate transmission for different altitudes
transmission_0km = simulator.simulate(wavelengths, altitude=0, zenith_angle=0)
transmission_10km = simulator.simulate(wavelengths, altitude=10000, zenith_angle=0)
transmission_20km = simulator.simulate(wavelengths, altitude=20000, zenith_angle=0)

# Find optimal wavelengths
optimal_wavelengths = simulator.find_optimal_wavelengths(
    wavelength_range=(300, 2500),
    num_wavelengths=5,
    altitude=0,
    zenith_angle=0
)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(wavelengths, transmission_0km, label='0 km')
plt.plot(wavelengths, transmission_10km, label='10 km')
plt.plot(wavelengths, transmission_20km, label='20 km')
plt.scatter(optimal_wavelengths, 
            simulator.simulate(optimal_wavelengths, altitude=0, zenith_angle=0),
            color='red', s=100, label='Optimal wavelengths')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Transmission')
plt.title('Atmospheric Transmission vs. Wavelength')
plt.legend()
plt.grid(True)
plt.savefig('docs/figures/transmission_vs_wavelength.png')
plt.close()
```

## Resource Profiling

To profile the resource usage of the algorithms:

```bash
python scripts/profile_resources.py
```

This will generate a report in `metrics/resource_profile.json` with memory and CPU usage for each algorithm.

## Docker Deployment

To run the framework in a Docker container:

```bash
# Build the Docker image
docker build -t atmospheric-characterization .

# Run the container
docker run -it atmospheric-characterization
```

To simulate CubeSat constraints:

```bash
# Build the CubeSat simulation image
docker build -t cubesat-sim --memory=200mb --cpus=1.0 -f Dockerfile.jetson .

# Run the CubeSat simulation
docker run -it cubesat-sim
```

## Next Steps

- Explore the Jupyter notebooks in the `notebooks` directory for examples and tutorials
- Check the API documentation in the `docs/api` directory
- Contribute to the project by submitting pull requests

For more detailed information, refer to the full documentation in the `docs` directory.
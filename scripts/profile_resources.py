#!/usr/bin/env python
"""
Resource profiling script for Earth's Atmospheric Characterization project.

This script profiles the memory and CPU usage of various algorithms in the project.
It's designed to help optimize the code for CubeSat deployment.

Usage:
    python profile_resources.py [--config CONFIG] [--output OUTPUT]

Options:
    --config CONFIG    Path to configuration file [default: config.yaml]
    --output OUTPUT    Path to output file [default: metrics/resource_profile.json]
"""

import os
import sys
import yaml
import json
import time
import argparse
import logging
import psutil
import numpy as np
from pathlib import Path
from functools import wraps
from datetime import datetime

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import project modules
from src.cloud_detection.modis import ModisCloudDetector
from src.cloud_detection.sentinel import Sentinel5PCloudDetector
from src.aerosol_analysis.optical_depth import Sentinel5POpticalDepthEstimator
from src.turbulence_modeling.lidar_processing import CalipsoLidarProcessor
from src.wavelength_optimizer.transmission import SimpleTransmissionSimulator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

def profile_resources(func):
    """
    Decorator to profile memory and CPU usage of a function.
    
    Args:
        func: Function to profile
        
    Returns:
        Wrapped function that profiles resources
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Get initial resource usage
        process = psutil.Process(os.getpid())
        start_mem = process.memory_info().rss / 1024 / 1024  # MB
        start_time = time.perf_counter()
        
        # Call the function
        result = func(*args, **kwargs)
        
        # Get final resource usage
        end_time = time.perf_counter()
        end_mem = process.memory_info().rss / 1024 / 1024  # MB
        
        # Calculate resource usage
        elapsed_time = end_time - start_time
        mem_used = end_mem - start_mem
        
        # Log resource usage
        logger.info(f"{func.__name__}:")
        logger.info(f"  Time: {elapsed_time:.2f} seconds")
        logger.info(f"  Memory: {mem_used:.2f} MB")
        
        # Add resource usage to result if it's a dictionary
        if isinstance(result, dict):
            result["profiling"] = {
                "time": elapsed_time,
                "memory": mem_used
            }
        
        return result, {
            "time": elapsed_time,
            "memory": mem_used
        }
    
    return wrapper

@profile_resources
def profile_cloud_detection(data_size=(5, 1000, 1000)):
    """
    Profile cloud detection algorithm.
    
    Args:
        data_size: Size of test data (bands, height, width)
        
    Returns:
        Cloud mask
    """
    # Create test data
    reflectance = np.random.random(data_size)
    
    # Create cloud detector
    detector = ModisCloudDetector()
    
    # Detect clouds
    cloud_mask = detector.detect(reflectance)
    
    return {"cloud_mask": cloud_mask}

@profile_resources
def profile_sentinel_cloud_detection(data_size=(5, 1000, 1000)):
    """
    Profile Sentinel-5P cloud detection algorithm.
    
    Args:
        data_size: Size of test data (bands, height, width)
        
    Returns:
        Cloud mask
    """
    # Create test data
    reflectance = np.random.random(data_size)
    
    # Create cloud detector
    detector = Sentinel5PCloudDetector()
    
    # Detect clouds
    cloud_mask = detector.detect(reflectance)
    
    return {"cloud_mask": cloud_mask}

@profile_resources
def profile_optical_depth_estimation(data_size=(5, 1000, 1000)):
    """
    Profile optical depth estimation algorithm.
    
    Args:
        data_size: Size of test data (bands, height, width)
        
    Returns:
        Optical depth
    """
    # Create test data
    reflectance = np.random.random(data_size)
    
    # Create optical depth estimator
    estimator = Sentinel5POpticalDepthEstimator()
    
    # Estimate optical depth
    optical_depth = estimator.estimate(reflectance)
    
    return {"optical_depth": optical_depth}

@profile_resources
def profile_lidar_processing(data_size=(100, 500)):
    """
    Profile lidar processing algorithm.
    
    Args:
        data_size: Size of test data (profiles, altitude_bins)
        
    Returns:
        Processed lidar data
    """
    # Create test data
    backscatter = np.random.random(data_size)
    altitude = np.linspace(0, 30000, data_size[1])
    
    # Create lidar processor
    processor = CalipsoLidarProcessor()
    
    # Process lidar data
    results = processor.process(backscatter, altitude)
    
    return results

@profile_resources
def profile_transmission_simulation(num_wavelengths=1000):
    """
    Profile transmission simulation algorithm.
    
    Args:
        num_wavelengths: Number of wavelengths to simulate
        
    Returns:
        Transmission values
    """
    # Create test data
    wavelengths = np.linspace(300, 2500, num_wavelengths)
    
    # Create transmission simulator
    simulator = SimpleTransmissionSimulator()
    
    # Simulate transmission
    transmission = simulator.simulate(
        wavelengths=wavelengths,
        altitude=0,
        zenith_angle=0,
        atmosphere_profile="midlatitude_summer"
    )
    
    return {"transmission": transmission}

@profile_resources
def profile_wavelength_optimization():
    """
    Profile wavelength optimization algorithm.
    
    Returns:
        Optimal wavelengths
    """
    # Create transmission simulator
    simulator = SimpleTransmissionSimulator()
    
    # Find optimal wavelengths
    optimal_wavelengths = simulator.find_optimal_wavelengths(
        wavelength_range=(300, 2500),
        num_wavelengths=10,
        altitude=0,
        zenith_angle=0,
        atmosphere_profile="midlatitude_summer",
        min_spacing=50
    )
    
    return {"optimal_wavelengths": optimal_wavelengths.tolist()}

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Profile resources for Earth's Atmospheric Characterization")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration file")
    parser.add_argument("--output", default="metrics/resource_profile.json", help="Path to output file")
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def main():
    """Main function."""
    # Parse command line arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create output directory if it doesn't exist
    output_path = Path(args.output)
    os.makedirs(output_path.parent, exist_ok=True)
    
    # Profile resources
    logger.info("Profiling resources...")
    
    # Profile cloud detection
    logger.info("Profiling cloud detection...")
    cloud_results, cloud_profile = profile_cloud_detection()
    
    # Profile Sentinel cloud detection
    logger.info("Profiling Sentinel-5P cloud detection...")
    sentinel_cloud_results, sentinel_cloud_profile = profile_sentinel_cloud_detection()
    
    # Profile optical depth estimation
    logger.info("Profiling optical depth estimation...")
    optical_depth_results, optical_depth_profile = profile_optical_depth_estimation()
    
    # Profile lidar processing
    logger.info("Profiling lidar processing...")
    lidar_results, lidar_profile = profile_lidar_processing()
    
    # Profile transmission simulation
    logger.info("Profiling transmission simulation...")
    transmission_results, transmission_profile = profile_transmission_simulation()
    
    # Profile wavelength optimization
    logger.info("Profiling wavelength optimization...")
    wavelength_results, wavelength_profile = profile_wavelength_optimization()
    
    # Combine results
    profile_results = {
        "timestamp": datetime.now().isoformat(),
        "system_info": {
            "cpu_count": psutil.cpu_count(),
            "total_memory": psutil.virtual_memory().total / 1024 / 1024  # MB
        },
        "profiles": {
            "cloud_detection": cloud_profile,
            "sentinel_cloud_detection": sentinel_cloud_profile,
            "optical_depth_estimation": optical_depth_profile,
            "lidar_processing": lidar_profile,
            "transmission_simulation": transmission_profile,
            "wavelength_optimization": wavelength_profile
        }
    }
    
    # Save results to file
    with open(output_path, "w") as f:
        json.dump(profile_results, f, indent=2)
    
    logger.info(f"Resource profiling complete! Results saved to {output_path}")
    
    # Print summary
    logger.info("Resource profiling summary:")
    logger.info(f"Cloud detection: {cloud_profile['time']:.2f}s, {cloud_profile['memory']:.2f}MB")
    logger.info(f"Sentinel cloud detection: {sentinel_cloud_profile['time']:.2f}s, {sentinel_cloud_profile['memory']:.2f}MB")
    logger.info(f"Optical depth estimation: {optical_depth_profile['time']:.2f}s, {optical_depth_profile['memory']:.2f}MB")
    logger.info(f"Lidar processing: {lidar_profile['time']:.2f}s, {lidar_profile['memory']:.2f}MB")
    logger.info(f"Transmission simulation: {transmission_profile['time']:.2f}s, {transmission_profile['memory']:.2f}MB")
    logger.info(f"Wavelength optimization: {wavelength_profile['time']:.2f}s, {wavelength_profile['memory']:.2f}MB")

if __name__ == "__main__":
    main()
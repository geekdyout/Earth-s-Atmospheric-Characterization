#!/usr/bin/env python
"""
Data download script for Earth's Atmospheric Characterization project.

This script downloads the required datasets from various sources:
- MODIS (MOD06_L2, MYD06_L2) from NASA CMR
- Sentinel-5P (AER_AI, CLOUD) from Copernicus
- CALIPSO (CAL_LID_L2_Cloud_Profile) from NASA CMR
- NEON (DP3.30006.001) from NEON API

Usage:
    python download_data.py [--config CONFIG] [--start-date START_DATE] [--end-date END_DATE]

Options:
    --config CONFIG         Path to configuration file [default: config.yaml]
    --start-date START_DATE Start date for data download (YYYY-MM-DD) [default: 2023-01-01]
    --end-date END_DATE     End date for data download (YYYY-MM-DD) [default: 2023-01-31]
"""

import os
import sys
import yaml
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_acquisition.nasa_cmr import NasaCMRClient
from src.data_acquisition.copernicus import CopernicusClient
from src.data_acquisition.neon import NeonClient

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Download data for Earth's Atmospheric Characterization")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration file")
    parser.add_argument("--start-date", default="2023-01-01", help="Start date for data download (YYYY-MM-DD)")
    parser.add_argument("--end-date", default="2023-01-31", help="End date for data download (YYYY-MM-DD)")
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def download_modis_data(client, start_date, end_date, raw_dir):
    """Download MODIS data."""
    logger.info("Downloading MODIS data...")
    
    # Download MOD06_L2 (Terra MODIS)
    logger.info("Downloading MOD06_L2 data...")
    mod06_files = client.download_modis_data(
        product="MOD06_L2",
        start_date=start_date,
        end_date=end_date,
        max_granules=10
    )
    logger.info(f"Downloaded {len(mod06_files)} MOD06_L2 granules")
    
    # Download MYD06_L2 (Aqua MODIS)
    logger.info("Downloading MYD06_L2 data...")
    myd06_files = client.download_modis_data(
        product="MYD06_L2",
        start_date=start_date,
        end_date=end_date,
        max_granules=10
    )
    logger.info(f"Downloaded {len(myd06_files)} MYD06_L2 granules")
    
    return mod06_files + myd06_files

def download_sentinel5p_data(client, start_date, end_date, raw_dir):
    """Download Sentinel-5P data."""
    logger.info("Downloading Sentinel-5P data...")
    
    # Download AER_AI (Aerosol Index)
    logger.info("Downloading S5P_L2_AER_AI data...")
    aer_ai_files = client.download_sentinel5p_data(
        product_type="S5P_L2_AER_AI",
        start_date=start_date,
        end_date=end_date,
        max_products=5
    )
    logger.info(f"Downloaded {len(aer_ai_files)} S5P_L2_AER_AI products")
    
    # Download CLOUD
    logger.info("Downloading S5P_L2_CLOUD data...")
    cloud_files = client.download_sentinel5p_data(
        product_type="S5P_L2_CLOUD",
        start_date=start_date,
        end_date=end_date,
        max_products=5
    )
    logger.info(f"Downloaded {len(cloud_files)} S5P_L2_CLOUD products")
    
    return aer_ai_files + cloud_files

def download_calipso_data(client, start_date, end_date, raw_dir):
    """Download CALIPSO data."""
    logger.info("Downloading CALIPSO data...")
    
    # Download CAL_LID_L2_Cloud_Profile
    logger.info("Downloading CAL_LID_L2_Cloud_Profile data...")
    calipso_files = client.download_calipso_data(
        product="CAL_LID_L2_Cloud_Profile",
        start_date=start_date,
        end_date=end_date,
        max_granules=5
    )
    logger.info(f"Downloaded {len(calipso_files)} CAL_LID_L2_Cloud_Profile granules")
    
    return calipso_files

def download_neon_data(client, start_date, end_date, raw_dir):
    """Download NEON data."""
    logger.info("Downloading NEON data...")
    
    # Download DP3.30006.001 (Spectrometer orthorectified surface directional reflectance)
    logger.info("Downloading DP3.30006.001 data...")
    
    # Extract year and month from start_date
    year = start_date.year
    month = start_date.month
    
    neon_files = client.download_data(
        product_code="DP3.30006.001",
        year=year,
        month=month
    )
    logger.info(f"Downloaded {len(neon_files)} DP3.30006.001 files")
    
    return neon_files

def main():
    """Main function."""
    # Parse command line arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Parse dates
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
    
    # Get data directory
    raw_dir = config["data"]["raw_dir"]
    
    # Create clients
    nasa_client = NasaCMRClient(
        username=config["api"]["nasa"]["username"],
        password=config["api"]["nasa"]["password"],
        data_dir=raw_dir
    )
    
    copernicus_client = CopernicusClient(
        username=config["api"]["copernicus"]["username"],
        password=config["api"]["copernicus"]["password"],
        data_dir=raw_dir
    )
    
    neon_client = NeonClient(
        data_dir=raw_dir
    )
    
    # Download data
    try:
        # Download MODIS data
        modis_files = download_modis_data(nasa_client, start_date, end_date, raw_dir)
        
        # Download Sentinel-5P data
        sentinel5p_files = download_sentinel5p_data(copernicus_client, start_date, end_date, raw_dir)
        
        # Download CALIPSO data
        calipso_files = download_calipso_data(nasa_client, start_date, end_date, raw_dir)
        
        # Download NEON data
        neon_files = download_neon_data(neon_client, start_date, end_date, raw_dir)
        
        # Print summary
        logger.info("Data download complete!")
        logger.info(f"Downloaded {len(modis_files)} MODIS files")
        logger.info(f"Downloaded {len(sentinel5p_files)} Sentinel-5P files")
        logger.info(f"Downloaded {len(calipso_files)} CALIPSO files")
        logger.info(f"Downloaded {len(neon_files)} NEON files")
        
    except Exception as e:
        logger.error(f"Error downloading data: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
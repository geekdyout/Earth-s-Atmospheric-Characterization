"""
NEON API client for downloading NEON data products.
"""

import os
import requests
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
import pandas as pd

logger = logging.getLogger(__name__)

class NeonClient:
    """Client for NEON API."""
    
    def __init__(self, api_token: Optional[str] = None, data_dir: str = "data/raw"):
        """
        Initialize the NEON client.
        
        Args:
            api_token: Optional NEON API token for authenticated requests
            data_dir: Directory to store downloaded data
        """
        self.api_token = api_token
        self.data_dir = Path(data_dir)
        self.base_url = "https://data.neonscience.org/api/v0"
        
        # Create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Set up session with API token if provided
        self.session = requests.Session()
        if api_token:
            self.session.headers.update({"X-API-Token": api_token})
    
    def get_products(self) -> List[Dict]:
        """
        Get a list of all available NEON data products.
        
        Returns:
            List of data product metadata dictionaries
        """
        url = f"{self.base_url}/products"
        response = self.session.get(url)
        response.raise_for_status()
        data = response.json()
        return data.get("data", [])
    
    def get_product_info(self, product_code: str) -> Dict:
        """
        Get detailed information about a specific NEON data product.
        
        Args:
            product_code: NEON product code (e.g., DP3.30006.001)
            
        Returns:
            Data product metadata dictionary
        """
        url = f"{self.base_url}/products/{product_code}"
        response = self.session.get(url)
        response.raise_for_status()
        data = response.json()
        return data.get("data", {})
    
    def get_sites(self) -> List[Dict]:
        """
        Get a list of all NEON sites.
        
        Returns:
            List of site metadata dictionaries
        """
        url = f"{self.base_url}/sites"
        response = self.session.get(url)
        response.raise_for_status()
        data = response.json()
        return data.get("data", [])
    
    def get_site_info(self, site_code: str) -> Dict:
        """
        Get detailed information about a specific NEON site.
        
        Args:
            site_code: NEON site code (e.g., HARV)
            
        Returns:
            Site metadata dictionary
        """
        url = f"{self.base_url}/sites/{site_code}"
        response = self.session.get(url)
        response.raise_for_status()
        data = response.json()
        return data.get("data", {})
    
    def get_product_availability(self, product_code: str, site_code: Optional[str] = None) -> List[Dict]:
        """
        Get availability information for a NEON data product.
        
        Args:
            product_code: NEON product code (e.g., DP3.30006.001)
            site_code: Optional NEON site code to filter by
            
        Returns:
            List of availability metadata dictionaries
        """
        url = f"{self.base_url}/products/{product_code}/availability"
        if site_code:
            url += f"?site={site_code}"
        
        response = self.session.get(url)
        response.raise_for_status()
        data = response.json()
        return data.get("data", [])
    
    def download_data(
        self, 
        product_code: str, 
        site_code: Optional[str] = None,
        year: Optional[int] = None,
        month: Optional[int] = None,
        output_dir: Optional[Path] = None
    ) -> List[Path]:
        """
        Download NEON data for a specific product, site, and time period.
        
        Args:
            product_code: NEON product code (e.g., DP3.30006.001)
            site_code: Optional NEON site code to filter by
            year: Optional year to filter by
            month: Optional month to filter by
            output_dir: Optional directory to save the data
            
        Returns:
            List of paths to downloaded files
        """
        if output_dir is None:
            output_dir = self.data_dir / product_code
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get data availability
        availability = self.get_product_availability(product_code, site_code)
        
        # Filter by year and month if specified
        filtered_availability = []
        for item in availability:
            url = item.get("url", "")
            # NEON URLs typically include year and month in the path
            if year and str(year) not in url:
                continue
            if month and f"{year:04d}-{month:02d}" not in url:
                continue
            filtered_availability.append(item)
        
        # Download data files
        downloaded_files = []
        for item in filtered_availability:
            url = item.get("url")
            if not url:
                continue
            
            # Get the filename from the URL
            filename = url.split("/")[-1]
            output_path = output_dir / filename
            
            # Download the file
            response = self.session.get(url, stream=True)
            response.raise_for_status()
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"Downloaded {url} to {output_path}")
            downloaded_files.append(output_path)
        
        return downloaded_files
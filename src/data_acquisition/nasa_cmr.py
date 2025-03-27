"""
NASA Common Metadata Repository (CMR) API client for downloading MODIS and CALIPSO datasets.
"""

import os
import requests
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple

logger = logging.getLogger(__name__)

class NasaCMRClient:
    """Client for NASA's Common Metadata Repository (CMR) API."""
    
    def __init__(self, username: str, password: str, data_dir: str = "data/raw"):
        """
        Initialize the NASA CMR client.
        
        Args:
            username: NASA Earthdata username
            password: NASA Earthdata password
            data_dir: Directory to store downloaded data
        """
        self.username = username
        self.password = password
        self.data_dir = Path(data_dir)
        self.session = requests.Session()
        self.session.auth = (username, password)
        
        # Create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Authenticate with NASA Earthdata
        self._authenticate()
    
    def _authenticate(self):
        """Authenticate with NASA Earthdata Login."""
        auth_url = "https://urs.earthdata.nasa.gov/oauth/authorize"
        response = self.session.get(auth_url)
        if response.status_code != 200:
            raise ConnectionError(f"Failed to authenticate with NASA Earthdata: {response.status_code}")
        logger.info("Successfully authenticated with NASA Earthdata")
    
    def search_granules(
        self, 
        short_name: str, 
        start_date: datetime, 
        end_date: datetime,
        bounding_box: Optional[Tuple[float, float, float, float]] = None,
        max_results: int = 100
    ) -> List[Dict]:
        """
        Search for granules in the CMR API.
        
        Args:
            short_name: Dataset short name (e.g., MOD06_L2)
            start_date: Start date for search
            end_date: End date for search
            bounding_box: Optional bounding box [west, south, east, north]
            max_results: Maximum number of results to return
            
        Returns:
            List of granule metadata dictionaries
        """
        cmr_url = "https://cmr.earthdata.nasa.gov/search/granules.json"
        
        params = {
            "short_name": short_name,
            "temporal": f"{start_date.strftime('%Y-%m-%dT%H:%M:%SZ')},{end_date.strftime('%Y-%m-%dT%H:%M:%SZ')}",
            "page_size": max_results,
        }
        
        if bounding_box:
            params["bounding_box"] = ",".join(map(str, bounding_box))
        
        response = self.session.get(cmr_url, params=params)
        if response.status_code != 200:
            raise ConnectionError(f"Failed to search granules: {response.status_code}")
        
        data = response.json()
        return data.get("feed", {}).get("entry", [])
    
    def download_granule(self, granule_url: str, output_path: Optional[Path] = None) -> Path:
        """
        Download a granule from NASA Earthdata.
        
        Args:
            granule_url: URL to the granule
            output_path: Optional path to save the granule
            
        Returns:
            Path to the downloaded file
        """
        if output_path is None:
            filename = granule_url.split("/")[-1]
            output_path = self.data_dir / filename
        
        # Create parent directories if they don't exist
        os.makedirs(output_path.parent, exist_ok=True)
        
        # Download the file
        with self.session.get(granule_url, stream=True) as response:
            response.raise_for_status()
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        
        logger.info(f"Downloaded {granule_url} to {output_path}")
        return output_path
    
    def download_modis_data(
        self, 
        product: str, 
        start_date: datetime, 
        end_date: datetime,
        bounding_box: Optional[Tuple[float, float, float, float]] = None,
        max_granules: int = 10
    ) -> List[Path]:
        """
        Download MODIS data.
        
        Args:
            product: MODIS product (e.g., MOD06_L2, MYD06_L2)
            start_date: Start date
            end_date: End date
            bounding_box: Optional bounding box [west, south, east, north]
            max_granules: Maximum number of granules to download
            
        Returns:
            List of paths to downloaded files
        """
        granules = self.search_granules(
            short_name=product,
            start_date=start_date,
            end_date=end_date,
            bounding_box=bounding_box,
            max_results=max_granules
        )
        
        downloaded_files = []
        for granule in granules:
            # Get download URL
            links = granule.get("links", [])
            download_url = None
            for link in links:
                if link.get("rel") == "download":
                    download_url = link.get("href")
                    break
            
            if download_url:
                # Create product-specific directory
                product_dir = self.data_dir / product
                os.makedirs(product_dir, exist_ok=True)
                
                # Download the file
                filename = download_url.split("/")[-1]
                output_path = product_dir / filename
                downloaded_file = self.download_granule(download_url, output_path)
                downloaded_files.append(downloaded_file)
        
        return downloaded_files
    
    def download_calipso_data(
        self, 
        product: str, 
        start_date: datetime, 
        end_date: datetime,
        bounding_box: Optional[Tuple[float, float, float, float]] = None,
        max_granules: int = 10
    ) -> List[Path]:
        """
        Download CALIPSO data.
        
        Args:
            product: CALIPSO product (e.g., CAL_LID_L2_Cloud_Profile)
            start_date: Start date
            end_date: End date
            bounding_box: Optional bounding box [west, south, east, north]
            max_granules: Maximum number of granules to download
            
        Returns:
            List of paths to downloaded files
        """
        # CALIPSO data is also available through CMR, so we can use the same method
        return self.download_modis_data(
            product=product,
            start_date=start_date,
            end_date=end_date,
            bounding_box=bounding_box,
            max_granules=max_granules
        )
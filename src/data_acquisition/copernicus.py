"""
Copernicus Open Access Hub client for downloading Sentinel-5P data.
"""

import os
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
import subprocess
import json

logger = logging.getLogger(__name__)

class CopernicusClient:
    """Client for Copernicus Open Access Hub API."""
    
    def __init__(self, username: str, password: str, data_dir: str = "data/raw"):
        """
        Initialize the Copernicus client.
        
        Args:
            username: Copernicus username
            password: Copernicus password
            data_dir: Directory to store downloaded data
        """
        self.username = username
        self.password = password
        self.data_dir = Path(data_dir)
        
        # Create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Configure Copernicus CLI
        self._configure()
    
    def _configure(self):
        """Configure Copernicus CLI with credentials."""
        try:
            # This is a placeholder for the actual Copernicus CLI configuration
            # In a real implementation, you would use the actual CLI tool
            subprocess.run(
                ["copernicus-ml", "configure", "--username", self.username, "--password", self.password],
                check=True,
                capture_output=True
            )
            logger.info("Successfully configured Copernicus CLI")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to configure Copernicus CLI: {e}")
            raise
    
    def search_products(
        self, 
        product_type: str, 
        start_date: datetime, 
        end_date: datetime,
        bounding_box: Optional[Tuple[float, float, float, float]] = None,
        max_results: int = 100
    ) -> List[Dict]:
        """
        Search for products in the Copernicus Open Access Hub.
        
        Args:
            product_type: Product type (e.g., S5P_L2_AER_AI)
            start_date: Start date for search
            end_date: End date for search
            bounding_box: Optional bounding box [west, south, east, north]
            max_results: Maximum number of results to return
            
        Returns:
            List of product metadata dictionaries
        """
        # Format dates for Copernicus API
        start_str = start_date.strftime("%Y-%m-%dT%H:%M:%S.000Z")
        end_str = end_date.strftime("%Y-%m-%dT%H:%M:%S.000Z")
        
        # Build search command
        cmd = [
            "copernicus-ml", "search",
            "--product", product_type,
            "--start", start_str,
            "--end", end_str,
            "--limit", str(max_results),
            "--format", "json"
        ]
        
        if bounding_box:
            # Format: west,south,east,north
            bbox_str = ",".join(map(str, bounding_box))
            cmd.extend(["--bbox", bbox_str])
        
        try:
            # Execute search command
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            products = json.loads(result.stdout)
            return products
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to search Copernicus products: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Copernicus search results: {e}")
            raise
    
    def download_product(self, product_id: str, output_dir: Optional[Path] = None) -> Path:
        """
        Download a product from Copernicus Open Access Hub.
        
        Args:
            product_id: ID of the product to download
            output_dir: Optional directory to save the product
            
        Returns:
            Path to the downloaded file
        """
        if output_dir is None:
            output_dir = self.data_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Build download command
        cmd = [
            "copernicus-ml", "download",
            "--id", product_id,
            "--output", str(output_dir)
        ]
        
        try:
            # Execute download command
            subprocess.run(cmd, check=True, capture_output=True)
            
            # The filename is typically the product ID with a .zip extension
            output_path = output_dir / f"{product_id}.zip"
            logger.info(f"Downloaded product {product_id} to {output_path}")
            return output_path
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to download Copernicus product {product_id}: {e}")
            raise
    
    def download_sentinel5p_data(
        self, 
        product_type: str, 
        start_date: datetime, 
        end_date: datetime,
        bounding_box: Optional[Tuple[float, float, float, float]] = None,
        max_products: int = 10
    ) -> List[Path]:
        """
        Download Sentinel-5P data.
        
        Args:
            product_type: Sentinel-5P product type (e.g., S5P_L2_AER_AI, S5P_L2_CLOUD)
            start_date: Start date
            end_date: End date
            bounding_box: Optional bounding box [west, south, east, north]
            max_products: Maximum number of products to download
            
        Returns:
            List of paths to downloaded files
        """
        # Search for products
        products = self.search_products(
            product_type=product_type,
            start_date=start_date,
            end_date=end_date,
            bounding_box=bounding_box,
            max_results=max_products
        )
        
        # Create product-specific directory
        product_dir = self.data_dir / product_type
        os.makedirs(product_dir, exist_ok=True)
        
        # Download products
        downloaded_files = []
        for product in products:
            product_id = product.get("id")
            if product_id:
                downloaded_file = self.download_product(product_id, product_dir)
                downloaded_files.append(downloaded_file)
        
        return downloaded_files
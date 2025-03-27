"""
Sentinel-5P-based cloud detection algorithms.
"""

import numpy as np
import xarray as xr
from typing import Dict, List, Optional, Union, Tuple
from .base import CloudDetector

class Sentinel5PCloudDetector(CloudDetector):
    """Cloud detection algorithm based on Sentinel-5P data."""
    
    def __init__(self, threshold: float = 0.5):
        """
        Initialize the Sentinel-5P cloud detector.
        
        Args:
            threshold: Cloud detection threshold
        """
        self.threshold = threshold
    
    def detect(self, reflectance: np.ndarray) -> np.ndarray:
        """
        Detect clouds in Sentinel-5P reflectance data.
        
        Args:
            reflectance: Normalized reflectance cube with shape (bands, height, width)
                         Assumes band order: [380nm, 440nm, 500nm, 675nm, 1020nm]
            
        Returns:
            Binary cloud mask with shape (height, width)
        """
        # Check input dimensions
        if reflectance.ndim != 3:
            raise ValueError(f"Expected 3D reflectance array, got shape {reflectance.shape}")
        
        # Get confidence values
        confidence = self.confidence(reflectance)
        
        # Apply threshold
        cloud_mask = confidence >= self.threshold
        
        return cloud_mask
    
    def confidence(self, reflectance: np.ndarray) -> np.ndarray:
        """
        Calculate cloud detection confidence for Sentinel-5P data.
        
        Args:
            reflectance: Normalized reflectance cube with shape (bands, height, width)
                         Assumes band order: [380nm, 440nm, 500nm, 675nm, 1020nm]
            
        Returns:
            Cloud confidence values (0-1) with shape (height, width)
        """
        # Check input dimensions
        if reflectance.ndim != 3:
            raise ValueError(f"Expected 3D reflectance array, got shape {reflectance.shape}")
        
        # Extract relevant bands
        # Assuming standard Sentinel-5P band order
        if reflectance.shape[0] < 5:
            raise ValueError(f"Expected at least 5 bands, got {reflectance.shape[0]}")
        
        # Extract bands
        band_380 = reflectance[0]  # 380nm (ultraviolet)
        band_440 = reflectance[1]  # 440nm (visible blue)
        band_500 = reflectance[2]  # 500nm (visible green)
        band_675 = reflectance[3]  # 675nm (visible red)
        band_1020 = reflectance[4]  # 1020nm (near-infrared)
        
        # Calculate spectral indices
        # NDVI (Normalized Difference Vegetation Index)
        ndvi = (band_1020 - band_675) / (band_1020 + band_675 + 1e-10)
        
        # UVAI (UV Aerosol Index) proxy
        # This is a simplified proxy, not the actual UVAI calculation
        uvai_proxy = (band_380 - band_440) / (band_380 + band_440 + 1e-10)
        
        # Cloud detection logic
        # High reflectance in visible bands
        high_reflectance = (band_440 > 0.4) & (band_500 > 0.4) & (band_675 > 0.4)
        
        # Low NDVI (clouds have low NDVI compared to vegetation)
        low_ndvi = ndvi < 0.2
        
        # Spectral test for clouds vs. aerosols
        # Clouds have different UV behavior than aerosols
        cloud_vs_aerosol = uvai_proxy > -0.1
        
        # Spectral test for clouds vs. water
        # Clouds have higher reflectance in NIR than water
        cloud_vs_water = band_1020 > 0.1
        
        # Combine tests
        cloud_confidence = np.zeros_like(band_675)
        
        # Basic cloud test
        cloud_confidence += 0.4 * high_reflectance
        
        # Vegetation test
        cloud_confidence += 0.2 * low_ndvi
        
        # Aerosol test
        cloud_confidence += 0.2 * cloud_vs_aerosol
        
        # Water test
        cloud_confidence += 0.2 * cloud_vs_water
        
        # Clip to [0, 1] range
        cloud_confidence = np.clip(cloud_confidence, 0, 1)
        
        return cloud_confidence
    
    def detect_from_dataset(self, dataset: xr.Dataset) -> xr.DataArray:
        """
        Detect clouds from an xarray Dataset containing Sentinel-5P data.
        
        Args:
            dataset: xarray Dataset containing Sentinel-5P reflectance data
            
        Returns:
            xarray DataArray containing binary cloud mask
        """
        # Extract reflectance bands
        # This assumes the dataset has variables named according to Sentinel-5P convention
        # Adjust as needed for your specific dataset
        band_names = ["reflectance_380", "reflectance_440", "reflectance_500", 
                      "reflectance_675", "reflectance_1020"]
        
        # Check if all required bands are present
        missing_bands = [band for band in band_names if band not in dataset]
        if missing_bands:
            # Try alternative band names
            alt_band_names = ["R_380", "R_440", "R_500", "R_675", "R_1020"]
            missing_alt_bands = [band for band in alt_band_names if band not in dataset]
            
            if len(missing_alt_bands) < len(missing_bands):
                band_names = alt_band_names
                missing_bands = missing_alt_bands
        
        if missing_bands:
            raise ValueError(f"Missing required bands: {missing_bands}")
        
        # Extract reflectance data
        reflectance = np.stack([dataset[band].values for band in band_names], axis=0)
        
        # Detect clouds
        cloud_mask = self.detect(reflectance)
        
        # Create xarray DataArray for the cloud mask
        mask_da = xr.DataArray(
            cloud_mask,
            dims=dataset[band_names[0]].dims,
            coords=dataset[band_names[0]].coords,
            name="cloud_mask",
            attrs={
                "long_name": "Binary cloud mask",
                "description": "1 = cloud, 0 = clear",
                "algorithm": "Sentinel5PCloudDetector",
                "threshold": self.threshold
            }
        )
        
        return mask_da
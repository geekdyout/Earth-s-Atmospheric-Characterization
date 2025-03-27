"""
MODIS-based cloud detection algorithms.
"""

import numpy as np
import xarray as xr
from typing import Dict, List, Optional, Union, Tuple
from .base import CloudDetector

class ModisCloudDetector(CloudDetector):
    """Cloud detection algorithm based on MODIS data."""
    
    def __init__(self, threshold: float = 0.5):
        """
        Initialize the MODIS cloud detector.
        
        Args:
            threshold: Cloud detection threshold
        """
        self.threshold = threshold
    
    def detect(self, reflectance: np.ndarray) -> np.ndarray:
        """
        Detect clouds in MODIS reflectance data.
        
        Args:
            reflectance: Normalized reflectance cube with shape (bands, height, width)
                         Assumes band order: [0.65µm, 0.86µm, 1.24µm, 1.63µm, 2.13µm]
            
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
        Calculate cloud detection confidence for MODIS data.
        
        Args:
            reflectance: Normalized reflectance cube with shape (bands, height, width)
                         Assumes band order: [0.65µm, 0.86µm, 1.24µm, 1.63µm, 2.13µm]
            
        Returns:
            Cloud confidence values (0-1) with shape (height, width)
        """
        # Check input dimensions
        if reflectance.ndim != 3:
            raise ValueError(f"Expected 3D reflectance array, got shape {reflectance.shape}")
        
        # Extract relevant bands
        # Assuming standard MODIS band order
        if reflectance.shape[0] < 5:
            raise ValueError(f"Expected at least 5 bands, got {reflectance.shape[0]}")
        
        # Extract bands
        band_065 = reflectance[0]  # 0.65µm (visible red)
        band_086 = reflectance[1]  # 0.86µm (near-infrared)
        band_124 = reflectance[2]  # 1.24µm (near-infrared)
        band_163 = reflectance[3]  # 1.63µm (shortwave infrared)
        band_213 = reflectance[4]  # 2.13µm (shortwave infrared)
        
        # Calculate spectral indices
        # NDVI (Normalized Difference Vegetation Index)
        ndvi = (band_086 - band_065) / (band_086 + band_065 + 1e-10)
        
        # NDSI (Normalized Difference Snow Index)
        ndsi = (band_065 - band_163) / (band_065 + band_163 + 1e-10)
        
        # NDWI (Normalized Difference Water Index)
        ndwi = (band_065 - band_124) / (band_065 + band_124 + 1e-10)
        
        # Cloud detection logic
        # High reflectance in visible and near-infrared bands
        high_reflectance = (band_065 > 0.4) & (band_086 > 0.4)
        
        # Low NDVI (clouds have low NDVI compared to vegetation)
        low_ndvi = ndvi < 0.2
        
        # Spectral test for clouds vs. snow
        # Clouds have higher reflectance in 1.63µm than snow
        cloud_vs_snow = band_163 > 0.1
        
        # Spectral test for clouds vs. water
        # Clouds have higher reflectance in 0.86µm than water
        cloud_vs_water = band_086 > 0.1
        
        # Combine tests
        cloud_confidence = np.zeros_like(band_065)
        
        # Basic cloud test
        cloud_confidence += 0.3 * high_reflectance
        
        # Vegetation test
        cloud_confidence += 0.2 * low_ndvi
        
        # Snow test
        cloud_confidence += 0.2 * cloud_vs_snow
        
        # Water test
        cloud_confidence += 0.2 * cloud_vs_water
        
        # Additional test using 2.13µm band
        # Clouds have higher reflectance in 2.13µm than most surfaces
        cloud_confidence += 0.1 * (band_213 > 0.1)
        
        # Clip to [0, 1] range
        cloud_confidence = np.clip(cloud_confidence, 0, 1)
        
        return cloud_confidence
    
    def detect_from_dataset(self, dataset: xr.Dataset) -> xr.DataArray:
        """
        Detect clouds from an xarray Dataset containing MODIS data.
        
        Args:
            dataset: xarray Dataset containing MODIS reflectance data
            
        Returns:
            xarray DataArray containing binary cloud mask
        """
        # Extract reflectance bands
        # This assumes the dataset has variables named according to MODIS convention
        # Adjust as needed for your specific dataset
        band_names = ["EV_250_Aggr1km_RefSB_1", "EV_250_Aggr1km_RefSB_2", 
                      "EV_500_Aggr1km_RefSB_3", "EV_500_Aggr1km_RefSB_6", 
                      "EV_500_Aggr1km_RefSB_7"]
        
        # Check if all required bands are present
        missing_bands = [band for band in band_names if band not in dataset]
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
                "algorithm": "ModisCloudDetector",
                "threshold": self.threshold
            }
        )
        
        return mask_da
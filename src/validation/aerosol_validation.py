"""
Aerosol analysis validation against reference datasets.
"""

import numpy as np
import xarray as xr
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
from .metrics import ValidationMetrics
from ..aerosol_analysis.optical_depth import OpticalDepthEstimator

class AerosolValidation:
    """Validation for aerosol analysis algorithms."""
    
    def __init__(self, metrics: Optional[ValidationMetrics] = None):
        """
        Initialize the aerosol validation.
        
        Args:
            metrics: Optional ValidationMetrics instance
        """
        self.metrics = metrics or ValidationMetrics()
    
    def validate_against_aeronet(
        self, 
        optical_depth_estimator: OpticalDepthEstimator,
        test_data: xr.Dataset,
        reference_data: xr.Dataset,
        aod_var: str = "aerosol_optical_depth",
        reference_var: str = "AERONET_L2_AOD",
        wavelength: float = 550.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """
        Validate aerosol optical depth against AERONET L2 reference data.
        
        Args:
            optical_depth_estimator: Optical depth estimator to validate
            test_data: Test dataset
            reference_data: Reference dataset
            aod_var: Variable name for AOD in test data
            reference_var: Variable name for reference AOD
            wavelength: Wavelength for AOD comparison
            metadata: Optional metadata for the validation report
            
        Returns:
            Dictionary of validation metrics
        """
        # Check if AOD already exists in test data
        if aod_var in test_data:
            # Use existing AOD
            aod = test_data[aod_var].values
        else:
            # Extract reflectance data
            # This assumes the dataset has variables named according to Sentinel-5P convention
            # Adjust as needed for your specific dataset
            band_names = ["reflectance_380", "reflectance_440", "reflectance_500", 
                          "reflectance_675", "reflectance_1020"]
            
            # Check if all required bands are present
            missing_bands = [band for band in band_names if band not in test_data]
            if missing_bands:
                # Try alternative band names
                alt_band_names = ["R_380", "R_440", "R_500", "R_675", "R_1020"]
                missing_alt_bands = [band for band in alt_band_names if band not in test_data]
                
                if len(missing_alt_bands) < len(missing_bands):
                    band_names = alt_band_names
                    missing_bands = missing_alt_bands
            
            if missing_bands:
                raise ValueError(f"Missing required bands: {missing_bands}")
            
            # Extract reflectance data
            reflectance = np.stack([test_data[band].values for band in band_names], axis=0)
            
            # Estimate optical depth
            aod = optical_depth_estimator.estimate(reflectance)
        
        # Extract reference AOD
        if reference_var not in reference_data:
            raise ValueError(f"Reference variable {reference_var} not found in reference data")
        
        reference_aod = reference_data[reference_var].values
        
        # Ensure AODs have the same shape
        if aod.shape != reference_aod.shape:
            raise ValueError(f"AOD shape {aod.shape} does not match reference AOD shape {reference_aod.shape}")
        
        # Calculate validation metrics
        metrics = self.metrics.calculate_regression_metrics(reference_aod.flatten(), aod.flatten())
        
        # Save metrics to files
        if metadata is None:
            metadata = {}
        
        metadata["algorithm"] = optical_depth_estimator.__class__.__name__
        metadata["reference_dataset"] = "AERONET_L2"
        metadata["wavelength"] = wavelength
        
        self.metrics.save_metrics_csv(metrics, "aerosol_rmse.csv", metadata)
        self.metrics.save_metrics_json(metrics, "aerosol_rmse.json", metadata)
        self.metrics.save_metrics_markdown(
            metrics, 
            "aerosol_validation.md", 
            title="Aerosol Optical Depth Validation", 
            metadata=metadata
        )
        
        return metrics
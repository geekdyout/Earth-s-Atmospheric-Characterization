"""
Cloud detection validation against reference datasets.
"""

import numpy as np
import xarray as xr
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
from .metrics import ValidationMetrics
from ..cloud_detection.base import CloudDetector

class CloudValidation:
    """Validation for cloud detection algorithms."""
    
    def __init__(self, metrics: Optional[ValidationMetrics] = None):
        """
        Initialize the cloud validation.
        
        Args:
            metrics: Optional ValidationMetrics instance
        """
        self.metrics = metrics or ValidationMetrics()
    
    def validate_against_modis_cosp(
        self, 
        cloud_detector: CloudDetector,
        test_data: xr.Dataset,
        reference_data: xr.Dataset,
        cloud_var: str = "cloud_mask",
        reference_var: str = "MODIS_COSP_L2_cloud_mask",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """
        Validate cloud detection against MODIS COSP L2 reference data.
        
        Args:
            cloud_detector: Cloud detector to validate
            test_data: Test dataset
            reference_data: Reference dataset
            cloud_var: Variable name for cloud mask in test data
            reference_var: Variable name for reference cloud mask
            metadata: Optional metadata for the validation report
            
        Returns:
            Dictionary of validation metrics
        """
        # Check if cloud mask already exists in test data
        if cloud_var in test_data:
            # Use existing cloud mask
            cloud_mask = test_data[cloud_var].values
        else:
            # Extract reflectance data
            # This assumes the dataset has variables named according to MODIS convention
            # Adjust as needed for your specific dataset
            band_names = ["EV_250_Aggr1km_RefSB_1", "EV_250_Aggr1km_RefSB_2", 
                          "EV_500_Aggr1km_RefSB_3", "EV_500_Aggr1km_RefSB_6", 
                          "EV_500_Aggr1km_RefSB_7"]
            
            # Check if all required bands are present
            missing_bands = [band for band in band_names if band not in test_data]
            if missing_bands:
                raise ValueError(f"Missing required bands: {missing_bands}")
            
            # Extract reflectance data
            reflectance = np.stack([test_data[band].values for band in band_names], axis=0)
            
            # Detect clouds
            cloud_mask = cloud_detector.detect(reflectance)
        
        # Extract reference cloud mask
        if reference_var not in reference_data:
            raise ValueError(f"Reference variable {reference_var} not found in reference data")
        
        reference_mask = reference_data[reference_var].values
        
        # Ensure masks have the same shape
        if cloud_mask.shape != reference_mask.shape:
            raise ValueError(f"Cloud mask shape {cloud_mask.shape} does not match reference mask shape {reference_mask.shape}")
        
        # Calculate validation metrics
        metrics = self.metrics.calculate_classification_metrics(reference_mask.flatten(), cloud_mask.flatten())
        
        # Save metrics to files
        if metadata is None:
            metadata = {}
        
        metadata["algorithm"] = cloud_detector.__class__.__name__
        metadata["reference_dataset"] = "MODIS_COSP_L2"
        
        self.metrics.save_metrics_csv(metrics, "cloud_f1_scores.csv", metadata)
        self.metrics.save_metrics_json(metrics, "cloud_validation.json", metadata)
        self.metrics.save_metrics_markdown(
            metrics, 
            "cloud_validation.md", 
            title="Cloud Detection Validation", 
            metadata=metadata
        )
        
        return metrics
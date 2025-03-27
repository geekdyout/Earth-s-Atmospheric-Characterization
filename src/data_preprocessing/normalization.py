"""
Normalization tools for standardizing data values.
"""

import os
import logging
from pathlib import Path
import numpy as np
import xarray as xr
from typing import Dict, List, Optional, Union, Tuple, Callable

logger = logging.getLogger(__name__)

class DataNormalizer:
    """Tools for normalizing data values."""
    
    def __init__(self, processed_dir: str = "data/processed"):
        """
        Initialize the data normalizer.
        
        Args:
            processed_dir: Directory containing processed data files
        """
        self.processed_dir = Path(processed_dir)
    
    def min_max_normalize(self, data: np.ndarray, min_val: Optional[float] = None, max_val: Optional[float] = None) -> np.ndarray:
        """
        Apply Min-Max normalization to data.
        
        Args:
            data: Input data array
            min_val: Optional minimum value for normalization (default: data minimum)
            max_val: Optional maximum value for normalization (default: data maximum)
            
        Returns:
            Normalized data array
        """
        if min_val is None:
            min_val = np.nanmin(data)
        if max_val is None:
            max_val = np.nanmax(data)
        
        # Avoid division by zero
        if min_val == max_val:
            return np.zeros_like(data)
        
        # Apply normalization
        normalized = (data - min_val) / (max_val - min_val)
        
        return normalized
    
    def z_score_normalize(self, data: np.ndarray, mean: Optional[float] = None, std: Optional[float] = None) -> np.ndarray:
        """
        Apply Z-score normalization to data.
        
        Args:
            data: Input data array
            mean: Optional mean value for normalization (default: data mean)
            std: Optional standard deviation for normalization (default: data std)
            
        Returns:
            Normalized data array
        """
        if mean is None:
            mean = np.nanmean(data)
        if std is None:
            std = np.nanstd(data)
        
        # Avoid division by zero
        if std == 0:
            return np.zeros_like(data)
        
        # Apply normalization
        normalized = (data - mean) / std
        
        return normalized
    
    def normalize_variable(
        self, 
        dataset: xr.Dataset, 
        variable: str, 
        method: str = "min-max",
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
        mean: Optional[float] = None,
        std: Optional[float] = None
    ) -> xr.Dataset:
        """
        Normalize a variable in a dataset.
        
        Args:
            dataset: Input xarray Dataset
            variable: Name of the variable to normalize
            method: Normalization method ("min-max" or "z-score")
            min_val: Optional minimum value for min-max normalization
            max_val: Optional maximum value for min-max normalization
            mean: Optional mean value for z-score normalization
            std: Optional standard deviation for z-score normalization
            
        Returns:
            Dataset with normalized variable
        """
        if variable not in dataset:
            raise ValueError(f"Variable {variable} not found in dataset")
        
        # Get the variable data
        data = dataset[variable].values
        
        # Apply normalization
        if method == "min-max":
            normalized = self.min_max_normalize(data, min_val, max_val)
        elif method == "z-score":
            normalized = self.z_score_normalize(data, mean, std)
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        # Create a new dataset with the normalized variable
        ds_normalized = dataset.copy()
        ds_normalized[variable].values = normalized
        
        # Add normalization metadata
        ds_normalized[variable].attrs["normalization_method"] = method
        if method == "min-max":
            ds_normalized[variable].attrs["normalization_min"] = min_val if min_val is not None else float(np.nanmin(data))
            ds_normalized[variable].attrs["normalization_max"] = max_val if max_val is not None else float(np.nanmax(data))
        elif method == "z-score":
            ds_normalized[variable].attrs["normalization_mean"] = mean if mean is not None else float(np.nanmean(data))
            ds_normalized[variable].attrs["normalization_std"] = std if std is not None else float(np.nanstd(data))
        
        return ds_normalized
    
    def normalize_dataset(
        self, 
        dataset: xr.Dataset, 
        variables: Optional[List[str]] = None,
        method: str = "min-max"
    ) -> xr.Dataset:
        """
        Normalize variables in a dataset.
        
        Args:
            dataset: Input xarray Dataset
            variables: List of variables to normalize (default: all variables)
            method: Normalization method ("min-max" or "z-score")
            
        Returns:
            Dataset with normalized variables
        """
        if variables is None:
            variables = list(dataset.data_vars)
        
        ds_normalized = dataset.copy()
        
        for variable in variables:
            if variable in dataset:
                ds_normalized = self.normalize_variable(ds_normalized, variable, method)
        
        # Add normalization metadata
        ds_normalized.attrs["normalized_by"] = "atmospheric-remote-sensing"
        ds_normalized.attrs["normalization_method"] = method
        ds_normalized.attrs["normalized_variables"] = ", ".join(variables)
        
        return ds_normalized
    
    def normalize_file(
        self, 
        file_path: Path, 
        variables: Optional[List[str]] = None,
        method: str = "min-max",
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Normalize variables in a NetCDF file.
        
        Args:
            file_path: Path to input NetCDF file
            variables: List of variables to normalize (default: all variables)
            method: Normalization method ("min-max" or "z-score")
            output_path: Optional path for output file
            
        Returns:
            Path to normalized file
        """
        if output_path is None:
            output_path = file_path.with_stem(f"{file_path.stem}_normalized")
        
        # Create parent directories if they don't exist
        os.makedirs(output_path.parent, exist_ok=True)
        
        # Open the NetCDF file
        ds = xr.open_dataset(file_path)
        
        # Normalize the dataset
        ds_normalized = self.normalize_dataset(ds, variables, method)
        
        # Save to NetCDF
        ds_normalized.to_netcdf(output_path)
        
        logger.info(f"Normalized {file_path} to {output_path} using {method} normalization")
        return output_path
    
    def normalize_directory(
        self, 
        directory: Path, 
        variables: Optional[List[str]] = None,
        method: str = "min-max"
    ) -> List[Path]:
        """
        Normalize variables in all NetCDF files in a directory.
        
        Args:
            directory: Path to directory containing NetCDF files
            variables: List of variables to normalize (default: all variables)
            method: Normalization method ("min-max" or "z-score")
            
        Returns:
            List of paths to normalized files
        """
        normalized_files = []
        
        for file_path in directory.glob("*.nc"):
            if file_path.is_file():
                try:
                    normalized_file = self.normalize_file(file_path, variables, method)
                    normalized_files.append(normalized_file)
                except Exception as e:
                    logger.error(f"Failed to normalize {file_path}: {e}")
        
        return normalized_files
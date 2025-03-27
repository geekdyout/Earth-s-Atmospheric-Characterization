"""
Standardize different data formats to a common NetCDF/HDF5 format.
"""

import os
import logging
from pathlib import Path
import numpy as np
import xarray as xr
import rasterio
from typing import Dict, List, Optional, Union, Tuple
import h5py
import netCDF4 as nc

logger = logging.getLogger(__name__)

class DataStandardizer:
    """Standardize different data formats to a common NetCDF/HDF5 format."""
    
    def __init__(self, raw_dir: str = "data/raw", processed_dir: str = "data/processed"):
        """
        Initialize the data standardizer.
        
        Args:
            raw_dir: Directory containing raw data files
            processed_dir: Directory to store processed data files
        """
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        
        # Create processed directory if it doesn't exist
        os.makedirs(self.processed_dir, exist_ok=True)
    
    def standardize_modis(self, file_path: Path) -> Path:
        """
        Standardize MODIS HDF file to NetCDF format.
        
        Args:
            file_path: Path to MODIS HDF file
            
        Returns:
            Path to standardized NetCDF file
        """
        # Create output directory if it doesn't exist
        product_name = file_path.parent.name
        output_dir = self.processed_dir / product_name
        os.makedirs(output_dir, exist_ok=True)
        
        # Output file path
        output_path = output_dir / f"{file_path.stem}.nc"
        
        # Open MODIS HDF file
        with h5py.File(file_path, "r") as hdf:
            # Create NetCDF file
            with nc.Dataset(output_path, "w", format="NETCDF4") as ncfile:
                # Copy global attributes
                for attr_name in hdf.attrs:
                    ncfile.setncattr(attr_name, hdf.attrs[attr_name])
                
                # Add standardization metadata
                ncfile.setncattr("standardized_by", "atmospheric-remote-sensing")
                ncfile.setncattr("original_file", str(file_path))
                
                # Process each dataset in the HDF file
                for group_name in hdf:
                    group = hdf[group_name]
                    
                    # Skip non-dataset items
                    if not isinstance(group, h5py.Dataset):
                        continue
                    
                    # Get dataset dimensions
                    dims = group.shape
                    
                    # Create dimensions in NetCDF file if they don't exist
                    for i, dim_size in enumerate(dims):
                        dim_name = f"{group_name}_dim{i}"
                        if dim_name not in ncfile.dimensions:
                            ncfile.createDimension(dim_name, dim_size)
                    
                    # Create variable
                    var = ncfile.createVariable(
                        group_name,
                        group.dtype,
                        tuple(f"{group_name}_dim{i}" for i in range(len(dims)))
                    )
                    
                    # Copy data
                    var[:] = group[:]
                    
                    # Copy variable attributes
                    for attr_name in group.attrs:
                        var.setncattr(attr_name, group.attrs[attr_name])
        
        logger.info(f"Standardized {file_path} to {output_path}")
        return output_path
    
    def standardize_sentinel5p(self, file_path: Path) -> Path:
        """
        Standardize Sentinel-5P NetCDF file to our standard format.
        
        Args:
            file_path: Path to Sentinel-5P NetCDF file
            
        Returns:
            Path to standardized NetCDF file
        """
        # Create output directory if it doesn't exist
        product_name = file_path.parent.name
        output_dir = self.processed_dir / product_name
        os.makedirs(output_dir, exist_ok=True)
        
        # Output file path
        output_path = output_dir / file_path.name
        
        # Open Sentinel-5P NetCDF file
        ds = xr.open_dataset(file_path)
        
        # Add standardization metadata
        ds.attrs["standardized_by"] = "atmospheric-remote-sensing"
        ds.attrs["original_file"] = str(file_path)
        
        # Save to NetCDF
        ds.to_netcdf(output_path)
        
        logger.info(f"Standardized {file_path} to {output_path}")
        return output_path
    
    def standardize_calipso(self, file_path: Path) -> Path:
        """
        Standardize CALIPSO HDF file to NetCDF format.
        
        Args:
            file_path: Path to CALIPSO HDF file
            
        Returns:
            Path to standardized NetCDF file
        """
        # Similar to MODIS standardization
        return self.standardize_modis(file_path)
    
    def standardize_neon(self, file_path: Path) -> Path:
        """
        Standardize NEON data to NetCDF format.
        
        Args:
            file_path: Path to NEON data file
            
        Returns:
            Path to standardized NetCDF file
        """
        # Create output directory if it doesn't exist
        product_name = file_path.parent.name
        output_dir = self.processed_dir / product_name
        os.makedirs(output_dir, exist_ok=True)
        
        # Output file path
        output_path = output_dir / f"{file_path.stem}.nc"
        
        # NEON data can be in various formats, here we assume it's a CSV
        if file_path.suffix.lower() == ".csv":
            # Read CSV data
            df = pd.read_csv(file_path)
            
            # Convert to xarray Dataset
            ds = df.to_xarray()
            
            # Add standardization metadata
            ds.attrs["standardized_by"] = "atmospheric-remote-sensing"
            ds.attrs["original_file"] = str(file_path)
            
            # Save to NetCDF
            ds.to_netcdf(output_path)
        else:
            # For other formats, we would need specific handling
            raise NotImplementedError(f"Standardization for {file_path.suffix} files not implemented")
        
        logger.info(f"Standardized {file_path} to {output_path}")
        return output_path
    
    def standardize_file(self, file_path: Path) -> Path:
        """
        Standardize a data file based on its type.
        
        Args:
            file_path: Path to data file
            
        Returns:
            Path to standardized NetCDF file
        """
        # Determine file type based on parent directory name
        product_name = file_path.parent.name
        
        if product_name.startswith("MOD") or product_name.startswith("MYD"):
            return self.standardize_modis(file_path)
        elif product_name.startswith("S5P"):
            return self.standardize_sentinel5p(file_path)
        elif product_name.startswith("CAL"):
            return self.standardize_calipso(file_path)
        elif product_name.startswith("DP"):
            return self.standardize_neon(file_path)
        else:
            raise ValueError(f"Unknown product type: {product_name}")
    
    def standardize_directory(self, directory: Path) -> List[Path]:
        """
        Standardize all files in a directory.
        
        Args:
            directory: Path to directory containing data files
            
        Returns:
            List of paths to standardized NetCDF files
        """
        standardized_files = []
        
        for file_path in directory.glob("*"):
            if file_path.is_file():
                try:
                    standardized_file = self.standardize_file(file_path)
                    standardized_files.append(standardized_file)
                except Exception as e:
                    logger.error(f"Failed to standardize {file_path}: {e}")
        
        return standardized_files
"""
Resampling tools for standardizing spatial resolution.
"""

import os
import logging
from pathlib import Path
import numpy as np
import xarray as xr
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from typing import Dict, List, Optional, Union, Tuple

logger = logging.getLogger(__name__)

class SpatialResampler:
    """Tools for resampling data to a common spatial resolution."""
    
    def __init__(self, processed_dir: str = "data/processed", target_resolution: float = 30.0):
        """
        Initialize the spatial resampler.
        
        Args:
            processed_dir: Directory containing processed data files
            target_resolution: Target resolution in meters
        """
        self.processed_dir = Path(processed_dir)
        self.target_resolution = target_resolution
    
    def resample_raster(self, file_path: Path, output_path: Optional[Path] = None) -> Path:
        """
        Resample a raster file to the target resolution using GDAL.
        
        Args:
            file_path: Path to input raster file
            output_path: Optional path for output file
            
        Returns:
            Path to resampled file
        """
        if output_path is None:
            output_path = file_path.with_stem(f"{file_path.stem}_resampled")
        
        # Create parent directories if they don't exist
        os.makedirs(output_path.parent, exist_ok=True)
        
        # Open the source dataset
        with rasterio.open(file_path) as src:
            # Calculate the transform for the target resolution
            dst_transform, dst_width, dst_height = calculate_default_transform(
                src.crs,
                src.crs,
                src.width,
                src.height,
                src.bounds.left,
                src.bounds.bottom,
                src.bounds.right,
                src.bounds.top,
                resolution=(self.target_resolution, self.target_resolution)
            )
            
            # Create the destination dataset
            dst_kwargs = src.meta.copy()
            dst_kwargs.update({
                "transform": dst_transform,
                "width": dst_width,
                "height": dst_height
            })
            
            with rasterio.open(output_path, "w", **dst_kwargs) as dst:
                # Reproject each band
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=dst_transform,
                        dst_crs=src.crs,
                        resampling=Resampling.bilinear
                    )
        
        logger.info(f"Resampled {file_path} to {output_path} at {self.target_resolution}m resolution")
        return output_path
    
    def resample_netcdf(self, file_path: Path, output_path: Optional[Path] = None) -> Path:
        """
        Resample a NetCDF file to the target resolution.
        
        Args:
            file_path: Path to input NetCDF file
            output_path: Optional path for output file
            
        Returns:
            Path to resampled file
        """
        if output_path is None:
            output_path = file_path.with_stem(f"{file_path.stem}_resampled")
        
        # Create parent directories if they don't exist
        os.makedirs(output_path.parent, exist_ok=True)
        
        # Open the NetCDF file
        ds = xr.open_dataset(file_path)
        
        # Identify spatial dimensions
        # This is a simplification - in practice, you would need to identify
        # the correct spatial dimensions based on metadata or naming conventions
        spatial_dims = [dim for dim in ds.dims if dim in ["lat", "latitude", "lon", "longitude", "x", "y"]]
        
        if len(spatial_dims) >= 2:
            # Get current resolution
            # This is a simplification - in practice, you would need to calculate
            # the actual resolution based on coordinate variables
            current_resolution = None
            for dim in spatial_dims:
                if dim in ds.coords:
                    coords = ds[dim].values
                    if len(coords) > 1:
                        res = abs(coords[1] - coords[0])
                        if current_resolution is None or res < current_resolution:
                            current_resolution = res
            
            if current_resolution is not None and current_resolution != self.target_resolution:
                # Calculate new dimension sizes
                scale_factor = current_resolution / self.target_resolution
                
                # Create new coordinates
                new_coords = {}
                for dim in spatial_dims:
                    if dim in ds.coords:
                        start = ds[dim].values[0]
                        end = ds[dim].values[-1]
                        num_points = int((end - start) / self.target_resolution) + 1
                        new_coords[dim] = np.linspace(start, end, num_points)
                
                # Resample the dataset
                ds_resampled = ds.interp(new_coords, method="linear")
                
                # Add resampling metadata
                ds_resampled.attrs["resampled_by"] = "atmospheric-remote-sensing"
                ds_resampled.attrs["original_resolution"] = str(current_resolution)
                ds_resampled.attrs["target_resolution"] = str(self.target_resolution)
                
                # Save to NetCDF
                ds_resampled.to_netcdf(output_path)
                
                logger.info(f"Resampled {file_path} to {output_path} at {self.target_resolution}m resolution")
                return output_path
        
        # If no resampling was performed, just copy the file
        ds.to_netcdf(output_path)
        logger.info(f"Copied {file_path} to {output_path} (no resampling needed)")
        return output_path
    
    def resample_file(self, file_path: Path, output_path: Optional[Path] = None) -> Path:
        """
        Resample a file to the target resolution based on its format.
        
        Args:
            file_path: Path to input file
            output_path: Optional path for output file
            
        Returns:
            Path to resampled file
        """
        # Determine file format based on extension
        if file_path.suffix.lower() in [".tif", ".tiff", ".img"]:
            return self.resample_raster(file_path, output_path)
        elif file_path.suffix.lower() in [".nc", ".h5", ".hdf", ".hdf5"]:
            return self.resample_netcdf(file_path, output_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    def resample_directory(self, directory: Path) -> List[Path]:
        """
        Resample all files in a directory.
        
        Args:
            directory: Path to directory containing data files
            
        Returns:
            List of paths to resampled files
        """
        resampled_files = []
        
        for file_path in directory.glob("*"):
            if file_path.is_file():
                try:
                    resampled_file = self.resample_file(file_path)
                    resampled_files.append(resampled_file)
                except Exception as e:
                    logger.error(f"Failed to resample {file_path}: {e}")
        
        return resampled_files
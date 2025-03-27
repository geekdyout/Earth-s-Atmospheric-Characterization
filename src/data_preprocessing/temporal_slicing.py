"""
Temporal slicing tools for creating time-based subsets of data.
"""

import os
import logging
from pathlib import Path
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple

logger = logging.getLogger(__name__)

class TemporalSlicer:
    """Tools for creating time-based subsets of data."""
    
    def __init__(self, processed_dir: str = "data/processed", temporal_dir: str = "data/temporal"):
        """
        Initialize the temporal slicer.
        
        Args:
            processed_dir: Directory containing processed data files
            temporal_dir: Directory to store temporal slices
        """
        self.processed_dir = Path(processed_dir)
        self.temporal_dir = Path(temporal_dir)
        
        # Create temporal directory if it doesn't exist
        os.makedirs(self.temporal_dir, exist_ok=True)
    
    def slice_by_time_range(
        self, 
        dataset: xr.Dataset, 
        start_time: datetime,
        end_time: datetime,
        time_dim: str = "time"
    ) -> xr.Dataset:
        """
        Slice a dataset by time range.
        
        Args:
            dataset: Input xarray Dataset
            start_time: Start time for slice
            end_time: End time for slice
            time_dim: Name of the time dimension
            
        Returns:
            Sliced dataset
        """
        if time_dim not in dataset.dims:
            raise ValueError(f"Time dimension '{time_dim}' not found in dataset")
        
        # Convert time dimension to datetime if it's not already
        if not np.issubdtype(dataset[time_dim].dtype, np.datetime64):
            try:
                dataset[time_dim] = xr.decode_cf(dataset).time
            except Exception as e:
                logger.warning(f"Failed to decode time dimension: {e}")
                raise ValueError(f"Time dimension '{time_dim}' is not in a recognized datetime format")
        
        # Slice the dataset by time range
        sliced = dataset.sel({time_dim: slice(start_time, end_time)})
        
        # Add slicing metadata
        sliced.attrs["time_slice_start"] = start_time.isoformat()
        sliced.attrs["time_slice_end"] = end_time.isoformat()
        
        return sliced
    
    def slice_by_time_step(
        self, 
        dataset: xr.Dataset, 
        time_step: timedelta,
        time_dim: str = "time"
    ) -> Dict[str, xr.Dataset]:
        """
        Slice a dataset into multiple time steps.
        
        Args:
            dataset: Input xarray Dataset
            time_step: Time step for slicing
            time_dim: Name of the time dimension
            
        Returns:
            Dictionary of sliced datasets, keyed by start time
        """
        if time_dim not in dataset.dims:
            raise ValueError(f"Time dimension '{time_dim}' not found in dataset")
        
        # Convert time dimension to datetime if it's not already
        if not np.issubdtype(dataset[time_dim].dtype, np.datetime64):
            try:
                dataset[time_dim] = xr.decode_cf(dataset).time
            except Exception as e:
                logger.warning(f"Failed to decode time dimension: {e}")
                raise ValueError(f"Time dimension '{time_dim}' is not in a recognized datetime format")
        
        # Get time values
        time_values = dataset[time_dim].values
        
        # Convert to Python datetime objects
        if isinstance(time_values[0], np.datetime64):
            time_values = [pd.Timestamp(t).to_pydatetime() for t in time_values]
        
        # Get start and end times
        start_time = min(time_values)
        end_time = max(time_values)
        
        # Create time steps
        current_time = start_time
        time_steps = []
        while current_time < end_time:
            next_time = current_time + time_step
            time_steps.append((current_time, min(next_time, end_time)))
            current_time = next_time
        
        # Slice the dataset for each time step
        sliced_datasets = {}
        for step_start, step_end in time_steps:
            sliced = self.slice_by_time_range(dataset, step_start, step_end, time_dim)
            sliced_datasets[step_start.isoformat()] = sliced
        
        return sliced_datasets
    
    def slice_file_by_time_range(
        self, 
        file_path: Path, 
        start_time: datetime,
        end_time: datetime,
        time_dim: str = "time",
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Slice a NetCDF file by time range.
        
        Args:
            file_path: Path to input NetCDF file
            start_time: Start time for slice
            end_time: End time for slice
            time_dim: Name of the time dimension
            output_path: Optional path for output file
            
        Returns:
            Path to sliced file
        """
        if output_path is None:
            # Create a directory for the time range
            time_range_dir = self.temporal_dir / f"{start_time.strftime('%Y%m%d')}-{end_time.strftime('%Y%m%d')}"
            os.makedirs(time_range_dir, exist_ok=True)
            
            # Output file path
            output_path = time_range_dir / file_path.name
        
        # Create parent directories if they don't exist
        os.makedirs(output_path.parent, exist_ok=True)
        
        # Open the NetCDF file
        ds = xr.open_dataset(file_path)
        
        # Slice the dataset
        try:
            sliced = self.slice_by_time_range(ds, start_time, end_time, time_dim)
            
            # Save to NetCDF
            sliced.to_netcdf(output_path)
            
            logger.info(f"Sliced {file_path} to {output_path} for time range {start_time} to {end_time}")
            return output_path
        except ValueError as e:
            logger.warning(f"Failed to slice {file_path}: {e}")
            # If slicing fails, just copy the file
            ds.to_netcdf(output_path)
            logger.info(f"Copied {file_path} to {output_path} (no slicing performed)")
            return output_path
    
    def slice_file_by_time_step(
        self, 
        file_path: Path, 
        time_step: timedelta,
        time_dim: str = "time",
        output_dir: Optional[Path] = None
    ) -> List[Path]:
        """
        Slice a NetCDF file into multiple time steps.
        
        Args:
            file_path: Path to input NetCDF file
            time_step: Time step for slicing
            time_dim: Name of the time dimension
            output_dir: Optional directory for output files
            
        Returns:
            List of paths to sliced files
        """
        if output_dir is None:
            # Create a directory for the time step
            time_step_str = f"{time_step.days}d"
            if time_step.seconds > 0:
                time_step_str += f"{time_step.seconds // 3600}h"
            output_dir = self.temporal_dir / f"step_{time_step_str}"
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Open the NetCDF file
        ds = xr.open_dataset(file_path)
        
        # Slice the dataset
        try:
            sliced_datasets = self.slice_by_time_step(ds, time_step, time_dim)
            
            # Save each slice to a separate file
            output_paths = []
            for start_time, sliced in sliced_datasets.items():
                # Format start time for filename
                start_time_dt = datetime.fromisoformat(start_time)
                start_time_str = start_time_dt.strftime("%Y%m%d%H%M%S")
                
                # Output file path
                output_path = output_dir / f"{file_path.stem}_{start_time_str}.nc"
                
                # Save to NetCDF
                sliced.to_netcdf(output_path)
                
                logger.info(f"Saved time slice for {start_time} to {output_path}")
                output_paths.append(output_path)
            
            return output_paths
        except ValueError as e:
            logger.warning(f"Failed to slice {file_path}: {e}")
            # If slicing fails, just copy the file
            output_path = output_dir / file_path.name
            ds.to_netcdf(output_path)
            logger.info(f"Copied {file_path} to {output_path} (no slicing performed)")
            return [output_path]
    
    def slice_directory_by_time_range(
        self, 
        directory: Path, 
        start_time: datetime,
        end_time: datetime,
        time_dim: str = "time"
    ) -> List[Path]:
        """
        Slice all NetCDF files in a directory by time range.
        
        Args:
            directory: Path to directory containing NetCDF files
            start_time: Start time for slice
            end_time: End time for slice
            time_dim: Name of the time dimension
            
        Returns:
            List of paths to sliced files
        """
        sliced_files = []
        
        for file_path in directory.glob("*.nc"):
            if file_path.is_file():
                try:
                    sliced_file = self.slice_file_by_time_range(file_path, start_time, end_time, time_dim)
                    sliced_files.append(sliced_file)
                except Exception as e:
                    logger.error(f"Failed to slice {file_path}: {e}")
        
        return sliced_files
    
    def slice_directory_by_time_step(
        self, 
        directory: Path, 
        time_step: timedelta,
        time_dim: str = "time"
    ) -> List[Path]:
        """
        Slice all NetCDF files in a directory into multiple time steps.
        
        Args:
            directory: Path to directory containing NetCDF files
            time_step: Time step for slicing
            time_dim: Name of the time dimension
            
        Returns:
            List of paths to sliced files
        """
        sliced_files = []
        
        for file_path in directory.glob("*.nc"):
            if file_path.is_file():
                try:
                    sliced_files_for_file = self.slice_file_by_time_step(file_path, time_step, time_dim)
                    sliced_files.extend(sliced_files_for_file)
                except Exception as e:
                    logger.error(f"Failed to slice {file_path}: {e}")
        
        return sliced_files
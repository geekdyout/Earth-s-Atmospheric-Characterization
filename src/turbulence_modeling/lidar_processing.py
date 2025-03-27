"""
CALIPSO lidar data processing for turbulence modeling.
"""

import numpy as np
import xarray as xr
from typing import Dict, List, Optional, Union, Tuple
from abc import ABC, abstractmethod

class LidarProcessor(ABC):
    """Base class for lidar data processors."""
    
    @abstractmethod
    def process(self, backscatter: np.ndarray, altitude: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Process lidar backscatter data.
        
        Args:
            backscatter: Lidar backscatter profiles with shape (profiles, altitude_bins)
            altitude: Altitude values for each bin with shape (altitude_bins,)
            
        Returns:
            Dictionary of processed data products
        """
        pass

class CalipsoLidarProcessor(LidarProcessor):
    """Processor for CALIPSO lidar data."""
    
    def __init__(self, snr_threshold: float = 1.5):
        """
        Initialize the CALIPSO lidar processor.
        
        Args:
            snr_threshold: Signal-to-noise ratio threshold for valid data
        """
        self.snr_threshold = snr_threshold
    
    def process(self, backscatter: np.ndarray, altitude: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Process CALIPSO lidar backscatter data.
        
        Args:
            backscatter: Lidar backscatter profiles with shape (profiles, altitude_bins)
            altitude: Altitude values for each bin with shape (altitude_bins,)
            
        Returns:
            Dictionary of processed data products
        """
        # Check input dimensions
        if backscatter.ndim != 2:
            raise ValueError(f"Expected 2D backscatter array, got shape {backscatter.shape}")
        if altitude.ndim != 1:
            raise ValueError(f"Expected 1D altitude array, got shape {altitude.shape}")
        if backscatter.shape[1] != altitude.shape[0]:
            raise ValueError(f"Backscatter altitude dimension {backscatter.shape[1]} does not match altitude array length {altitude.shape[0]}")
        
        # Calculate signal-to-noise ratio
        # This is a simplified approach - in practice, more sophisticated methods are used
        # Estimate noise as the standard deviation of high-altitude backscatter
        # where signal is expected to be minimal
        high_altitude_idx = altitude > 30000  # Above 30 km
        if np.any(high_altitude_idx):
            noise_estimate = np.std(backscatter[:, high_altitude_idx], axis=1, keepdims=True)
        else:
            # If no high-altitude data, use the highest 10% of altitude bins
            top_10_percent = int(0.1 * altitude.shape[0])
            noise_estimate = np.std(backscatter[:, -top_10_percent:], axis=1, keepdims=True)
        
        # Avoid division by zero
        noise_estimate = np.maximum(noise_estimate, 1e-10)
        
        # Calculate SNR
        snr = backscatter / noise_estimate
        
        # Create valid data mask based on SNR threshold
        valid_data = snr > self.snr_threshold
        
        # Calculate cloud boundaries
        cloud_boundaries = self._detect_cloud_boundaries(backscatter, altitude, valid_data)
        
        # Calculate cloud optical depth
        cloud_optical_depth = self._estimate_optical_depth(backscatter, altitude, valid_data)
        
        # Calculate turbulence parameters
        turbulence_params = self._estimate_turbulence(backscatter, altitude, valid_data)
        
        # Combine results
        results = {
            "snr": snr,
            "valid_data": valid_data,
            "cloud_top_altitude": cloud_boundaries["top"],
            "cloud_base_altitude": cloud_boundaries["base"],
            "cloud_optical_depth": cloud_optical_depth,
            "turbulence_intensity": turbulence_params["intensity"],
            "turbulence_scale": turbulence_params["scale"]
        }
        
        return results
    
    def _detect_cloud_boundaries(
        self, 
        backscatter: np.ndarray, 
        altitude: np.ndarray, 
        valid_data: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Detect cloud top and base altitudes.
        
        Args:
            backscatter: Lidar backscatter profiles
            altitude: Altitude values for each bin
            valid_data: Mask of valid data points
            
        Returns:
            Dictionary with cloud top and base altitudes
        """
        n_profiles = backscatter.shape[0]
        
        # Initialize cloud boundaries
        cloud_top = np.full(n_profiles, np.nan)
        cloud_base = np.full(n_profiles, np.nan)
        
        # Cloud detection threshold
        # This is a simplified approach - in practice, more sophisticated methods are used
        cloud_threshold = np.median(backscatter) + 2 * np.std(backscatter)
        
        # Process each profile
        for i in range(n_profiles):
            # Find potential cloud regions
            cloud_mask = (backscatter[i] > cloud_threshold) & valid_data[i]
            
            if np.any(cloud_mask):
                # Find continuous cloud regions
                # This is a simplified approach - in practice, more sophisticated methods are used
                cloud_regions = self._find_continuous_regions(cloud_mask)
                
                if cloud_regions:
                    # Get the most significant cloud region (highest backscatter)
                    max_backscatter = 0
                    max_region = None
                    
                    for start, end in cloud_regions:
                        region_max = np.max(backscatter[i, start:end])
                        if region_max > max_backscatter:
                            max_backscatter = region_max
                            max_region = (start, end)
                    
                    if max_region:
                        # Set cloud top and base
                        cloud_top[i] = altitude[max_region[1] - 1]  # Top is at higher altitude (end of region)
                        cloud_base[i] = altitude[max_region[0]]     # Base is at lower altitude (start of region)
        
        return {"top": cloud_top, "base": cloud_base}
    
    def _find_continuous_regions(self, mask: np.ndarray) -> List[Tuple[int, int]]:
        """
        Find continuous regions in a boolean mask.
        
        Args:
            mask: Boolean mask
            
        Returns:
            List of (start, end) indices for continuous regions
        """
        regions = []
        in_region = False
        start = 0
        
        for i in range(len(mask)):
            if mask[i] and not in_region:
                # Start of a new region
                in_region = True
                start = i
            elif not mask[i] and in_region:
                # End of a region
                in_region = False
                regions.append((start, i))
        
        # Handle case where the last region extends to the end
        if in_region:
            regions.append((start, len(mask)))
        
        return regions
    
    def _estimate_optical_depth(
        self, 
        backscatter: np.ndarray, 
        altitude: np.ndarray, 
        valid_data: np.ndarray
    ) -> np.ndarray:
        """
        Estimate cloud optical depth.
        
        Args:
            backscatter: Lidar backscatter profiles
            altitude: Altitude values for each bin
            valid_data: Mask of valid data points
            
        Returns:
            Cloud optical depth for each profile
        """
        n_profiles = backscatter.shape[0]
        
        # Initialize optical depth
        optical_depth = np.full(n_profiles, np.nan)
        
        # Lidar ratio (extinction-to-backscatter ratio)
        # This is a simplified approach - in practice, this varies by cloud type
        lidar_ratio = 20.0  # sr
        
        # Process each profile
        for i in range(n_profiles):
            # Get cloud boundaries
            cloud_boundaries = self._detect_cloud_boundaries(
                backscatter[i:i+1], altitude, valid_data[i:i+1])
            
            cloud_top = cloud_boundaries["top"][0]
            cloud_base = cloud_boundaries["base"][0]
            
            if not np.isnan(cloud_top) and not np.isnan(cloud_base):
                # Find altitude indices for cloud region
                cloud_base_idx = np.argmin(np.abs(altitude - cloud_base))
                cloud_top_idx = np.argmin(np.abs(altitude - cloud_top))
                
                # Ensure correct order
                if cloud_base_idx > cloud_top_idx:
                    cloud_base_idx, cloud_top_idx = cloud_top_idx, cloud_base_idx
                
                # Extract cloud region
                cloud_backscatter = backscatter[i, cloud_base_idx:cloud_top_idx+1]
                cloud_altitude = altitude[cloud_base_idx:cloud_top_idx+1]
                
                # Calculate altitude differences
                altitude_diff = np.diff(cloud_altitude, prepend=cloud_altitude[0])
                
                # Calculate extinction
                extinction = lidar_ratio * cloud_backscatter
                
                # Calculate optical depth (integrate extinction over altitude)
                od = np.sum(extinction * altitude_diff)
                
                optical_depth[i] = od
        
        return optical_depth
    
    def _estimate_turbulence(
        self, 
        backscatter: np.ndarray, 
        altitude: np.ndarray, 
        valid_data: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Estimate turbulence parameters.
        
        Args:
            backscatter: Lidar backscatter profiles
            altitude: Altitude values for each bin
            valid_data: Mask of valid data points
            
        Returns:
            Dictionary with turbulence intensity and scale
        """
        n_profiles = backscatter.shape[0]
        
        # Initialize turbulence parameters
        turbulence_intensity = np.full(n_profiles, np.nan)
        turbulence_scale = np.full(n_profiles, np.nan)
        
        # Process each profile
        for i in range(n_profiles):
            # Get cloud boundaries
            cloud_boundaries = self._detect_cloud_boundaries(
                backscatter[i:i+1], altitude, valid_data[i:i+1])
            
            cloud_top = cloud_boundaries["top"][0]
            cloud_base = cloud_boundaries["base"][0]
            
            if not np.isnan(cloud_top) and not np.isnan(cloud_base):
                # Find altitude indices for cloud region
                cloud_base_idx = np.argmin(np.abs(altitude - cloud_base))
                cloud_top_idx = np.argmin(np.abs(altitude - cloud_top))
                
                # Ensure correct order
                if cloud_base_idx > cloud_top_idx:
                    cloud_base_idx, cloud_top_idx = cloud_top_idx, cloud_base_idx
                
                # Extract cloud region
                cloud_backscatter = backscatter[i, cloud_base_idx:cloud_top_idx+1]
                
                if len(cloud_backscatter) > 1:
                    # Calculate backscatter gradient
                    backscatter_gradient = np.gradient(cloud_backscatter)
                    
                    # Estimate turbulence intensity as the standard deviation of the gradient
                    turbulence_intensity[i] = np.std(backscatter_gradient)
                    
                    # Estimate turbulence scale using autocorrelation
                    # This is a simplified approach - in practice, more sophisticated methods are used
                    if len(cloud_backscatter) > 10:
                        # Calculate autocorrelation
                        autocorr = np.correlate(cloud_backscatter, cloud_backscatter, mode="full")
                        autocorr = autocorr[len(cloud_backscatter)-1:]
                        autocorr = autocorr / autocorr[0]
                        
                        # Find the lag where autocorrelation drops below 1/e
                        below_threshold = np.where(autocorr < 1/np.e)[0]
                        if len(below_threshold) > 0:
                            lag = below_threshold[0]
                            # Convert lag to physical scale
                            altitude_resolution = np.mean(np.diff(altitude))
                            turbulence_scale[i] = lag * altitude_resolution
        
        return {"intensity": turbulence_intensity, "scale": turbulence_scale}
    
    def process_dataset(self, dataset: xr.Dataset) -> xr.Dataset:
        """
        Process an xarray Dataset containing CALIPSO lidar data.
        
        Args:
            dataset: xarray Dataset containing CALIPSO lidar data
            
        Returns:
            xarray Dataset with processed data products
        """
        # Extract backscatter and altitude data
        # This assumes the dataset has variables named according to CALIPSO convention
        # Adjust as needed for your specific dataset
        backscatter_var = None
        altitude_var = None
        
        # Try common variable names for backscatter
        backscatter_names = ["Total_Backscatter_532", "Attenuated_Backscatter_532", "backscatter"]
        for name in backscatter_names:
            if name in dataset:
                backscatter_var = name
                break
        
        # Try common variable names for altitude
        altitude_names = ["altitude", "height", "Altitude"]
        for name in altitude_names:
            if name in dataset:
                altitude_var = name
                break
        
        if backscatter_var is None or altitude_var is None:
            raise ValueError("Could not find backscatter or altitude variables in dataset")
        
        # Extract data
        backscatter = dataset[backscatter_var].values
        altitude = dataset[altitude_var].values
        
        # Handle different dimension orders
        if backscatter.ndim > 2:
            # Reshape to (profiles, altitude_bins)
            orig_shape = backscatter.shape
            backscatter = backscatter.reshape(-1, orig_shape[-1])
        
        # Process data
        results = self.process(backscatter, altitude)
        
        # Create output dataset
        ds_out = xr.Dataset()
        
        # Add variables to output dataset
        for name, data in results.items():
            # Reshape data back to original dimensions if needed
            if backscatter.ndim != data.ndim and data.ndim == 2:
                data = data.reshape(orig_shape)
            
            # Create variable
            ds_out[name] = xr.Variable(
                dims=dataset[backscatter_var].dims,
                data=data,
                attrs={
                    "long_name": name.replace("_", " ").title(),
                    "processor": "CalipsoLidarProcessor"
                }
            )
        
        # Add specific attributes
        if "cloud_optical_depth" in ds_out:
            ds_out["cloud_optical_depth"].attrs["units"] = "dimensionless"
        
        if "turbulence_intensity" in ds_out:
            ds_out["turbulence_intensity"].attrs["units"] = "m^-1 sr^-1"
        
        if "turbulence_scale" in ds_out:
            ds_out["turbulence_scale"].attrs["units"] = "m"
        
        if "cloud_top_altitude" in ds_out:
            ds_out["cloud_top_altitude"].attrs["units"] = "m"
        
        if "cloud_base_altitude" in ds_out:
            ds_out["cloud_base_altitude"].attrs["units"] = "m"
        
        # Add global attributes
        ds_out.attrs["processor"] = "CalipsoLidarProcessor"
        ds_out.attrs["snr_threshold"] = float(self.snr_threshold)
        ds_out.attrs["source_dataset"] = dataset.attrs.get("title", "CALIPSO data")
        
        return ds_out
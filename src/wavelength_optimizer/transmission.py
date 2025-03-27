"""
Transmission simulation tools for wavelength optimization.
"""

import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from abc import ABC, abstractmethod

class TransmissionSimulator(ABC):
    """Base class for atmospheric transmission simulators."""
    
    @abstractmethod
    def simulate(
        self, 
        wavelengths: np.ndarray, 
        altitude: float, 
        zenith_angle: float,
        atmosphere_profile: str
    ) -> np.ndarray:
        """
        Simulate atmospheric transmission.
        
        Args:
            wavelengths: Wavelengths in nanometers
            altitude: Observer altitude in meters
            zenith_angle: Zenith angle in degrees
            atmosphere_profile: Atmosphere profile name
            
        Returns:
            Transmission values for each wavelength
        """
        pass

class SimpleTransmissionSimulator(TransmissionSimulator):
    """Simple atmospheric transmission simulator."""
    
    def __init__(self):
        """Initialize the simple transmission simulator."""
        # Define atmospheric absorption bands
        # (center_wavelength, width, strength)
        self.absorption_bands = [
            (940.0, 80.0, 0.8),    # Water vapor
            (1140.0, 100.0, 0.9),   # Water vapor
            (1400.0, 150.0, 0.95),  # Water vapor
            (1900.0, 200.0, 0.98),  # Water vapor
            (2700.0, 250.0, 0.99),  # Water vapor
            (4300.0, 300.0, 0.99),  # Carbon dioxide
            (6000.0, 350.0, 0.99),  # Water vapor
            (760.0, 10.0, 0.7),     # Oxygen
            (690.0, 5.0, 0.3),      # Oxygen
            (630.0, 5.0, 0.2),      # Oxygen
            (9600.0, 400.0, 0.99),  # Ozone
            (3000.0, 200.0, 0.9),   # Carbon dioxide
        ]
        
        # Define atmospheric profiles
        # Each profile modifies the strength of absorption bands
        self.profiles = {
            "tropical": 1.2,       # Higher water vapor
            "midlatitude_summer": 1.0,  # Reference profile
            "midlatitude_winter": 0.7,  # Lower water vapor
            "subarctic_summer": 0.8,    # Lower water vapor
            "subarctic_winter": 0.5,    # Much lower water vapor
            "us_standard": 0.9,         # Standard profile
        }
    
    def simulate(
        self, 
        wavelengths: np.ndarray, 
        altitude: float, 
        zenith_angle: float,
        atmosphere_profile: str = "midlatitude_summer"
    ) -> np.ndarray:
        """
        Simulate atmospheric transmission.
        
        Args:
            wavelengths: Wavelengths in nanometers
            altitude: Observer altitude in meters
            zenith_angle: Zenith angle in degrees
            atmosphere_profile: Atmosphere profile name
            
        Returns:
            Transmission values for each wavelength
        """
        # Check inputs
        if not isinstance(wavelengths, np.ndarray):
            wavelengths = np.array(wavelengths)
        
        if atmosphere_profile not in self.profiles:
            raise ValueError(f"Unknown atmosphere profile: {atmosphere_profile}")
        
        # Calculate air mass
        # Air mass is the path length through the atmosphere relative to zenith
        # This is a simplified model valid for zenith angles < 80 degrees
        if zenith_angle >= 90:
            air_mass = 40.0  # Very high air mass for horizontal path
        elif zenith_angle >= 80:
            air_mass = 6.0  # High air mass for near-horizontal path
        else:
            air_mass = 1.0 / np.cos(np.radians(zenith_angle))
        
        # Adjust air mass for altitude
        # This is a simplified model
        # Scale factor decreases with altitude
        scale_height = 8000.0  # Scale height of atmosphere in meters
        altitude_factor = np.exp(-altitude / scale_height)
        air_mass *= altitude_factor
        
        # Get profile factor
        profile_factor = self.profiles[atmosphere_profile]
        
        # Initialize transmission with Rayleigh scattering
        # Rayleigh scattering ~ λ^-4
        transmission = np.ones_like(wavelengths, dtype=float)
        
        # Apply Rayleigh scattering
        rayleigh = np.exp(-0.008 * (550.0 / wavelengths)**4 * air_mass)
        transmission *= rayleigh
        
        # Apply aerosol extinction (simplified model)
        # Aerosol extinction ~ λ^-1
        angstrom_exponent = 1.3  # Typical value
        aerosol_optical_depth_550nm = 0.1  # Typical value for clear conditions
        aerosol = np.exp(-aerosol_optical_depth_550nm * (550.0 / wavelengths)**angstrom_exponent * air_mass)
        transmission *= aerosol
        
        # Apply absorption bands
        for center, width, strength in self.absorption_bands:
            # Calculate Gaussian absorption band
            absorption = strength * profile_factor * np.exp(-((wavelengths - center) / width)**2)
            
            # Apply to transmission
            transmission *= (1.0 - absorption * (1.0 - np.exp(-air_mass)))
        
        # Ensure transmission is in [0, 1]
        transmission = np.clip(transmission, 0.0, 1.0)
        
        return transmission
    
    def find_optimal_wavelengths(
        self, 
        wavelength_range: Tuple[float, float],
        num_wavelengths: int,
        altitude: float,
        zenith_angle: float,
        atmosphere_profile: str = "midlatitude_summer",
        min_spacing: Optional[float] = None
    ) -> np.ndarray:
        """
        Find optimal wavelengths for atmospheric transmission.
        
        Args:
            wavelength_range: (min_wavelength, max_wavelength) in nanometers
            num_wavelengths: Number of wavelengths to select
            altitude: Observer altitude in meters
            zenith_angle: Zenith angle in degrees
            atmosphere_profile: Atmosphere profile name
            min_spacing: Minimum spacing between selected wavelengths
            
        Returns:
            Array of optimal wavelengths
        """
        min_wavelength, max_wavelength = wavelength_range
        
        # Generate fine wavelength grid for simulation
        wavelength_step = 1.0  # 1 nm step
        wavelengths = np.arange(min_wavelength, max_wavelength + wavelength_step, wavelength_step)
        
        # Simulate transmission
        transmission = self.simulate(wavelengths, altitude, zenith_angle, atmosphere_profile)
        
        # Find local maxima in transmission
        # A point is a local maximum if it's greater than its neighbors
        is_local_max = np.zeros_like(transmission, dtype=bool)
        is_local_max[1:-1] = (transmission[1:-1] > transmission[:-2]) & (transmission[1:-1] > transmission[2:])
        
        # Get wavelengths at local maxima
        max_wavelengths = wavelengths[is_local_max]
        max_transmissions = transmission[is_local_max]
        
        # Sort by transmission value (descending)
        sort_idx = np.argsort(-max_transmissions)
        max_wavelengths = max_wavelengths[sort_idx]
        max_transmissions = max_transmissions[sort_idx]
        
        # Apply minimum spacing constraint if specified
        if min_spacing is not None and len(max_wavelengths) > 0:
            selected_wavelengths = [max_wavelengths[0]]
            selected_transmissions = [max_transmissions[0]]
            
            for i in range(1, len(max_wavelengths)):
                # Check if this wavelength is far enough from all selected wavelengths
                if all(abs(max_wavelengths[i] - selected) >= min_spacing for selected in selected_wavelengths):
                    selected_wavelengths.append(max_wavelengths[i])
                    selected_transmissions.append(max_transmissions[i])
                
                # Stop if we have enough wavelengths
                if len(selected_wavelengths) >= num_wavelengths:
                    break
            
            max_wavelengths = np.array(selected_wavelengths)
            max_transmissions = np.array(selected_transmissions)
        
        # Select the top wavelengths
        if len(max_wavelengths) >= num_wavelengths:
            optimal_wavelengths = max_wavelengths[:num_wavelengths]
        else:
            # If we don't have enough local maxima, add wavelengths with highest transmission
            sort_idx = np.argsort(-transmission)
            additional_needed = num_wavelengths - len(max_wavelengths)
            
            # Add wavelengths that aren't already in max_wavelengths
            additional_wavelengths = []
            for idx in sort_idx:
                if wavelengths[idx] not in max_wavelengths:
                    additional_wavelengths.append(wavelengths[idx])
                    if len(additional_wavelengths) >= additional_needed:
                        break
            
            optimal_wavelengths = np.concatenate([max_wavelengths, np.array(additional_wavelengths)])
        
        # Sort by wavelength
        optimal_wavelengths = np.sort(optimal_wavelengths)
        
        return optimal_wavelengths
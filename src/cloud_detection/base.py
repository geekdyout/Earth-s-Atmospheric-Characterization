"""
Base classes for cloud detection algorithms.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, List, Optional, Union, Tuple

class CloudDetector(ABC):
    """Base class for cloud detection algorithms."""
    
    @abstractmethod
    def detect(self, reflectance: np.ndarray) -> np.ndarray:
        """
        Detect clouds in reflectance data.
        
        Args:
            reflectance: Normalized reflectance cube
            
        Returns:
            Binary cloud mask
        """
        pass
    
    @abstractmethod
    def confidence(self, reflectance: np.ndarray) -> np.ndarray:
        """
        Calculate cloud detection confidence.
        
        Args:
            reflectance: Normalized reflectance cube
            
        Returns:
            Cloud confidence values (0-1)
        """
        pass
"""
Validation metrics for evaluating algorithm performance.
"""

import numpy as np
import pandas as pd
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)

class ValidationMetrics:
    """Validation metrics for evaluating algorithm performance."""
    
    def __init__(self, metrics_dir: str = "metrics"):
        """
        Initialize the validation metrics.
        
        Args:
            metrics_dir: Directory to store metrics reports
        """
        self.metrics_dir = Path(metrics_dir)
        
        # Create metrics directory if it doesn't exist
        os.makedirs(self.metrics_dir, exist_ok=True)
    
    def calculate_classification_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        sample_weight: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            sample_weight: Optional sample weights
            
        Returns:
            Dictionary of metric names and values
        """
        metrics = {}
        
        # Accuracy
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred, sample_weight=sample_weight))
        
        # Precision, recall, F1 score
        metrics["precision"] = float(precision_score(y_true, y_pred, sample_weight=sample_weight, zero_division=0))
        metrics["recall"] = float(recall_score(y_true, y_pred, sample_weight=sample_weight, zero_division=0))
        metrics["f1"] = float(f1_score(y_true, y_pred, sample_weight=sample_weight, zero_division=0))
        
        return metrics
    
    def calculate_regression_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        sample_weight: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate regression metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            sample_weight: Optional sample weights
            
        Returns:
            Dictionary of metric names and values
        """
        metrics = {}
        
        # Mean squared error
        metrics["mse"] = float(mean_squared_error(y_true, y_pred, sample_weight=sample_weight))
        
        # Root mean squared error
        metrics["rmse"] = float(np.sqrt(metrics["mse"]))
        
        # Mean absolute error
        metrics["mae"] = float(mean_absolute_error(y_true, y_pred, sample_weight=sample_weight))
        
        # R-squared
        metrics["r2"] = float(r2_score(y_true, y_pred, sample_weight=sample_weight))
        
        # Mean bias
        metrics["bias"] = float(np.mean(y_pred - y_true))
        
        # Normalized RMSE
        if np.max(y_true) > np.min(y_true):
            metrics["nrmse"] = float(metrics["rmse"] / (np.max(y_true) - np.min(y_true)))
        else:
            metrics["nrmse"] = float("nan")
        
        return metrics
    
    def save_metrics_csv(
        self, 
        metrics: Dict[str, float], 
        filename: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Save metrics to a CSV file.
        
        Args:
            metrics: Dictionary of metric names and values
            filename: Output filename
            metadata: Optional metadata to include
            
        Returns:
            Path to the saved file
        """
        # Create a DataFrame from the metrics
        df = pd.DataFrame([metrics])
        
        # Add metadata if provided
        if metadata:
            for key, value in metadata.items():
                df[key] = value
        
        # Ensure the filename has a .csv extension
        if not filename.endswith(".csv"):
            filename += ".csv"
        
        # Save to CSV
        output_path = self.metrics_dir / filename
        df.to_csv(output_path, index=False)
        
        return output_path
    
    def save_metrics_json(
        self, 
        metrics: Dict[str, float], 
        filename: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Save metrics to a JSON file.
        
        Args:
            metrics: Dictionary of metric names and values
            filename: Output filename
            metadata: Optional metadata to include
            
        Returns:
            Path to the saved file
        """
        # Create a dictionary with metrics and metadata
        data = {"metrics": metrics}
        
        if metadata:
            data["metadata"] = metadata
        
        # Ensure the filename has a .json extension
        if not filename.endswith(".json"):
            filename += ".json"
        
        # Save to JSON
        output_path = self.metrics_dir / filename
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        
        return output_path
    
    def save_metrics_markdown(
        self, 
        metrics: Dict[str, float], 
        filename: str,
        title: str = "Validation Metrics",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Save metrics to a Markdown file.
        
        Args:
            metrics: Dictionary of metric names and values
            filename: Output filename
            title: Title for the Markdown document
            metadata: Optional metadata to include
            
        Returns:
            Path to the saved file
        """
        # Ensure the filename has a .md extension
        if not filename.endswith(".md"):
            filename += ".md"
        
        # Create Markdown content
        content = f"# {title}\n\n"
        
        # Add metadata if provided
        if metadata:
            content += "## Metadata\n\n"
            for key, value in metadata.items():
                content += f"- **{key}**: {value}\n"
            content += "\n"
        
        # Add metrics
        content += "## Metrics\n\n"
        content += "| Metric | Value |\n"
        content += "|--------|-------|\n"
        for key, value in metrics.items():
            content += f"| {key} | {value:.6f} |\n"
        
        # Save to Markdown
        output_path = self.metrics_dir / filename
        with open(output_path, "w") as f:
            f.write(content)
        
        return output_path
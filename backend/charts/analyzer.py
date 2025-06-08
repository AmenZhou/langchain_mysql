"""
Chart Eligibility Analyzer

Analyzes data to determine if it's suitable for chart generation.
Handles PII filtering, data quality assessment, and eligibility criteria.
"""

from typing import Dict, List, Any, Tuple
import numpy as np
import logging

from .models import EligibilityResult, ChartSettings
from .exceptions import InsufficientDataError, PIIOnlyDataError, DataIneligibleError

logger = logging.getLogger(__name__)


class ChartEligibilityAnalyzer:
    """Analyzes data to determine if it's suitable for chart generation."""
    
    def __init__(self, settings: ChartSettings = None):
        self.settings = settings or ChartSettings()
        
    def analyze(self, data: List[Dict[str, Any]]) -> EligibilityResult:
        """
        Analyze data to determine chart eligibility.
        
        Args:
            data: List of data dictionaries
            
        Returns:
            EligibilityResult with analysis details
            
        Raises:
            DataIneligibleError: If data is fundamentally unsuitable
        """
        if not data:
            raise DataIneligibleError("No data to visualize", data_size=0)
        
        data_points = len(data)
        
        # Check data size limits
        if data_points < self.settings.min_rows:
            raise InsufficientDataError(data_points, self.settings.min_rows)
            
        if data_points > self.settings.max_rows:
            raise DataIneligibleError(
                f"Too many data points ({data_points}, maximum {self.settings.max_rows} supported)",
                data_size=data_points
            )
        
        # Analyze columns
        column_analysis = self._analyze_columns(data)
        
        # Check for PII-only data
        if column_analysis['non_pii_columns'] == 0:
            raise PIIOnlyDataError()
        
        # Check for chartable data types
        if not self._has_chartable_data(data, column_analysis):
            raise DataIneligibleError(
                "No numeric or categorical data suitable for visualization",
                data_size=data_points
            )
        
        return EligibilityResult(
            eligible=True,
            reason="Data is suitable for chart generation",
            data_points=data_points,
            columns_analyzed=column_analysis['total_columns'],
            pii_columns_excluded=column_analysis['pii_columns']
        )
    
    def _analyze_columns(self, data: List[Dict[str, Any]]) -> Dict[str, int]:
        """Analyze column types and PII status."""
        if not data:
            return {'total_columns': 0, 'pii_columns': 0, 'non_pii_columns': 0}
        
        sample_row = data[0]
        total_columns = len(sample_row)
        pii_columns = 0
        
        for key, value in sample_row.items():
            if str(value) == "[PRIVATE]":
                pii_columns += 1
        
        return {
            'total_columns': total_columns,
            'pii_columns': pii_columns,
            'non_pii_columns': total_columns - pii_columns
        }
    
    def _has_chartable_data(self, data: List[Dict[str, Any]], column_analysis: Dict[str, int]) -> bool:
        """Check if data contains numeric or categorical columns suitable for charts."""
        if not data or column_analysis['non_pii_columns'] == 0:
            return False
            
        sample_row = data[0]
        numeric_cols = 0
        categorical_cols = 0
        
        for key, value in sample_row.items():
            if str(value) == "[PRIVATE]":
                continue
                
            # Check if column is numeric across all rows
            if self._is_numeric_column(data, key):
                numeric_cols += 1
            elif self._is_categorical_column(data, key):
                categorical_cols += 1
                
        # Need at least one numeric OR two categorical columns
        return numeric_cols >= 1 or categorical_cols >= 2

    def _is_numeric_column(self, data: List[Dict[str, Any]], column: str) -> bool:
        """Check if a column contains numeric data with sufficient variance."""
        try:
            non_private_values = [
                row[column] for row in data 
                if str(row.get(column)) != "[PRIVATE]" and row.get(column) is not None
            ]
            
            if len(non_private_values) < 2:
                return False
                
            # Try converting to float
            numeric_values = []
            for val in non_private_values:
                try:
                    numeric_values.append(float(val))
                except (ValueError, TypeError):
                    return False
                    
            # Check for sufficient variance
            if len(set(numeric_values)) < 2:
                return False
                
            return np.var(numeric_values) > self.settings.min_numeric_variance
        except Exception as e:
            logger.debug(f"Error checking numeric column {column}: {e}")
            return False
    
    def _is_categorical_column(self, data: List[Dict[str, Any]], column: str) -> bool:
        """Check if a column contains categorical data suitable for charts."""
        try:
            non_private_values = [
                str(row[column]) for row in data 
                if str(row.get(column)) != "[PRIVATE]" and row.get(column) is not None
            ]
            
            if len(non_private_values) < 2:
                return False
                
            # Should have limited unique values (categorical nature)
            unique_values = set(non_private_values)
            return 2 <= len(unique_values) <= len(non_private_values) * 0.7
        except Exception as e:
            logger.debug(f"Error checking categorical column {column}: {e}")
            return False
    
    def get_column_info(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get detailed information about columns for debugging."""
        if not data:
            return {}
        
        sample_row = data[0]
        column_info = {}
        
        for column in sample_row.keys():
            is_pii = str(sample_row[column]) == "[PRIVATE]"
            is_numeric = self._is_numeric_column(data, column) if not is_pii else False
            is_categorical = self._is_categorical_column(data, column) if not is_pii else False
            
            non_private_count = sum(
                1 for row in data 
                if str(row.get(column)) != "[PRIVATE]" and row.get(column) is not None
            )
            
            column_info[column] = {
                'is_pii': is_pii,
                'is_numeric': is_numeric,
                'is_categorical': is_categorical,
                'non_private_values': non_private_count,
                'sample_values': [
                    row.get(column) for row in data[:5] 
                    if str(row.get(column)) != "[PRIVATE]"
                ][:3]
            }
        
        return column_info 
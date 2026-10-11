"""
Centralized missing value detection utilities.

This module provides a single authoritative implementation for detecting
and counting missing values across the entire codebase.

All missing value detection logic should use these functions to ensure
consistency across:
- dataset profiler
- dataset understanding
- quality engine
- ML readiness
- large dataset processing
- visualizations
- AI explanations
"""

import pandas as pd
import numpy as np


# Standard missing value tokens that should be treated as missing
MISSING_TOKENS = {
    "", " ", "nan", "NaN", "NAN", "na", "NA", "N/A", 
    "null", "NULL", "None", "none", "missing", "?", "-"
}


def convert_missing_tokens_to_nan(df):
    """
    Convert common missing value representations to np.nan in-place.
    
    This should be called once after loading a CSV to normalize
    all missing value representations to np.nan for consistent
    downstream processing.
    
    Modifies the DataFrame in place.
    """
    missing_tokens = MISSING_TOKENS
    
    for col in df.columns:
        dtype = df[col].dtype
        is_string_dtype = dtype == object or isinstance(dtype, pd.StringDtype)
        if is_string_dtype:
            # Convert missing tokens to NaN
            df[col] = df[col].apply(
                lambda x: np.nan if (isinstance(x, str) and x.strip().lower() in missing_tokens) else x
            )
            
            # Also convert empty/whitespace-only strings to NaN
            df[col] = df[col].apply(
                lambda x: np.nan if (isinstance(x, str) and x.strip() == "") else x
            )
    
    return df


def is_missing_value(value):
    """
    Check if a single value should be considered missing.
    
    Args:
        value: A scalar value to check
        
    Returns:
        bool: True if the value should be considered missing
    """
    if pd.isna(value):
        return True
    
    if isinstance(value, str):
        return value.strip() == "" or value.strip().lower() in MISSING_TOKENS
    
    return False


def count_missing_values(series):
    """
    Count missing values in a pandas Series.
    
    Handles both NaN and empty/whitespace-only strings.
    
    Args:
        series: pandas Series to analyze
        
    Returns:
        tuple: (total_missing_count, empty_string_count, nan_count)
    """
    # Count NaN values
    nan_count = int(series.isna().sum())
    
    # Count empty/whitespace-only strings as missing
    empty_count = 0
    dtype = series.dtype
    is_string_dtype = dtype == object or isinstance(dtype, pd.StringDtype)
    if is_string_dtype:
        empty_count = int(
            series.fillna("").astype(str).str.strip().eq("").sum()
        ) - nan_count
        if empty_count < 0:
            empty_count = 0
    
    total_missing = nan_count + empty_count
    
    return total_missing, empty_count, nan_count


def count_missing_values_df(df):
    """
    Count missing values across all columns in a DataFrame.
    
    Uses the centralized missing value semantics.
    
    Args:
        df: pandas DataFrame
        
    Returns:
        dict: Per-column missing value counts and percentages
    """
    results = {}
    total_rows = len(df)
    
    for column in df.columns:
        total_missing, empty_count, nan_count = count_missing_values(df[column])
        
        missing_percentage = (
            (total_missing / total_rows * 100) if total_rows > 0 else 0
        )
        
        results[column] = {
            "missing_count": total_missing,
            "missing_percentage": missing_percentage,
            "empty_count": empty_count,
            "nan_count": nan_count,
            "has_issue": total_missing > 0
        }
    
    return results


def get_missing_value_summary(df):
    """
    Get a summary of missing values across the entire DataFrame.
    
    Returns:
        dict: Summary with total missing cells, affected columns, etc.
    """
    total_rows = len(df)
    total_cells = total_rows * len(df.columns) if total_rows > 0 else 0
    
    missing_info = count_missing_values_df(df)
    
    total_missing = sum(info["missing_count"] for info in missing_info.values())
    affected_columns = [col for col, info in missing_info.items() if info["has_issue"]]
    
    return {
        "total_rows": total_rows,
        "total_cells": total_cells,
        "total_missing_cells": total_missing,
        "missing_percentage": (total_missing / total_cells * 100) if total_cells > 0 else 0,
        "affected_columns": affected_columns,
        "per_column": missing_info
    }
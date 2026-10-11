"""
Shared metadata computation for Data Sentinel.

This module computes common dataset statistics once and provides them
to all analysis modules to avoid redundant DataFrame scans.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

from app.profiler.missing_values import count_missing_values, count_missing_values_df
from app.profiler.sentinel_detector import get_cached_sentinel_detection
from app.profiler.column_roles import analyze_column_roles


def compute_column_metadata(
    df: pd.DataFrame,
    column_roles: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """
    Compute comprehensive column metadata in a single pass.
    
    This replaces the separate computations in profile_dataset and understand_dataset.
    
    Returns:
        Dict with per-column metadata including:
        - data_type, missing_count, missing_percentage
        - unique_count, non_null_unique_count
        - numerical statistics (min, max, mean, median, std, q1, q3, iqr)
        - possible_sentinels
        - column_role
    """
    rows = len(df)
    columns = len(df.columns)
    
    if column_roles is None:
        column_roles = analyze_column_roles(df)
    
    column_metadata = {}
    
    for column in df.columns:
        series = df[column]
        role = column_roles.get(column, "unknown")
        
        # Basic stats
        total_missing, empty_count, nan_count = count_missing_values(series)
        non_null_count = rows - total_missing
        
        unique_count = int(series.nunique(dropna=True))
        non_null_unique_count = int(series.dropna().nunique())
        
        # Detect column type
        if pd.api.types.is_datetime64_any_dtype(series):
            column_type = "datetime"
        elif pd.api.types.is_numeric_dtype(series):
            column_type = "numerical"
        elif pd.api.types.is_bool_dtype(series):
            column_type = "boolean"
        else:
            column_type = "categorical"
        
        # Numerical statistics
        raw_statistics = {}
        clean_statistics = {}
        possible_sentinels = []
        
        if column_type == "numerical":
            numeric_series = pd.to_numeric(series, errors="coerce").dropna()
            
            if not numeric_series.empty:
                q1 = numeric_series.quantile(0.25)
                q3 = numeric_series.quantile(0.75)
                iqr = q3 - q1
                
                raw_statistics = {
                    "min": float(numeric_series.min()),
                    "max": float(numeric_series.max()),
                    "mean": float(numeric_series.mean()),
                    "median": float(numeric_series.median()),
                    "std": float(numeric_series.std()) if pd.notna(numeric_series.std()) else 0.0,
                    "q1": float(q1),
                    "q3": float(q3),
                    "iqr": float(iqr)
                }
                
                # Detect sentinels for clean statistics
                if unique_count <= 10000:
                    possible_sentinels = get_cached_sentinel_detection(series, column)
                
                # Compute clean statistics (excluding high-confidence sentinels)
                clean_series = numeric_series.copy()
                if possible_sentinels:
                    sentinel_values = []
                    for candidate in possible_sentinels:
                        confidence = candidate.get("confidence", 0)
                        classification = candidate.get("classification")
                        if classification == "possible_sentinel" and confidence >= 0.8:
                            sentinel_values.append(candidate["value"])
                    
                    if sentinel_values:
                        clean_series = clean_series[~clean_series.isin(sentinel_values)]
                
                if not clean_series.empty:
                    q1_clean = clean_series.quantile(0.25)
                    q3_clean = clean_series.quantile(0.75)
                    iqr_clean = q3_clean - q1_clean
                    
                    clean_statistics = {
                        "min": float(clean_series.min()),
                        "max": float(clean_series.max()),
                        "mean": float(clean_series.mean()),
                        "median": float(clean_series.median()),
                        "std": float(clean_series.std()) if pd.notna(clean_series.std()) else 0.0,
                        "q1": float(q1_clean),
                        "q3": float(q3_clean),
                        "iqr": float(iqr_clean),
                        "count": int(len(clean_series))
                    }
        
        # Categorical statistics
        categories = []
        if column_type == "categorical":
            values = series.dropna().astype(str).str.strip()
            values = values[values != ""]
            if not values.empty:
                counts = values.value_counts().head(20)
                categories = [{"category": str(v), "count": int(c)} for v, c in counts.items()]
        
        column_metadata[column] = {
            "column": column,
            "type": column_type,
            "role": role,
            "rows": rows,
            "non_null_count": non_null_count,
            "missing_count": total_missing,
            "missing_percentage": round((total_missing / rows * 100) if rows > 0 else 0, 2),
            "unique_count": unique_count,
            "non_null_unique_count": non_null_unique_count,
            "uniqueness_ratio": round(unique_count / rows if rows > 0 else 0, 4),
            "raw_statistics": raw_statistics,
            "clean_statistics": clean_statistics,
            "possible_sentinels": possible_sentinels,
            "categories": categories
        }
    
    return column_metadata


def compute_dataset_profile(
    df: pd.DataFrame,
    column_metadata: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Compute dataset-level profile from pre-computed column metadata.
    
    Avoids re-scanning the DataFrame.
    """
    rows = len(df)
    columns = len(df.columns)
    
    # Duplicate detection (single scan)
    duplicated_mask = df.duplicated(keep=False)
    duplicate_count = int((~duplicated_mask).sum())  # Actually this is wrong - let me fix
    duplicate_count = int(df.duplicated().sum())
    duplicate_rows_involved = int(duplicated_mask.sum())
    duplicate_percentage = (duplicate_count / rows * 100) if rows > 0 else 0
    
    # Column details for profile (subset of column_metadata)
    column_details = {}
    for column, meta in column_metadata.items():
        column_details[column] = {
            "data_type": meta["type"],
            # Additive keys consumed by the chat assistant (column/project/
            # model answers). 'data_type' remains for existing consumers.
            "type": meta["type"],
            "role": meta["role"],
            "missing_count": meta["missing_count"],
            "missing_percentage": meta["missing_percentage"],
            "unique_count": meta["unique_count"],
            "non_null_unique_count": meta["non_null_unique_count"],
            "empty_count": meta["missing_count"] - meta.get("nan_count", 0),  # Approximation
        }
        
        # Add numerical stats if available
        if meta["raw_statistics"]:
            column_details[column].update({
                "min": meta["raw_statistics"].get("min"),
                "max": meta["raw_statistics"].get("max"),
                "mean": meta["raw_statistics"].get("mean"),
                "median": meta["raw_statistics"].get("median"),
            })
    
    return {
        "rows": rows,
        "columns": columns,
        "column_names": df.columns.tolist(),
        "duplicate_count": duplicate_count,
        "duplicate_rows_involved": duplicate_rows_involved,
        "duplicate_percentage": duplicate_percentage,
        "column_details": column_details
    }


def compute_understanding(
    df: pd.DataFrame,
    column_metadata: Dict[str, Any],
    column_roles: Dict[str, str]
) -> Dict[str, Any]:
    """
    Compute dataset understanding from pre-computed column metadata.
    
    Avoids re-scanning the DataFrame.
    """
    rows = len(df)
    columns = len(df.columns)
    
    column_analysis = []
    numerical_statistics = {}
    categorical_statistics = {}
    sentinel_detection = {"columns": {}}
    
    for column, meta in column_metadata.items():
        col_info = {
            "column": column,
            "type": meta["type"],
            "role": meta["role"],
            "rows": meta["rows"],
            "non_null_count": meta["non_null_count"],
            "missing_count": meta["missing_count"],
            "missing_percentage": meta["missing_percentage"],
            "unique_count": meta["unique_count"],
            "uniqueness_ratio": meta["uniqueness_ratio"],
            "raw_statistics": meta["raw_statistics"],
            "clean_statistics": meta["clean_statistics"],
            "possible_sentinels": meta["possible_sentinels"]
        }
        
        if meta["type"] == "categorical":
            col_info["categories"] = meta["categories"]
            categorical_statistics[column] = meta["categories"]
        
        if meta["raw_statistics"]:
            numerical_statistics[column] = {
                "raw_statistics": meta["raw_statistics"],
                "clean_statistics": meta["clean_statistics"],
                "possible_sentinels": meta["possible_sentinels"]
            }
            
            # Build sentinel detection format
            sentinel_candidates = []
            placeholder_candidates = []
            sentinel_count = 0
            placeholder_count = 0
            sentinel_occurrences = 0
            placeholder_occurrences = 0
            
            for sentinel in meta["possible_sentinels"]:
                classification = sentinel.get("classification")
                confidence = sentinel.get("confidence", 0)
                
                if classification == "possible_sentinel":
                    sentinel_candidates.append(sentinel)
                    sentinel_count += 1
                    sentinel_occurrences += sentinel.get("count", 0)
                elif classification == "possible_placeholder":
                    placeholder_candidates.append(sentinel)
                    placeholder_count += 1
                    placeholder_occurrences += sentinel.get("count", 0)
            
            if sentinel_candidates or placeholder_candidates:
                sentinel_detection["columns"][column] = {
                    "candidate_count": len(meta["possible_sentinels"]),
                    "sentinel_count": sentinel_count,
                    "placeholder_count": placeholder_count,
                    "sentinel_occurrences": sentinel_occurrences,
                    "placeholder_occurrences": placeholder_occurrences,
                    "sentinel_candidates": sentinel_candidates,
                    "placeholder_candidates": placeholder_candidates,
                    "has_issue": (sentinel_count > 0 or placeholder_count > 0)
                }
        
        column_analysis.append(col_info)
    
    # Aggregate sentinel stats
    total_sentinels = sum(c.get("sentinel_count", 0) for c in sentinel_detection["columns"].values())
    total_placeholders = sum(c.get("placeholder_count", 0) for c in sentinel_detection["columns"].values())
    affected_rows = sum(c.get("sentinel_occurrences", 0) + c.get("placeholder_occurrences", 0) for c in sentinel_detection["columns"].values())
    affected_row_rate = affected_rows / rows if rows > 0 else 0
    
    sentinel_detection["total_sentinels"] = total_sentinels
    sentinel_detection["total_placeholders"] = total_placeholders
    sentinel_detection["affected_rows"] = affected_rows
    sentinel_detection["affected_row_rate"] = affected_row_rate
    
    return {
        "rows": rows,
        "columns": columns,
        "column_analysis": column_analysis,
        "column_roles": column_roles,
        "numerical_statistics": numerical_statistics,
        "categorical_statistics": categorical_statistics,
        "sentinel_detection": sentinel_detection,
        "column_types": {col: meta["type"] for col, meta in column_metadata.items()},
        "column_names": df.columns.tolist()
    }
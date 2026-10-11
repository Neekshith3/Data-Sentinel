import pandas as pd

from app.profiler.column_roles import detect_column_role


def detect_data_type_issues(df, expected_schema):
    """
    Compare expected column data types with actual data types.
    Also detects semantic type mismatches (e.g., numerical data stored as strings).
    
    IMPORTANT: Missing values (NaN) must NOT trigger data type issues.
    Only the NON-NULL values are checked for type correctness.

    Parameters:
        df (pandas.DataFrame): Dataset to analyze.
        expected_schema (dict): Expected data types for each column.

    Returns:
        dict: Data type validation results.
    """

    results = {}

    # Check expected schema if provided
    for column, expected_type in expected_schema.items():

        if column not in df.columns:

            results[column] = {
                "expected_type": expected_type,
                "actual_type": None,
                "has_issue": True,
                "issue": "column_missing"
            }

            continue

        actual_dtype = str(df[column].dtype)

        if expected_type == "integer":
            type_matches = pd.api.types.is_integer_dtype(df[column])

        elif expected_type == "float":
            type_matches = pd.api.types.is_float_dtype(df[column])

        elif expected_type == "numeric":
            type_matches = pd.api.types.is_numeric_dtype(df[column])

        elif expected_type == "string":
            type_matches = pd.api.types.is_string_dtype(df[column])

        else:
            type_matches = False

        results[column] = {
            "expected_type": expected_type,
            "actual_type": actual_dtype,
            "has_issue": not type_matches,
            "issue": (
                None
                if type_matches
                else "data_type_mismatch"
            )
        }

    # Semantic type validation: detect columns that should be numerical/datetime
    # but are stored as object/string
    # 
    # CRITICAL: Only check NON-NULL values. Missing values (NaN) must NOT 
    # trigger data type issues. A column with missing values stored as object
    # dtype is NOT a data type issue if the non-null values match the 
    # expected semantic type.
    for column in df.columns:
        if column in results:
            continue  # already checked via expected_schema

        series = df[column]
        actual_dtype = str(series.dtype)
        
        # Only check object/string columns for semantic type mismatches
        if (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
        ):
            # Check if this column should be numerical/datetime/etc.
            # Use ONLY non-null values for role detection
            non_null_series = series.dropna()
            
            if non_null_series.empty:
                # All values are missing - can't determine semantic type
                continue
            
            # Check if non-null values match the expected semantic type
            role = detect_column_role(column, non_null_series)
            
            if role in ("numerical_measurement", "numerical"):
                # Check if non-null values are actually numeric
                numeric_attempt = pd.to_numeric(non_null_series, errors="coerce")
                numeric_success_rate = numeric_attempt.notna().mean()
                
                if numeric_success_rate > 0.8:
                    # Non-null values ARE numeric - this is fine, just stored as object due to NaN
                    # This is NOT a data type issue
                    pass
                else:
                    # Non-null values are NOT numeric - this IS a data type issue
                    results[column] = {
                        "expected_type": "numerical",
                        "actual_type": actual_dtype,
                        "has_issue": True,
                        "issue": "semantic_type_mismatch",
                        "detected_role": role,
                        "message": f"Column '{column}' contains non-numeric values but is expected to be numerical"
                    }
            elif role == "datetime":
                # Check if non-null values are parseable as dates
                parsed = 0
                sample = non_null_series.head(100).astype(str)
                for v in sample:
                    try:
                        pd.to_datetime(v)
                        parsed += 1
                    except:
                        pass
                if parsed / len(sample) > 0.8:
                    # Non-null values ARE dates - this is fine, just stored as object due to NaN
                    pass
                else:
                    results[column] = {
                        "expected_type": "datetime",
                        "actual_type": actual_dtype,
                        "has_issue": True,
                        "issue": "semantic_type_mismatch",
                        "detected_role": role,
                        "message": f"Column '{column}' contains unparseable date values"
                    }
            elif role == "postal_code":
                # Check if non-null values match postal code pattern
                import re
                postal_pattern = re.compile(r'^\d{5,6}$')
                sample = non_null_series.head(100).astype(str)
                matches = sum(1 for v in sample if postal_pattern.match(v.strip()))
                if matches / len(sample) > 0.8:
                    pass
                else:
                    results[column] = {
                        "expected_type": "postal_code",
                        "actual_type": actual_dtype,
                        "has_issue": True,
                        "issue": "semantic_type_mismatch",
                        "detected_role": role,
                        "message": f"Column '{column}' contains invalid postal code values"
                    }
            elif role == "email":
                # Check if non-null values match email pattern
                import re
                email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
                sample = non_null_series.head(100).astype(str)
                matches = sum(1 for v in sample if email_pattern.match(v))
                if matches / len(sample) > 0.8:
                    pass
                else:
                    results[column] = {
                        "expected_type": "email",
                        "actual_type": actual_dtype,
                        "has_issue": True,
                        "issue": "semantic_type_mismatch",
                        "detected_role": role,
                        "message": f"Column '{column}' contains invalid email values"
                    }

    return {
        "issue_type": "data_type_validation",
        "has_issue": any(
            result["has_issue"]
            for result in results.values()
        ),
        "columns": results
    }
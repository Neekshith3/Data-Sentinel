import pandas as pd
import re

from app.profiler.column_roles import detect_column_role, looks_like_email, looks_like_postal_code


def detect_invalid_values(df, rules):
    """
    Detect values that violate configured validation rules.
    Also performs semantic validation for email, postal codes, and numerical ranges.

    Parameters:
        df (pandas.DataFrame): Dataset to analyze.
        rules (dict): Validation rules for columns.

    Returns:
        dict: Invalid-value detection results.
    """

    results = {}

    # User-provided validation rules
    for column, rule in rules.items():

        if column not in df.columns:
            results[column] = {
                "has_issue": True,
                "invalid_count": 0,
                "issue": "column_missing"
            }
            continue

        column_results = {}

        if "min" in rule:
            min_value = rule["min"]

            invalid_mask = df[column] < min_value

            column_results["below_minimum"] = {
                "count": int(invalid_mask.sum()),
                "rows": df[invalid_mask].index.tolist()
            }

        if "max" in rule:
            max_value = rule["max"]

            invalid_mask = df[column] > max_value

            column_results["above_maximum"] = {
                "count": int(invalid_mask.sum()),
                "rows": df[invalid_mask].index.tolist()
            }

        invalid_count = sum(
            result["count"]
            for result in column_results.values()
        )

        results[column] = {
            "has_issue": invalid_count > 0,
            "invalid_count": invalid_count,
            "rules": column_results
        }

    # Semantic validation for columns not in user rules
    for column in df.columns:
        if column in results:
            continue

        series = df[column].dropna()
        if series.empty:
            continue

        role = detect_column_role(column, df[column])
        column_results = {}
        invalid_count = 0

        # Email validation
        if role == "email" or looks_like_email(df[column]):
            email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
            invalid_mask = ~series.astype(str).str.match(email_pattern)
            count = int(invalid_mask.sum())
            if count > 0:
                column_results["invalid_email_format"] = {
                    "count": count,
                    "rows": series[invalid_mask].index.tolist()
                }
                invalid_count += count

        # Postal code validation (Indian PIN: 6 digits, US ZIP: 5 digits)
        if role == "postal_code" or looks_like_postal_code(df[column]):
            postal_pattern = re.compile(r'^\d{5,6}$')
            invalid_mask = ~series.astype(str).str.match(postal_pattern)
            count = int(invalid_mask.sum())
            if count > 0:
                column_results["invalid_postal_format"] = {
                    "count": count,
                    "rows": series[invalid_mask].index.tolist()
                }
                invalid_count += count

        # Auto-detect numerical range issues for semantic numerical columns
        if role in ("numerical_measurement", "numerical"):
            numeric_series = pd.to_numeric(series, errors="coerce").dropna()
            if not numeric_series.empty:
                # Check for negative values in columns that should be non-negative
                if column.lower() in ("age", "quantity", "count", "price", "amount", "salary", "income", "distance", "weight", "height"):
                    negative_mask = numeric_series < 0
                    count = int(negative_mask.sum())
                    if count > 0:
                        column_results["negative_values"] = {
                            "count": count,
                            "rows": numeric_series[negative_mask].index.tolist()
                        }
                        invalid_count += count
                
                # Check for zero/near-zero variance (constant columns)
                if numeric_series.nunique() <= 1:
                    column_results["zero_variance"] = {
                        "count": len(numeric_series),
                        "rows": numeric_series.index.tolist()
                    }
                    invalid_count += len(numeric_series)

        if invalid_count > 0:
            results[column] = {
                "has_issue": True,
                "invalid_count": invalid_count,
                "rules": column_results
            }

    return {
        "issue_type": "invalid_values",
        "has_issue": any(
            result["has_issue"]
            for result in results.values()
        ),
        "columns": results
    }
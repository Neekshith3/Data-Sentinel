import pandas as pd

from app.quality_engine.duplicates import detect_duplicates
from app.quality_engine.missing_values import detect_missing_values
from app.quality_engine.consistency import (
    detect_categorical_inconsistencies
)
from app.quality_engine.data_types import detect_data_type_issues
from app.quality_engine.validation import detect_invalid_values
from app.quality_engine.outliers import detect_iqr_outliers
from app.quality_engine.ml_anomaly import detect_ml_anomalies
from app.quality_engine.sentinels import (
    detect_sentinel_values
)

from app.quality_engine.findings import create_finding


def standardize_finding(
    result
):
    """
    Convert a detector result into a common finding format.
    """

    issue_type = result["issue_type"]


    # --------------------------------------------------
    # Duplicate rows
    # --------------------------------------------------

    if issue_type == "duplicate_rows":

        return create_finding(

            issue_type=issue_type,

            has_issue=result[
                "has_issue"
            ],

            count=result[
                "duplicate_count"
            ],

            affected_rows=(
                result[
                    "duplicate_rows"
                ]
                .index
                .tolist()
            ),

            details={
                "percentage":
                    result[
                        "duplicate_percentage"
                    ],
                "duplicate_rows_involved":
                    result.get("duplicate_rows_involved", 0)
            }

        )


    # --------------------------------------------------
    # Missing values
    # --------------------------------------------------

    if issue_type == "missing_values":

        affected_columns = []

        total_missing = 0


        for (
            column,
            column_result
        ) in result[
            "columns"
        ].items():

            if column_result[
                "has_issue"
            ]:

                affected_columns.append(
                    column
                )

                total_missing += (
                    column_result[
                        "missing_count"
                    ]
                )


        return create_finding(

            issue_type=issue_type,

            has_issue=result[
                "has_issue"
            ],

            count=total_missing,

            affected_columns=
                affected_columns,

            details=result[
                "columns"
            ]

        )


    # --------------------------------------------------
    # Categorical inconsistencies
    # --------------------------------------------------

    if issue_type == "categorical_inconsistency":

        affected_columns = []

        total_inconsistencies = 0


        for (
            column,
            column_result
        ) in result[
            "columns"
        ].items():

            if column_result[
                "has_issue"
            ]:

                affected_columns.append(
                    column
                )

                total_inconsistencies += len(
                    column_result[
                        "inconsistent_groups"
                    ]
                )


        return create_finding(

            issue_type=issue_type,

            has_issue=result[
                "has_issue"
            ],

            count=total_inconsistencies,

            affected_columns=
                affected_columns,

            details=result[
                "columns"
            ]

        )


    # --------------------------------------------------
    # Data type validation
    # --------------------------------------------------

    if issue_type == "data_type_validation":

        affected_columns = []
        total_issues = 0


        for (
            column,
            column_result
        ) in result[
            "columns"
        ].items():

            if column_result[
                "has_issue"
            ]:

                affected_columns.append(
                    column
                )
                total_issues += 1


        return create_finding(

            issue_type=issue_type,

            has_issue=result[
                "has_issue"
            ],

            count=total_issues,

            affected_columns=
                affected_columns,

            details=result[
                "columns"
            ]

        )


    # --------------------------------------------------
    # Invalid values
    # --------------------------------------------------

    if issue_type == "invalid_values":

        affected_columns = []

        total_invalid = 0


        for (
            column,
            column_result
        ) in result[
            "columns"
        ].items():

            if column_result[
                "has_issue"
            ]:

                affected_columns.append(
                    column
                )

                total_invalid += (
                    column_result[
                        "invalid_count"
                    ]
                )


        return create_finding(

            issue_type=issue_type,

            has_issue=result[
                "has_issue"
            ],

            count=total_invalid,

            affected_columns=
                affected_columns,

            details=result[
                "columns"
            ]

        )


    # --------------------------------------------------
    # Statistical outliers
    # --------------------------------------------------

    if issue_type == "statistical_outliers":

        affected_columns = []

        total_outliers = 0

        details = {}


        for (
            column,
            column_result
        ) in result[
            "columns"
        ].items():

            if column_result[
                "has_issue"
            ]:

                affected_columns.append(
                    column
                )

                total_outliers += (
                    column_result[
                        "outlier_count"
                    ]
                )


            # Get the outlier rows DataFrame
            outlier_rows_df = column_result.get("outlier_rows", None)

            # Extract actual outlier values if available
            outlier_values = []
            if outlier_rows_df is not None and len(outlier_rows_df) > 0:
                # Get the actual values for this column from the outlier rows
                outlier_values = outlier_rows_df[column].tolist()

            details[column] = {

                "q1":
                    column_result[
                        "q1"
                    ],

                "q3":
                    column_result[
                        "q3"
                    ],

                "iqr":
                    column_result[
                        "iqr"
                    ],

                "lower_bound":
                    column_result[
                        "lower_bound"
                    ],

                "upper_bound":
                    column_result[
                        "upper_bound"
                    ],

                "outlier_count":
                    column_result[
                        "outlier_count"
                    ],

                "has_issue":
                    column_result[
                        "has_issue"
                    ],

                "outlier_rows": (
                    column_result[
                        "outlier_rows"
                    ]
                    .index
                    .tolist()
                ),

                "outlier_values": outlier_values

            }


        return create_finding(

            issue_type=issue_type,

            has_issue=result[
                "has_issue"
            ],

            count=total_outliers,

            affected_columns=
                affected_columns,

            details=details

        )


    # --------------------------------------------------
    # Possible sentinel and placeholder values
    # --------------------------------------------------

    if issue_type == "possible_sentinel_values":

        affected_columns = []

        total_sentinel_occurrences = 0
        total_placeholder_occurrences = 0

        details = {}


        for (
            column,
            column_result
        ) in result[
            "columns"
        ].items():

            if column_result[
                "has_issue"
            ]:

                affected_columns.append(
                    column
                )

                total_sentinel_occurrences += (
                    column_result.get(
                        "sentinel_occurrences", 0
                    )
                )
                total_placeholder_occurrences += (
                    column_result.get(
                        "placeholder_occurrences", 0
                    )
                )


                details[column] = {

                    "candidate_count":
                        column_result[
                            "candidate_count"
                        ],

                    "sentinel_count":
                        column_result[
                            "sentinel_count"
                        ],

                    "placeholder_count":
                        column_result[
                            "placeholder_count"
                        ],

                    "sentinel_occurrences":
                        column_result.get(
                            "sentinel_occurrences", 0
                        ),

                    "placeholder_occurrences":
                        column_result.get(
                            "placeholder_occurrences", 0
                        ),

                    "sentinel_candidates":
                        column_result[
                            "sentinel_candidates"
                        ],

                    "placeholder_candidates":
                        column_result[
                            "placeholder_candidates"
                        ]

                }


        total_occurrences = total_sentinel_occurrences + total_placeholder_occurrences

        return create_finding(

            issue_type=issue_type,

            has_issue=result[
                "has_issue"
            ],

            count=total_occurrences,

            affected_columns=
                affected_columns,

            details={
                "total_candidates":
                    sum(c["candidate_count"] for c in details.values()),
                "total_sentinels":
                    sum(c["sentinel_count"] for c in details.values()),
                "total_placeholders":
                    sum(c["placeholder_count"] for c in details.values()),
                "total_sentinel_occurrences":
                    total_sentinel_occurrences,
                "total_placeholder_occurrences":
                    total_placeholder_occurrences,
                "total_occurrences":
                    total_occurrences,
                "columns":
                    details

            }

        )


    # --------------------------------------------------
    # ML anomalies
    # --------------------------------------------------

    if issue_type == "ml_anomalies":

        return create_finding(

            issue_type=issue_type,

            has_issue=result[
                "has_issue"
            ],

            count=result[
                "anomaly_count"
            ],

            affected_rows=result[
                "anomaly_rows"
            ],

            details={

                "anomaly_scores":
                    result[
                        "anomaly_scores"
                    ],

                "feature_columns":
                    result[
                        "feature_columns"
                    ],

                "contamination":
                    result[
                        "contamination"
                    ]

            }

        )


    # --------------------------------------------------
    # Unknown detector
    # --------------------------------------------------

    return create_finding(

        issue_type=issue_type,

        has_issue=result[
            "has_issue"
        ]

    )


def analyze_dataset(
    df,
    expected_schema,
    validation_rules,
    ml_feature_columns=None,
    column_metadata=None,
    understanding=None,
    column_roles=None
):
    """
    Run all data-quality and ML anomaly detectors
    on a dataset.
    
    If column_metadata, understanding, and column_roles are provided,
    they will be used to avoid redundant DataFrame scans.
    """

    # If metadata is provided, use optimized path
    if column_metadata is not None and understanding is not None and column_roles is not None:
        return analyze_dataset_with_metadata(
            df,
            expected_schema,
            validation_rules,
            ml_feature_columns,
            column_metadata,
            understanding,
            column_roles
        )

    # Original path (backward compatible)
    duplicate_result = detect_duplicates(
        df
    )


    missing_result = detect_missing_values(
        df
    )


    consistency_result = (
        detect_categorical_inconsistencies(
            df
        )
    )


    type_result = detect_data_type_issues(
        df,
        expected_schema
    )


    validation_result = detect_invalid_values(
        df,
        validation_rules
    )


    outlier_result = detect_iqr_outliers(
        df
    )


    raw_results = [

        duplicate_result,

        missing_result,

        consistency_result,

        type_result,

        validation_result,

        outlier_result

    ]


    # --------------------------------------------------
    # Possible sentinel / placeholder detection
    # --------------------------------------------------

    sentinel_result = detect_sentinel_values(
        df
    )


    raw_results.append(
        sentinel_result
    )


    # --------------------------------------------------
    # ML anomaly detection
    # --------------------------------------------------

    if ml_feature_columns:

        ml_result = detect_ml_anomalies(

            df,

            feature_columns=
                ml_feature_columns,

            contamination=0.10

        )


        raw_results.append(
            ml_result
        )


    # --------------------------------------------------
    # Standardize all findings
    # --------------------------------------------------

    findings = [

        standardize_finding(
            result
        )

        for result
        in raw_results

    ]


    return {

        "has_issues":
            any(

                finding[
                    "has_issue"
                ]

                for finding
                in findings

            ),

        "findings":
            findings

    }


def analyze_dataset_with_metadata(
    df,
    expected_schema,
    validation_rules,
    ml_feature_columns,
    column_metadata,
    understanding,
    column_roles
):
    """
    Optimized analysis using precomputed column metadata and understanding.
    Avoids redundant DataFrame scans.
    """
    
    # Extract precomputed data from column_metadata
    missing_counts = {}
    for column, meta in column_metadata.items():
        missing_counts[column] = meta["missing_count"]
    
    # Missing values - use precomputed counts
    missing_result = {
        "issue_type": "missing_values",
        "has_issue": any(c > 0 for c in missing_counts.values()),
        "columns": {
            column: {
                "missing_count": meta["missing_count"],
                "missing_percentage": meta["missing_percentage"],
                "has_issue": meta["missing_count"] > 0
            }
            for column, meta in column_metadata.items()
        }
    }
    
    # Duplicates - still need to scan (cheap operation)
    duplicate_result = detect_duplicates(df)
    
    # Categorical inconsistencies - use precomputed categories from understanding
    consistency_result = detect_categorical_inconsistencies_with_metadata(
        df, understanding, column_metadata
    )
    
    # Data type validation - use column_roles
    type_result = detect_data_type_issues_with_metadata(
        df, expected_schema, column_metadata, column_roles
    )
    
    # Invalid values - still need to scan
    validation_result = detect_invalid_values(
        df,
        validation_rules
    )
    
    # Outliers - use precomputed clean_statistics
    outlier_result = detect_iqr_outliers_with_metadata(
        df, column_metadata
    )
    
    # Sentinels - use precomputed possible_sentinels from understanding
    sentinel_result = detect_sentinel_values_with_metadata(
        df, understanding, column_metadata
    )
    
    raw_results = [
        duplicate_result,
        missing_result,
        consistency_result,
        type_result,
        validation_result,
        outlier_result,
        sentinel_result
    ]
    
    # ML anomaly detection
    if ml_feature_columns:
        ml_result = detect_ml_anomalies(
            df,
            feature_columns=ml_feature_columns,
            contamination=0.10
        )
        raw_results.append(ml_result)
    
    # Standardize all findings
    findings = [
        standardize_finding(result)
        for result in raw_results
    ]
    
    return {
        "has_issues": any(
            finding["has_issue"]
            for finding in findings
        ),
        "findings": findings
    }


def detect_categorical_inconsistencies_with_metadata(
    df, understanding, column_metadata
):
    """
    Detect categorical inconsistencies using precomputed categories from understanding.
    """
    results = {}
    
    # Get categorical columns from understanding
    categorical_columns = [
        col for col, meta in column_metadata.items()
        if meta["type"] == "categorical"
    ]
    
    # Use categories from understanding if available
    cat_stats = understanding.get("categorical_statistics", {})
    
    for column in categorical_columns:
        # Get categories from understanding (precomputed)
        categories = cat_stats.get(column, [])
        
        if not categories:
            # Fallback: compute from DataFrame
            values = df[column].dropna().astype(str)
            normalized_values = values.str.strip().str.lower()
            
            groups = {}
            for original, normalized in zip(values, normalized_values):
                if normalized not in groups:
                    groups[normalized] = set()
                groups[normalized].add(original)
            
            # Check for known aliases
            from app.quality_engine.consistency import SEMAPHORIC_ALIASES
            alias_groups = {}
            processed = set()
            
            for norm_val, variants in groups.items():
                if norm_val in processed:
                    continue
                    
                canonical = SEMAPHORIC_ALIASES.get(norm_val)
                if canonical and canonical in groups:
                    combined = set()
                    for v in [norm_val, canonical]:
                        combined.update(groups[v])
                    alias_groups[canonical] = list(combined)
                    processed.add(norm_val)
                    processed.add(canonical)
                    continue
                
                reverse_alias = {v: k for k, v in SEMAPHORIC_ALIASES.items()}
                canonical_rev = reverse_alias.get(norm_val)
                if canonical_rev and canonical_rev in groups:
                    combined = set()
                    for v in [norm_val, canonical_rev]:
                        combined.update(groups[v])
                    alias_groups[canonical_rev] = list(combined)
                    processed.add(norm_val)
                    processed.add(canonical_rev)
                    continue
            
            exact_groups = {
                normalized: list(original_values)
                for normalized, original_values in groups.items()
                if len(original_values) > 1 and normalized not in processed
            }
            
            all_groups = {**alias_groups, **exact_groups}
        else:
            # Use precomputed categories - check for case/whitespace variants
            # Build normalized groups from category list
            groups = {}
            for cat in categories:
                norm = cat.get("category", "").strip().lower()
                if norm:
                    if norm not in groups:
                        groups[norm] = set()
                    groups[norm].add(cat.get("category", ""))
            
            # Check for aliases and exact variants
            from app.quality_engine.consistency import SEMAPHORIC_ALIASES
            alias_groups = {}
            processed = set()
            
            for norm_val, variants in groups.items():
                if norm_val in processed:
                    continue
                    
                canonical = SEMAPHORIC_ALIASES.get(norm_val)
                if canonical and canonical in groups:
                    combined = set()
                    for v in [norm_val, canonical]:
                        combined.update(groups[v])
                    alias_groups[canonical] = list(combined)
                    processed.add(norm_val)
                    processed.add(canonical)
                    continue
                
                reverse_alias = {v: k for k, v in SEMAPHORIC_ALIASES.items()}
                canonical_rev = reverse_alias.get(norm_val)
                if canonical_rev and canonical_rev in groups:
                    combined = set()
                    for v in [norm_val, canonical_rev]:
                        combined.update(groups[v])
                    alias_groups[canonical_rev] = list(combined)
                    processed.add(norm_val)
                    processed.add(canonical_rev)
                    continue
            
            exact_groups = {
                normalized: list(original_values)
                for normalized, original_values in groups.items()
                if len(original_values) > 1 and normalized not in processed
            }
            
            all_groups = {**alias_groups, **exact_groups}
        
        results[column] = {
            "has_issue": len(all_groups) > 0,
            "inconsistent_groups": all_groups
        }
    
    return {
        "issue_type": "categorical_inconsistency",
        "has_issue": any(
            result["has_issue"]
            for result in results.values()
        ),
        "columns": results
    }


def detect_data_type_issues_with_metadata(
    df, expected_schema, column_metadata, column_roles
):
    """
    Detect data type issues using precomputed column roles.
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
            "issue": None if type_matches else "data_type_mismatch"
        }
    
    # Semantic type validation using precomputed column_roles
    for column in df.columns:
        if column in results:
            continue
        
        series = df[column]
        actual_dtype = str(series.dtype)
        
        # Only check object/string columns for semantic type mismatches
        if (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
        ):
            role = column_roles.get(column, "unknown")
            
            if role in ("numerical_measurement", "numerical"):
                non_null_series = series.dropna()
                if non_null_series.empty:
                    continue
                
                numeric_attempt = pd.to_numeric(non_null_series, errors="coerce")
                numeric_success_rate = numeric_attempt.notna().mean()
                
                if numeric_success_rate > 0.8:
                    pass
                else:
                    results[column] = {
                        "expected_type": "numerical",
                        "actual_type": actual_dtype,
                        "has_issue": True,
                        "issue": "semantic_type_mismatch",
                        "detected_role": role,
                        "message": f"Column '{column}' contains non-numeric values but is expected to be numerical"
                    }
            elif role == "datetime":
                non_null_series = series.dropna()
                if non_null_series.empty:
                    continue
                
                parsed = 0
                sample = non_null_series.head(100).astype(str)
                for v in sample:
                    try:
                        pd.to_datetime(v)
                        parsed += 1
                    except:
                        pass
                if parsed / len(sample) > 0.8:
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
                import re
                postal_pattern = re.compile(r'^\d{5,6}$')
                non_null_series = series.dropna()
                if non_null_series.empty:
                    continue
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
                import re
                email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
                non_null_series = series.dropna()
                if non_null_series.empty:
                    continue
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


def detect_iqr_outliers_with_metadata(
    df, column_metadata
):
    """
    Detect IQR outliers using precomputed clean_statistics.
    """
    from app.profiler.sentinel_detector import get_cached_sentinel_detection
    
    SENTINEL_CONFIDENCE_THRESHOLD = 0.8
    
    results = {}
    
    numerical_columns = [
        column for column, meta in column_metadata.items()
        if meta["type"] == "numerical"
    ]
    
    for column in numerical_columns:
        meta = column_metadata[column]
        
        # Use precomputed clean statistics if available
        clean_stats = meta.get("clean_statistics", {})
        
        if clean_stats and "q1" in clean_stats and "q3" in clean_stats:
            q1 = clean_stats["q1"]
            q3 = clean_stats["q3"]
            iqr = clean_stats["iqr"]
            
            # Also need to get sentinel values to exclude
            sentinel_values = []
            for candidate in meta.get("possible_sentinels", []):
                confidence = candidate.get("confidence", 0)
                classification = candidate.get("classification")
                if (
                    classification == "possible_sentinel"
                    and confidence >= SENTINEL_CONFIDENCE_THRESHOLD
                ):
                    sentinel_values.append(candidate["value"])
            
            # Get actual values (excluding sentinels)
            numeric_series = pd.to_numeric(df[column], errors="coerce")
            valid_mask = numeric_series.notna()
            if sentinel_values:
                valid_mask = valid_mask & ~numeric_series.isin(sentinel_values)
            
            clean_values = numeric_series[valid_mask]
            
            if len(clean_values) > 0:
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                outlier_mask = (clean_values < lower_bound) | (clean_values > upper_bound)
                outlier_count = int(outlier_mask.sum())
                
                outlier_rows = df[valid_mask & outlier_mask]
                
                results[column] = {
                    "q1": q1,
                    "q3": q3,
                    "iqr": iqr,
                    "lower_bound": lower_bound,
                    "upper_bound": upper_bound,
                    "outlier_count": outlier_count,
                    "has_issue": outlier_count > 0,
                    "outlier_rows": outlier_rows,
                    "outlier_values": clean_values[outlier_mask].tolist()
                }
            else:
                results[column] = {
                    "q1": q1,
                    "q3": q3,
                    "iqr": iqr,
                    "lower_bound": q1 - 1.5 * iqr,
                    "upper_bound": q3 + 1.5 * iqr,
                    "outlier_count": 0,
                    "has_issue": False,
                    "outlier_rows": pd.DataFrame(),
                    "outlier_values": []
                }
        else:
            # Fallback to original computation
            from app.quality_engine.outliers import detect_iqr_outliers
            fallback_result = detect_iqr_outliers(df[[column]])
            if column in fallback_result.get("columns", {}):
                results[column] = fallback_result["columns"][column]
            else:
                results[column] = {
                    "q1": 0, "q3": 0, "iqr": 0,
                    "lower_bound": 0, "upper_bound": 0,
                    "outlier_count": 0, "has_issue": False,
                    "outlier_rows": pd.DataFrame(), "outlier_values": []
                }
    
    return {
        "issue_type": "statistical_outliers",
        "has_issue": any(r.get("has_issue", False) for r in results.values()),
        "columns": results
    }


def detect_sentinel_values_with_metadata(
    df, understanding, column_metadata
):
    """
    Detect sentinel values using precomputed possible_sentinels from understanding.
    """
    columns = {}
    total_sentinel_occurrences = 0
    total_placeholder_occurrences = 0
    
    for column, meta in column_metadata.items():
        possible_sentinels = meta.get("possible_sentinels", [])
        
        sentinel_candidates = []
        placeholder_candidates = []
        
        for candidate in possible_sentinels:
            classification = candidate.get("classification")
            
            if classification == "possible_sentinel":
                sentinel_candidates.append(candidate)
            elif classification == "possible_placeholder":
                placeholder_candidates.append(candidate)
        
        sentinel_occurrences = sum(c.get("count", 0) for c in sentinel_candidates)
        placeholder_occurrences = sum(c.get("count", 0) for c in placeholder_candidates)
        
        candidate_count = len(sentinel_candidates) + len(placeholder_candidates)
        
        total_sentinel_occurrences += sentinel_occurrences
        total_placeholder_occurrences += placeholder_occurrences
        
        columns[column] = {
            "has_issue": candidate_count > 0,
            "candidate_count": candidate_count,
            "sentinel_count": len(sentinel_candidates),
            "placeholder_count": len(placeholder_candidates),
            "sentinel_occurrences": sentinel_occurrences,
            "placeholder_occurrences": placeholder_occurrences,
            "sentinel_candidates": sentinel_candidates,
            "placeholder_candidates": placeholder_candidates
        }
    
    return {
        "issue_type": "possible_sentinel_values",
        "has_issue": total_sentinel_occurrences > 0 or total_placeholder_occurrences > 0,
        "candidate_count": sum(c["candidate_count"] for c in columns.values()),
        "sentinel_count": sum(c["sentinel_count"] for c in columns.values()),
        "placeholder_count": sum(c["placeholder_count"] for c in columns.values()),
        "sentinel_occurrences": total_sentinel_occurrences,
        "placeholder_occurrences": total_placeholder_occurrences,
        "total_occurrences": total_sentinel_occurrences + total_placeholder_occurrences,
        "columns": columns
    }
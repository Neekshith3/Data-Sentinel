import pandas as pd


from app.profiler.column_roles import (
    analyze_column_roles
)


from app.profiler.dataset_understanding import (
    understand_dataset
)

from app.profiler.missing_values import count_missing_values_df

from app.profiler.shared_metadata import compute_column_metadata


# --------------------------------------------------
# Configuration
# --------------------------------------------------

SENTINEL_CONFIDENCE_THRESHOLD = 0.8


# --------------------------------------------------
# Check missing values
# --------------------------------------------------

def check_missing_values(
    df
):

    total_cells = (
        df.shape[0]
        *
        df.shape[1]
    )


    missing_counts = count_missing_values_df(df)
    missing_cells = sum(info["missing_count"] for info in missing_counts.values())


    missing_percentage = (

        missing_cells
        /
        total_cells
        *
        100

        if total_cells > 0

        else 0

    )


    return {

        "check":
            "missing_values",

        "passed":
            missing_cells == 0,

        "missing_cells":
            missing_cells,

        "missing_percentage":
            missing_percentage

    }


# --------------------------------------------------
# Check numerical features
# --------------------------------------------------

def check_numerical_features(
    df,
    column_roles
):

    numerical_columns = [

        column

        for column in df.columns

        if column_roles.get(column)
        ==
        "numerical_measurement"

    ]


    return {

        "check":
            "numerical_features",

        "passed":
            len(numerical_columns) > 0,

        "count":
            len(numerical_columns),

        "columns":
            numerical_columns

    }


# --------------------------------------------------
# Check categorical features
# --------------------------------------------------

def check_categorical_features(
    df,
    column_roles
):

    categorical_columns = [

        column

        for column in df.columns

        if column_roles.get(column)
        ==
        "categorical"

    ]


    return {

        "check":
            "categorical_features",

        "passed":
            True,

        "count":
            len(categorical_columns),

        "columns":
            categorical_columns

    }


# --------------------------------------------------
# Check constant columns
# --------------------------------------------------

def check_constant_columns(
    df
):

    constant_columns = []


    for column in df.columns:

        if (
            df[column]
            .nunique(
                dropna=False
            )
            <= 1
        ):

            constant_columns.append(
                column
            )


    return {

        "check":
            "constant_columns",

        "passed":
            len(constant_columns) == 0,

        "count":
            len(constant_columns),

        "columns":
            constant_columns

    }


# --------------------------------------------------
# Check categorical usability
# --------------------------------------------------

def check_categorical_usability(
    df,
    column_roles
):

    categorical_columns = [

        column

        for column in df.columns

        if column_roles.get(column)
        ==
        "categorical"

    ]


    column_results = {}


    for column in categorical_columns:

        unique_count = int(

            df[column]
            .nunique(
                dropna=True
            )

        )


        missing_count = int(

            df[column]
            .isnull()
            .sum()

        )


        non_missing_count = (

            len(df)
            -
            missing_count

        )


        cardinality_ratio = (

            unique_count
            /
            non_missing_count

            if non_missing_count > 0

            else 0

        )


        high_cardinality = (

            cardinality_ratio
            >
            0.50

        )


        column_results[column] = {

            "unique_count":
                unique_count,

            "missing_count":
                missing_count,

            "cardinality_ratio":
                cardinality_ratio,

            "high_cardinality":
                high_cardinality

        }


    return {

        "check":
            "categorical_usability",

        "passed":
            True,

        "count":
            len(categorical_columns),

        "columns":
            column_results

    }


# --------------------------------------------------
# Check numerical variance
#
# Uses sentinel-aware clean statistics so that
# suspicious numerical values do not distort
# ML-readiness calculations.
# --------------------------------------------------

def check_numerical_variance(
    df,
    column_roles,
    understanding
):

    numerical_columns = [

        column

        for column in df.columns

        if column_roles.get(column)
        ==
        "numerical_measurement"

    ]


    column_results = {}


    understanding_columns = {

        item["column"]:
            item

        for item
        in understanding.get(
            "column_analysis",
            []
        )

    }


    for column in numerical_columns:

        column_information = (
            understanding_columns.get(
                column,
                {}
            )
        )


        clean_statistics = (
            column_information.get(
                "clean_statistics",
                {}
            )
        )


        raw_series = (

            pd.to_numeric(
                df[column],
                errors="coerce"
            )
            .dropna()

        )


        unique_count = int(

            raw_series
            .nunique()

        )


        if clean_statistics:

            mean = float(

                clean_statistics.get(
                    "mean",
                    0
                )

            )


            standard_deviation = float(

                clean_statistics.get(
                    "standard_deviation",
                    0
                )

            )


            variance = (

                standard_deviation
                **
                2

            )


        else:

            if len(raw_series) > 1:

                mean = float(
                    raw_series.mean()
                )


                standard_deviation = float(
                    raw_series.std()
                )


                variance = float(
                    raw_series.var()
                )

            else:

                mean = (

                    float(
                        raw_series.mean()
                    )

                    if len(raw_series) > 0

                    else 0

                )


                standard_deviation = 0.0

                variance = 0.0


        low_variance = (

            unique_count <= 2

        )


        column_results[column] = {

            "unique_count":
                unique_count,

            "mean":
                mean,

            "standard_deviation":
                standard_deviation,

            "variance":
                variance,

            "low_variance":
                low_variance

        }


    return {

        "check":
            "numerical_variance",

        "passed":
            True,

        "count":
            len(numerical_columns),

        "columns":
            column_results

    }


# --------------------------------------------------
# Check identifier columns
#
# Uses the centralized column-role detector.
# --------------------------------------------------

def check_identifier_columns(
    df,
    column_roles
):

    identifier_columns = []


    column_results = {}


    for column in df.columns:

        unique_count = int(

            df[column]
            .nunique(
                dropna=True
            )

        )


        non_missing_count = int(

            df[column]
            .notna()
            .sum()

        )


        uniqueness_ratio = (

            unique_count
            /
            non_missing_count

            if non_missing_count > 0

            else 0

        )


        role = column_roles.get(

            column,

            "unknown"

        )


        potential_identifier = (

            role
            in
            {
                "identifier",
                "identifier_like"
            }

        )


        if potential_identifier:

            identifier_columns.append(
                column
            )


        column_results[column] = {

            "unique_count":
                unique_count,

            "uniqueness_ratio":
                uniqueness_ratio,

            "detected_role":
                role,

            "potential_identifier":
                potential_identifier

        }


    return {

        "check":
            "identifier_columns",

        "severity":
            "information",

        "passed":
            True,  # Never fails - identifiers are informational

        "count":
            len(identifier_columns),

        "columns":
            identifier_columns,

        "details":
            column_results,

        "message":
            "Identifier columns detected. Exclude from model features unless there is a justified predictive reason."
            if identifier_columns else "No identifier columns detected."
    }


# --------------------------------------------------
# Check sentinel and placeholder values
#
# Only high-confidence candidates are considered.
#
# This prevents weak statistical signals from
# automatically making a dataset fail ML readiness.
# --------------------------------------------------

def check_sentinel_values(
    understanding,
    sentinel_finding=None
):
    """
    Check for sentinel and placeholder values.
    
    If sentinel_finding is provided (from quality engine), it is used
    as the canonical source for sentinel counts. Otherwise, falls back
    to the understanding's possible_sentinels.
    """
    
    # Use canonical sentinel finding if available
    if sentinel_finding is not None:
        sentinel_details = sentinel_finding.get("details", {})
        sentinel_columns = sentinel_details.get("columns", {})
        affected_columns = list(sentinel_columns.keys())
        
        total_sentinel_occurrences = sentinel_details.get("total_sentinel_occurrences", 0)
        total_placeholder_occurrences = sentinel_details.get("total_placeholder_occurrences", 0)
        
        column_results = {}
        for col, info in sentinel_columns.items():
            column_results[col] = {
                "candidate_count": info.get("candidate_count", 0),
                "sentinel_occurrences": info.get("sentinel_occurrences", 0),
                "placeholder_occurrences": info.get("placeholder_occurrences", 0),
                "candidates": info.get("sentinel_candidates", []) + info.get("placeholder_candidates", []),
                "has_issue": info.get("sentinel_occurrences", 0) > 0 or info.get("placeholder_occurrences", 0) > 0
            }
    else:
        # Fallback to understanding's possible_sentinels
        affected_columns = []

        total_sentinel_occurrences = 0

        total_placeholder_occurrences = 0

        column_results = {}


        for column_information in understanding.get(
            "column_analysis",
            []
        ):

            column = column_information[
                "column"
            ]


            candidates = (
                column_information.get(
                    "possible_sentinels",
                    []
                )
            )


            high_confidence_candidates = []


            for candidate in candidates:

                confidence = candidate.get(
                    "confidence",
                    0
                )


                if (
                    confidence
                    >=
                    SENTINEL_CONFIDENCE_THRESHOLD
                ):

                    high_confidence_candidates.append(
                        candidate
                    )


            if high_confidence_candidates:

                affected_columns.append(
                    column
                )

                # Sum actual occurrences (not just unique candidates)
                sentinel_occurrences = sum(c.get("count", 0) for c in high_confidence_candidates 
                                           if c.get("classification") == "possible_sentinel")
                placeholder_occurrences = sum(c.get("count", 0) for c in high_confidence_candidates 
                                              if c.get("classification") == "possible_placeholder")

                total_sentinel_occurrences += sentinel_occurrences
                total_placeholder_occurrences += placeholder_occurrences


            column_results[column] = {

                "candidate_count":
                    len(
                        high_confidence_candidates
                    ),

                "sentinel_occurrences":
                    sum(c.get("count", 0) for c in high_confidence_candidates 
                        if c.get("classification") == "possible_sentinel"),

                "placeholder_occurrences":
                    sum(c.get("count", 0) for c in high_confidence_candidates 
                        if c.get("classification") == "possible_placeholder"),

                "candidates":
                    high_confidence_candidates,

                "has_issue":
                    len(
                        high_confidence_candidates
                    ) > 0

            }


    return {

        "check":
            "sentinel_values",

        "passed":
            total_sentinel_occurrences == 0 and total_placeholder_occurrences == 0,

        "sentinel_occurrences":
            total_sentinel_occurrences,

        "placeholder_occurrences":
            total_placeholder_occurrences,

        "total_occurrences":
            total_sentinel_occurrences + total_placeholder_occurrences,

        "affected_columns":
            affected_columns,

        "columns":
            column_results

    }


# --------------------------------------------------
# Check target column
# --------------------------------------------------

def check_target_column(
    df,
    target_column,
    task_type
):

    if target_column not in df.columns:

        return {

            "check":
                "target_column",

            "passed":
                False,

            "target_column":
                target_column,

            "task_type":
                task_type,

            "issue":
                "target_column_missing"

        }


    target = df[
        target_column
    ]


    missing_count = int(

        target
        .isnull()
        .sum()

    )


    if missing_count == len(df):

        return {

            "check":
                "target_column",

            "passed":
                False,

            "target_column":
                target_column,

            "task_type":
                task_type,

            "data_type":
                str(target.dtype),

            "unique_count":
                0,

            "missing_count":
                missing_count,

            "issue":
                "target_completely_missing"

        }


    data_type = str(
        target.dtype
    )


    unique_count = int(

        target
        .nunique(
            dropna=True
        )

    )


    if task_type == "regression":

        type_valid = (

            pd.api.types
            .is_numeric_dtype(
                target
            )

        )


    elif task_type == "classification":

        type_valid = (

            pd.api.types
            .is_object_dtype(
                target
            )

            or

            pd.api.types
            .is_string_dtype(
                target
            )

            or

            pd.api.types
            .is_categorical_dtype(
                target
            )

            or

            pd.api.types
            .is_integer_dtype(
                target
            )

            or

            pd.api.types
            .is_bool_dtype(
                target
            )

        )


    else:

        return {

            "check":
                "target_column",

            "passed":
                False,

            "target_column":
                target_column,

            "task_type":
                task_type,

            "issue":
                "unsupported_task_type"

        }


    return {

        "check":
            "target_column",

        "passed":
            type_valid,

        "target_column":
            target_column,

        "task_type":
            task_type,

        "data_type":
            data_type,

        "unique_count":
            unique_count,

        "missing_count":
            missing_count,

        "issue": (

            None

            if type_valid

            else
            "target_type_incompatible"

        )

    }


# --------------------------------------------------
# Assess ML readiness
# --------------------------------------------------

def assess_ml_readiness(
    df,
    understanding=None,
    sentinel_finding=None,
    column_metadata=None,
    column_roles=None
):
    """
    Assess whether the dataset is structurally
    suitable for machine-learning preparation.

    The function does not train a model.

    It combines structural checks with the dataset
    understanding layer so that suspicious sentinel
    values are also considered.

    If sentinel_finding is provided (from quality engine),
    it is used as the canonical source for sentinel counts.
    
    If column_metadata and column_roles are provided,
    they are used to avoid redundant DataFrame scans.
    """

    # --------------------------------------------------
    # Reuse dataset understanding when available.
    # --------------------------------------------------

    if understanding is None:
        understanding = understand_dataset(df)

    if column_roles is None:
        column_roles = analyze_column_roles(df)

    # Compute missing values from precomputed metadata if available
    if column_metadata is not None:
        missing_counts = {col: meta["missing_count"] for col, meta in column_metadata.items()}
        missing_cells = sum(missing_counts.values())
        total_cells = df.shape[0] * df.shape[1]
        missing_percentage = (missing_cells / total_cells * 100) if total_cells > 0 else 0
        missing_check = {
            "check": "missing_values",
            "passed": missing_cells == 0,
            "missing_cells": missing_cells,
            "missing_percentage": missing_percentage
        }
    else:
        missing_check = check_missing_values(df)

    # Compute constant columns from precomputed metadata if available
    if column_metadata is not None:
        constant_columns = [col for col, meta in column_metadata.items() if meta["unique_count"] <= 1]
        constant_check = {
            "check": "constant_columns",
            "passed": len(constant_columns) == 0,
            "count": len(constant_columns),
            "columns": constant_columns
        }
    else:
        constant_check = check_constant_columns(df)

    checks = [
        missing_check,
        check_numerical_features(df, column_roles),
        check_categorical_features(df, column_roles),
        constant_check,
        check_categorical_usability(df, column_roles),
        check_numerical_variance(df, column_roles, understanding),
        check_identifier_columns(df, column_roles),
        check_sentinel_values(understanding, sentinel_finding)
    ]


    # Add severity to checks that don't have it
    def _add_severity(check, default_severity="warning"):
        if "severity" not in check:
            check = dict(check)
            check["severity"] = default_severity
        return check

    checks_with_severity = [
        _add_severity(missing_check, "blocking"),
        _add_severity(check_numerical_features(df, column_roles), "warning"),
        _add_severity(check_categorical_features(df, column_roles), "information"),
        _add_severity(constant_check, "warning"),
        _add_severity(check_categorical_usability(df, column_roles), "warning"),
        _add_severity(check_numerical_variance(df, column_roles, understanding), "warning"),
        _add_severity(check_identifier_columns(df, column_roles), "information"),
        _add_severity(check_sentinel_values(understanding, sentinel_finding), "warning")
    ]


    # Calculate readiness based on severity
    # Blocking problems = 0 points, Warnings = 50 points, Information = 100 points
    severity_weights = {
        "blocking": 0,
        "warning": 50,
        "information": 100
    }
    
    total_weight = 0
    weighted_score = 0
    
    for check in checks_with_severity:
        severity = check.get("severity", "warning")
        weight = 1  # All checks have equal weight for now
        total_weight += weight
        weighted_score += severity_weights.get(severity, 50) * weight
    
    readiness_percentage = (
        weighted_score / total_weight
        if total_weight > 0
        else 0
    )

    # Also compute traditional pass/fail for backward compatibility
    passed_checks = sum(
        1 for check in checks_with_severity if check.get("passed", True)
    )
    total_checks = len(checks_with_severity)

    return {

        "readiness_percentage":
            readiness_percentage,

        "passed_checks":
            passed_checks,

        "total_checks":
            total_checks,

        "checks":
            checks_with_severity

    }


# --------------------------------------------------
# Local testing
# --------------------------------------------------
# --------------------------------------------------
# Local testing
# --------------------------------------------------

if __name__ == "__main__":

    from app.profiler.dataset_profiler import (
        load_dataset
    )


    df = load_dataset(
        "data/sample/customers.csv"
    )


    result = assess_ml_readiness(
        df
    )


    print(
        "=" * 50
    )


    print(
        "ML READINESS"
    )


    print(
        "=" * 50
    )


    print(

        f"Readiness: "
        f"{result['readiness_percentage']:.2f}%"

    )


    print(

        f"Checks Passed: "
        f"{result['passed_checks']}/"
        f"{result['total_checks']}"

    )


    print(
        "\nChecks"
    )


    print(
        "-" * 50
    )


    for check in result["checks"]:

        print(

            f"{check['check']:<25}"
            f"Passed: {check['passed']}"

        )


        if (
            check["check"]
            ==
            "sentinel_values"
        ):

            print(
                "\nSentinel Analysis"
            )


            print(
                "-" * 50
            )


            print(

                f"Candidates: "
                f"{check['count']}"

            )


            print(

                f"Numerical Sentinels: "
                f"{check['numerical_sentinels']}"

            )


            print(

                f"Text Placeholders: "
                f"{check['textual_placeholders']}"

            )


            print(

                f"Affected Columns: "
                f"{check['affected_columns']}"

            )


        if (
            check["check"]
            ==
            "categorical_usability"
        ):

            print(
                "\nCategorical Details"
            )


            print(
                "-" * 50
            )


            for (
                column,
                details
            ) in check[
                "columns"
            ].items():

                print(

                    f"{column:<15}"
                    f"Unique: "
                    f"{details['unique_count']:<5}"
                    f"Cardinality: "
                    f"{details['cardinality_ratio'] * 100:.2f}%"

                )


                print(

                    f"{'':15}"
                    f"Missing: "
                    f"{details['missing_count']:<5}"
                    f"High Cardinality: "
                    f"{details['high_cardinality']}"

                )


        if (
            check["check"]
            ==
            "numerical_variance"
        ):

            print(
                "\nNumerical Variance Details"
            )


            print(
                "-" * 50
            )


            for (
                column,
                details
            ) in check[
                "columns"
            ].items():

                print(

                    f"{column:<15}"
                    f"Unique: "
                    f"{details['unique_count']:<5}"
                    f"Mean: "
                    f"{details['mean']:<12.2f}"

                )


                print(

                    f"{'':15}"
                    f"Std Dev: "
                    f"{details['standard_deviation']:<12.2f}"
                    f"Variance: "
                    f"{details['variance']:<12.2f}"

                )


                print(

                    f"{'':15}"
                    f"Low Variance: "
                    f"{details['low_variance']}"

                )


        if (
            check["check"]
            ==
            "identifier_columns"
        ):

            print(
                "\nIdentifier Analysis"
            )


            print(
                "-" * 50
            )


            for (
                column,
                details
            ) in check[
                "details"
            ].items():

                print(

                    f"{column:<15}"
                    f"Unique: "
                    f"{details['unique_count']:<5}"
                    f"Uniqueness: "
                    f"{details['uniqueness_ratio'] * 100:.2f}%"

                )


                print(

                    f"{'':15}"
                    f"Detected Role: "
                    f"{details['detected_role']}"

                )


                print(

                    f"{'':15}"
                    f"Potential Identifier: "
                    f"{details['potential_identifier']}"

                )
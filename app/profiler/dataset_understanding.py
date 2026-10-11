from app.profiler.column_roles import (
    analyze_column_roles
)

from app.profiler.sentinel_detector import (
    detect_possible_sentinels,
    get_cached_sentinel_detection
)

from app.profiler.missing_values import count_missing_values

import pandas as pd


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MAX_UNIQUE_VALUES_TO_ANALYZE = 10000


# --------------------------------------------------
# Detect basic column type
# --------------------------------------------------

def detect_column_type(
    series
):

    if pd.api.types.is_datetime64_any_dtype(
        series
    ):

        return "datetime"


    if pd.api.types.is_numeric_dtype(
        series
    ):

        return "numerical"


    if pd.api.types.is_bool_dtype(
        series
    ):

        return "boolean"


    return "categorical"


# --------------------------------------------------
# Calculate numerical statistics
# --------------------------------------------------

def calculate_numerical_statistics(
    series
):

    numeric_series = (
        pd.to_numeric(
            series,
            errors="coerce"
        )
        .dropna()
    )


    if numeric_series.empty:

        return {}


    q1 = (
        numeric_series
        .quantile(0.25)
    )


    q3 = (
        numeric_series
        .quantile(0.75)
    )


    iqr = (
        q3 - q1
    )


    mean = (
        numeric_series
        .mean()
    )


    median = (
        numeric_series
        .median()
    )


    standard_deviation = (
        numeric_series
        .std()
    )


    return {

        "min":
            float(
                numeric_series.min()
            ),

        "max":
            float(
                numeric_series.max()
            ),

        "mean":
            float(mean),

        "median":
            float(median),

        "standard_deviation":
            float(
                standard_deviation
            )
            if pd.notna(
                standard_deviation
            )
            else 0.0,

        "q1":
            float(q1),

        "q3":
            float(q3),

        "iqr":
            float(iqr)

    }


# --------------------------------------------------
# Remove high-confidence numerical sentinels
#
# This creates a clean statistical view without
# modifying the original dataset.
# --------------------------------------------------

def remove_detected_numerical_sentinels(
    series,
    possible_sentinels
):

    numeric_series = (
        pd.to_numeric(
            series,
            errors="coerce"
        )
    )


    if numeric_series.empty:

        return numeric_series


    values_to_remove = []


    for candidate in possible_sentinels:

        confidence = candidate.get(
            "confidence",
            0
        )


        classification = candidate.get(
            "classification"
        )


        if (
            classification
            ==
            "possible_sentinel"
            and
            confidence >= 0.8
        ):

            values_to_remove.append(
                candidate["value"]
            )


    if not values_to_remove:

        return numeric_series


    clean_series = (
        numeric_series[
            ~numeric_series.isin(
                values_to_remove
            )
        ]
        .dropna()
    )


    return clean_series


# --------------------------------------------------
# Analyze one column
# --------------------------------------------------

def analyze_column(
    column_name,
    series,
    column_role
):

    column_type = detect_column_type(
        series
    )


    total_count = len(series)

    total_missing, empty_count, nan_count = count_missing_values(series)

    non_null_count = total_count - total_missing

    unique_count = (
        int(
            series.nunique(
                dropna=True
            )
        )
    )


    uniqueness_ratio = (
        unique_count
        /
        total_count
        if total_count > 0
        else 0
    )


    raw_statistics = {}

    clean_statistics = {}

    possible_sentinels = []


    # --------------------------------------------------
    # Detect possible sentinels first.
    #
    # The result is then used to build a clean
    # statistical view.
    # --------------------------------------------------

    if unique_count <= (
        MAX_UNIQUE_VALUES_TO_ANALYZE
    ):

        possible_sentinels = (
            get_cached_sentinel_detection(
                series, column_name
            )
        )


    # --------------------------------------------------
    # Numerical analysis
    # --------------------------------------------------

    if column_type == "numerical":

        raw_statistics = (
            calculate_numerical_statistics(
                series
            )
        )


        clean_series = (
            remove_detected_numerical_sentinels(
                series,
                possible_sentinels
            )
        )


        clean_statistics = (
            calculate_numerical_statistics(
                clean_series
            )
        )


    # --------------------------------------------------
    # Categorical analysis
    # --------------------------------------------------

    elif column_type == "categorical":

        raw_statistics = {}
        clean_statistics = {}

    # --------------------------------------------------
    # Datetime analysis
    # --------------------------------------------------

    elif column_type == "datetime":

        raw_statistics = {}
        clean_statistics = {}

    # --------------------------------------------------
    # Boolean analysis
    # --------------------------------------------------

    elif column_type == "boolean":

        raw_statistics = {}
        clean_statistics = {}

    # --------------------------------------------------
    # Other types (email, postal_code, identifier, etc.)
    # --------------------------------------------------

    else:

        raw_statistics = {}
        clean_statistics = {}


    return {
        "column":
            column_name,

        "type":
            column_type,

        "role":
            column_role,

        "rows":
            total_count,

        "non_null_count":
            non_null_count,

        "missing_count":
            total_missing,

        "missing_percentage": (
            (
                total_missing
                /
                total_count
            )
            *
            100
            if total_count > 0
            else 0
        ),

        "unique_count":
            unique_count,

        "uniqueness_ratio":
            uniqueness_ratio,

        "raw_statistics":
            raw_statistics,

        "clean_statistics":
            clean_statistics,

        "possible_sentinels":
            possible_sentinels

    }


# --------------------------------------------------
# Understand complete dataset
# --------------------------------------------------

def understand_dataset(
    df
):

    # --------------------------------------------------
    # Determine column roles first.
    # --------------------------------------------------

    column_roles = (
        analyze_column_roles(
            df
        )
    )


    columns = []


    # --------------------------------------------------
    # Analyze every column.
    # --------------------------------------------------

    for column in df.columns:

        column_information = (
            analyze_column(
                column_name=column,
                series=df[column],
                column_role=column_roles.get(
                    column,
                    "unknown"
                )
            )
        )


        columns.append(
            column_information
        )


    return {

        "rows":
            len(df),

        "columns":
            len(df.columns),

        "column_analysis":
            columns

    }
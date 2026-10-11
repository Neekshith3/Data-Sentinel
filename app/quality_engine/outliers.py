import pandas as pd

from app.profiler.sentinel_detector import (
    detect_possible_sentinels,
    get_cached_sentinel_detection
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

SENTINEL_CONFIDENCE_THRESHOLD = 0.8


# --------------------------------------------------
# Get values that should be excluded from
# statistical outlier calculations.
# --------------------------------------------------

def get_sentinel_values(
    series,
    column_name=None
):

    possible_sentinels = (
        get_cached_sentinel_detection(
            series,
            column_name or "unknown"
        )
    )

    sentinel_values = []

    for candidate in possible_sentinels:

        classification = (
            candidate.get(
                "classification"
            )
        )

        confidence = (
            candidate.get(
                "confidence",
                0
            )
        )

        if (
            classification
            ==
            "possible_sentinel"
            and
            confidence
            >=
            SENTINEL_CONFIDENCE_THRESHOLD
        ):

            sentinel_values.append(
                candidate["value"]
            )


    return sentinel_values


# --------------------------------------------------
# Detect IQR outliers
# --------------------------------------------------

def detect_iqr_outliers(
    df
):
    """
    Detect statistical outliers in numerical columns
    using the IQR method.

    Possible high-confidence sentinel values are
    excluded from the statistical calculation.

    The original DataFrame is never modified.

    Parameters:
        df (pandas.DataFrame): Dataset to analyze.

    Returns:
        dict: IQR outlier detection results.
    """

    results = {}


    numeric_columns = (
        df.select_dtypes(
            include="number"
        ).columns
    )


    for column in numeric_columns:

        series = (
            df[column]
            .dropna()
        )


        if series.empty:

            continue


        # --------------------------------------------------
        # Detect possible sentinel values.
        # --------------------------------------------------

        sentinel_values = (
            get_sentinel_values(
                df[column]
            )
        )


        # --------------------------------------------------
        # Create clean statistical series.
        #
        # Only suspicious sentinel values are removed.
        # The original DataFrame remains unchanged.
        # --------------------------------------------------

        if sentinel_values:

            statistical_series = (
                series[
                    ~series.isin(
                        sentinel_values
                    )
                ]
            )

        else:

            statistical_series = series


        if statistical_series.empty:

            results[column] = {

                "q1": None,

                "q3": None,

                "iqr": None,

                "lower_bound": None,

                "upper_bound": None,

                "outlier_count": 0,

                "has_issue": False,

                "outlier_rows":
                    df.iloc[0:0],

                "excluded_sentinel_values":
                    sentinel_values

            }

            continue


        # --------------------------------------------------
        # Calculate IQR using clean statistical values.
        # --------------------------------------------------

        q1 = (
            statistical_series
            .quantile(0.25)
        )


        q3 = (
            statistical_series
            .quantile(0.75)
        )


        iqr = (
            q3 - q1
        )


        lower_bound = (
            q1
            -
            (
                1.5
                *
                iqr
            )
        )


        upper_bound = (
            q3
            +
            (
                1.5
                *
                iqr
            )
        )


        # --------------------------------------------------
        # Identify outliers only among values that
        # were not classified as high-confidence
        # sentinels.
        # --------------------------------------------------

        valid_mask = (
            df[column].notna()
        )


        if sentinel_values:

            valid_mask = (
                valid_mask
                &
                ~df[column].isin(
                    sentinel_values
                )
            )


        outlier_mask = (
            valid_mask
            &
            (
                (
                    df[column]
                    <
                    lower_bound
                )
                |
                (
                    df[column]
                    >
                    upper_bound
                )
            )
        )


        outlier_rows = (
            df[outlier_mask]
        )


        results[column] = {

            "q1":
                float(q1),

            "q3":
                float(q3),

            "iqr":
                float(iqr),

            "lower_bound":
                float(
                    lower_bound
                ),

            "upper_bound":
                float(
                    upper_bound
                ),

            "outlier_count":
                int(
                    len(outlier_rows)
                ),

            "has_issue":
                len(outlier_rows) > 0,

            "outlier_rows":
                outlier_rows,

            "excluded_sentinel_values":
                sentinel_values

        }


    return {

        "issue_type":
            "statistical_outliers",

        "has_issue":
            any(
                result[
                    "has_issue"
                ]

                for result
                in results.values()
            ),

        "columns":
            results

    }
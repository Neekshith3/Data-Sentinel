# app/quality_engine/statistical_drift.py


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DRIFT_THRESHOLD = 0.20


# --------------------------------------------------
# Calculate relative change
# --------------------------------------------------

def calculate_relative_change(
    previous_value,
    current_value
):
    """
    Calculate the relative change between two values.

    Example:

        Previous = 100
        Current = 120

        Relative change = 0.20

    The result is returned as a decimal.
    """

    if previous_value is None:
        return None


    if current_value is None:
        return None


    previous_value = float(
        previous_value
    )


    current_value = float(
        current_value
    )


    # --------------------------------------------------
    # When the previous value is zero, normal relative
    # change is undefined.
    # --------------------------------------------------

    if previous_value == 0:

        if current_value == 0:

            return 0.0


        return None


    return (
        abs(
            current_value
            -
            previous_value
        )
        /
        abs(
            previous_value
        )
    )


# --------------------------------------------------
# Check whether a value has drifted
# --------------------------------------------------

def has_significant_change(
    previous_value,
    current_value,
    threshold=DRIFT_THRESHOLD
):

    relative_change = (
        calculate_relative_change(
            previous_value,
            current_value
        )
    )


    if relative_change is None:

        return False


    return (
        relative_change
        >=
        threshold
    )


# --------------------------------------------------
# Compare one numerical column
# --------------------------------------------------

def compare_numerical_column(
    column_name,
    previous_column,
    current_column,
    threshold=DRIFT_THRESHOLD
):
    """
    Compare numerical statistics for one column.

    The following statistics are compared:

        - mean
        - median
        - standard deviation

    The function does not modify either snapshot.
    """

    previous_statistics = (
        previous_column.get(
            "clean_statistics",
            {}
        )
    )


    current_statistics = (
        current_column.get(
            "clean_statistics",
            {}
        )
    )


    statistics_to_compare = [
        "mean",
        "median",
        "standard_deviation"
    ]


    changes = []


    for statistic in (
        statistics_to_compare
    ):

        previous_value = (
            previous_statistics.get(
                statistic
            )
        )


        current_value = (
            current_statistics.get(
                statistic
            )
        )


        if (
            previous_value is None
            or
            current_value is None
        ):

            continue


        relative_change = (
            calculate_relative_change(
                previous_value,
                current_value
            )
        )


        if relative_change is None:

            continue


        significant = (
            relative_change
            >=
            threshold
        )


        changes.append({

            "statistic":
                statistic,

            "previous_value":
                previous_value,

            "current_value":
                current_value,

            "relative_change":
                relative_change,

            "relative_change_percentage":
                relative_change * 100,

            "significant":
                significant

        })


    significant_changes = [

        change

        for change in changes

        if change["significant"]

    ]


    return {

        "column":
            column_name,

        "has_drift":
            len(
                significant_changes
            ) > 0,

        "changes":
            changes,

        "significant_changes":
            significant_changes

    }


# --------------------------------------------------
# Build column map
# --------------------------------------------------

def build_column_map(
    snapshot
):

    column_map = {}


    for column in snapshot.get(
        "columns",
        []
    ):

        column_name = (
            column.get(
                "name"
            )
        )


        if column_name is None:

            continue


        column_map[
            column_name
        ] = column


    return column_map


# --------------------------------------------------
# Detect numerical statistical drift
# --------------------------------------------------

def detect_numerical_statistical_drift(
    previous_snapshot,
    current_snapshot,
    threshold=DRIFT_THRESHOLD
):
    """
    Compare numerical statistics between two
    dataset snapshots.

    This detector focuses on:

        - mean changes
        - median changes
        - standard-deviation changes

    Only columns that are numerical in both
    snapshots are compared.
    """

    previous_columns = (
        build_column_map(
            previous_snapshot
        )
    )


    current_columns = (
        build_column_map(
            current_snapshot
        )
    )


    common_columns = (

        set(previous_columns)
        &
        set(current_columns)

    )


    column_results = []


    for column_name in sorted(
        common_columns
    ):

        previous_column = (
            previous_columns[
                column_name
            ]
        )


        current_column = (
            current_columns[
                column_name
            ]
        )


        previous_type = (
            previous_column.get(
                "type"
            )
        )


        current_type = (
            current_column.get(
                "type"
            )
        )


        # --------------------------------------------------
        # Only compare columns that remain numerical
        # in both dataset versions.
        # --------------------------------------------------

        if (
            previous_type != "numerical"
            or
            current_type != "numerical"
        ):

            continue


        comparison = (
            compare_numerical_column(

                column_name,

                previous_column,

                current_column,

                threshold

            )
        )


        column_results.append(
            comparison
        )


    drifted_columns = [

        result

        for result in column_results

        if result["has_drift"]

    ]


    return {

        "issue_type":
            "statistical_drift",

        "has_issue":
            len(
                drifted_columns
            ) > 0,

        "drifted_column_count":
            len(
                drifted_columns
            ),

        "compared_column_count":
            len(
                column_results
            ),

        "threshold":
            threshold,

        "columns":
            column_results

    }
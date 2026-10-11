# app/quality_engine/categorical_drift.py


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DISTRIBUTION_DRIFT_THRESHOLD = 0.20


# --------------------------------------------------
# Build category distribution map
# --------------------------------------------------

def get_category_distribution(
    column_snapshot
):
    """
    Return the categorical distribution stored
    inside a column snapshot.
    """

    return column_snapshot.get(
        "categorical_distribution",
        {}
    )


# --------------------------------------------------
# Calculate absolute percentage-point change
# --------------------------------------------------

def calculate_percentage_point_change(
    previous_percentage,
    current_percentage
):
    """
    Calculate the absolute change in percentage points.

    Example:

        Previous = 70%
        Current = 40%

        Change = 30 percentage points
    """

    return abs(
        float(current_percentage)
        -
        float(previous_percentage)
    )


# --------------------------------------------------
# Detect significant category change
# --------------------------------------------------

def has_significant_category_change(
    previous_percentage,
    current_percentage,
    threshold=DISTRIBUTION_DRIFT_THRESHOLD
):
    """
    Determine whether a category changed significantly.

    The threshold is applied to percentage points.

    A threshold of 0.20 means a 20 percentage-point
    change is considered significant.
    """

    percentage_point_change = (
        calculate_percentage_point_change(

            previous_percentage,

            current_percentage

        )
    )


    return (
        percentage_point_change
        >=
        threshold * 100
    )


# --------------------------------------------------
# Compare one categorical column
# --------------------------------------------------

def compare_categorical_column(
    column_name,
    previous_column,
    current_column,
    threshold=DISTRIBUTION_DRIFT_THRESHOLD
):
    """
    Compare categorical distributions between two
    dataset snapshots.

    Detects:

        - newly appearing categories
        - disappearing categories
        - significant frequency changes
    """

    previous_distribution = (
        get_category_distribution(
            previous_column
        )
    )


    current_distribution = (
        get_category_distribution(
            current_column
        )
    )


    previous_categories = set(
        previous_distribution
    )


    current_categories = set(
        current_distribution
    )


    added_categories = sorted(

        current_categories
        -
        previous_categories

    )


    removed_categories = sorted(

        previous_categories
        -
        current_categories

    )


    common_categories = (

        previous_categories
        &
        current_categories

    )


    distribution_changes = []


    for category in sorted(
        common_categories
    ):

        previous_percentage = (
            previous_distribution[
                category
            ].get(
                "percentage",
                0
            )
        )


        current_percentage = (
            current_distribution[
                category
            ].get(
                "percentage",
                0
            )
        )


        percentage_point_change = (
            calculate_percentage_point_change(

                previous_percentage,

                current_percentage

            )
        )


        significant = (
            has_significant_category_change(

                previous_percentage,

                current_percentage,

                threshold

            )
        )


        distribution_changes.append({

            "category":
                category,

            "previous_percentage":
                previous_percentage,

            "current_percentage":
                current_percentage,

            "percentage_point_change":
                percentage_point_change,

            "significant":
                significant

        })


    significant_changes = [

        change

        for change in distribution_changes

        if change["significant"]

    ]


    has_drift = (

        len(added_categories) > 0

        or

        len(removed_categories) > 0

        or

        len(significant_changes) > 0

    )


    return {

        "column":
            column_name,

        "has_drift":
            has_drift,

        "added_categories":
            added_categories,

        "removed_categories":
            removed_categories,

        "distribution_changes":
            distribution_changes,

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
# Detect categorical distribution drift
# --------------------------------------------------

def detect_categorical_distribution_drift(
    previous_snapshot,
    current_snapshot,
    threshold=DISTRIBUTION_DRIFT_THRESHOLD
):
    """
    Compare categorical distributions between two
    dataset snapshots.

    Only columns that are categorical in both
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
        # Only compare columns that remain categorical
        # in both dataset versions.
        # --------------------------------------------------

        if (
            previous_type != "categorical"
            or
            current_type != "categorical"
        ):

            continue


        comparison = (
            compare_categorical_column(

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
            "categorical_distribution_drift",

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
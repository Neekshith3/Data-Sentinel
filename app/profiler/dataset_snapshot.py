from datetime import datetime, timezone


# --------------------------------------------------
# Snapshot configuration
# --------------------------------------------------

SNAPSHOT_VERSION = "1.1"

MAX_CATEGORICAL_VALUES = 20


# --------------------------------------------------
# Create categorical distribution
# --------------------------------------------------

def create_categorical_distribution(
    series
):
    """
    Create a compact categorical distribution.

    Only the most frequent categories are stored.

    Remaining categories are grouped into
    __OTHER__ so the snapshot remains compact.
    """

    values = (
        series
        .dropna()
        .astype(str)
        .str.strip()
    )


    if values.empty:

        return {}


    value_counts = (
        values
        .value_counts()
    )


    total_count = (
        int(
            value_counts.sum()
        )
    )


    if total_count == 0:

        return {}


    top_values = (
        value_counts
        .head(
            MAX_CATEGORICAL_VALUES
        )
    )


    distribution = {}


    for value, count in (
        top_values.items()
    ):

        distribution[str(value)] = {

            "count":
                int(count),

            "percentage":
                (
                    float(count)
                    /
                    total_count
                    *
                    100
                )

        }


    # --------------------------------------------------
    # Group categories outside the top-N list.
    # --------------------------------------------------

    remaining_count = (
        total_count
        -
        int(
            top_values.sum()
        )
    )


    if remaining_count > 0:

        distribution[
            "__OTHER__"
        ] = {

            "count":
                remaining_count,

            "percentage":
                (
                    float(
                        remaining_count
                    )
                    /
                    total_count
                    *
                    100
                )

        }


    return distribution


# --------------------------------------------------
# Create a compact snapshot of one column
# --------------------------------------------------

def create_column_snapshot(
    column_information,
    series=None
):
    """
    Convert detailed dataset-understanding information
    for one column into a compact snapshot.

    The snapshot stores metadata and statistics only.

    It does not store the original column values.
    """

    snapshot = {

        "name":
            column_information.get(
                "column"
            ),

        "type":
            column_information.get(
                "type"
            ),

        "role":
            column_information.get(
                "role"
            ),

        "rows":
            column_information.get(
                "rows"
            ),

        "non_null_count":
            column_information.get(
                "non_null_count"
            ),

        "missing_count":
            column_information.get(
                "missing_count"
            ),

        "missing_percentage":
            column_information.get(
                "missing_percentage"
            ),

        "unique_count":
            column_information.get(
                "unique_count"
            ),

        "uniqueness_ratio":
            column_information.get(
                "uniqueness_ratio"
            ),

        "raw_statistics":
            column_information.get(
                "raw_statistics",
                {}
            ),

        "clean_statistics":
            column_information.get(
                "clean_statistics",
                {}
            )

    }


    # --------------------------------------------------
    # Store compact sentinel information.
    # --------------------------------------------------

    possible_sentinels = (
        column_information.get(
            "possible_sentinels",
            []
        )
    )


    sentinel_summary = []


    for candidate in (
        possible_sentinels
    ):

        sentinel_summary.append({

            "value":
                candidate.get(
                    "value"
                ),

            "count":
                candidate.get(
                    "count"
                ),

            "confidence":
                candidate.get(
                    "confidence"
                ),

            "classification":
                candidate.get(
                    "classification"
                )

        })


    snapshot[
        "possible_sentinels"
    ] = sentinel_summary


    # --------------------------------------------------
    # Store categorical distribution.
    #
    # Only categorical columns need this information.
    # --------------------------------------------------

    if (
        series is not None
        and
        column_information.get(
            "type"
        ) == "categorical"
    ):

        snapshot[
            "categorical_distribution"
        ] = (
            create_categorical_distribution(
                series
            )
        )

    else:

        snapshot[
            "categorical_distribution"
        ] = {}


    return snapshot


# --------------------------------------------------
# Create complete dataset snapshot
# --------------------------------------------------

def create_dataset_snapshot(
    df,
    profile,
    understanding
):
    """
    Create a compact representation of the dataset
    that can be stored as a historical baseline.

    The snapshot contains:

        - dataset dimensions
        - column structure
        - column roles
        - data types
        - missing-value statistics
        - uniqueness statistics
        - numerical statistics
        - categorical distributions
        - sentinel summaries

    The original dataset values are not stored.
    """

    column_analysis = (
        understanding.get(
            "column_analysis",
            []
        )
    )


    columns = []


    for column_information in (
        column_analysis
    ):

        column_name = (
            column_information.get(
                "column"
            )
        )


        columns.append(

            create_column_snapshot(

                column_information,

                series=df[
                    column_name
                ]

            )

        )


    snapshot = {

        "snapshot_version":
            SNAPSHOT_VERSION,

        "created_at":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "dataset": {

            "rows":
                profile.get(
                    "rows",
                    len(df)
                ),

            "columns":
                profile.get(
                    "columns",
                    len(df.columns)
                ),

            "column_names":
                profile.get(
                    "column_names",
                    df.columns.tolist()
                )

        },

        "columns":
            columns

    }


    return snapshot
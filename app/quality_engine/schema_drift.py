# app/quality_engine/schema_drift.py


# --------------------------------------------------
# Normalize column information
# --------------------------------------------------

def build_column_map(
    snapshot
):
    """
    Convert the snapshot column list into a dictionary
    keyed by column name.

    This makes comparison between two snapshots
    efficient and easy to understand.
    """

    column_map = {}


    for column in snapshot.get(
        "columns",
        []
    ):

        column_name = column.get(
            "name"
        )


        if column_name is None:

            continue


        column_map[column_name] = (
            column
        )


    return column_map


# --------------------------------------------------
# Detect added columns
# --------------------------------------------------

def detect_added_columns(
    previous_columns,
    current_columns
):

    return sorted(

        set(current_columns)
        -
        set(previous_columns)

    )


# --------------------------------------------------
# Detect removed columns
# --------------------------------------------------

def detect_removed_columns(
    previous_columns,
    current_columns
):

    return sorted(

        set(previous_columns)
        -
        set(current_columns)

    )


# --------------------------------------------------
# Detect data-type changes
# --------------------------------------------------

def detect_type_changes(
    previous_map,
    current_map
):

    changes = []


    common_columns = (

        set(previous_map)
        &
        set(current_map)

    )


    for column in sorted(
        common_columns
    ):

        previous_type = (
            previous_map[column]
            .get("type")
        )


        current_type = (
            current_map[column]
            .get("type")
        )


        if previous_type != current_type:

            changes.append({

                "column":
                    column,

                "previous_type":
                    previous_type,

                "current_type":
                    current_type

            })


    return changes


# --------------------------------------------------
# Detect semantic role changes
# --------------------------------------------------

def detect_role_changes(
    previous_map,
    current_map
):

    changes = []


    common_columns = (

        set(previous_map)
        &
        set(current_map)

    )


    for column in sorted(
        common_columns
    ):

        previous_role = (
            previous_map[column]
            .get("role")
        )


        current_role = (
            current_map[column]
            .get("role")
        )


        if previous_role != current_role:

            changes.append({

                "column":
                    column,

                "previous_role":
                    previous_role,

                "current_role":
                    current_role

            })


    return changes


# --------------------------------------------------
# Detect column order changes
# --------------------------------------------------

def detect_column_order_changes(
    previous_snapshot,
    current_snapshot
):

    previous_columns = (
        previous_snapshot
        .get(
            "dataset",
            {}
        )
        .get(
            "column_names",
            []
        )
    )


    current_columns = (
        current_snapshot
        .get(
            "dataset",
            {}
        )
        .get(
            "column_names",
            []
        )
    )


    # --------------------------------------------------
    # Order is only meaningful when the same columns
    # exist in both snapshots.
    # --------------------------------------------------

    if (
        set(previous_columns)
        !=
        set(current_columns)
    ):

        return {

            "changed":
                False,

            "previous_order":
                previous_columns,

            "current_order":
                current_columns,

            "reason":
                "column_sets_are_different"

        }


    changed = (

        previous_columns
        !=
        current_columns

    )


    return {

        "changed":
            changed,

        "previous_order":
            previous_columns,

        "current_order":
            current_columns

    }


# --------------------------------------------------
# Detect schema drift
# --------------------------------------------------

def detect_schema_drift(
    previous_snapshot,
    current_snapshot
):
    """
    Compare two dataset snapshots.

    This detector focuses only on structural changes:

        - added columns
        - removed columns
        - data-type changes
        - semantic role changes
        - column-order changes

    It does not modify either snapshot.
    """

    previous_dataset = (
        previous_snapshot.get(
            "dataset",
            {}
        )
    )


    current_dataset = (
        current_snapshot.get(
            "dataset",
            {}
        )
    )


    previous_columns = (
        previous_dataset.get(
            "column_names",
            []
        )
    )


    current_columns = (
        current_dataset.get(
            "column_names",
            []
        )
    )


    previous_map = build_column_map(
        previous_snapshot
    )


    current_map = build_column_map(
        current_snapshot
    )


    added_columns = (
        detect_added_columns(
            previous_columns,
            current_columns
        )
    )


    removed_columns = (
        detect_removed_columns(
            previous_columns,
            current_columns
        )
    )


    type_changes = (
        detect_type_changes(
            previous_map,
            current_map
        )
    )


    role_changes = (
        detect_role_changes(
            previous_map,
            current_map
        )
    )


    column_order = (
        detect_column_order_changes(
            previous_snapshot,
            current_snapshot
        )
    )


    has_schema_drift = (

        len(added_columns) > 0

        or

        len(removed_columns) > 0

        or

        len(type_changes) > 0

        or

        len(role_changes) > 0

        or

        column_order["changed"]

    )


    return {

        "issue_type":
            "schema_drift",

        "has_issue":
            has_schema_drift,

        "added_columns":
            added_columns,

        "removed_columns":
            removed_columns,

        "type_changes":
            type_changes,

        "role_changes":
            role_changes,

        "column_order":
            column_order

    }
import pandas as pd


def detect_duplicates(df):
    """
    Detect duplicate rows in a dataset.

    Parameters:
        df (pandas.DataFrame): Dataset to analyze.

    Returns:
        dict: Duplicate detection result.
    """

    duplicate_mask = df.duplicated()

    duplicate_rows = df[duplicate_mask]

    duplicate_count = len(duplicate_rows)

    # Also count all rows involved in duplicate groups (including first occurrence)
    duplicate_rows_involved = int(df.duplicated(keep=False).sum())

    result = {
        "issue_type": "duplicate_rows",
        "duplicate_count": duplicate_count,
        "duplicate_rows_involved": duplicate_rows_involved,
        "duplicate_percentage": (
            duplicate_count / len(df) * 100
            if len(df) > 0
            else 0
        ),
        "has_issue": duplicate_count > 0,
        "duplicate_rows": duplicate_rows
    }

    return result
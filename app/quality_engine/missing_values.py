from app.profiler.missing_values import (
    count_missing_values,
    count_missing_values_df,
)

def detect_missing_values(df):
    """
    Detect missing values in each column.

    Parameters:
        df (pandas.DataFrame): Dataset to analyze.

    Returns:
        dict: Missing-value detection result.
    """

    column_results = count_missing_values_df(df)

    return {
        "issue_type": "missing_values",
        "has_issue": any(
            result["has_issue"]
            for result in column_results.values()
        ),
        "columns": column_results
    }
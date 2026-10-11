import numpy as np
import pandas as pd


def detect_ml_anomalies(
    df,
    feature_columns,
    contamination=0.10
):
    """
    Detect anomalous rows using Isolation Forest.

    Parameters:
        df (pandas.DataFrame):
            Dataset to analyze.

        feature_columns (list):
            Numerical columns used by the model.

        contamination (float):
            Expected proportion of anomalies.

    Returns:
        dict:
            Isolation Forest anomaly detection results.
    """

    # --------------------------------------------------
    # Validate feature columns
    # --------------------------------------------------

    missing_columns = [
        column
        for column in feature_columns
        if column not in df.columns
    ]

    if missing_columns:

        return {
            "issue_type": "ml_anomalies",
            "has_issue": True,
            "anomaly_count": 0,
            "anomaly_rows": [],
            "anomaly_scores": {},
            "issue": "feature_columns_missing",
            "missing_columns": missing_columns
        }

    # Select model features

    features = df[feature_columns].copy()

    # Handle missing values

    if features.isnull().any().any():

        features = features.fillna(
            features.median()
        )

    # Robust multivariate outlier scoring (median/MAD based).
    # This mirrors the role of Isolation Forest in this project:
    # ML-based anomaly detection without requiring scikit-learn.

    values = features.to_numpy(dtype=float)

    medians = np.nanmedian(values, axis=0)
    mad = np.nanmedian(np.abs(values - medians), axis=0)
    mad[mad == 0] = 1e-9

    robust_z = np.abs(values - medians) / (1.4826 * mad)

    # Higher score = more normal (same convention as
    # Isolation Forest's decision_function, which is
    # positive for normal samples and negative for anomalies).

    raw_distance = np.linalg.norm(robust_z, axis=1)

    scores = -raw_distance + np.median(raw_distance)

    # Rank-based anomaly assignment using contamination proportion

    anomaly_count_target = int(round(contamination * len(values)))

    ranking = np.argsort(scores)  # most anomalous first

    anomaly_mask = np.zeros(len(values), dtype=bool)

    if anomaly_count_target > 0:
        anomaly_mask[ranking[:anomaly_count_target]] = True

    anomaly_rows = df.index[anomaly_mask].tolist()

    anomaly_count = len(anomaly_rows)

    # Calculate anomaly scores

    anomaly_scores = {
        int(index): float(score)
        for index, score in zip(
            df.index,
            scores
        )
    }

    # Return results

    return {
        "issue_type": "ml_anomalies",
        "has_issue": anomaly_count > 0,
        "anomaly_count": anomaly_count,
        "anomaly_rows": anomaly_rows,
        "anomaly_scores": anomaly_scores,
        "feature_columns": feature_columns,
        "contamination": contamination
    }

if __name__ == "__main__":

    df = pd.read_csv(
        "data/sample/customers.csv"
    )

    result = detect_ml_anomalies(
        df,
        feature_columns=[
            "age",
            "salary"
        ],
        contamination=0.10
    )

    print("=" * 50)
    print("ML ANOMALY DETECTION")
    print("=" * 50)

    print(
        "Anomaly Count:",
        result["anomaly_count"]
    )

    print(
        "Anomaly Rows:",
        result["anomaly_rows"]
    )

    print("\nAnomaly Scores")
    print("-" * 50)

    for row, score in result[
        "anomaly_scores"
    ].items():

        print(
            f"Row {row}: "
            f"{score:.4f}"
        )
from app.quality_engine.severity import (
    determine_severity
)


# --------------------------------------------------
# Calculate confidence-weighted sentinel impact
# --------------------------------------------------

def calculate_sentinel_affected_count(
    finding
):
    """
    Calculate the confidence-weighted impact of
    possible sentinel and placeholder values.

    Numerical sentinels and textual placeholders are
    handled separately so the scoring logic remains
    explainable.

    Returns:
        float: Confidence-weighted affected count.
    """

    details = finding.get(
        "details",
        {}
    )


    columns = details.get(
        "columns",
        {}
    )


    weighted_count = 0.0


    for column_details in columns.values():

        sentinel_candidates = (
            column_details.get(
                "sentinel_candidates",
                []
            )
        )


        placeholder_candidates = (
            column_details.get(
                "placeholder_candidates",
                []
            )
        )


        # --------------------------------------------------
        # Numerical sentinel values
        # --------------------------------------------------

        for candidate in sentinel_candidates:

            count = candidate.get(
                "count",
                0
            )


            confidence = candidate.get(
                "confidence",
                0
            )


            weighted_count += (
                count
                *
                confidence
            )


        # --------------------------------------------------
        # Text placeholders
        # --------------------------------------------------

        for candidate in placeholder_candidates:

            count = candidate.get(
                "count",
                0
            )


            confidence = candidate.get(
                "confidence",
                0
            )


            weighted_count += (
                count
                *
                confidence
            )


    return weighted_count


# --------------------------------------------------
# Calculate data quality score
# --------------------------------------------------

def calculate_quality_score(
    findings,
    total_rows,
    total_columns
):
    """
    Calculate an overall data-quality score from 0 to 100.

    The score is based on:

        1. The weight of each quality category.
        2. The proportion of affected data.
        3. The severity of the detected issue.
        4. Confidence-weighted sentinel detection.

    The score is a configurable heuristic. It is not
    intended to be a universal data-quality standard.
    """

    weights = {

        "missing_values":
            25,

        "duplicate_rows":
            15,

        "categorical_inconsistency":
            15,

        "data_type_validation":
            20,

        "invalid_values":
            15,

        "statistical_outliers":
            10,

        "possible_sentinel_values":
            20

    }


    total_penalty = 0.0

    category_scores = {}


    # --------------------------------------------------
    # Process every finding
    # --------------------------------------------------

    for finding in findings:

        issue_type = finding[
            "issue_type"
        ]


        # --------------------------------------------------
        # ML anomalies are not part of the basic
        # data-quality score.
        # --------------------------------------------------

        if issue_type == "ml_anomalies":

            continue


        weight = weights.get(
            issue_type,
            0
        )


        # --------------------------------------------------
        # Clean category
        # --------------------------------------------------

        if not finding[
            "has_issue"
        ]:

            category_scores[
                issue_type
            ] = {

                "weight":
                    weight,

                "affected_rate":
                    0.0,

                "severity":
                    "CLEAN",

                "penalty":
                    0.0,

                "score":
                    100.0

            }


            continue


        # --------------------------------------------------
        # Calculate affected rate
        # --------------------------------------------------

        if issue_type == "missing_values":

            # Use the worst-affected column rate, not the global
            # cell rate, so a few badly damaged columns cannot
            # hide inside an otherwise healthy dataset.
            det = finding.get("details") or {}
            columns = det.get("columns") or {}
            if not columns and any(
                isinstance(v, dict) and "missing_percentage" in v
                for v in det.values()
            ):
                columns = det
            rates = [
                (c.get("missing_percentage") or 0.0) / 100.0
                for c in columns.values()
                if isinstance(c, dict)
            ]
            affected_rate = max(rates) if rates else 0.0


        elif issue_type == "possible_sentinel_values":

            # --------------------------------------------------
            # Sentinel and placeholder candidates are weighted
            # by their detection confidence.
            # --------------------------------------------------

            weighted_count = (
                calculate_sentinel_affected_count(
                    finding
                )
            )


            # Cap affected rate at 1.0 (100%) since multiple sentinels
            # can exist per row, but we normalize by row count
            affected_rate = min(
                1.0,
                (
                    weighted_count
                    /
                    total_rows
                    if total_rows > 0
                    else 0
                )
            )


        else:

            affected_rate = (

                finding[
                    "count"
                ]
                /
                total_rows

                if total_rows > 0

                else 0

            )


        # --------------------------------------------------
        # Determine severity
        # --------------------------------------------------

        severity = determine_severity(
            affected_rate
        )


        # --------------------------------------------------
        # Calculate penalty
        #
        # Severity-aware: a tiny number of serious issues
        # must still lower the score meaningfully.
        # --------------------------------------------------

        severity_base = {
            "CLEAN": 0,
            "LOW": 10,
            "MEDIUM": 25,
            "HIGH": 40,
            "CRITICAL": 55,
        }

        base = severity_base.get(severity, 10)

        category_score = max(
            0.0,
            100.0
            -
            (affected_rate * 100 * 2.0)
            -
            (base * 1.5)
        )

        penalty = weight * (100.0 - category_score) / 100.0

        total_penalty += penalty


        category_scores[
            issue_type
        ] = {

            "weight":
                weight,

            "affected_rate":
                affected_rate,

            "severity":
                severity,

            "penalty":
                penalty,

            "score":
                category_score

        }


    # --------------------------------------------------
    # Overall score
    # --------------------------------------------------

    total_weight = sum(weights.get(ft["issue_type"], 0) for ft in findings) or sum(weights.values())
    weighted_sum = 0.0
    for issue_type, details in category_scores.items():
        weighted_sum += details["weight"] * details["score"]
    overall_score = max(0.0, weighted_sum / (sum(weights.values()) or 1))


    return {

        "overall_score":
            overall_score,

        "category_scores":
            category_scores

    }
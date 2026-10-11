from app.profiler.sentinel_detector import (
    detect_possible_sentinels
)


def detect_sentinel_values(
    df
):
    """
    Detect possible sentinel and placeholder values
    across the dataset.

    This detector does not modify the dataset.

    It reports suspicious numerical sentinels and
    textual placeholders separately.

    Returns actual occurrence counts, not just unique candidate counts.
    """

    columns = {}

    total_sentinel_occurrences = 0

    total_placeholder_occurrences = 0


    # --------------------------------------------------
    # Analyze every column independently.
    # --------------------------------------------------

    for column in df.columns:

        candidates = (
            detect_possible_sentinels(
                df[column]
            )
        )


        sentinel_candidates = []

        placeholder_candidates = []


        # --------------------------------------------------
        # Separate numerical sentinels from textual
        # placeholders.
        # --------------------------------------------------

        for candidate in candidates:

            classification = (
                candidate.get(
                    "classification"
                )
            )


            if (
                classification
                ==
                "possible_sentinel"
            ):

                sentinel_candidates.append(
                    candidate
                )


            elif (
                classification
                ==
                "possible_placeholder"
            ):

                placeholder_candidates.append(
                    candidate
                )


        # Sum actual occurrences (not just unique candidates)
        sentinel_occurrences = sum(c.get("count", 0) for c in sentinel_candidates)
        placeholder_occurrences = sum(c.get("count", 0) for c in placeholder_candidates)

        candidate_count = (
            len(sentinel_candidates)
            +
            len(placeholder_candidates)
        )


        total_sentinel_occurrences += sentinel_occurrences

        total_placeholder_occurrences += placeholder_occurrences


        columns[column] = {

            "has_issue":
                candidate_count > 0,

            "candidate_count":
                candidate_count,

            "sentinel_count":
                len(sentinel_candidates),

            "placeholder_count":
                len(placeholder_candidates),

            "sentinel_occurrences":
                sentinel_occurrences,

            "placeholder_occurrences":
                placeholder_occurrences,

            "sentinel_candidates":
                sentinel_candidates,

            "placeholder_candidates":
                placeholder_candidates

        }


    return {

        "issue_type":
            "possible_sentinel_values",

        "has_issue":
            total_sentinel_occurrences > 0 or total_placeholder_occurrences > 0,

        "candidate_count":
            sum(c["candidate_count"] for c in columns.values()),

        "sentinel_count":
            sum(c["sentinel_count"] for c in columns.values()),

        "placeholder_count":
            sum(c["placeholder_count"] for c in columns.values()),

        "sentinel_occurrences":
            total_sentinel_occurrences,

        "placeholder_occurrences":
            total_placeholder_occurrences,

        "total_occurrences":
            total_sentinel_occurrences + total_placeholder_occurrences,

        "columns":
            columns

    }
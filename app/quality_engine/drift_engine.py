# app/quality_engine/drift_engine.py


from app.quality_engine.schema_drift import (
    detect_schema_drift
)

from app.quality_engine.statistical_drift import (
    detect_numerical_statistical_drift
)

from app.quality_engine.categorical_drift import (
    detect_categorical_distribution_drift
)


# --------------------------------------------------
# Detect complete dataset drift
# --------------------------------------------------

def detect_dataset_drift(
    previous_snapshot,
    current_snapshot,
    numerical_threshold=0.20,
    categorical_threshold=0.20
):
    """
    Run all Data Sentinel drift detectors.

    The drift engine combines:

        1. Schema drift
        2. Numerical statistical drift
        3. Categorical distribution drift

    The individual detectors remain responsible
    for their own detection logic.

    This function only orchestrates them.
    """

    # --------------------------------------------------
    # Schema drift
    # --------------------------------------------------

    schema_drift = (
        detect_schema_drift(

            previous_snapshot,

            current_snapshot

        )
    )


    # --------------------------------------------------
    # Numerical statistical drift
    # --------------------------------------------------

    numerical_drift = (
        detect_numerical_statistical_drift(

            previous_snapshot,

            current_snapshot,

            threshold=
                numerical_threshold

        )
    )


    # --------------------------------------------------
    # Categorical distribution drift
    # --------------------------------------------------

    categorical_drift = (
        detect_categorical_distribution_drift(

            previous_snapshot,

            current_snapshot,

            threshold=
                categorical_threshold

        )
    )


    # --------------------------------------------------
    # Determine whether any type of drift exists.
    # --------------------------------------------------

    has_drift = (

        schema_drift["has_issue"]

        or

        numerical_drift["has_issue"]

        or

        categorical_drift["has_issue"]

    )


    # --------------------------------------------------
    # Return the combined drift analysis.
    # --------------------------------------------------

    return {

        "has_drift":
            has_drift,

        "schema_drift":
            schema_drift,

        "numerical_drift":
            numerical_drift,

        "categorical_drift":
            categorical_drift

    }
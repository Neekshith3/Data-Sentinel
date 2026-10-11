import pandas as pd


from app.profiler.shared_metadata import (
    compute_column_metadata,
    compute_dataset_profile,
    compute_understanding
)

from app.profiler.column_roles import (

    analyze_column_roles
)


from app.profiler.dataset_snapshot import (

    create_dataset_snapshot
)


from app.quality_engine.quality_engine import (

    analyze_dataset as run_quality_analysis
)


from app.quality_engine.quality_score import (

    calculate_quality_score
)


from app.quality_engine.ml_readiness import (

    assess_ml_readiness
)


from app.quality_engine.drift_engine import (

    detect_dataset_drift
)

# --------------------------------------------------

# Visualization configuration

# --------------------------------------------------


MAX_CATEGORICAL_VALUES = 20


MAX_HISTOGRAM_VALUES = 5000


MAX_SCATTER_VALUES = 5000



# --------------------------------------------------
# Detect visualization roles
# --------------------------------------------------


def detect_visualization_roles(
    df
):


    detected_roles = (
        analyze_column_roles(
            df
        )
    )


    visualization_roles = {}


    for column, role in (
        detected_roles.items()
    ):


        if role == "numerical_measurement":


            visualization_roles[
                column
            ] = "numerical"


        elif role == "categorical":


            visualization_roles[
                column
            ] = "categorical"


        elif role == "datetime":


            visualization_roles[
                column
            ] = "datetime"


        else:


            visualization_roles[
                column
            ] = role


    return visualization_roles


# --------------------------------------------------
# Generate quality issue visualization
# --------------------------------------------------


def generate_quality_issue_visualization(
    findings
):


    issue_data = []


    for finding in findings:


        if not finding.get(
            "has_issue",
            False
        ):


            continue


        issue_data.append({
            "category":
                finding.get(
                    "issue_type",
                    "unknown"
                ),
            "count":
                finding.get(
                    "count",
                    0
                )
        })


    if not issue_data:


        return None


    return {
        "type":
            "bar",
        "chart_id":
            "qualityIssues",
        "title":
            "Data Quality Issues",
        "description":
            "Detected data-quality issues that require attention.",
        "data":
            issue_data
    }


# --------------------------------------------------
# Generate missing-value visualization
# --------------------------------------------------


def generate_missing_visualization(
    df
):


    missing_data = []


    for column in df.columns:


        missing_count = int(
            df[column].isna().sum()
        )


        if missing_count > 0:


            missing_data.append({
                "column":
                    column,
                "count":
                    missing_count
            })


    if not missing_data:


        return None


    return {
        "type":
            "bar",
        "chart_id":
            "missingValues",
        "title":
            "Missing Values by Column",
        "description":
            "Columns containing missing data.",
        "data":
            missing_data
    }


# --------------------------------------------------
# Generate categorical visualizations
# --------------------------------------------------

# --------------------------------------------------
# Build visualization exclusion values
# --------------------------------------------------
def build_visualization_exclusions(
    understanding,
    minimum_confidence=0.8
):

    exclusions = {}
    column_analysis = understanding.get(
        "column_analysis",
        []
    )

    for column_information in column_analysis:

        column = column_information.get(
            "column"
        )

        if not column:

            continue

        candidates = column_information.get(
            "possible_sentinels",
            []
        )

        excluded_values = []
        for candidate in candidates:
            confidence = float(
                candidate.get(
                    "confidence",
                    0
                )
            )

            classification = candidate.get(
                "classification"
            )

            if (
                confidence >= minimum_confidence
                and classification in {
                    "possible_sentinel",
                    "possible_placeholder"
                }
            ):

                excluded_values.append(
                    candidate.get("value")
                )

        if excluded_values:

            exclusions[column] = excluded_values

    return exclusions


# --------------------------------------------------
# Filter numerical visualization values
# --------------------------------------------------

def filter_numerical_values(
    series,
    excluded_values
):

    values = pd.to_numeric(
        series,
        errors="coerce"
    )

    values = values.dropna()

    if not excluded_values:

        return values

    excluded_numeric = set()

    for value in excluded_values:

        try:

            excluded_numeric.add(
                float(value)
            )

        except (
            TypeError,
            ValueError
        ):

            continue

    if not excluded_numeric:

        return values

    return values[
        ~values.isin(
            excluded_numeric
        )
    ]


# --------------------------------------------------
# Filter categorical visualization values
# --------------------------------------------------

def filter_categorical_values(
    series,
    excluded_values
):

    values = (
        series
        .dropna()
        .astype(str)
    )

    if not excluded_values:

        return values

    excluded_text = {
        str(value).strip().lower()
        for value in excluded_values
    }

    normalized_values = (
        values
        .str.strip()
        .str.lower()
    )

    return values[
        ~normalized_values.isin(
            excluded_text
        )
    ]


def generate_categorical_visualizations(
    df,
    roles,
    visualization_exclusions=None
):

    visualizations = []

    if visualization_exclusions is None:

        visualization_exclusions = {}

    categorical_columns = [
        column
        for column in df.columns
        if roles.get(column) == "categorical"
    ]

    for column in categorical_columns:

        values = filter_categorical_values(
            df[column],
            visualization_exclusions.get(
                column,
                []
            )
        )

        counts = (
            values
            .value_counts()
            .head(
                MAX_CATEGORICAL_VALUES
            )
        )

        if counts.empty:

            continue

        category_data = [
            {
                "category": str(value),
                "count": int(count)
            }
            for value, count
            in counts.items()
        ]

        visualizations.append({
            "type": "bar",
            "chart_id": f"category_{column}",
            "title": f"{column.title()} Distribution",
            "description": (
                f"Distribution of values in {column}."
            ),
            "column": column,
            "data": category_data
        })

    return visualizations


def generate_numerical_visualizations(
    df,
    roles,
    visualization_exclusions=None
):

    visualizations = []

    if visualization_exclusions is None:

        visualization_exclusions = {}

    numerical_columns = [
        column
        for column in df.columns
        if roles.get(column) == "numerical"
    ]

    for column in numerical_columns:

        values = filter_numerical_values(
            df[column],
            visualization_exclusions.get(
                column,
                []
            )
        )

        if values.empty:

            continue

        if len(values) > MAX_HISTOGRAM_VALUES:

            values = values.sample(
                MAX_HISTOGRAM_VALUES,
                random_state=42
            )

        visualizations.append({
            "type": "histogram",
            "chart_id": f"distribution_{column}",
            "title": f"{column.title()} Distribution",
            "description": (
                f"Distribution of values in {column}."
            ),
            "column": column,
            "data": [
                {
                    "value": float(value)
                }
                for value in values
            ]
        })

    return visualizations


def generate_scatter_visualization(
    df,
    roles,
    visualization_exclusions=None
):

    if visualization_exclusions is None:

        visualization_exclusions = {}

    numerical_columns = [
        column
        for column in df.columns
        if roles.get(column) == "numerical"
    ]

    if len(numerical_columns) < 2:

        return None

    x_column = numerical_columns[0]

    y_column = numerical_columns[1]

    relationship_df = df[
        [
            x_column,
            y_column
        ]
    ].copy()

    relationship_df[x_column] = pd.to_numeric(
        relationship_df[x_column],
        errors="coerce"
    )

    relationship_df[y_column] = pd.to_numeric(
        relationship_df[y_column],
        errors="coerce"
    )

    relationship_df = relationship_df.dropna()

    if x_column in visualization_exclusions:

        excluded_x = set()

        for value in visualization_exclusions[x_column]:

            try:

                excluded_x.add(float(value))

            except (TypeError, ValueError):

                continue

        if excluded_x:

            relationship_df = relationship_df[
                ~relationship_df[x_column].isin(
                    excluded_x
                )
            ]

    if y_column in visualization_exclusions:

        excluded_y = set()

        for value in visualization_exclusions[y_column]:

            try:

                excluded_y.add(float(value))

            except (TypeError, ValueError):

                continue

        if excluded_y:

            relationship_df = relationship_df[
                ~relationship_df[y_column].isin(
                    excluded_y
                )
            ]

    if relationship_df.empty:

        return None

    if len(relationship_df) > MAX_SCATTER_VALUES:

        relationship_df = relationship_df.sample(
            MAX_SCATTER_VALUES,
            random_state=42
        )

    data = [
        {
            "x": float(row[x_column]),
            "y": float(row[y_column])
        }
        for _, row in relationship_df.iterrows()
    ]

    return {
        "type": "scatter",
        "chart_id": (
            f"relationship_{x_column}_{y_column}"
        ),
        "title": (
            f"{x_column.title()} vs "
            f"{y_column.title()}"
        ),
        "description": (
            "Relationship between two numerical variables."
        ),
        "x_column": x_column,
        "y_column": y_column,
        "data": data
    }


def generate_grouped_visualizations(
    df,
    roles,
    visualization_exclusions=None
):

    visualizations = []

    if visualization_exclusions is None:

        visualization_exclusions = {}

    categorical_columns = [
        column
        for column in df.columns
        if roles.get(column) == "categorical"
    ]

    numerical_columns = [
        column
        for column in df.columns
        if roles.get(column) == "numerical"
    ]

    if not categorical_columns or not numerical_columns:

        return visualizations

    for categorical_column in categorical_columns[:3]:

        numerical_column = numerical_columns[0]

        grouped_df = df[
            [
                categorical_column,
                numerical_column
            ]
        ].copy()

        grouped_df[numerical_column] = pd.to_numeric(
            grouped_df[numerical_column],
            errors="coerce"
        )

        grouped_df = grouped_df.dropna()

        numerical_exclusions = visualization_exclusions.get(
            numerical_column,
            []
        )

        if numerical_exclusions:

            excluded_numeric = set()

            for value in numerical_exclusions:

                try:

                    excluded_numeric.add(float(value))

                except (TypeError, ValueError):

                    continue

                if excluded_numeric:

                    grouped_df = grouped_df[
                        ~grouped_df[numerical_column].isin(
                            excluded_numeric
                        )
                    ]

        categorical_exclusions = visualization_exclusions.get(
            categorical_column,
            []
        )

        if categorical_exclusions:

            excluded_text = {
                str(value).strip().lower()
                for value in categorical_exclusions
            }

            normalized_categories = (
                grouped_df[categorical_column]
                .astype(str)
                .str.strip()
                .str.lower()
            )

            grouped_df = grouped_df[
                ~normalized_categories.isin(
                    excluded_text
                )
            ]

        if grouped_df.empty:

            continue

        grouped_df = (
            grouped_df
            .groupby(categorical_column)[numerical_column]
            .mean()
            .sort_values(ascending=False)
            .head(MAX_CATEGORICAL_VALUES)
        )

        if grouped_df.empty:

            continue

        grouped_data = [
            {
                "category": str(category),
                "value": float(value)
            }
            for category, value
            in grouped_df.items()
        ]

        visualizations.append({
            "type": "bar",
            "chart_id": (
                f"grouped_{numerical_column}_"
                f"{categorical_column}"
            ),
            "title": (
                f"Average {numerical_column.title()} "
                f"by {categorical_column.title()}"
            ),
            "description": (
                f"Average {numerical_column} "
                f"across {categorical_column}."
            ),
            "x_column": categorical_column,
            "y_column": numerical_column,
            "data": grouped_data
        })

    return visualizations


def generate_visualizations(
    df,
    findings,
    understanding=None
):

    visualizations = []

    if understanding is None:

        understanding = {}

    roles = detect_visualization_roles(
        df
    )

    visualization_exclusions = (
        build_visualization_exclusions(
            understanding
        )
    )

    quality_visualization = (
        generate_quality_issue_visualization(
            findings
        )
    )

    if quality_visualization:

        visualizations.append(
            quality_visualization
        )

    missing_visualization = (
        generate_missing_visualization(
            df
        )
    )

    if missing_visualization:

        visualizations.append(
            missing_visualization
        )

    visualizations.extend(
        generate_categorical_visualizations(
            df,
            roles,
            visualization_exclusions
        )
    )

    visualizations.extend(
        generate_numerical_visualizations(
            df,
            roles,
            visualization_exclusions
        )
    )

    scatter_visualization = (
        generate_scatter_visualization(
            df,
            roles,
            visualization_exclusions
        )
    )

    if scatter_visualization:

        visualizations.append(
            scatter_visualization
        )

    visualizations.extend(
        generate_grouped_visualizations(
            df,
            roles,
            visualization_exclusions
        )
    )

    return visualizations


def analyze_dataset(

    df,

    expected_schema=None,

    validation_rules=None,

    ml_feature_columns=None,

    previous_snapshot=None

):

    """

    Orchestrate the complete Data Sentinel analysis.


    The analysis includes:


        - dataset profiling
        - dataset understanding
        - deterministic quality checks
        - quality scoring
        - ML readiness assessment
        - dataset snapshot creation
        - optional complete drift detection
        - visualizations


    The previous snapshot is optional.


    If no previous snapshot is supplied,
    drift detection is not performed.

    """



    if expected_schema is None:


        expected_schema = {}


    if validation_rules is None:


        validation_rules = {}



    # --------------------------------------------------
    # Compute shared column metadata (single pass)
    # --------------------------------------------------

    column_roles = analyze_column_roles(df)
    column_metadata = compute_column_metadata(df, column_roles)

    # --------------------------------------------------
    # Dataset profiling (from shared metadata)
    # --------------------------------------------------

    profile = compute_dataset_profile(df, column_metadata)


    # --------------------------------------------------
    # Dataset understanding (from shared metadata)
    # --------------------------------------------------

    understanding = compute_understanding(df, column_metadata, column_roles)


    # --------------------------------------------------
    # Deterministic quality analysis
    # --------------------------------------------------


    quality_analysis = (
        run_quality_analysis(


            df,


            expected_schema=


                expected_schema,


            validation_rules=


                validation_rules,


            ml_feature_columns=


                ml_feature_columns,


            column_metadata=


                column_metadata,


            understanding=


                understanding,


            column_roles=


                column_roles


        )
    )


    findings = (
        quality_analysis[
            "findings"
        ]
    )



    # --------------------------------------------------
    # Calculate quality score
    # --------------------------------------------------


    quality_score = (
        calculate_quality_score(


            findings,


            total_rows=


                len(df),


            total_columns=


                len(df.columns)


        )
    )



    # --------------------------------------------------
    # Assess ML readiness
    # --------------------------------------------------


    ml_readiness = (
        assess_ml_readiness(


            df,


            understanding=


                understanding,


            sentinel_finding=None,


            column_metadata=


                column_metadata,


            column_roles=


                column_roles


        )
    )



    # --------------------------------------------------
    # Create current dataset snapshot
    # --------------------------------------------------


    current_snapshot = (
        create_dataset_snapshot(


            df,


            profile,


            understanding


        )
    )



    # --------------------------------------------------
    # Detect complete historical drift
    #
    # Drift requires a previous snapshot.
    # --------------------------------------------------


    drift = None


    if previous_snapshot is not None:


        drift = (
            detect_dataset_drift(


                previous_snapshot,


                current_snapshot


            )
        )



    # --------------------------------------------------
    # Generate visualizations
    # --------------------------------------------------


    visualizations = (
        generate_visualizations(


            df,


            findings,


            understanding


        )
    )



    # --------------------------------------------------
    # Return complete analysis
    # --------------------------------------------------


    return {
        "profile":
            profile,
        "understanding":
            understanding,
        "findings":
            findings,
        "quality_score":
            quality_score,
        "ml_readiness":
            ml_readiness,
        "snapshot":
            current_snapshot,
        "drift":
            drift,
        "visualizations":
            visualizations
    }
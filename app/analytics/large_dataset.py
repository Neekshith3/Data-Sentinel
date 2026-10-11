import os



import random



from collections import Counter







import numpy as np



import pandas as pd











# --------------------------------------------------



# Configuration



# --------------------------------------------------







DEFAULT_CHUNK_SIZE = 100_000







MAX_CATEGORY_VALUES = 20







MAX_SCATTER_POINTS = 5_000







MAX_UNIQUE_VALUES_TO_TRACK = 10_000







MAX_NUMERICAL_SAMPLE_VALUES = 20_000







NUMERICAL_SENTINEL_CONFIDENCE_THRESHOLD = 0.8







EXTREME_ROBUST_SCORE = 10.0







EXTREME_IQR_MULTIPLIER = 3.0







MINIMUM_OCCURRENCES = 2







PLACEHOLDER_VALUES = {



    "na",



    "n/a",



    "nan",



    "null",



    "none",



    "unknown",



    "not available",



    "not_applicable",



    "missing",



    "?",



    "-"



}











# --------------------------------------------------



# Get file size in MB



# --------------------------------------------------







def get_file_size_mb(



    file_path



):







    size_bytes = os.path.getsize(



        file_path



    )







    return (



        size_bytes



        /



        (1024 * 1024)



    )











# --------------------------------------------------



# Detect whether a value is a textual placeholder



# --------------------------------------------------







def looks_like_placeholder(



    value



):







    if not isinstance(



        value,



        str



    ):







        return False











    normalized_value = (



        value



        .strip()



        .lower()



    )











    return (



        normalized_value



        in PLACEHOLDER_VALUES



    )











# --------------------------------------------------



# Detect column role



# --------------------------------------------------







def detect_column_role(



    column_name,



    column_type,



    unique_count,



    total_rows



):







    if total_rows == 0:







        return "unknown"











    column_name_lower = (



        str(column_name)



        .strip()



        .lower()



    )











    identifier_tokens = (







        column_name_lower



        .replace("-", "_")



        .replace(" ", "_")



        .split("_")







    )











    identifier_keywords = {







        "id",



        "uuid",



        "identifier",



        "key"







    }











    if any(







        token in identifier_keywords







        for token



        in identifier_tokens







    ):







        return "identifier"











    if column_type == "numerical":







        return "numerical"











    uniqueness_ratio = (







        unique_count



        /



        total_rows







    )











    if uniqueness_ratio >= 0.80:







        return "identifier_like"











    if column_type == "categorical":







        return "categorical"











    return "unknown"











# --------------------------------------------------



# Update top categorical values



# --------------------------------------------------







def update_top_categories(



    category_counts,



    series,



    max_categories=MAX_CATEGORY_VALUES



):







    values = (







        series



        .dropna()



        .astype(str)



        .str.strip()







    )











    values = values[



        values != ""



    ]











    if values.empty:







        return











    counts = (



        values.value_counts()



    )











    for value, count in counts.items():







        category_counts[value] = (







            category_counts.get(



                value,



                0



            )



            +



            int(count)







        )











    if len(category_counts) > (







        max_categories * 10







    ):







        top_categories = sorted(







            category_counts.items(),







            key=lambda item:



                item[1],







            reverse=True







        )[:max_categories]











        category_counts.clear()







        category_counts.update(



            top_categories



        )











# --------------------------------------------------



# Reservoir sampling



# --------------------------------------------------







def reservoir_sample(



    reservoir,



    rows,



    total_seen,



    max_samples=MAX_SCATTER_POINTS



):







    for row in rows:







        total_seen += 1











        if len(reservoir) < max_samples:







            reservoir.append(



                row



            )







            continue











        random_index = random.randint(







            0,







            total_seen - 1







        )











        if random_index < max_samples:







            reservoir[random_index] = row











    return total_seen











# --------------------------------------------------



# Numerical reservoir sampling



# --------------------------------------------------







def update_numerical_sample(



    reservoir,



    values,



    total_seen,



    max_samples=MAX_NUMERICAL_SAMPLE_VALUES



):







    for value in values:







        if pd.isna(value):







            continue











        total_seen += 1











        value = float(value)











        if len(reservoir) < max_samples:







            reservoir.append(



                value



            )







            continue











        random_index = random.randint(







            0,







            total_seen - 1







        )











        if random_index < max_samples:







            reservoir[random_index] = value











    return total_seen











# --------------------------------------------------



# Determine whether a column chunk is numerical



#



# We allow textual placeholders to coexist with



# numerical measurements.



# --------------------------------------------------







def infer_chunk_column_type(



    series



):







    if pd.api.types.is_bool_dtype(



        series



    ):







        return "categorical"











    numeric_values = pd.to_numeric(



        series,



        errors="coerce"



    )











    non_null_count = int(



        series.notna().sum()



    )











    if non_null_count == 0:







        return "unknown"











    numeric_count = int(



        numeric_values.notna().sum()



    )











    numeric_ratio = (







        numeric_count



        /



        non_null_count







    )











    if numeric_ratio >= 0.95:







        return "numerical"











    return "categorical"











# --------------------------------------------------



# Calculate robust numerical statistics



# --------------------------------------------------







def get_robust_statistics(



    values



):







    numeric_values = pd.to_numeric(



        pd.Series(values),



        errors="coerce"



    ).dropna()











    if numeric_values.empty:







        return None











    median = float(



        numeric_values.median()



    )











    mad = float(



        (



            numeric_values



            -



            median



        )



        .abs()



        .median()



    )











    q1 = float(



        numeric_values.quantile(



            0.25



        )



    )











    q3 = float(



        numeric_values.quantile(



            0.75



        )



    )











    iqr = (



        q3



        -



        q1



    )











    return {







        "median":



            median,







        "mad":



            mad,







        "q1":



            q1,







        "q3":



            q3,







        "iqr":



            float(iqr)







    }











# --------------------------------------------------



# Calculate robust score



# --------------------------------------------------







def calculate_robust_score(



    value,



    statistics



):







    median = statistics[



        "median"



    ]











    mad = statistics[



        "mad"



    ]











    if mad > 0:







        return (







            abs(



                value



                -



                median



            )



            /



            mad







        )











    iqr = statistics[



        "iqr"



    ]











    if iqr > 0:







        return (







            abs(



                value



                -



                median



            )



            /



            iqr







        )











    return 0.0











# --------------------------------------------------



# Check extreme IQR distance



# --------------------------------------------------







def is_extreme_iqr_value(



    value,



    statistics



):







    q1 = statistics[



        "q1"



    ]











    q3 = statistics[



        "q3"



    ]











    iqr = statistics[



        "iqr"



    ]











    if iqr <= 0:







        return False











    lower_limit = (







        q1



        -



        (



            EXTREME_IQR_MULTIPLIER



            *



            iqr



        )







    )











    upper_limit = (







        q3



        +



        (



            EXTREME_IQR_MULTIPLIER



            *



            iqr



        )







    )











    return (







        value < lower_limit







        or







        value > upper_limit







    )











# --------------------------------------------------



# Build numerical sentinel candidate



# --------------------------------------------------







def build_numerical_candidate(



    value,



    count,



    statistics



):







    robust_score = (



        calculate_robust_score(



            value,



            statistics



        )



    )











    extreme_iqr = (



        is_extreme_iqr_value(



            value,



            statistics



        )



    )











    repeated = (



        count



        >=



        MINIMUM_OCCURRENCES



    )











    has_strong_signal = (







        robust_score



        >=



        EXTREME_ROBUST_SCORE







        or







        extreme_iqr







    )











    if not has_strong_signal:







        return None











    signals = []











    if (



        robust_score



        >=



        EXTREME_ROBUST_SCORE



    ):







        signals.append(



            "extreme_from_robust_center"



        )











    if extreme_iqr:







        signals.append(



            "far_outside_iqr"



        )











    if repeated:







        signals.append(



            "repeated_value"



        )











    confidence = 0.0











    if (



        robust_score



        >=



        EXTREME_ROBUST_SCORE



    ):







        confidence += 0.45











    if extreme_iqr:







        confidence += 0.35











    if repeated:







        confidence += 0.20











    confidence = min(



        1.0,



        confidence



    )











    return {







        "value":



            value,







        "count":



            int(count),







        "confidence":



            float(confidence),







        "robust_score":



            float(robust_score),







        "extreme_iqr":



            bool(extreme_iqr),







        "repeated":



            bool(repeated),







        "signals":



            signals,







        "classification":



            "possible_sentinel",







        "reason":



            (



                "Numerical value shows "



                "strong evidence of being "



                "outside the normal distribution"



            )







    }











# --------------------------------------------------



# Find numerical sentinel candidates



#



# This operates on a representative sample.



# A second full chunk pass later validates the



# candidates against the complete dataset.



# --------------------------------------------------







def detect_numerical_sentinel_candidates(



    sample_values



):







    statistics = (



        get_robust_statistics(



            sample_values



        )



    )











    if statistics is None:







        return []











    value_counts = (



        pd.Series(



            sample_values



        )



        .value_counts()



    )











    candidates = []











    for value, count in (



        value_counts.items()



    ):







        candidate = (



            build_numerical_candidate(







                float(value),







                int(count),







                statistics







            )



        )











        if candidate is not None:







            candidates.append(



                candidate



            )











    return candidates











# --------------------------------------------------



# Discover textual placeholders



# --------------------------------------------------







def discover_textual_placeholders(



    category_counts



):







    placeholders = []











    for value, count in (



        category_counts.items()



    ):







        if not looks_like_placeholder(



            value



        ):







            continue











        placeholders.append({







            "value":



                value,







            "count":



                int(count),







            "confidence":



                0.8,







            "repeated":



                int(count)



                >=



                MINIMUM_OCCURRENCES,







            "signals": [



                "placeholder_pattern"



            ],







            "classification":



                "possible_placeholder",







            "reason":



                (



                    "Text value matches "



                    "a possible placeholder pattern"



                )







        })











    return placeholders











# --------------------------------------------------



# Create empty streaming statistics



# --------------------------------------------------







def create_numerical_stat():







    return {







        "count":



            0,







        "sum":



            0.0,







        "sum_squared":



            0.0,







        "min":



            None,







        "max":



            None







    }











# --------------------------------------------------



# Update numerical statistics



# --------------------------------------------------







def update_numerical_stat(



    statistics,



    series



):







    numeric_series = pd.to_numeric(



        series,



        errors="coerce"



    ).dropna()











    if numeric_series.empty:







        return











    values = (



        numeric_series



        .astype(float)



    )











    statistics[



        "count"



    ] += len(values)











    statistics[



        "sum"



    ] += float(



        values.sum()



    )











    statistics[



        "sum_squared"



    ] += float(



        (



            values ** 2



        ).sum()



    )











    current_min = float(



        values.min()



    )











    current_max = float(



        values.max()



    )











    if (







        statistics["min"] is None







        or







        current_min



        <



        statistics["min"]







    ):







        statistics[



            "min"



        ] = current_min











    if (







        statistics["max"] is None







        or







        current_max



        >



        statistics["max"]







    ):







        statistics[



            "max"



        ] = current_max











# --------------------------------------------------



# Finalize numerical statistics



# --------------------------------------------------







def finalize_numerical_statistics(



    statistics



):







    count = statistics[



        "count"



    ]











    if count == 0:







        return {







            "count": 0,







            "min": None,







            "max": None,







            "mean": None,







            "variance": None,







            "standard_deviation": None







        }











    mean = (







        statistics["sum"]



        /



        count







    )











    variance = max(







        0.0,







        (







            statistics[



                "sum_squared"



            ]



            /



            count







        )



        -



        (



            mean ** 2



        )







    )











    return {







        "count":



            int(count),







        "min":



            statistics["min"],







        "max":



            statistics["max"],







        "mean":



            float(mean),







        "variance":



            float(variance),







        "standard_deviation":



            float(



                variance ** 0.5



            )







    }











# --------------------------------------------------



# Clean a value using detected candidates



# --------------------------------------------------







def is_detected_bad_value(



    value,



    bad_values



):







    if pd.isna(value):







        return True











    if isinstance(



        value,



        str



    ):







        normalized = (



            value



            .strip()



            .lower()



        )







        return (



            normalized



            in bad_values



        )











    try:







        numeric_value = float(



            value



        )







    except (



        TypeError,



        ValueError



    ):







        return False











    return numeric_value in (



        bad_values



    )











# --------------------------------------------------



# Build clean value sets



# --------------------------------------------------







def build_bad_value_map(



    sentinel_candidates



):







    bad_value_map = {}











    for column, candidates in (



        sentinel_candidates.items()



    ):







        values = set()











        for candidate in candidates:







            if (



                candidate.get(



                    "confidence",



                    0



                )



                >=



                NUMERICAL_SENTINEL_CONFIDENCE_THRESHOLD



            ):







                values.add(



                    candidate[



                        "value"



                    ]



                )











        bad_value_map[column] = values











    return bad_value_map











# --------------------------------------------------



# Recalculate exact clean statistics



# --------------------------------------------------







def calculate_clean_statistics(



    file_path,



    chunk_size,



    column_types,



    bad_value_map



):







    clean_stats = {}











    for column, column_type in (



        column_types.items()



    ):







        if column_type != "numerical":







            continue











        clean_stats[column] = (



            create_numerical_stat()



        )











    for chunk in pd.read_csv(







        file_path,







        chunksize=chunk_size,







        keep_default_na=False,







        na_values=[""]







    ):







        for column in clean_stats:







            numeric_values = pd.to_numeric(







                chunk[column],







                errors="coerce"







            )











            bad_values = (



                bad_value_map.get(



                    column,



                    set()



                )



            )











            if bad_values:







                mask = numeric_values.isin(



                    list(



                        bad_values



                    )



                )







                numeric_values = (



                    numeric_values



                    .where(



                        ~mask



                    )



                )











            update_numerical_stat(







                clean_stats[column],







                numeric_values







            )











    finalized = {}











    for column, stats in (



        clean_stats.items()



    ):







        finalized[column] = (



            finalize_numerical_statistics(



                stats



            )



        )











    return finalized











# --------------------------------------------------



# Validate and count detected candidates



# against the complete dataset



# --------------------------------------------------







def validate_candidates(



    file_path,



    chunk_size,



    candidate_map



):







    exact_counts = {







        column: Counter()







        for column



        in candidate_map







    }











    for chunk in pd.read_csv(







        file_path,







        chunksize=chunk_size,







        keep_default_na=False,







        na_values=[""]







    ):







        for column, candidates in (



            candidate_map.items()



        ):







            if not candidates:







                continue











            series = chunk[



                column



            ]











            numeric_series = pd.to_numeric(







                series,







                errors="coerce"







            )











            for candidate in candidates:







                candidate_value = candidate[



                    "value"



                ]











                mask = (







                    numeric_series



                    ==



                    float(



                        candidate_value



                    )







                )











                count = int(



                    mask.sum()



                )











                if count > 0:







                    exact_counts[



                        column



                    ][



                        candidate_value



                    ] += count











    validated = {}











    for column, candidates in (



        candidate_map.items()



    ):







        validated_candidates = []











        for candidate in candidates:







            value = candidate[



                "value"



            ]











            count = exact_counts[



                column



            ].get(



                value,



                0



            )











            if count == 0:







                continue











            updated_candidate = (



                candidate.copy()



            )











            updated_candidate[



                "count"



            ] = int(count)











            if count >= MINIMUM_OCCURRENCES:







                updated_candidate[



                    "repeated"



                ] = True











                if (



                    "repeated_value"



                    not in



                    updated_candidate[



                        "signals"



                    ]



                ):







                    updated_candidate[



                        "signals"



                    ].append(



                        "repeated_value"



                    )











                updated_candidate[



                    "confidence"



                ] = min(







                    1.0,







                    float(



                        updated_candidate[



                            "confidence"



                        ]



                    )



                    +



                    0.20







                )











            validated_candidates.append(



                updated_candidate



            )











        validated[



            column



        ] = validated_candidates











    return validated











# --------------------------------------------------



# Generate compact visualizations from a cleaned



# representative sample



# --------------------------------------------------







def generate_visualizations(



    sample_df,



    column_roles,



    bad_value_map



):







    visualizations = []











    numerical_columns = [







        column







        for column in sample_df.columns







        if column_roles.get(



            column



        ) == "numerical"







    ]











    categorical_columns = [







        column







        for column in sample_df.columns







        if column_roles.get(



            column



        ) == "categorical"







    ]











    cleaned_df = (



        sample_df.copy()



    )











    for column in cleaned_df.columns:







        bad_values = (



            bad_value_map.get(



                column,



                set()



            )



        )











        if not bad_values:







            continue











        if column in numerical_columns:







            numeric_values = pd.to_numeric(







                cleaned_df[column],







                errors="coerce"







            )











            cleaned_df[column] = (



                numeric_values



                .where(



                    ~numeric_values.isin(



                        list(



                            bad_values



                        )



                    )



                )



            )











        else:







            normalized_bad_values = {







                str(value)



                .strip()



                .lower()







                for value



                in bad_values







            }











            mask = (







                cleaned_df[column]



                .astype(str)



                .str.strip()



                .str.lower()



                .isin(



                    normalized_bad_values



                )







            )











            cleaned_df.loc[



                mask,



                column



            ] = np.nan











    # --------------------------------------------------



    # Categorical distributions



    # --------------------------------------------------







    for column in categorical_columns:







        counts = (







            cleaned_df[column]



            .dropna()



            .astype(str)



            .str.strip()



            .value_counts()



            .head(



                MAX_CATEGORY_VALUES



            )







        )











        if counts.empty:







            continue











        visualizations.append({







            "type":



                "bar",







            "chart_id":



                f"category_{column}",







            "title":



                f"{column.title()} Distribution",







            "description":



                (



                    f"Distribution of values "



                    f"in {column}, excluding "



                    f"detected placeholders."



                ),







            "column":



                column,







            "data": [







                {







                    "category":



                        str(value),







                    "count":



                        int(count)







                }







                for value, count



                in counts.items()







            ]







        })











    # --------------------------------------------------



    # Numerical distributions



    # --------------------------------------------------







    for column in numerical_columns:







        values = pd.to_numeric(







            cleaned_df[column],







            errors="coerce"







        ).dropna()











        if values.empty:







            continue











        if len(values) > 5000:







            values = values.sample(







                5000,







                random_state=42







            )











        visualizations.append({







            "type":



                "histogram",







            "chart_id":



                f"distribution_{column}",







            "title":



                f"{column.title()} Distribution",







            "description":



                (



                    f"Distribution of values "



                    f"in {column}, excluding "



                    f"detected sentinel values."



                ),







            "column":



                column,







            "data": [







                {







                    "value":



                        float(value)







                }







                for value



                in values







            ]







        })











    # --------------------------------------------------



    # Numerical relationship



    # --------------------------------------------------







    if len(



        numerical_columns



    ) >= 2:







        x_column = (



            numerical_columns[0]



        )







        y_column = (



            numerical_columns[1]



        )











        relationship_df = (







            cleaned_df[



                [



                    x_column,



                    y_column



                ]



            ]



            .copy()







        )











        relationship_df[



            x_column



        ] = pd.to_numeric(







            relationship_df[



                x_column



            ],







            errors="coerce"







        )











        relationship_df[



            y_column



        ] = pd.to_numeric(







            relationship_df[



                y_column



            ],







            errors="coerce"







        )











        relationship_df = (



            relationship_df



            .dropna()



        )











        if len(



            relationship_df



        ) > MAX_SCATTER_POINTS:







            relationship_df = (



                relationship_df.sample(







                    MAX_SCATTER_POINTS,







                    random_state=42







                )



            )











        if not relationship_df.empty:







            visualizations.append({







                "type":



                    "scatter",







                "chart_id":



                    (



                        f"relationship_"



                        f"{x_column}_"



                        f"{y_column}"



                    ),







                "title":



                    (



                        f"{x_column.title()} "



                        f"vs {y_column.title()}"



                    ),







                "description":



                    (



                        "Relationship between "



                        "two numerical variables "



                        "after detected sentinel "



                        "values are excluded."



                    ),







                "x_column":



                    x_column,







                "y_column":



                    y_column,







                "data": [







                    {







                        "x":



                            float(



                                row[



                                    x_column



                                ]



                            ),







                        "y":



                            float(



                                row[



                                    y_column



                                ]



                            )







                    }







                    for _, row



                    in relationship_df.iterrows()







                ]







            })











    # --------------------------------------------------



    # Grouped numerical averages



    # --------------------------------------------------







    if (







        categorical_columns







        and







        numerical_columns







    ):







        numerical_column = (



            numerical_columns[0]



        )











        for categorical_column in (



            categorical_columns[:3]



        ):







            grouped_df = (







                cleaned_df[



                    [



                        categorical_column,



                        numerical_column



                    ]



                ]



                .copy()







            )











            grouped_df[



                numerical_column



            ] = pd.to_numeric(







                grouped_df[



                    numerical_column



                ],







                errors="coerce"







            )











            grouped_df = (



                grouped_df



                .dropna()



            )











            if grouped_df.empty:







                continue











            grouped = (







                grouped_df



                .groupby(



                    categorical_column



                )[numerical_column]



                .mean()



                .sort_values(



                    ascending=False



                )



                .head(



                    MAX_CATEGORY_VALUES



                )







            )











            if grouped.empty:







                continue











            visualizations.append({







                "type":



                    "bar",







                "chart_id":



                    (



                        f"grouped_"



                        f"{numerical_column}_"



                        f"{categorical_column}"



                    ),







                "title":



                    (



                        f"Average "



                        f"{numerical_column.title()} "



                        f"by "



                        f"{categorical_column.title()}"



                    ),







                "description":



                    (



                        f"Average {numerical_column} "



                        f"across {categorical_column}, "



                        f"excluding detected bad values."



                    ),







                "x_column":



                    categorical_column,







                "y_column":



                    numerical_column,







                "data": [







                    {







                        "category":



                            str(category),







                        "value":



                            float(value)







                    }







                    for category, value



                    in grouped.items()







                ]







            })











    return visualizations











# --------------------------------------------------



# Analyze large CSV in chunks



# --------------------------------------------------







def analyze_csv_in_chunks(



    file_path,



    chunk_size=DEFAULT_CHUNK_SIZE



):







    """



    Analyze a large CSV without loading the



    complete dataset into memory.







    The dataset is processed in multiple passes:







    Pass 1:



        schema, counts, samples, and category



        information.







    Pass 2:



        validate detected numerical sentinel



        candidates and calculate clean statistics.







    A small representative sample is retained



    for compact browser visualizations.



    """











    total_rows = 0







    column_names = None







    column_types = {}







    column_roles = {}







    missing_counts = {}







    unique_values = {}







    categorical_counts = {}







    numerical_stats = {}







    numerical_samples = {}







    numerical_samples_seen = {}







    sample_rows = []







    sample_rows_seen = 0











    # ==================================================



    # PASS 1



    # ==================================================







    for chunk in pd.read_csv(







        file_path,







        chunksize=chunk_size,







        keep_default_na=False,







        na_values=[""]







    ):







        if column_names is None:







            column_names = (



                chunk.columns.tolist()



            )











            for column in column_names:







                missing_counts[



                    column



                ] = 0











                unique_values[



                    column



                ] = set()











                numerical_samples[



                    column



                ] = []











                numerical_samples_seen[



                    column



                ] = 0











                numerical_stats[



                    column



                ] = create_numerical_stat()











        total_rows += len(



            chunk



        )











        # --------------------------------------------------



        # Representative row sample



        # --------------------------------------------------







        rows_for_sample = (







            chunk



            .head(



                min(



                    len(chunk),



                    1000



                )



            )



            .to_dict(



                orient="records"



            )







        )











        sample_rows_seen = (



            reservoir_sample(







                sample_rows,







                rows_for_sample,







                sample_rows_seen







            )



        )











        # --------------------------------------------------



        # Process each column



        # --------------------------------------------------







        for column in chunk.columns:







            series = chunk[



                column



            ]











            missing_counts[



                column



            ] += int(



                series.isna().sum()



            )











            chunk_type = (



                infer_chunk_column_type(



                    series



                )



            )











            previous_type = (



                column_types.get(



                    column



                )



            )











            if (



                chunk_type



                ==



                "numerical"



            ):







                if previous_type != "categorical":







                    column_types[



                        column



                    ] = "numerical"











            elif chunk_type == "categorical":







                column_types[



                    column



                ] = "categorical"











            if (



                column_types.get(



                    column



                )



                ==



                "numerical"



            ):







                numeric_values = (



                    pd.to_numeric(



                        series,



                        errors="coerce"



                    )



                    .dropna()



                )











                numerical_samples_seen[



                    column



                ] = update_numerical_sample(







                    numerical_samples[



                        column



                    ],







                    numeric_values,







                    numerical_samples_seen[



                        column



                    ]







                )











                update_numerical_stat(







                    numerical_stats[



                        column



                    ],







                    numeric_values







                )











            else:







                update_top_categories(







                    categorical_counts.setdefault(



                        column,



                        {}



                    ),







                    series







                )











            # --------------------------------------------------



            # Limited unique-value tracking



            # --------------------------------------------------







            if len(



                unique_values[



                    column



                ]



            ) < MAX_UNIQUE_VALUES_TO_TRACK:







                non_null_values = (







                    series



                    .dropna()



                    .astype(str)



                    .str.strip()



                    .unique()







                )











                remaining_capacity = (







                    MAX_UNIQUE_VALUES_TO_TRACK



                    -



                    len(



                        unique_values[



                            column



                        ]



                    )







                )











                unique_values[



                    column



                ].update(







                    non_null_values[



                        :remaining_capacity



                    ]







                )











    if column_names is None:







        column_names = []











    # --------------------------------------------------



    # Determine final column roles



    # --------------------------------------------------







    for column in column_names:







        column_roles[



            column



        ] = detect_column_role(







            column_name=column,







            column_type=column_types.get(



                column,



                "unknown"



            ),







            unique_count=len(



                unique_values.get(



                    column,



                    set()



                )



            ),







            total_rows=total_rows







        )











    # ==================================================



    # Discover candidate sentinels



    # ==================================================







    numerical_candidates = {}











    textual_placeholders = {}











    for column in column_names:







        if (



            column_types.get(



                column



            )



            ==



            "numerical"



        ):







            candidates = (



                detect_numerical_sentinel_candidates(







                    numerical_samples[



                        column



                    ]







                )



            )











            numerical_candidates[



                column



            ] = candidates











        else:







            textual_placeholders[



                column



            ] = discover_textual_placeholders(







                categorical_counts.get(



                    column,



                    {}



                )







            )











    # ==================================================



    # Validate numerical candidates against complete



    # dataset and obtain exact occurrence counts.



    # ==================================================







    validated_numerical_candidates = (



        validate_candidates(







            file_path,







            chunk_size,







            numerical_candidates







        )



    )











    # --------------------------------------------------



    # Build values that should be excluded from clean



    # numerical statistics and visualizations.



    # --------------------------------------------------







    bad_value_map = (



        build_bad_value_map(







            validated_numerical_candidates







        )



    )











    # Textual placeholders are case-insensitive,



    # therefore normalized strings are used.



    # --------------------------------------------------







    for column, candidates in (



        textual_placeholders.items()



    ):







        normalized_values = set()











        for candidate in candidates:







            if (



                candidate.get(



                    "confidence",



                    0



                )



                >=



                NUMERICAL_SENTINEL_CONFIDENCE_THRESHOLD



            ):







                normalized_values.add(







                    str(



                        candidate[



                            "value"



                        ]



                    )



                    .strip()



                    .lower()







                )











        if normalized_values:







            bad_value_map[



                column



            ] = normalized_values











    # ==================================================



    # PASS 2



    #



    # Calculate clean numerical statistics after



    # detected sentinel values have been identified.



    # ==================================================







    clean_numerical_statistics = (



        calculate_clean_statistics(







            file_path,







            chunk_size,







            column_types,







            bad_value_map







        )



    )











    # --------------------------------------------------



    # Raw numerical statistics



    # --------------------------------------------------







    raw_numerical_statistics = {}











    for column, statistics in (



        numerical_stats.items()



    ):







        raw_numerical_statistics[



            column



        ] = finalize_numerical_statistics(



            statistics



        )











    # --------------------------------------------------



    # Compact categorical statistics



    # --------------------------------------------------







    compact_categories = {}











    for column, counts in (



        categorical_counts.items()



    ):







        sorted_counts = sorted(







            counts.items(),







            key=lambda item:



                item[1],







            reverse=True







        )











        compact_categories[



            column



        ] = [







            {







                "category":



                    value,







                "count":



                    int(count)







            }







            for value, count



            in sorted_counts[



                :MAX_CATEGORY_VALUES



            ]







        ]











    # --------------------------------------------------



    # Remove detected placeholders from the



    # representative sample used for visualization.



    # --------------------------------------------------







    visualization_sample = pd.DataFrame(



        sample_rows



    )











    if not visualization_sample.empty:







        for column in visualization_sample.columns:







            bad_values = (



                bad_value_map.get(



                    column,



                    set()



                )



            )











            if not bad_values:







                continue











            if (



                column_types.get(



                    column



                )



                ==



                "numerical"



            ):







                numeric_values = pd.to_numeric(







                    visualization_sample[



                        column



                    ],







                    errors="coerce"







                )











                visualization_sample[



                    column



                ] = numeric_values.where(







                    ~numeric_values.isin(



                        list(



                            bad_values



                        )



                    )







                )







            else:







                normalized = (







                    visualization_sample[



                        column



                    ]



                    .astype(str)



                    .str.strip()



                    .str.lower()







                )











                mask = normalized.isin(



                    bad_values



                )











                visualization_sample.loc[



                    mask,



                    column



                ] = np.nan











    visualizations = generate_visualizations(







        visualization_sample,







        column_roles,







        bad_value_map







    )











    # --------------------------------------------------



    # Calculate the number of rows affected by detected

    # sentinel or placeholder values.



    # This uses chunked processing so a large CSV is never

    # loaded into memory as one complete DataFrame.



    sentinel_affected_rows = 0



    for chunk in pd.read_csv(



        file_path,



        chunksize=chunk_size,



        keep_default_na=False,



        na_values=[""]



    ):



        row_has_bad_value = pd.Series(



            False,



            index=chunk.index



        )



        for column, bad_values in bad_value_map.items():



            if not bad_values:



                continue



            if column not in chunk.columns:



                continue



            series = chunk[column]



            if column_types.get(column) == "numerical":



                numeric_values = pd.to_numeric(



                    series,



                    errors="coerce"



                )



                mask = numeric_values.isin(



                    list(bad_values)



                )



            else:



                normalized = (



                    series



                    .astype(str)



                    .str.strip()



                    .str.lower()



                )



                mask = normalized.isin(



                    list(bad_values)



                )



            row_has_bad_value = (



                row_has_bad_value



                | mask



            )



        sentinel_affected_rows += int(



            row_has_bad_value.sum()



        )





    # Build compact sentinel findings



    # --------------------------------------------------







    sentinel_columns = {}











    total_sentinels = 0







    total_placeholders = 0














    for column, candidates in (



        validated_numerical_candidates.items()



    ):







        if candidates:







            sentinel_columns[

                column

            ] = {

                "candidate_count":

                    len(candidates),

                "sentinel_count":

                    sum(1 for c in candidates if c.get("classification") == "possible_sentinel"),

                "placeholder_count":

                    sum(1 for c in candidates if c.get("classification") == "possible_placeholder"),

                "sentinel_occurrences":

                    sum(c.get("count", 0) for c in candidates if c.get("classification") == "possible_sentinel"),

                "placeholder_occurrences":

                    sum(c.get("count", 0) for c in candidates if c.get("classification") == "possible_placeholder"),

                "sentinel_candidates":

                    candidates,

                "placeholder_candidates":

                    []

            }












            # Sum occurrence counts for this column's sentinel candidates
            sentinel_occs = sum(c.get("count", 0) for c in candidates if c.get("classification") == "possible_sentinel")
            placeholder_occs = sum(c.get("count", 0) for c in candidates if c.get("classification") == "possible_placeholder")
            total_sentinels += sentinel_occs

            total_placeholders += placeholder_occs












    for column, candidates in (



        textual_placeholders.items()



    ):







        if candidates:







            existing = (



                sentinel_columns.setdefault(







                    column,







                    {







                        "candidate_count":

                            0,

                        "sentinel_count":

                            0,

                        "placeholder_count":

                            0,

                        "sentinel_occurrences":

                            0,

                        "placeholder_occurrences":

                            0,

                        "sentinel_candidates":

                            [],

                        "placeholder_candidates":

                            []
                    }







                )



            )




            existing["candidate_count"] += len(candidates)

            existing["placeholder_candidates"].extend(candidates)

            sentinel_occs = sum(c.get("count", 0) for c in candidates if c.get("classification") == "possible_sentinel")
            placeholder_occs = sum(c.get("count", 0) for c in candidates if c.get("classification") == "possible_placeholder")
            existing["sentinel_count"] += sum(1 for c in candidates if c.get("classification") == "possible_sentinel")
            existing["placeholder_count"] += sum(1 for c in candidates if c.get("classification") == "possible_placeholder")
            existing["sentinel_occurrences"] += sentinel_occs
            existing["placeholder_occurrences"] += placeholder_occs

            total_sentinels += sentinel_occs

            total_placeholders += placeholder_occs












    return {







        "rows":



            total_rows,







        "columns":



            len(column_names),







        "column_names":



            column_names,







        "column_types":



            column_types,







        "column_roles":



            column_roles,







        "missing_counts":



            missing_counts,







        "numerical_statistics":



            {







                column: {







                    "raw_statistics":



                        raw_numerical_statistics.get(



                            column,



                            {}



                        ),







                    "clean_statistics":



                        clean_numerical_statistics.get(



                            column,



                            {}



                        ),







                    "possible_sentinels":



                        validated_numerical_candidates.get(



                            column,



                            []



                        )







                }







                for column



                in column_types







                if column_types[



                    column



                ] == "numerical"







            },







        "categorical_statistics":



            compact_categories,







        "sentinel_detection": {







            "has_issue":



                bool(



                    sentinel_columns



                ),







            "total_candidates":



                (



                    total_sentinels



                    +



                    total_placeholders



                ),







            "total_sentinels":



                total_sentinels,







            "total_placeholders":



                total_placeholders,



            "affected_rows":



                sentinel_affected_rows,



            "affected_row_rate":



                (



                    (



                        sentinel_affected_rows



                        /



                        total_rows



                    )



                    if total_rows



                    else 0.0



                ),







            "columns":



                sentinel_columns







        },







        "sample_rows":



            visualization_sample.to_dict(



                orient="records"



            )



            if not visualization_sample.empty



            else [],







        "sample_size":



            len(



                visualization_sample



            ),







        "file_size_mb":



            get_file_size_mb(



                file_path



            ),







        "processing_mode":



            "chunked",







        "visualizations":



            visualizations







    }

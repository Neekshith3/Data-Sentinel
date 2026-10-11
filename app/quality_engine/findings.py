def create_finding(
    issue_type,
    has_issue,
    count=0,
    affected_columns=None,
    affected_rows=None,
    details=None
):
    """
    Create a standardized data-quality finding.

    Parameters:
        issue_type (str): Type of quality issue.
        has_issue (bool): Whether an issue was detected.
        count (int): Number of affected records.
        affected_columns (list): Columns affected by the issue.
        affected_rows (list): Row indexes affected by the issue.
        details (dict): Additional information.

    Returns:
        dict: Standardized finding.
    """

    return {

        "issue_type":
            issue_type,

        "has_issue":
            has_issue,

        "count":
            count,

        "affected_columns":
            affected_columns or [],

        "affected_rows":
            affected_rows or [],

        "details":
            details or {}

    }


def standardize_finding(
    result
):
    """
    Convert a detector result into a common finding format.
    """

    issue_type = result["issue_type"]


    # --------------------------------------------------
    # Duplicate rows
    # --------------------------------------------------

    if issue_type == "duplicate_rows":

        return create_finding(

            issue_type=issue_type,

            has_issue=result[
                "has_issue"
            ],

            count=result[
                "duplicate_count"
            ],

            affected_rows=(
                result[
                    "duplicate_rows"
                ]
                .index
                .tolist()
            ),

            details={
                "percentage":
                    result[
                        "duplicate_percentage"
                    ]
            }

        )


    # --------------------------------------------------
    # Missing values
    # --------------------------------------------------

    if issue_type == "missing_values":

        affected_columns = []

        total_missing = 0


        for (
            column,
            column_result
        ) in result[
            "columns"
        ].items():

            if column_result[
                "has_issue"
            ]:

                affected_columns.append(
                    column
                )

                total_missing += (
                    column_result[
                        "missing_count"
                    ]
                )


        return create_finding(

            issue_type=issue_type,

            has_issue=result[
                "has_issue"
            ],

            count=total_missing,

            affected_columns=
                affected_columns,

            details=result[
                "columns"
            ]

        )


    # --------------------------------------------------
    # Categorical inconsistencies
    # --------------------------------------------------

    if issue_type == "categorical_inconsistency":

        affected_columns = []

        total_inconsistencies = 0


        for (
            column,
            column_result
        ) in result[
            "columns"
        ].items():

            if column_result[
                "has_issue"
            ]:

                affected_columns.append(
                    column
                )

                total_inconsistencies += len(
                    column_result[
                        "inconsistent_groups"
                    ]
                )


        return create_finding(

            issue_type=issue_type,

            has_issue=result[
                "has_issue"
            ],

            count=total_inconsistencies,

            affected_columns=
                affected_columns,

            details=result[
                "columns"
            ]

        )


    # --------------------------------------------------
    # Data type validation
    # --------------------------------------------------

    if issue_type == "data_type_validation":

        affected_columns = []


        for (
            column,
            column_result
        ) in result[
            "columns"
        ].items():

            if column_result[
                "has_issue"
            ]:

                affected_columns.append(
                    column
                )


        return create_finding(

            issue_type=issue_type,

            has_issue=result[
                "has_issue"
            ],

            affected_columns=
                affected_columns,

            details=result[
                "columns"
            ]

        )


    # --------------------------------------------------
    # Invalid values
    # --------------------------------------------------

    if issue_type == "invalid_values":

        affected_columns = []

        total_invalid = 0


        for (
            column,
            column_result
        ) in result[
            "columns"
        ].items():

            if column_result[
                "has_issue"
            ]:

                affected_columns.append(
                    column
                )

                total_invalid += (
                    column_result[
                        "invalid_count"
                    ]
                )


        return create_finding(

            issue_type=issue_type,

            has_issue=result[
                "has_issue"
            ],

            count=total_invalid,

            affected_columns=
                affected_columns,

            details=result[
                "columns"
            ]

        )


    # --------------------------------------------------
    # Statistical outliers
    # --------------------------------------------------

    if issue_type == "statistical_outliers":

        affected_columns = []

        total_outliers = 0

        details = {}


        for (
            column,
            column_result
        ) in result[
            "columns"
        ].items():

            if column_result[
                "has_issue"
            ]:

                affected_columns.append(
                    column
                )

                total_outliers += (
                    column_result[
                        "outlier_count"
                    ]
                )


            details[column] = {

                "q1":
                    column_result[
                        "q1"
                    ],

                "q3":
                    column_result[
                        "q3"
                    ],

                "iqr":
                    column_result[
                        "iqr"
                    ],

                "lower_bound":
                    column_result[
                        "lower_bound"
                    ],

                "upper_bound":
                    column_result[
                        "upper_bound"
                    ],

                "outlier_count":
                    column_result[
                        "outlier_count"
                    ],

                "has_issue":
                    column_result[
                        "has_issue"
                    ],

                "outlier_rows": (
                    column_result[
                        "outlier_rows"
                    ]
                    .index
                    .tolist()
                )

            }


        return create_finding(

            issue_type=issue_type,

            has_issue=result[
                "has_issue"
            ],

            count=total_outliers,

            affected_columns=
                affected_columns,

            details=details

        )


    # --------------------------------------------------
    # Possible sentinel and placeholder values
    # --------------------------------------------------

    if issue_type == "possible_sentinel_values":

        affected_columns = []

        total_candidates = 0

        details = {}


        for (
            column,
            column_result
        ) in result[
            "columns"
        ].items():

            if column_result[
                "has_issue"
            ]:

                affected_columns.append(
                    column
                )


                total_candidates += (
                    column_result[
                        "candidate_count"
                    ]
                )


                details[column] = {

                    "candidate_count":
                        column_result[
                            "candidate_count"
                        ],

                    "sentinel_count":
                        column_result[
                            "sentinel_count"
                        ],

                    "placeholder_count":
                        column_result[
                            "placeholder_count"
                        ],

                    "sentinel_candidates":
                        column_result[
                            "sentinel_candidates"
                        ],

                    "placeholder_candidates":
                        column_result[
                            "placeholder_candidates"
                        ]

                }


        return create_finding(

            issue_type=issue_type,

            has_issue=result[
                "has_issue"
            ],

            count=total_candidates,

            affected_columns=
                affected_columns,

            details={

                "total_candidates":
                    result[
                        "candidate_count"
                    ],

                "total_sentinels":
                    result[
                        "sentinel_count"
                    ],

                "total_placeholders":
                    result[
                        "placeholder_count"
                    ],

                "columns":
                    details

            }

        )


    # --------------------------------------------------
    # ML anomalies
    # --------------------------------------------------

    if issue_type == "ml_anomalies":

        return create_finding(

            issue_type=issue_type,

            has_issue=result[
                "has_issue"
            ],

            count=result[
                "anomaly_count"
            ],

            affected_rows=result[
                "anomaly_rows"
            ],

            details={

                "anomaly_scores":
                    result[
                        "anomaly_scores"
                    ],

                "feature_columns":
                    result[
                        "feature_columns"
                    ],

                "contamination":
                    result[
                        "contamination"
                    ]

            }

        )


    # --------------------------------------------------
    # Unknown detector
    # --------------------------------------------------

    return create_finding(

        issue_type=issue_type,

        has_issue=result[
            "has_issue"
        ]

    )
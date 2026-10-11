import os
import uuid

import tempfile

import pandas as pd
from typing import Any
from pydantic import BaseModel

from fastapi import (

    APIRouter,

    UploadFile,

    File

)

from app.schemas.analysis import AnalysisResponse, ChatRequest, ChatResponse
from app.ai.assistant import explain as explain_findings, answer_question

from app.services.analysis_service import (

    analyze_dataset as run_analysis

)

from app.profiler.dataset_profiler import (

    load_dataset

)

from app.analytics.large_dataset import (

    analyze_csv_in_chunks

)

router = APIRouter()

# --------------------------------------------------

# Large dataset threshold

# --------------------------------------------------

LARGE_DATASET_THRESHOLD_MB = 100

# --------------------------------------------------

# Convert NumPy/Pandas values into JSON-safe values

# --------------------------------------------------

def make_json_safe(

    value

):

    if isinstance(

        value,

        dict

    ):

        return {

            str(key):

                make_json_safe(item)

            for key, item in value.items()

        }

    if isinstance(

        value,

        (list, tuple)

    ):

        return [

            make_json_safe(item)

            for item in value

        ]

    if hasattr(

        value,

        "item"

    ):

        try:

            return value.item()

        except (

            ValueError,

            TypeError

        ):

            pass

    if pd.isna(

        value

    ):

        return None

    return value

# --------------------------------------------------

# Build compact findings for large datasets

#

# The chunked analyzer already performs the expensive

# dataset-wide detection. We convert those results into

# the same standardized finding shape used by the API.

# --------------------------------------------------

# --------------------------------------------------

# Build compact findings for large datasets

#

# The chunked analyzer already performs the expensive

# dataset-wide detection. We convert those results into

# the same standardized finding shape used by the API.

# --------------------------------------------------

def get_large_sentinel_columns(

    large_analysis

):

    sentinel_detection = large_analysis.get(

        "sentinel_detection",

        {}

    )

    explicit_columns = sentinel_detection.get(

        "affected_columns",

        []

    )

    if isinstance(explicit_columns, list) and explicit_columns:

        return explicit_columns

    detailed_columns = sentinel_detection.get(

        "columns",

        {}

    )

    if not isinstance(detailed_columns, dict):

        return []

    return [

        column

        for column, details in detailed_columns.items()

        if isinstance(details, dict)

        and (

            details.get("has_issue", False)

            or int(details.get("candidate_count", 0)) > 0

            or len(details.get("candidates", [])) > 0

            or len(details.get("sentinel_candidates", [])) > 0

            or len(details.get("placeholder_candidates", [])) > 0

        )

    ]

def build_large_findings(

    large_analysis

):

    findings = []

    missing_counts = (

        large_analysis.get(

            "missing_counts",

            {}

        )

    )

    total_rows = int(

        large_analysis.get(

            "rows",

            0

        )

    )

    missing_columns = {

        column: int(count)

        for column, count in missing_counts.items()

        if int(count) > 0

    }

    missing_total = sum(

        missing_columns.values()

    )

    findings.append({

        "issue_type":

            "missing_values",

        "has_issue":

            missing_total > 0,

        "count":

            missing_total,

        "affected_columns":

            list(

                missing_columns.keys()

            ),

        "affected_rows":

            [],

        "details": {

            "total_rows":

                total_rows,

            "columns":

                missing_columns

        }

    })

    sentinel_detection = (

        large_analysis.get(

            "sentinel_detection",

            {}

        )

    )

    numerical_sentinels = int(

        sentinel_detection.get(

            "total_sentinels",

            0

        )

    )

    textual_placeholders = int(

        sentinel_detection.get(

            "total_placeholders",

            0

        )

    )

     # First use the explicit affected-columns

     # list when the analyzer provides it.

     #

    sentinel_columns = get_large_sentinel_columns(

        large_analysis

    )

    sentinel_count = (

        numerical_sentinels

        +

        textual_placeholders

    )

    findings.append({

        "issue_type":

            "possible_sentinel_values",

        "has_issue":

            sentinel_count > 0,

        "count":

            sentinel_count,

        "affected_columns":

            sentinel_columns,

        "affected_rows":

            [],

        "details": {

            **sentinel_detection,

            "affected_row_count":

                int(

                    sentinel_detection.get(

                        "affected_rows",

                        0

                    )

                ),

            "affected_row_rate":

                float(

                    sentinel_detection.get(

                        "affected_row_rate",

                        0.0

                    )

                )

        }

    })

    return findings

# --------------------------------------------------

# Build compact quality score for large datasets

#

# This score uses dataset-wide affected rates rather

# than raw row-level objects, so the API stays compact.

# --------------------------------------------------

def build_large_quality_score(

    large_analysis

):

    total_rows = int(

        large_analysis.get(

            "rows",

            0

        )

    )

    missing_counts = (

        large_analysis.get(

            "missing_counts",

            {}

        )

    )

    missing_total = sum(

        int(count)

        for count in missing_counts.values()

    )

    sentinel_detection = (

        large_analysis.get(

            "sentinel_detection",

            {}

        )

    )

    sentinel_total = int(

        sentinel_detection.get(

            "affected_rows",

            0

        )

    )

    missing_rate = (

        missing_total / total_rows

        if total_rows > 0

        else 0

    )

    sentinel_rate = (

        sentinel_total / total_rows

        if total_rows > 0

        else 0

    )

    missing_penalty = min(

        50.0,

        missing_rate * 100

    )

    sentinel_penalty = min(

        50.0,

        sentinel_rate * 100

    )

    penalty = (

        missing_penalty

        +

        sentinel_penalty

    )

    score = max(

        0.0,

        round(

            100.0 - penalty,

            2

        )

    )

    severity = "LOW"

    if score < 70:

        severity = "HIGH"

    elif score < 90:

        severity = "MEDIUM"

    return {

        "overall_score":

            score,

        "severity":

            severity,

        "penalty":

            round(

                penalty,

                2

            ),

        "components": {

            "missing_values": {

                "affected_rows":

                    missing_total,

                "affected_rate":

                    round(

                        missing_rate,

                        4

                    ),

                "penalty":

                    round(

                        missing_penalty,

                        2

                    )

            },

            "possible_sentinel_values": {

                "affected_rows":

                    sentinel_total,

                "affected_rate":

                    round(

                        sentinel_rate,

                        4

                    ),

                "penalty":

                    round(

                        sentinel_penalty,

                        2

                    )

            }

        },

        "processing_mode":

            "chunked"

    }

# --------------------------------------------------

# Build ML readiness for large datasets

# --------------------------------------------------

def build_large_ml_readiness(

    large_analysis

):

    column_roles = (

        large_analysis.get(

            "column_roles",

            {}

        )

    )

    numerical_columns = [

        column

        for column, role

        in column_roles.items()

        if role == "numerical"

        or role == "numerical_measurement"

    ]

    categorical_columns = [

        column

        for column, role

        in column_roles.items()

        if role == "categorical"

    ]

    identifier_columns = [

        column

        for column, role

        in column_roles.items()

        if role == "identifier"

        or role == "identifier_like"

    ]

    numerical_statistics = (

        large_analysis.get(

            "numerical_statistics",

            {}

        )

    )

    constant_columns = []

    for column, statistics in (

        numerical_statistics.items()

    ):

        clean_statistics = (

            statistics.get(

                "clean_statistics",

                {}

            )

        )

        if (

            clean_statistics

            and

            clean_statistics.get(

                "count",

                0

            ) > 0

        ):

            if (

                clean_statistics.get(

                    "min"

                )

                ==

                clean_statistics.get(

                    "max"

                )

            ):

                constant_columns.append(

                    column

                )

    sentinel_detection = (

        large_analysis.get(

            "sentinel_detection",

            {}

        )

    )

    sentinel_count = (

        int(

            sentinel_detection.get(

                "total_sentinels",

                0

            )

        )

        +

        int(

            sentinel_detection.get(

                "total_placeholders",

                0

            )

        )

    )

    checks = [

        {

            "check": "numerical_features",

            "passed": len(

                numerical_columns

            ) > 0,

            "count": len(

                numerical_columns

            )

        },

        {

            "check": "categorical_features",

            "passed": True,

            "count": len(

                categorical_columns

            )

        },

        {

            "check": "constant_columns",

            "passed": len(

                constant_columns

            ) == 0,

            "count": len(

                constant_columns

            ),

            "columns":

                constant_columns

        },

        {

            "check": "identifier_columns",

            "passed": True,

            "count": len(

                identifier_columns

            ),

            "columns":

                identifier_columns

        },

        {

            "check": "sentinel_values",

            "passed":

                sentinel_count == 0,

            "count":

                sentinel_count,

            "numerical_sentinels":

                int(

                    sentinel_detection.get(

                        "total_sentinels",

                        0

                    )

                ),

            "textual_placeholders":

                int(

                    sentinel_detection.get(

                        "total_placeholders",

                        0

                    )

                )

        }

    ]

    passed_checks = sum(

        1

        for check in checks

        if check["passed"]

    )

    total_checks = len(

        checks

    )

    readiness_percentage = (

        round(

            (

                passed_checks

                /

                total_checks

            )

            * 100,

            2

        )

        if total_checks > 0

        else 0.0

    )

    return {

        "readiness_percentage":

            readiness_percentage,

        "passed_checks":

            passed_checks,

        "total_checks":

            total_checks,

        "checks":

            checks,

        "processing_mode":

            "chunked"

    }

# --------------------------------------------------

# Build compact understanding for large datasets

# --------------------------------------------------

def build_large_understanding(

    large_analysis

):

    column_roles = large_analysis.get(

        "column_roles",

        {}

    )

    numerical_statistics = large_analysis.get(

        "numerical_statistics",

        {}

    )

    categorical_statistics = large_analysis.get(

        "categorical_statistics",

        {}

    )

    missing_counts = large_analysis.get(

        "missing_counts",

        {}

    )

    total_rows = int(

        large_analysis.get(

            "rows",

            0

        )

    )

    column_types = large_analysis.get(

        "column_types",

        {}

    )

    column_names = large_analysis.get(

        "column_names",

        []

    )

    column_analysis = []

    for column in column_names:

        role = column_roles.get(

            column,

            "unknown"

        )

        missing_count = int(

            missing_counts.get(

                column,

                0

            )

        )

        non_null_count = max(

            total_rows - missing_count,

            0

        )

        missing_percentage = (

            round(

                (

                    missing_count

                    /

                    total_rows

                )

                * 100,

                2

            )

            if total_rows > 0

            else 0.0

        )

        if column in numerical_statistics:

            statistics = numerical_statistics.get(

                column,

                {}

            )

            raw_statistics = statistics.get(

                "raw_statistics",

                {}

            )

            clean_statistics = statistics.get(

                "clean_statistics",

                {}

            )

            possible_sentinels = statistics.get(

                "possible_sentinels",

                []

            )

            unique_count = int(

                raw_statistics.get(

                    "unique_count",

                    0

                )

            )

            column_analysis.append({

                "column":

                    column,

                "type":

                    "numerical",

                "role":

                    role,

                "rows":

                    total_rows,

                "non_null_count":

                    non_null_count,

                "missing_count":

                    missing_count,

                "missing_percentage":

                    missing_percentage,

                "unique_count":

                    unique_count,

                "uniqueness_ratio":

                    (

                        round(

                            unique_count

                            /

                            total_rows,

                            4

                        )

                        if total_rows > 0

                        else 0.0

                    ),

                "raw_statistics":

                    raw_statistics,

                "clean_statistics":

                    clean_statistics,

                "possible_sentinels":

                    possible_sentinels

            })

        elif column in categorical_statistics:

            categories = categorical_statistics.get(

                column,

                []

            )

            unique_count = len(

                categories

            )

            column_analysis.append({

                "column":

                    column,

                "type":

                    "categorical",

                "role":

                    role,

                "rows":

                    total_rows,

                "non_null_count":

                    non_null_count,

                "missing_count":

                    missing_count,

                "missing_percentage":

                    missing_percentage,

                "unique_count":

                    unique_count,

                "uniqueness_ratio":

                    (

                        round(

                            unique_count

                            /

                            total_rows,

                            4

                        )

                        if total_rows > 0

                        else 0.0

                    ),

                "categories":

                    categories

            })

        else:

            column_analysis.append({

                "column":

                    column,

                "type":

                    column_types.get(

                        column,

                        "unknown"

                    ),

                "role":

                    role,

                "rows":

                    total_rows,

                "non_null_count":

                    non_null_count,

                "missing_count":

                    missing_count,

                "missing_percentage":

                    missing_percentage

            })

    return {

        "rows":

            total_rows,

        "columns":

            len(

                column_names

            ),

        "column_analysis":

            column_analysis,

        "column_roles":

            column_roles,

        "numerical_statistics":

            numerical_statistics,

        "categorical_statistics":

            categorical_statistics,

        "sentinel_detection":

            large_analysis.get(

                "sentinel_detection",

                {}

            ),

        "processing_mode":

            "chunked"

    }

# Build compact snapshot

#

# This is intentionally a lightweight snapshot. The

# complete in-memory snapshot/drift system remains used

# by the small-dataset pipeline.

# --------------------------------------------------

def build_large_snapshot(

    large_analysis

):

    return {

        "snapshot_version":

            "large-1.0",

        "dataset": {

            "rows":

                large_analysis.get(

                    "rows",

                    0

                ),

            "columns":

                large_analysis.get(

                    "columns",

                    0

                ),

            "column_names":

                large_analysis.get(

                    "column_names",

                    []

                ),

            "column_types":

                large_analysis.get(

                    "column_types",

                    {}

                ),

            "column_roles":

                large_analysis.get(

                    "column_roles",

                    {}

                )

        },

        "processing": {

            "mode":

                "chunked",

            "file_size_mb":

                large_analysis.get(

                    "file_size_mb",

                    0

                ),

            "sample_size":

                large_analysis.get(

                    "sample_size",

                    0

                )

        }

    }

# --------------------------------------------------

# Health check

# --------------------------------------------------

@router.get(

    "/health"

)

def health_check():

    return {

        "status":

            "success",

        "message":

            "Data Sentinel API is running"

    }

# --------------------------------------------------

# Analyze uploaded dataset

# --------------------------------------------------

@router.post(

    "/analyze",

    response_model=AnalysisResponse

)

async def analyze_dataset(

    file: UploadFile = File(...)

):

    if not file.filename:

        return {

            "status":

                "error",

            "message":

                "No file was uploaded",

            "filename":

                "",

            "profile":

                {},

            "understanding":

                {},

            "findings":

                [],

            "quality_score":

                {},

            "ml_readiness":

                {},

            "snapshot":

                {},

            "drift":

                None,

            "visualizations":

                []

        }

    SUPPORTED_EXTENSIONS = (".csv", ".xlsx", ".xls")

    if not file.filename.lower().endswith(SUPPORTED_EXTENSIONS):

        return {

            "status":

                "error",

            "message":

                "Only CSV and Excel (.xlsx, .xls) files are supported",

            "filename":

                file.filename,

            "profile":

                {},

            "understanding":

                {},

            "findings":

                [],

            "quality_score":

                {},

            "ml_readiness":

                {},

            "snapshot":

                {},

            "drift":

                None,

            "visualizations":

                []

        }

    temporary_file_path = None

    try:

        # --------------------------------------------------

        # Create temporary file

        # --------------------------------------------------

        with tempfile.NamedTemporaryFile(

            delete=False,

            suffix=".csv"

        ) as temporary_file:

            temporary_file_path = (

                temporary_file.name

            )

            # --------------------------------------------------

            # Stream upload to disk.

            #

            # The complete uploaded file is never kept

            # in application memory.

            # --------------------------------------------------

            while True:

                chunk = await file.read(

                    1024 * 1024

                )

                if not chunk:

                    break

                temporary_file.write(

                    chunk

                )

        # --------------------------------------------------

        # Determine uploaded file size

        # --------------------------------------------------

        # --------------------------------------------------
        # Excel upload: convert the first sheet to CSV so the
        # same (chunk-capable) pipeline handles every format.
        # Cells are read as raw text so placeholders such as
        # N/A or NULL survive for the detectors.
        # --------------------------------------------------

        lower_name = file.filename.lower()

        if lower_name.endswith((".xlsx", ".xls")):

            engine = (
                "openpyxl"
                if lower_name.endswith(".xlsx")
                else "xlrd"
            )

            excel_df = pd.read_excel(
                temporary_file_path,
                engine=engine,
                dtype=object,
                keep_default_na=False,
                na_filter=False
            )

            excel_df.to_csv(
                temporary_file_path,
                index=False
            )

        file_size_bytes = os.path.getsize(

            temporary_file_path

        )

        file_size_mb = (

            file_size_bytes

            /

            (1024 * 1024)

        )

        # ==================================================

        # LARGE DATASET PATH

        # ==================================================

        if (

            file_size_mb

            >=

            LARGE_DATASET_THRESHOLD_MB

        ):

            large_analysis = (

                analyze_csv_in_chunks(

                    temporary_file_path

                )

            )

            sentinel_detection = large_analysis.get(

                "sentinel_detection",

                {}

            )

            sentinel_detection["affected_columns"] = (

                get_large_sentinel_columns(

                    large_analysis

                )

            )

            large_analysis["sentinel_detection"] = sentinel_detection

            findings = (

                build_large_findings(

                    large_analysis

                )

            )

            quality_score = (

                build_large_quality_score(

                    large_analysis

                )

            )

            ml_readiness = (

                build_large_ml_readiness(

                    large_analysis

                )

            )

            understanding = (

                build_large_understanding(

                    large_analysis

                )

            )

            snapshot = (

                build_large_snapshot(

                    large_analysis

                )

            )

            response = {

                "status":

                    "success",

                "message":

                    "Large dataset analyzed using chunked processing",

                "filename":

                    file.filename,

                "profile": {

                    "rows":

                        large_analysis[

                            "rows"

                        ],

                    "columns":

                        large_analysis[

                            "columns"

                        ],

                    "column_names":

                        large_analysis[

                            "column_names"

                        ],

                    "column_types":

                        large_analysis[

                            "column_types"

                        ],

                    "column_roles":

                        large_analysis[

                            "column_roles"

                        ],

                    "missing_counts":

                        large_analysis[

                            "missing_counts"

                        ],

                    "numerical_statistics":

                        large_analysis[

                            "numerical_statistics"

                        ],

                    "categorical_statistics":

                        large_analysis[

                            "categorical_statistics"

                        ],

                    "file_size_mb":

                        large_analysis[

                            "file_size_mb"

                        ],

                    "processing_mode":

                        large_analysis[

                            "processing_mode"

                        ],

                    "sample_size":

                        large_analysis[

                            "sample_size"

                        ]

                },

                "understanding":

                    understanding,

                "findings":

                    findings,

                "quality_score":

                    quality_score,

                "ml_readiness":

                    ml_readiness,

                "snapshot":

                    snapshot,

                # --------------------------------------------------

                # Drift requires a previous persisted snapshot.

                #

                # The current upload endpoint does not yet receive

                # one, so null is intentional here.

                # --------------------------------------------------

                "drift":

                    None,

                "visualizations":

                    large_analysis.get(

                        "visualizations",

                        []

                    )

            }

            return make_json_safe(

                response

            )

        # ==================================================

        # SMALL DATASET PATH

        # ==================================================

        # --------------------------------------------------

        # Use the centralized loader.

        #

        # This preserves raw placeholder values such as:

        # N/A, NA, NULL, and None.

        # --------------------------------------------------

        df = load_dataset(

            temporary_file_path

        )

        # --------------------------------------------------

        # Run complete small-dataset analysis.

        # --------------------------------------------------

        analysis = run_analysis(

            df

        )

        response = {

            "status":

                "success",

            "message":

                "Dataset analyzed successfully",

            "filename":

                file.filename,

            "profile":

                analysis[

                    "profile"

                ],

            "understanding":

                analysis[

                    "understanding"

                ],

            "findings":

                analysis[

                    "findings"

                ],

            "quality_score":

                analysis[

                    "quality_score"

                ],

            "ml_readiness":

                analysis[

                    "ml_readiness"

                ],

            "snapshot":

                analysis[

                    "snapshot"

                ],

            "drift":

                analysis[

                    "drift"

                ],

            "visualizations":

                analysis[

                    "visualizations"

                ]

        }

        return make_json_safe(

            response

        )

    finally:

        # --------------------------------------------------

        # Remove temporary file

        # --------------------------------------------------

        if (

            temporary_file_path

            and

            os.path.exists(

                temporary_file_path

            )

        ):

            os.remove(

                temporary_file_path

            )


# --------------------------------------------------
# AI explanation layer
#
# Receives the structured findings produced by the
# deterministic engine and returns explanations and
# recommended actions. Detection is never done here.
# --------------------------------------------------

class ExplainRequest(BaseModel):

    findings: list[dict[str, Any]]

    quality_score: dict[str, Any] = {}

    ml_readiness: dict[str, Any] = {}


@router.post("/explain")
def explain_analysis(
    request: ExplainRequest
):

    return make_json_safe(
        explain_findings(
            request.findings,
            request.quality_score,
            request.ml_readiness
        )
    )


# --------------------------------------------------
# Dataset-aware AI chat
# --------------------------------------------------
#
# The chat endpoint receives a user question along with the current
# analysis context (findings, quality_score, ml_readiness, profile)
# and conversation history, and returns a grounded response from the AI assistant.
# --------------------------------------------------


@router.post("/chat")
async def chat_analysis(
    request: ChatRequest
):
    # Generate a conversation ID
    import uuid
    conversation_id = str(uuid.uuid4())[:8]

    # Extract analysis context from request
    # The frontend sends the analysis context with each request
    findings = request.findings
    quality_score = request.quality_score
    ml_readiness = request.ml_readiness
    profile = request.profile
    dataset_name = request.dataset_name
    
    # Use question if message is not provided (frontend sends 'question')
    question = request.message if request.message else request.question
    
    # Convert conversation history to the format expected by answer_question
    # The frontend sends conversation as list of {role, content}
    conversation_history = [{"role": msg.role, "content": msg.content} for msg in request.conversation]
    
    # Call the answer_question function with the analysis context and conversation history
    answer = answer_question(
        question=question,
        findings=findings,
        quality_score=quality_score,
        ml_readiness=ml_readiness,
        profile=profile,
        conversation_history=conversation_history
    )
    
    # Prepend conversation context to the answer if we have history
    # The answer_question function doesn't currently use conversation history,
    # but we include it in the response for reference
    response = {
        "response": answer.get("answer", "I couldn't generate a response. Please try again."),
        "conversation_id": conversation_id,
        "grounded": True
    }

    return make_json_safe(response)

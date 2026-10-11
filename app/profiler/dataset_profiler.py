import os

import numpy as np
import pandas as pd

from app.profiler.missing_values import (
    convert_missing_tokens_to_nan,
    count_missing_values,
    count_missing_values_df,
)


# --------------------------------------------------
# Delimiter detection
# --------------------------------------------------

def detect_delimiter(file_path, sample_size=1024):
    """
    Detect the delimiter used in a delimited text file.
    
    Tries common delimiters and selects the one that produces
    the most consistent column count across rows.
    
    Args:
        file_path: Path to the file
        sample_size: Number of bytes to sample for detection
        
    Returns:
        tuple: (delimiter, confidence_score)
    """
    # Read a sample of the file
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        sample = f.read(sample_size)
    
    if not sample:
        return ',', 0.0
    
    # Common delimiters to test
    delimiters = [',', ';', '\t', '|', ':']
    
    best_delimiter = ','
    best_score = 0.0
    
    lines = sample.strip().split('\n')
    if len(lines) < 2:
        return ',', 0.5  # Not enough data to detect
    
    for delim in delimiters:
        # Count columns for each line
        col_counts = [len(line.split(delim)) for line in lines if line.strip()]
        if not col_counts:
            continue
            
        # Score based on consistency (all rows should have same column count)
        # and number of columns (more columns is better, up to a point)
        unique_counts = set(col_counts)
        if len(unique_counts) == 1:
            # Perfect consistency
            num_cols = col_counts[0]
            if num_cols > 1:
                score = 1.0 + min(num_cols / 20, 0.5)  # Bonus for more columns
            else:
                score = 0.1
        else:
            # Inconsistent - check if most rows agree
            from collections import Counter
            count_freq = Counter(col_counts)
            most_common_count, most_common_freq = count_freq.most_common(1)[0]
            consistency = most_common_freq / len(col_counts)
            if consistency > 0.8 and most_common_count > 1:
                score = consistency * 0.8
            else:
                score = 0.0
        
        if score > best_score:
            best_score = score
            best_delimiter = delim
    
    return best_delimiter, best_score


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

def load_dataset(
    file_path
):

    # --------------------------------------------------
    # Validate file existence
    # --------------------------------------------------

    if not os.path.exists(file_path):

        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )


    # --------------------------------------------------
    # Validate file format
    # --------------------------------------------------

    ext = file_path.lower().split('.')[-1]
    supported_formats = {'csv', 'tsv', 'txt', 'xlsx', 'xls'}
    if ext not in supported_formats:
        raise ValueError(
            f"Unsupported file format: .{ext}. Supported formats: {', '.join(supported_formats)}"
        )

    # --------------------------------------------------
    # Read the file based on format
    # --------------------------------------------------
    
    if ext in {'xlsx', 'xls'}:
        # Excel file - read first sheet by default
        # keep_default_na=False prevents Pandas from automatically converting
        # values like N/A, NA, NULL, None into NaN
        df = pd.read_excel(
            file_path,
            sheet_name=0,  # First sheet
            keep_default_na=False,
            na_filter=False
        )
    else:
        # Delimited text file (CSV, TSV, TXT)
        # --------------------------------------------------
        # Detect delimiter
        # --------------------------------------------------
        
        delimiter, confidence = detect_delimiter(file_path)
        
        # --------------------------------------------------
        # Read the file while preserving raw values.
        #
        # keep_default_na=False prevents Pandas from
        # automatically converting values such as:
        #
        # N/A
        # NA
        # NULL
        # None
        #
        # into NaN before Data Sentinel analyzes them.
        # We also preserve empty strings.
        # --------------------------------------------------

        df = pd.read_csv(
            file_path,
            sep=delimiter,
            keep_default_na=False,
            na_filter=False  # Don't convert anything to NaN automatically
        )

        # If delimiter detection failed (single column), try comma as fallback
        if len(df.columns) == 1 and delimiter != ',':
            try:
                df = pd.read_csv(
                    file_path,
                    sep=',',
                    keep_default_na=False,
                    na_filter=False
                )
                delimiter = ','
            except:
                pass

    # Now explicitly convert common missing representations to NaN
    # for consistent downstream processing
    missing_tokens = {"", " ", "nan", "NaN", "NAN", "na", "NA", "N/A", "null", "NULL", "None", "none", "missing", "?", "-"}
    
    for col in df.columns:
        dtype = df[col].dtype
        is_string_dtype = dtype == object or isinstance(dtype, pd.StringDtype)
        if is_string_dtype:
            df[col] = df[col].apply(
                lambda x: np.nan if (isinstance(x, str) and x.strip().lower() in missing_tokens) else x
            )
            
            # Also convert empty/whitespace-only strings to NaN
            df[col] = df[col].apply(
                lambda x: np.nan if (isinstance(x, str) and x.strip() == "") else x
            )

    return df


# --------------------------------------------------
# Profile dataset
# --------------------------------------------------

def profile_dataset(
    df
):

    rows = len(df)

    columns = len(df.columns)


    duplicate_count = int(
        df.duplicated().sum()
    )


    duplicate_percentage = (
        (
            duplicate_count
            /
            rows
        )
        *
        100
        if rows > 0
        else 0
    )


    # Also count rows involved in duplicate groups (including the first occurrence)
    duplicate_rows_involved = int(
        df.duplicated(keep=False).sum()
    )


    column_details = {}


    for column in df.columns:

        series = df[column]

        total_missing, empty_count, nan_count = count_missing_values(series)

        unique_count = int(
            series.nunique(
                dropna=True
            )
        )

        non_null_unique_count = int(
            series.dropna().nunique()
        )

        column_information = {

            "data_type":
                str(
                    series.dtype
                ),

            "missing_count":
                total_missing,

            "missing_percentage": (
                (
                    total_missing
                    /
                    rows
                )
                *
                100
                if rows > 0
                else 0
            ),

            "unique_count":
                unique_count,

            "non_null_unique_count":
                non_null_unique_count,

            "empty_count":
                empty_count

        }


        # --------------------------------------------------
        # Numerical statistics
        # --------------------------------------------------

        if pd.api.types.is_numeric_dtype(
            series
        ):

            numeric_series = (
                pd.to_numeric(
                    series,
                    errors="coerce"
                )
                .dropna()
            )


            if not numeric_series.empty:

                column_information[
                    "min"
                ] = float(
                    numeric_series.min()
                )


                column_information[
                    "max"
                ] = float(
                    numeric_series.max()
                )


                column_information[
                    "mean"
                ] = float(
                    numeric_series.mean()
                )


                column_information[
                    "median"
                ] = float(
                    numeric_series.median()
                )


        column_details[column] = (
            column_information
        )


    return {

        "rows":
            rows,

        "columns":
            columns,

        "column_names":
            df.columns.tolist(),

        "duplicate_count":
            duplicate_count,

        "duplicate_rows_involved":
            duplicate_rows_involved,

        "duplicate_percentage":
            duplicate_percentage,

        "column_details":
            column_details

    }


# --------------------------------------------------
# Display profile
# --------------------------------------------------

def display_profile(
    profile
):

    print(
        "\nDataset Profile"
    )


    print(
        "-------------------------"
    )


    print(
        f"Rows: {profile['rows']}"
    )


    print(
        f"Columns: {profile['columns']}"
    )


    print(
        "Column Details:"
    )


    for column, details in (
        profile[
            "column_details"
        ].items()
    ):

        print(
            f"\n{column}"
        )


        print(
            f"  Type: {details['data_type']}"
        )


        print(
            f"  Missing: {details['missing_count']} ({details['missing_percentage']:.1f}%)"
        )


        print(
            f"  Unique: {details['unique_count']} (non-null: {details.get('non_null_unique_count', 'N/A')})"
        )


        if "min" in details:

            print(
                f"  Min: {details['min']}"
            )


            print(
                f"  Max: {details['max']}"
            )


            print(
                f"  Mean: {details['mean']}"
            )


            print(
                f"  Median: {details['median']}"
            )
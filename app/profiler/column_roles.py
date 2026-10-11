import re

import pandas as pd


# --------------------------------------------------
# Identifier keywords
# --------------------------------------------------

IDENTIFIER_KEYWORDS = {
    "id",
    "ids",
    "uuid",
    "guid",
    "identifier",
    "key",
    "code",
    "no",
    "num",
    "number",
    "index",
    "idx",
    "ref",
    "reference",
    "serial"
}


# --------------------------------------------------
# Measurement keywords
# --------------------------------------------------

MEASUREMENT_KEYWORDS = {
    "age",
    "amount",
    "balance",
    "count",
    "distance",
    "height",
    "humidity",
    "income",
    "latitude",
    "longitude",
    "pressure",
    "precipitation",
    "price",
    "quantity",
    "rainfall",
    "rate",
    "revenue",
    "salary",
    "score",
    "speed",
    "temperature",
    "wind",
    "weight"
}


# --------------------------------------------------
# Postal code keywords
# --------------------------------------------------

POSTAL_KEYWORDS = {
    "pincode",
    "zipcode",
    "zip_code",
    "postal",
    "postcode",
    "zip"
}


# --------------------------------------------------
# Email/contact keywords
# --------------------------------------------------

EMAIL_KEYWORDS = {
    "email",
    "e_mail",
    "mail",
    "contact_email"
}


# --------------------------------------------------
# Date/time keywords
# --------------------------------------------------

DATE_KEYWORDS = {
    "date",
    "datetime",
    "timestamp",
    "time",
    "created",
    "updated",
    "modified",
    "order_date",
    "order_time"
}


# --------------------------------------------------
# Normalize column name
# --------------------------------------------------

def normalize_column_name(
    column_name
):

    name = (
        str(column_name)
        .strip()
        .lower()
    )

    name = re.sub(
        r"[^a-z0-9]+",
        "_",
        name
    )


    return name.strip("_")


# --------------------------------------------------
# Get column name tokens
# --------------------------------------------------

# Splits a name on separators AND camelCase humps, so concatenated
# identifier names such as "InvoiceNo", "CustomerID", or "StockCode"
# yield the tokens ['invoice','no'], ['customer','id'], ['stock','code']
# and can be matched by the shared keyword vocabularies below. Without
# this, every concatenated name is one opaque token and identifier
# evidence is silently lost.
_CAMEL_HUMP = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z0-9]*|[a-z0-9]+")


def get_column_tokens(
    column_name
):

    humps = _CAMEL_HUMP.findall(
        str(column_name).strip()
    )

    return [
        hump.lower()
        for hump
        in humps
        if hump
    ]


# --------------------------------------------------
# Check whether column name suggests identifier
# --------------------------------------------------

def has_identifier_name(
    column_name
):

    tokens = (
        get_column_tokens(
            column_name
        )
    )


    return any(
        token in IDENTIFIER_KEYWORDS
        for token in tokens
    )


# --------------------------------------------------
# Check whether column name suggests measurement
# --------------------------------------------------

def has_measurement_name(
    column_name
):

    tokens = (
        get_column_tokens(
            column_name
        )
    )


    return any(
        token in MEASUREMENT_KEYWORDS
        for token in tokens
    )


# --------------------------------------------------
# Check whether column name suggests postal code
# --------------------------------------------------

def has_postal_name(
    column_name
):

    tokens = (
        get_column_tokens(
            column_name
        )
    )


    return any(
        token in POSTAL_KEYWORDS
        for token in tokens
    )


# --------------------------------------------------
# Check whether column name suggests email
# --------------------------------------------------

def has_email_name(
    column_name
):

    tokens = (
        get_column_tokens(
            column_name
        )
    )


    return any(
        token in EMAIL_KEYWORDS
        for token in tokens
    )


# --------------------------------------------------
# Check whether column name suggests date/time
# --------------------------------------------------

def has_date_name(
    column_name
):

    tokens = (
        get_column_tokens(
            column_name
        )
    )


    return any(
        token in DATE_KEYWORDS
        for token in tokens
    )


# --------------------------------------------------
# Check whether values look like emails
# --------------------------------------------------

def looks_like_email(
    series
):
    """Check if values in series match email pattern."""
    email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    non_null = series.dropna().astype(str)
    if non_null.empty:
        return False
    # Sample up to 100 values for efficiency
    sample = non_null.head(100)
    matches = sum(1 for v in sample if email_pattern.match(v))
    return matches / len(sample) > 0.8


# --------------------------------------------------
# Check whether values look like postal codes
# --------------------------------------------------

def looks_like_postal_code(
    series
):
    """Check if values in series match postal code pattern."""
    # Indian PIN codes (6 digits) or US ZIP codes (5 digits)
    postal_pattern = re.compile(r'^\d{5,6}$')
    non_null = series.dropna().astype(str)
    if non_null.empty:
        return False
    sample = non_null.head(100)
    matches = sum(1 for v in sample if postal_pattern.match(v.strip()))
    return matches / len(sample) > 0.8


# --------------------------------------------------
# Check whether values look like dates
# --------------------------------------------------

def looks_like_date(
    series
):
    """Check if values in series can be parsed as dates."""
    non_null = series.dropna().astype(str)
    if non_null.empty:
        return False
    sample = non_null.head(100)
    parsed = 0
    for v in sample:
        try:
            pd.to_datetime(v)
            parsed += 1
        except:
            pass
    return parsed / len(sample) > 0.8


# --------------------------------------------------
# Check whether numerical values resemble an ID
# --------------------------------------------------

def looks_like_sequential_identifier(
    series
):

    numeric_series = (

        pd.to_numeric(
            series,
            errors="coerce"
        )
        .dropna()

    )


    if numeric_series.empty:

        return False


    unique_values = (

        numeric_series
        .drop_duplicates()
        .sort_values()

    )


    if len(
        unique_values
    ) < 3:

        return False


    differences = (

        unique_values
        .diff()
        .dropna()

    )


    if differences.empty:

        return False


    sequential_ratio = (

        (
            differences == 1
        )
        .mean()

    )


    return (

        sequential_ratio
        >=
        0.90

    )


# --------------------------------------------------
# Detect column role (enhanced with semantic inference)
# --------------------------------------------------

def detect_column_role(
    column_name,
    series
):

    non_null = (
        series
        .dropna()
    )


    if non_null.empty:

        return "unknown"


    if pd.api.types.is_bool_dtype(
        series
    ):

        return "boolean"


    if pd.api.types.is_datetime64_any_dtype(
        series
    ):

        return "datetime"


    explicit_identifier = (
        has_identifier_name(
            column_name
        )
    )


    explicit_measurement = (
        has_measurement_name(
            column_name
        )
    )
    
    explicit_postal = (
        has_postal_name(
            column_name
        )
    )
    
    explicit_email = (
        has_email_name(
            column_name
        )
    )
    
    explicit_date = (
        has_date_name(
            column_name
        )
    )


    # First check for semantic types based on column name + value patterns
    if explicit_postal and looks_like_postal_code(series):
        return "postal_code"
    
    if explicit_email and looks_like_email(series):
        return "email"
    
    if explicit_date and looks_like_date(series):
        return "datetime"
    
    # Also check for numerical data stored as text (e.g., quantity, price with missing values)
    # Try to convert to numeric and see if most values parse successfully
    if (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
    ):
        non_null = series.dropna().astype(str)
        if not non_null.empty:
            sample = non_null.head(100)
            # Try to convert to numeric
            numeric_attempt = pd.to_numeric(sample, errors="coerce")
            numeric_success_rate = numeric_attempt.notna().mean()
            
            # If >80% convert to numeric and column name suggests measurement, treat as numerical
            if numeric_success_rate > 0.8 and explicit_measurement:
                return "numerical_measurement"


    if pd.api.types.is_numeric_dtype(
        series
    ):

        if explicit_identifier:

            return "identifier"


        # Measurement names take precedence over the
        # sequential-value identifier heuristic.
        #
        # This prevents columns such as humidity,
        # temperature, pressure, and salary from being
        # classified as identifiers merely because their
        # values happen to be sequential in a small sample.

        if explicit_measurement:

            return "numerical_measurement"


        if looks_like_sequential_identifier(
            series
        ):

            return "identifier"


        return "numerical_measurement"


    if (
        pd.api.types.is_object_dtype(
            series
        )

        or

        pd.api.types.is_string_dtype(
            series
        )

    ):

        # Check value-based semantic types for object/string columns
        if looks_like_email(series):
            return "email"

        if looks_like_date(series):
            return "datetime"

        unique_count = (
            non_null.nunique()
        )


        row_count = (
            len(non_null)
        )


        uniqueness_ratio = (

            unique_count
            /
            row_count

            if row_count > 0

            else 0

        )


        # Identifier NAME evidence beats a bare numeric value pattern:
        # invoice numbers, stock codes, and customer identifiers often
        # look like 5-6 digit strings but are never postal codes.
        if explicit_identifier:

            return "identifier"


        # Postal codes require actual postal evidence: a postal-style
        # name (handled earlier) OR the numeric postal pattern on a
        # NON-identifier column whose values repeat. Near-unique 5-6
        # digit strings are identifiers, not postal codes.
        if looks_like_postal_code(series) and uniqueness_ratio < 0.98:

            return "postal_code"


        if (
            uniqueness_ratio >= 0.98

            and

            unique_count >= 20

        ):

            return "identifier_like"


        return "categorical"


    return "unknown"


# --------------------------------------------------
# Get primary role and semantic type for display
# --------------------------------------------------

def get_role_display(role):
    """
    Convert internal role to display-friendly primary role + semantic type.
    
    Returns tuple: (primary_role, semantic_type)
    """
    role_mapping = {
        "identifier": ("Identifier", "Numeric ID"),
        "identifier_like": ("Identifier", "High-cardinality ID"),
        "numerical_measurement": ("Numerical", "Measurement"),
        "numerical": ("Numerical", "Measurement"),
        "categorical": ("Categorical", "Category"),
        "datetime": ("Date/Time", "Timestamp"),
        "email": ("Contact", "Email"),
        "postal_code": ("Postal Code", "Postal Identifier"),
        "boolean": ("Boolean", "True/False"),
        "unknown": ("Unknown", "Unknown"),
    }
    return role_mapping.get(role, (role.replace("_", " ").title(), role))


def analyze_column_roles(
    df
):

    roles = {}


    for column in df.columns:

        roles[column] = (

            detect_column_role(

                column,

                df[column]

            )

        )


    return roles
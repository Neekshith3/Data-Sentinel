import pandas as pd


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MINIMUM_OCCURRENCES = 2


# --------------------------------------------------
# Known sentinel patterns (common missing value encodings)
# These are checked explicitly regardless of statistical evidence
# --------------------------------------------------

KNOWN_NUMERICAL_SENTINELS = {
    -9999, -999, -99, -9,
    9999, 99999, 999999,
    -99999, -999999,
}

KNOWN_TEXT_PLACEHOLDERS = {
    "na", "n/a", "nan", "null", "none", "unknown",
    "not available", "not_applicable", "missing", "?", "-",
    "na ", " n/a", " na",  # with whitespace variations
}


# --------------------------------------------------
# Sentinel detection cache
# --------------------------------------------------

# Simple in-memory cache for sentinel detection results
# Key: (column_name, series_hash), Value: detection result
_sentinel_cache = {}


def _get_series_hash(series):
    """Generate a hash for a series based on its content."""
    import hashlib
    try:
        # Create a hash from the series values (sample for large series)
        sample = series.dropna().astype(str).head(1000).str.cat(sep='|')
        return hashlib.md5(sample.encode()).hexdigest()
    except Exception:
        return str(id(series))


def clear_sentinel_cache():
    """Clear the sentinel detection cache."""
    global _sentinel_cache
    _sentinel_cache.clear()


def get_cached_sentinel_detection(series, column_name):
    """Get cached sentinel detection result or compute and cache it."""
    global _sentinel_cache
    series_hash = _get_series_hash(series)
    cache_key = (column_name, series_hash)
    
    if cache_key in _sentinel_cache:
        return _sentinel_cache[cache_key]
    
    result = detect_possible_sentinels(series)
    _sentinel_cache[cache_key] = result
    return result


# --------------------------------------------------
# Detect textual placeholder patterns
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


    placeholder_values = {
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


    return (
        normalized_value
        in placeholder_values
    )


# --------------------------------------------------
# Calculate numerical evidence for sentinel detection
# --------------------------------------------------
#
# SENTINEL SEMANTICS:
# A sentinel is a VALUE THAT ENCODES MISSING DATA.
# Statistical extremeness (outliers) is NOT sentinel evidence.
# Only known missing-data patterns + repetition are evidence.
#
# This function ONLY evaluates KNOWN SENTINEL PATTERNS.
# Statistical outlier detection is handled separately by outliers.py.
# --------------------------------------------------

def calculate_numerical_evidence(
    value,
    count,
    statistics
):
    """
    Evaluate whether a numerical value is a known sentinel pattern.
    
    Returns evidence dict if value matches a known sentinel pattern.
    Returns None for all other values (including statistical outliers).
    
    Known sentinel patterns: -9999, -999, -99, -9, 9999, 99999, 999999, -99999, -999999
    """
    
    # Check if this value is a known sentinel pattern
    val = float(value)
    if val not in KNOWN_NUMERICAL_SENTINELS:
        # Not a known sentinel pattern - no evidence
        return None
    
    # Known sentinel pattern detected
    repeated = count >= MINIMUM_OCCURRENCES
    
    signals = ["known_sentinel_pattern"]
    if repeated:
        signals.append("repeated_value")
    
    # Confidence based on how common/reliable this sentinel pattern is
    if val == -9999:
        confidence = 0.95
    elif val in (-999, -99, -9):
        confidence = 0.90
    elif val in (9999, 99999, 999999):
        confidence = 0.85
    else:
        confidence = 0.80
    
    return {
        "robust_score": None,  # Not used for sentinel evidence
        "extreme_iqr": False,  # Statistical extremeness is NOT sentinel evidence
        "repeated": bool(repeated),
        "signals": signals,
        "confidence": float(confidence)
    }


# --------------------------------------------------
# Detect numerical sentinel candidates (known patterns only)
# --------------------------------------------------

def detect_numerical_sentinels(
    series
):
    """
    Detect known numerical sentinel patterns in a series.
    
    Only values matching KNOWN_NUMERICAL_SENTINELS are flagged.
    Statistical outliers are NOT considered sentinels.
    """
    
    numeric_series = (
        pd.to_numeric(
            series,
            errors="coerce"
        )
        .dropna()
    )

    if numeric_series.empty:
        return []

    value_counts = numeric_series.value_counts()
    candidates = []

    for value, count in value_counts.items():
        val = float(value)
        if val in KNOWN_NUMERICAL_SENTINELS:
            # Calculate confidence based on how common this sentinel is
            # -9999 is the most common, so highest confidence
            if val == -9999:
                confidence = 0.95
            elif val in (-999, -99, -9):
                confidence = 0.90
            elif val in (9999, 99999, 999999):
                confidence = 0.85
            else:
                confidence = 0.80

            candidates.append({
                "value": val,
                "count": int(count),
                "confidence": confidence,
                "robust_score": None,
                "extreme_iqr": False,
                "repeated": True,
                "signals": ["known_sentinel_pattern", "repeated_value"],
                "classification": "possible_sentinel",
                "reason": f"Value {val} is a known missing data encoding pattern"
            })

    return candidates


# --------------------------------------------------
# Calculate textual evidence
# --------------------------------------------------

def calculate_text_evidence(
    value,
    count
):

    signals = []


    # --------------------------------------------------
    # Placeholder pattern is the primary signal.
    # --------------------------------------------------

    if looks_like_placeholder(
        value
    ):

        signals.append(
            "placeholder_pattern"
        )


    # --------------------------------------------------
    # Repetition is only supporting information.
    # --------------------------------------------------

    repeated = (
        count
        >=
        MINIMUM_OCCURRENCES
    )


    if repeated:

        signals.append(
            "repeated_value"
        )


    # --------------------------------------------------
    # A normal categorical value should not become
    # a placeholder just because it repeats.
    # --------------------------------------------------

    if "placeholder_pattern" not in signals:

        return None


    confidence = 0.80


    if repeated:

        confidence += 0.20


    return {

        "signals":
            signals,

        "confidence":
            min(
                1.0,
                confidence
            ),

        "repeated":
            repeated

    }


# --------------------------------------------------
# Detect textual sentinel candidates
# --------------------------------------------------

def detect_text_sentinels(
    series
):

    values = (
        series
        .dropna()
        .astype(str)
    )


    if values.empty:

        return []


    value_counts = (
        values
        .str.strip()
        .str.lower()
        .value_counts()
    )


    candidates = []


    for value, count in (
        value_counts.items()
    ):

        evidence = (
            calculate_text_evidence(
                value,
                int(count)
            )
        )


        if evidence is None:

            continue


        candidates.append(
            {
                "value":
                    value,

                "count":
                    int(count),

                "confidence":
                    evidence[
                        "confidence"
                    ],

                "repeated":
                    evidence[
                        "repeated"
                    ],

                "signals":
                    evidence[
                        "signals"
                    ],

                "classification":
                    "possible_placeholder",

                "reason":
                    "Text value matches a possible placeholder pattern"

            }
        )


    return candidates


# --------------------------------------------------
# Detect possible sentinels (known patterns only)
# --------------------------------------------------

def detect_possible_sentinels(
    series
):
    if pd.api.types.is_numeric_dtype(
        series
    ):
        return detect_numerical_sentinels(series)

    return detect_text_sentinels(series)


def get_cached_sentinel_detection(series, column_name):
    """Get cached sentinel detection result or compute and cache it."""
    global _sentinel_cache
    series_hash = _get_series_hash(series)
    cache_key = (column_name, series_hash)
    
    if cache_key in _sentinel_cache:
        return _sentinel_cache[cache_key]
    
    result = detect_possible_sentinels(series)
    _sentinel_cache[cache_key] = result
    return result
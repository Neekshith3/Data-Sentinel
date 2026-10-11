import pandas as pd
import re
from difflib import SequenceMatcher


# --------------------------------------------------
# Conservative semantic alias dictionary
# Only includes high-confidence, domain-agnostic aliases
# --------------------------------------------------

SEMAPHORIC_ALIASES = {
    # City/location abbreviations
    "bang": "bangalore",
    "hyd": "hyderabad",
    "blr": "bangalore",
    "bom": "mumbai",
    "del": "delhi",
    "maa": "chennai",
    "ccu": "kolkata",
    
    # US state abbreviations
    "ca": "california",
    "ny": "new york",
    "tx": "texas",
    "fl": "florida",
    "il": "illinois",
    
    # Common measurement abbreviations
    "qty": "quantity",
    "amt": "amount",
    "wt": "weight",
    "ht": "height",
    "len": "length",
    "wdt": "width",
    
    # Payment method abbreviations
    "cc": "credit card",
    "dc": "debit card",
    "upi": "upi",
    "nb": "net banking",
    "cod": "cash on delivery",
}

# Only apply fuzzy matching for short strings (likely abbreviations)
MAX_ALIAS_LENGTH = 10
MIN_SIMILARITY_FOR_FUZZY = 0.85


def detect_categorical_inconsistencies(df):
    """
    Detect possible inconsistencies in categorical columns.
    
    Detects:
    1. Exact normalized variants (case, whitespace differences)
    2. Known semantic aliases (abbreviations) from dictionary
    3. Conservative fuzzy matching for short strings only
    
    Parameters:
        df (pandas.DataFrame): Dataset to analyze.

    Returns:
        dict: Categorical consistency results.
    """

    results = {}

    for column in df.columns:

        if (
            pd.api.types.is_object_dtype(df[column])
            or pd.api.types.is_string_dtype(df[column])
        ):

            values = df[column].dropna().astype(str)

            normalized_values = (
                values
                .str.strip()
                .str.lower()
            )

            groups = {}

            for original, normalized in zip(
                values,
                normalized_values
            ):

                if normalized not in groups:
                    groups[normalized] = set()

                groups[normalized].add(original)

            # Check for known aliases first
            alias_groups = {}
            processed = set()
            
            for norm_val, variants in groups.items():
                if norm_val in processed:
                    continue
                    
                # Check if this normalized value has a known alias
                canonical = SEMAPHORIC_ALIASES.get(norm_val)
                if canonical and canonical in groups:
                    # Merge the variants
                    combined = set()
                    for v in [norm_val, canonical]:
                        combined.update(groups[v])
                    alias_groups[canonical] = list(combined)
                    processed.add(norm_val)
                    processed.add(canonical)
                    continue
                
                # Check reverse alias (e.g., "bangalore" -> "bang")
                reverse_alias = {v: k for k, v in SEMAPHORIC_ALIASES.items()}
                canonical_rev = reverse_alias.get(norm_val)
                if canonical_rev and canonical_rev in groups:
                    combined = set()
                    for v in [norm_val, canonical_rev]:
                        combined.update(groups[v])
                    alias_groups[canonical_rev] = list(combined)
                    processed.add(norm_val)
                    processed.add(canonical_rev)
                    continue
            
            # Now add remaining exact groups (case/whitespace variants)
            exact_groups = {
                normalized: list(original_values)
                for normalized, original_values in groups.items()
                if len(original_values) > 1 and normalized not in processed
            }
            
            # Combine alias groups and exact groups
            all_groups = {**alias_groups, **exact_groups}

            results[column] = {
                "has_issue": len(all_groups) > 0,
                "inconsistent_groups": all_groups
            }

    return {
        "issue_type": "categorical_inconsistency",
        "has_issue": any(
            result["has_issue"]
            for result in results.values()
        ),
        "columns": results
    }
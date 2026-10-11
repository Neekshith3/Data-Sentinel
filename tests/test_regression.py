"""
Regression tests for Data Sentinel - ecommerce_dirty_data.csv
"""
import pytest
import pandas as pd
import numpy as np
from app.profiler.dataset_profiler import load_dataset
from app.services.analysis_service import analyze_dataset


@pytest.fixture(scope="module")
def dataset():
    """Load the ecommerce dirty dataset."""
    return load_dataset('data/sample/ecommerce_dirty_data.csv')


@pytest.fixture(scope="module")
def analysis_result(dataset):
    """Run full analysis on the dataset."""
    return analyze_dataset(dataset)


class TestDatasetShape:
    """Test basic dataset properties."""

    def test_shape(self, dataset):
        assert dataset.shape == (228, 13), f"Expected (228, 13), got {dataset.shape}"

    def test_columns(self, dataset):
        expected_cols = [
            "order_id", "firstname", "lastname", "email", "product",
            "quantity", "price", "city", "state", "pincode",
            "order_date", "payment_method", "status"
        ]
        assert list(dataset.columns) == expected_cols


class TestMissingValues:
    """Test missing value detection."""

    def test_missing_counts(self, analysis_result):
        profile = analysis_result["profile"]
        details = profile["column_details"]

        # Expected missing counts from ground truth
        expected_missing = {
            "email": 26,
            "quantity": 44,
            "price": 42,
            "city": 26,
            "pincode": 24,
            "payment_method": 37,
        }

        for col, expected in expected_missing.items():
            actual = details[col]["missing_count"]
            assert actual == expected, f"{col}: expected {expected} missing, got {actual}"

    def test_total_missing_cells(self, analysis_result):
        findings = analysis_result["findings"]
        missing_finding = next(f for f in findings if f["issue_type"] == "missing_values")
        assert missing_finding["count"] == 199, f"Expected 199 total missing, got {missing_finding['count']}"

    def test_affected_columns(self, analysis_result):
        findings = analysis_result["findings"]
        missing_finding = next(f for f in findings if f["issue_type"] == "missing_values")
        affected = set(missing_finding["affected_columns"])
        expected = {"email", "quantity", "price", "city", "pincode", "payment_method"}
        assert affected == expected, f"Affected columns mismatch: {affected} vs {expected}"


class TestDuplicates:
    """Test duplicate detection."""

    def test_duplicate_occurrences(self, analysis_result):
        profile = analysis_result["profile"]
        assert profile["duplicate_count"] == 13, f"Expected 13 duplicate occurrences, got {profile['duplicate_count']}"

    def test_duplicate_rows_involved(self, analysis_result):
        profile = analysis_result["profile"]
        assert profile["duplicate_rows_involved"] == 26, f"Expected 26 rows involved in duplicates, got {profile['duplicate_rows_involved']}"

    def test_duplicate_finding(self, analysis_result):
        findings = analysis_result["findings"]
        dup_finding = next(f for f in findings if f["issue_type"] == "duplicate_rows")
        assert dup_finding["count"] == 13
        assert dup_finding["details"]["duplicate_rows_involved"] == 26


class TestCategoricalInconsistency:
    """Test categorical inconsistency detection."""

    def test_inconsistency_count(self, analysis_result):
        findings = analysis_result["findings"]
        incons_finding = next(f for f in findings if f["issue_type"] == "categorical_inconsistency")
        # Should detect: Credit Card/Credit card, UPI/Upi, Bang/Bangalore, Hyd/Hyderabad (4 groups)
        assert incons_finding["count"] == 4, f"Expected 4 inconsistency groups, got {incons_finding['count']}"

    def test_affected_columns(self, analysis_result):
        findings = analysis_result["findings"]
        incons_finding = next(f for f in findings if f["issue_type"] == "categorical_inconsistency")
        assert "payment_method" in incons_finding["affected_columns"]

    def test_no_false_positives_on_emails(self, analysis_result):
        """Ensure emails are not flagged as inconsistent (high cardinality)."""
        from app.quality_engine.consistency import detect_categorical_inconsistencies
        from app.profiler.dataset_profiler import load_dataset

        df = load_dataset('data/sample/ecommerce_dirty_data.csv')
        result = detect_categorical_inconsistencies(df)
        assert not result["columns"].get("email", {}).get("has_issue", False), \
            "Emails should not be flagged as inconsistent (high cardinality)"

    def test_no_false_positives_on_timestamps(self, analysis_result):
        """Ensure timestamps are not flagged as inconsistent (high cardinality)."""
        from app.quality_engine.consistency import detect_categorical_inconsistencies
        from app.profiler.dataset_profiler import load_dataset

        df = load_dataset('data/sample/ecommerce_dirty_data.csv')
        result = detect_categorical_inconsistencies(df)
        assert not result["columns"].get("order_date", {}).get("has_issue", False), \
            "Timestamps should not be flagged as inconsistent (high cardinality)"


class TestUniqueValues:
    """Test unique value counts exclude nulls."""

    def test_unique_counts_exclude_nulls(self, analysis_result):
        details = analysis_result["profile"]["column_details"]

        # unique_count should equal non_null_unique_count
        for col, details_col in analysis_result["profile"]["column_details"].items():
            assert details_col["unique_count"] == details_col["non_null_unique_count"], \
                f"{col}: unique_count ({details_col['unique_count']}) should equal non_null_unique_count ({details_col['non_null_unique_count']})"

    def test_specific_unique_counts(self, analysis_result):
        details = analysis_result["profile"]["column_details"]
        expected = {
            "email": 192,
            "quantity": 3,
            "price": 4,
            "city": 10,
            "pincode": 6,
            "payment_method": 6,
        }
        for col, expected_count in expected.items():
            actual = analysis_result["profile"]["column_details"][col]["non_null_unique_count"]
            assert actual == expected_count, f"{col}: expected {expected_count} unique non-null, got {actual}"


class TestSemanticRoles:
    """Test semantic column role inference."""

    def test_roles(self, analysis_result):
        expected_roles = {
            "order_id": "identifier",
            "firstname": "categorical",
            "lastname": "categorical",
            "email": "email",
            "product": "categorical",
            "quantity": "numerical_measurement",
            "price": "numerical_measurement",
            "city": "categorical",
            "state": "categorical",
            "pincode": "postal_code",
            "order_date": "datetime",
            "payment_method": "categorical",
            "status": "categorical",
        }

        for col, expected_role in expected_roles.items():
            actual_role = None
            for ca in analysis_result["understanding"]["column_analysis"]:
                if ca["column"] == col:
                    actual_role = ca["role"]
                    break
            assert actual_role == expected_role, \
                f"{col}: expected role {expected_role}, got {actual_role}"


class TestDatetimeDetection:
    """Test datetime column detection."""

    def test_order_date_is_datetime(self, analysis_result):
        for ca in analysis_result["understanding"]["column_analysis"]:
            if ca["column"] == "order_date":
                assert ca["role"] == "datetime", f"order_date should be datetime, got {ca['role']}"
                break


class TestDataTypeValidation:
    """Test semantic data type validation."""

    def test_data_type_validation_finding(self, analysis_result):
        findings = analysis_result["findings"]
        dtype_finding = next(f for f in findings if f["issue_type"] == "data_type_validation")
        
        # With correct implementation, missing values should NOT create data type errors.
        # All non-null values in this dataset have correct types.
        assert dtype_finding["has_issue"] is False
        assert dtype_finding["count"] == 0
        assert dtype_finding["affected_columns"] == []


class TestInvalidValues:
    """Test invalid value detection."""

    def test_no_invalid_emails_in_test_data(self, analysis_result):
        """Test data has no invalid emails (ground truth)."""
        findings = analysis_result["findings"]
        invalid_finding = next(f for f in findings if f["issue_type"] == "invalid_values")
        # Test data has 0 malformed emails
        assert invalid_finding["has_issue"] is False


class TestQualityScore:
    """Test quality score calculation."""

    def test_score_range(self, analysis_result):
        score = analysis_result["quality_score"]["overall_score"]
        assert 0 <= score <= 100, f"Score out of range: {score}"

    def test_score_deterministic(self, dataset):
        """Score should be deterministic across runs."""
        r1 = analyze_dataset(dataset)
        r2 = analyze_dataset(dataset)
        assert r1["quality_score"]["overall_score"] == r2["quality_score"]["overall_score"]

    def test_category_scores_present(self, analysis_result):
        scores = analysis_result["quality_score"]["category_scores"]
        expected_categories = [
            "missing_values", "duplicate_rows", "categorical_inconsistency",
            "data_type_validation", "invalid_values", "statistical_outliers",
            "possible_sentinel_values"
        ]
        for cat in expected_categories:
            assert cat in scores, f"Missing category: {cat}"

    def test_score_components(self, analysis_result):
        scores = analysis_result["quality_score"]["category_scores"]
        for cat, details in scores.items():
            assert "weight" in details
            assert "affected_rate" in details
            assert "severity" in details
            assert "penalty" in details
            assert "score" in details
            assert 0 <= details["score"] <= 100


class TestMLReadiness:
    """Test ML readiness assessment."""

    def test_readiness_percentage(self, analysis_result):
        ml = analysis_result["ml_readiness"]
        assert 0 <= ml["readiness_percentage"] <= 100

    def test_numerical_features_pass(self, analysis_result):
        ml = analysis_result["ml_readiness"]
        num_check = next(c for c in ml["checks"] if c["check"] == "numerical_features")
        assert num_check["passed"] is True, "quantity and price should be numerical features"
        assert num_check["count"] == 2
        assert set(num_check["columns"]) == {"quantity", "price"}

    def test_categorical_features_excludes_numerical(self, analysis_result):
        ml = analysis_result["ml_readiness"]
        cat_check = next(c for c in ml["checks"] if c["check"] == "categorical_features")
        # quantity and price should NOT be in categorical features
        assert "quantity" not in cat_check["columns"]
        assert "price" not in cat_check["columns"]

    def test_numerical_variance_pass(self, analysis_result):
        ml = analysis_result["ml_readiness"]
        var_check = next(c for c in ml["checks"] if c["check"] == "numerical_variance")
        assert var_check["passed"] is True
        assert var_check["count"] == 2
        assert set(var_check["columns"].keys()) == {"quantity", "price"}

    def test_identifier_column_flagged(self, analysis_result):
        ml = analysis_result["ml_readiness"]
        id_check = next(c for c in ml["checks"] if c["check"] == "identifier_columns")
        # Identifier columns are informational, not a failure
        assert id_check["passed"] is True
        assert id_check["severity"] == "information"
        assert id_check["count"] == 1
        assert "order_id" in id_check["columns"]
        assert "Exclude from model features" in id_check["message"]

    def test_missing_values_check_fails(self, analysis_result):
        """Missing values check should fail when there are missing values."""
        ml = analysis_result["ml_readiness"]
        miss_check = next(c for c in ml["checks"] if c["check"] == "missing_values")
        assert miss_check["passed"] is False


class TestSentinels:
    """Test sentinel/placeholder detection."""

    def test_no_sentinels_in_test_data(self, analysis_result):
        """Test data has no sentinel values."""
        findings = analysis_result["findings"]
        sentinel_finding = next(f for f in findings if f["issue_type"] == "possible_sentinel_values")
        assert sentinel_finding["has_issue"] is False


class TestSummary:
    """Test overall analysis summary consistency."""

    def test_findings_count_matches(self, analysis_result):
        findings = analysis_result["findings"]
        issues = [f for f in findings if f["has_issue"]]
        # Count of issue types with has_issue=True
        assert len(issues) >= 3  # At least missing, duplicates, data_type_validation

    def test_no_metric_calculated_twice(self, analysis_result):
        """Ensure metrics are consistent between profile and findings."""
        profile_missing = sum(
            d["missing_count"] for d in analysis_result["profile"]["column_details"].values()
        )
        missing_finding = next(f for f in analysis_result["findings"] if f["issue_type"] == "missing_values")
        assert profile_missing == missing_finding["count"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
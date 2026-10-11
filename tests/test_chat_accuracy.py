"""
Focused regression tests for the final targeted chatbot accuracy fixes.

Defects covered (each verified against the real /api/chat endpoint and the
real resolver / role-detection code paths):

1. Remediation follow-ups ("How can I fix all of them?") after a problems
   report must give per-issue remediation, never re-list the problems.
2. Semantic column roles: identifier-like names (invoice numbers, stock
   codes, customer ids) must never be classified as postal codes, while
   genuine postal columns stay postal codes.
3. Outlier questions must resolve to outlier evidence, not a column
   inventory, and must report only what the canonical finding retains.
4. Project suggestions must be dataset-aware (transactional structure ->
   market basket / RFM / forecasting) and must not invent targets or
   training.
5. Feature importance must stay honest (no invented rankings).
"""
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from main import app
from app.ai.chat_resolver import resolve, answer_is_relevant, Resolution
from app.ai.assistant import _avoid_expanded_repeat
from app.profiler.column_roles import detect_column_role, get_column_tokens

client = TestClient(app)


@pytest.fixture(autouse=True)
def _deterministic_rules_path(monkeypatch):
    """Pin the deterministic contract: disable the optional LLM.

    The live LLM path is verified separately by the relevance-gate unit
    tests below (which use the real gate function) and by the browser
    acceptance run against the running application.
    """
    monkeypatch.setenv("GEMINI_API_KEY", "")


# ============================================================
# Fixtures: a retail-transaction-shaped analysis (the shape that
# reproduced every defect). All values are synthetic.
# ============================================================

RETAIL_FINDINGS = [
    {
        "issue_type": "missing_values",
        "has_issue": True,
        "count": 135080,
        "affected_columns": ["CustomerID"],
        "details": {"columns": {"CustomerID": {"missing_count": 135080,
                                               "missing_percentage": 24.9}}},
    },
    {
        "issue_type": "statistical_outliers",
        "has_issue": True,
        "count": 8000,
        "affected_columns": ["Quantity", "UnitPrice"],
        "details": {
            "Quantity": {"has_issue": True, "outlier_count": 5000,
                         "q1": 2.0, "q3": 12.0, "iqr": 10.0,
                         "lower_bound": -13.0, "upper_bound": 27.0,
                         "outlier_rows": [17, 42, 108, 260, 999],
                         "outlier_values": [100, -200, 80, -150, 95]},
            "UnitPrice": {"has_issue": True, "outlier_count": 3000,
                          "q1": 1.25, "q3": 4.95, "iqr": 3.7,
                          "lower_bound": -4.3, "upper_bound": 10.5,
                          "outlier_rows": [5, 77, 311],
                          "outlier_values": [39.8, 295.0, 45.0]},
        },
    },
    {
        "issue_type": "invalid_values",
        "has_issue": True,
        "count": 10600,
        "affected_columns": ["Quantity", "UnitPrice"],
        "details": {"Quantity": {"has_issue": True, "invalid_count": 5000},
                    "UnitPrice": {"has_issue": True, "invalid_count": 5600}},
    },
    {
        "issue_type": "duplicate_rows",
        "has_issue": True,
        "count": 4000,
        "affected_columns": [],
        "details": {"percentage": 0.74, "duplicate_rows_involved": 8000},
    },
    {
        "issue_type": "possible_sentinel_values",
        "has_issue": True,
        "count": 397,
        "affected_columns": ["UnitPrice"],
        "details": {"columns": {"UnitPrice": {"sentinel_occurrences": 397,
                                              "sentinel_count": 1}},
                    "total_sentinels": 1, "total_placeholders": 0},
    },
]

RETAIL_PROFILE = {
    "rows": 541909,
    "columns": 8,
    "name": "online_retail",
    "column_details": {
        "InvoiceNo": {"type": "categorical", "role": "identifier",
                      "unique_count": 25899, "uniqueness_ratio": 0.0478,
                      "missing_count": 0, "missing_percentage": 0.0},
        "StockCode": {"type": "categorical", "role": "identifier",
                      "unique_count": 4070, "uniqueness_ratio": 0.0075,
                      "missing_count": 0, "missing_percentage": 0.0},
        "Description": {"type": "categorical", "role": "categorical",
                        "unique_count": 4223, "uniqueness_ratio": 0.0078,
                        "missing_count": 1454, "missing_percentage": 0.27},
        "Quantity": {"type": "numerical", "role": "numerical_measurement",
                     "unique_count": 722, "uniqueness_ratio": 0.0013,
                     "missing_count": 0, "missing_percentage": 0.0,
                     "min": -80995, "max": 80995},
        "InvoiceDate": {"type": "categorical", "role": "datetime",
                        "unique_count": 247455, "uniqueness_ratio": 0.4568,
                        "missing_count": 0, "missing_percentage": 0.0},
        "UnitPrice": {"type": "numerical", "role": "numerical_measurement",
                      "unique_count": 4014, "uniqueness_ratio": 0.0074,
                      "missing_count": 0, "missing_percentage": 0.0,
                      "min": 0.0, "max": 38970.0},
        "CustomerID": {"type": "categorical", "role": "identifier",
                       "unique_count": 4372, "uniqueness_ratio": 0.0081,
                       "missing_count": 135080, "missing_percentage": 24.93},
        "Country": {"type": "categorical", "role": "categorical",
                    "unique_count": 38, "uniqueness_ratio": 0.0001,
                    "missing_count": 0, "missing_percentage": 0.0},
    },
}

RETAIL_QUALITY = {"overall_score": 62.0}
RETAIL_ML = {"readiness_percentage": 55}


def ask(message, conversation=None, findings=None, profile=None,
        quality=None, ml=None):
    resp = client.post("/api/chat", json={
        "message": message,
        "conversation": conversation or [],
        "findings": findings if findings is not None else RETAIL_FINDINGS,
        "quality_score": quality or RETAIL_QUALITY,
        "ml_readiness": ml or RETAIL_ML,
        "profile": profile or RETAIL_PROFILE,
    })
    assert resp.status_code == 200
    return resp.json()["response"]


def problems_conversation():
    """Real two-turn prefix: ask about problems, capture the real answer."""
    answer = ask("What problems does this dataset have?")
    return [
        {"role": "user", "content": "What problems does this dataset have?"},
        {"role": "assistant", "content": answer},
    ], answer


# ============================================================
# Defect 2: remediation follow-ups after a problems report
# ============================================================

class TestRemediationFollowUp:
    PARAPHRASES = [
        "How can I fix all of them?",
        "How do I resolve these issues?",
        "What methods should I use to clean them?",
        "How can I remove these problems?",
    ]

    @pytest.fixture(scope="class")
    def history_and_problems(self):
        return problems_conversation()

    def test_problems_report_lists_canonical_findings(self, history_and_problems):
        _, answer = history_and_problems
        low = answer.lower()
        for word in ("missing", "outlier", "invalid", "duplicate", "sentinel"):
            assert word in low, f"problems report should mention {word}: {answer}"

    @pytest.mark.parametrize("follow_up", PARAPHRASES)
    def test_follow_up_gives_remediation_not_problems(self, follow_up, history_and_problems):
        history, _ = history_and_problems
        answer = ask(follow_up, conversation=history)
        low = answer.lower()
        # Must NOT repeat the problem report.
        assert "problems detected" not in low
        # Must cover every issue type the previous answer referenced.
        for word in ("missing", "outlier", "invalid", "duplicate", "sentinel"):
            assert word in low, f"{follow_up!r} should cover {word}: {answer}"
        # Must give actionable methods that distinguish handling strategies.
        assert "imput" in low, answer          # imputation guidance
        assert "investigat" in low, answer     # investigation guidance
        assert "correct" in low, answer        # correction guidance
        assert "retain" in low or "keep" in low, answer  # retention guidance
        # Must ground in the actual affected columns.
        assert "customerid" in low.replace(" ", ""), answer
        assert "quantity" in low, answer

    def test_business_meaning_caution_included(self, history_and_problems):
        history, _ = history_and_problems
        answer = ask("How can I fix all of them?", conversation=history)
        low = answer.lower()
        # Must not blindly recommend deletion; must caution about
        # legitimate business events (returns, cancellations, free items).
        assert "delete" in low or "deletion" in low or "drop" in low
        assert "return" in low or "cancellation" in low or "free" in low

    def test_resolver_resolves_paraphrases_to_multi_remediation(self, history_and_problems):
        history, _ = history_and_problems
        for follow_up in self.PARAPHRASES:
            r = resolve(follow_up, history,
                        available_issues={"missing_values", "statistical_outliers",
                                          "invalid_values", "duplicate_rows",
                                          "possible_sentinel_values"})
            assert r.intent == "remediation", (follow_up, r)
            assert r.action == "remediate", (follow_up, r)
            assert r.topic is None, f"{follow_up!r} must not pin one topic: {r}"

    def test_single_issue_remediation_still_works(self):
        # An explicit single-issue remediation request keeps its topic.
        r = resolve("How do I fix the missing values?", None,
                    available_issues={"missing_values"})
        assert r.topic == "missing_values"
        assert r.action == "remediate"

    def test_quality_score_fix_still_resolves_to_score(self):
        # "How do I fix it?" after a quality-score question stays anchored
        # to the score (regression guard for the (g) gate).
        history = [
            {"role": "user", "content": "What is the quality score?"},
            {"role": "assistant", "content": "Quality score: 62.0. Categories dragging it down: x."},
        ]
        r = resolve("How do I fix it?", history, available_issues=set())
        assert r.topic == "quality_score"
        assert r.action == "remediate"


# ============================================================
# Defect 3: semantic column roles vs physical types
# ============================================================

def _retail_like_df(n=2000, seed=7):
    rng = np.random.default_rng(seed)
    invoice_no = [("C" + str(536365 + i)) if rng.random() < 0.12
                  else str(536365 + i) for i in range(n)]
    stock_code = [rng.choice(["85123A", "71053", "40023", "22423", "84029G"])
                  for _ in range(n)]
    customer_id = [str(rng.integers(12000, 19000)) if rng.random() < 0.8 else np.nan
                   for _ in range(n)]
    invoice_date = [f"{rng.integers(1, 28):02d}/{rng.integers(1, 13):02d}/2011 "
                    f"{rng.integers(8, 20):02d}:{rng.integers(0, 60):02d}"
                    for _ in range(n)]
    return pd.DataFrame({
        "InvoiceNo": invoice_no,
        "StockCode": stock_code,
        "Quantity": rng.integers(-10, 50, n).tolist(),
        "InvoiceDate": invoice_date,
        "UnitPrice": [float(rng.choice([0.0, 0.85, 1.25, 2.5, 4.95]))
                      for _ in range(n)],
        "CustomerID": customer_id,
        "Country": [rng.choice(["United Kingdom", "France", "Germany"])
                    for _ in range(n)],
    })


class TestSemanticColumnRoles:
    def test_camel_case_identifier_tokens(self):
        assert get_column_tokens("InvoiceNo") == ["invoice", "no"]
        assert get_column_tokens("CustomerID") == ["customer", "id"]
        assert get_column_tokens("StockCode") == ["stock", "code"]
        # Separator-based names keep working.
        assert get_column_tokens("order_id") == ["order", "id"]
        assert get_column_tokens("e_mail") == ["e", "mail"]

    def test_identifier_like_names_are_never_postal(self):
        df = _retail_like_df()
        assert detect_column_role("InvoiceNo", df["InvoiceNo"]) == "identifier"
        assert detect_column_role("StockCode", df["StockCode"]) == "identifier"
        # CustomerID arrives as float in the raw file -> still identifier.
        df["CustomerIDFloat"] = pd.to_numeric(df["CustomerID"], errors="coerce")
        assert detect_column_role("CustomerIDFloat", df["CustomerIDFloat"]) == "identifier"
        # Object-dtype customer id (stringified) -> identifier too.
        assert detect_column_role("CustomerID", df["CustomerID"]) == "identifier"
        # And no identifier-like name may ever be labelled postal.
        for col in ("InvoiceNo", "StockCode", "CustomerID"):
            role = detect_column_role(col, df[col])
            assert role != "postal_code", f"{col} misclassified as postal_code"

    def test_transaction_and_doc_identifier_names(self):
        series = pd.Series([str(100000 + i) if i % 5 else "C" + str(100000 + i)
                            for i in range(500)])
        for name in ("TransactionID", "OrderNumber", "InvoiceNo",
                     "ReceiptNo", "TicketID", "AccountRef"):
            assert detect_column_role(name, series) == "identifier", name

    def test_genuine_named_postal_stays_postal(self):
        rng = np.random.default_rng(3)
        zips = pd.Series([str(rng.integers(100000, 899999)) for _ in range(1000)])
        assert detect_column_role("pincode", zips) == "postal_code"
        assert detect_column_role("zip_code", zips) == "postal_code"
        assert detect_column_role("PostalCode", zips) == "postal_code"

    def test_genuine_unnamed_repeating_postal_stays_postal(self):
        # A real postal column repeats heavily (many rows per code).
        rng = np.random.default_rng(4)
        zips = pd.Series([str(rng.integers(10000, 99999)) for _ in range(5000)])
        assert detect_column_role("delivery_zone", zips) == "postal_code"

    def test_near_unique_numeric_codes_are_identifier_like_not_postal(self):
        # Boundary: bare unique 5-6 digit strings with no postal name are
        # identifiers, not postal codes.
        series = pd.Series([str(10000 + i) for i in range(1000)])
        role = detect_column_role("col", series)
        assert role in ("identifier_like", "identifier")
        assert role != "postal_code"

    def test_numeric_measurement_and_categorical_unchanged(self):
        floats = pd.Series([1.5, 2.25, 3.0, 4.75, 5.5, 2.25, 3.0] * 50)
        assert detect_column_role("price", floats) == "numerical_measurement"
        cats = pd.Series(["red", "blue", "green"] * 200)
        assert detect_column_role("color", cats) == "categorical"

    def test_date_stored_as_string_is_datetime_role(self):
        df = _retail_like_df()
        assert detect_column_role("InvoiceDate", df["InvoiceDate"]) == "datetime"
        # Generic string date, no date name.
        generic = pd.Series(["01/12/2010 08:26", "02/12/2010 09:30"] * 100)
        assert detect_column_role("when_seen", generic) == "datetime"

    def test_chat_column_answer_uses_corrected_roles(self):
        # End-to-end: roles detected from the real dataframe flow into the
        # chat answer; categorical ask must not label ids as postal codes.
        df = _retail_like_df()
        profile = {
            "rows": len(df), "columns": len(df.columns), "name": "retail",
            "column_details": {
                c: {"type": ("numerical" if pd.api.types.is_numeric_dtype(df[c])
                             else "categorical"),
                    "role": detect_column_role(c, df[c]),
                    "unique_count": int(df[c].nunique()),
                    "missing_count": int(df[c].isna().sum()),
                    "missing_percentage": float(df[c].isna().mean() * 100)}
                for c in df.columns
            },
        }
        answer = ask("What columns are categorical?", findings=[], profile=profile,
                     quality={"overall_score": 90.0}, ml={"readiness_percentage": 80})
        low = answer.lower()
        assert "postal" not in low, answer
        # The identifier columns are reported as identifiers, not categories.
        assert "invoiceno" in low.replace(" ", ""), answer
        assert "identifier" in low, answer


# ============================================================
# Defect 4: outlier questions must select outlier evidence
# ============================================================

class TestOutlierEvidence:
    def test_resolver_selects_outlier_topic_not_columns(self):
        r = resolve("What are the outliers in the identified columns?", None)
        assert r.topic == "statistical_outliers"
        assert r.action == "report"
        assert r.intent == "statistical_outliers"

    def test_exact_outlier_question_reports_outlier_evidence(self):
        answer = ask("What are the outliers in the identified columns?")
        low = answer.lower()
        assert "outlier" in low
        # Affected columns, count, bounds and retained representative values.
        assert "quantity" in low
        assert "unitprice" in low
        assert "5000" in low or "5,000" in low
        assert "-13.00" in low and "27.00" in low       # IQR bounds
        assert "100" in low and "-200" in low           # representative values
        # Must NOT be a full column inventory.
        assert "country" not in low
        assert "stockcode" not in low.replace(" ", "")
        assert "invoiceno" not in low.replace(" ", "")

    def test_outlier_question_paraphrase(self):
        answer = ask("Show me the extreme values in the data")
        low = answer.lower()
        assert "outlier" in low
        assert "quantity" in low
        assert "country" not in low

    def test_outlier_follow_up_after_column_turn(self):
        history = [
            {"role": "user", "content": "What columns are categorical?"},
            {"role": "assistant", "content": "Categorical columns: Country, Description."},
        ]
        answer = ask("What are the outliers in the identified columns?",
                     conversation=history)
        low = answer.lower()
        assert "outlier" in low
        assert "quantity" in low
        assert "categorical columns: country" not in low

    def test_outlier_categories_are_distinguished(self):
        answer = ask("What are the outliers in the identified columns?")
        low = answer.lower()
        assert "invalid" in low and "sentinel" in low  # explicitly distinguished

    def test_no_outlier_case_is_honest(self):
        findings = [f for f in RETAIL_FINDINGS if f["issue_type"] != "statistical_outliers"]
        findings.append({
            "issue_type": "statistical_outliers", "has_issue": False,
            "count": 0, "affected_columns": [],
            "details": {"Quantity": {"has_issue": False, "outlier_count": 0,
                                     "lower_bound": -13.0, "upper_bound": 27.0,
                                     "outlier_rows": [], "outlier_values": []}},
        })
        answer = ask("What are the outliers in the identified columns?", findings=findings)
        assert "no statistical outliers" in answer.lower()
        # Must not substitute another finding.
        assert "sentinel" not in answer.lower()

    def test_single_outlier_column_remediation_still_scoped(self):
        answer = ask("How do I fix the outliers?")
        low = answer.lower()
        assert "outlier" in low
        assert "missing value remediation" not in low


# ============================================================
# Defect 5: dataset-aware project suggestions
# ============================================================

PATIENT_PROFILE = {
    "rows": 2400, "columns": 6, "name": "patients",
    "column_details": {
        "PatientID": {"type": "categorical", "role": "identifier",
                      "unique_count": 2400, "uniqueness_ratio": 1.0,
                      "missing_count": 0, "missing_percentage": 0.0},
        "Age": {"type": "numerical", "role": "numerical_measurement",
                "unique_count": 80, "uniqueness_ratio": 0.033,
                "missing_count": 0, "missing_percentage": 0.0},
        "BloodPressure": {"type": "numerical", "role": "numerical_measurement",
                          "unique_count": 400, "uniqueness_ratio": 0.167,
                          "missing_count": 12, "missing_percentage": 0.5},
        "Ward": {"type": "categorical", "role": "categorical",
                 "unique_count": 8, "uniqueness_ratio": 0.003,
                 "missing_count": 0, "missing_percentage": 0.0},
        "AdmitDate": {"type": "datetime", "role": "datetime",
                      "unique_count": 900, "uniqueness_ratio": 0.375,
                      "missing_count": 0, "missing_percentage": 0.0},
        "Readmitted": {"type": "categorical", "role": "target",
                       "unique_count": 2, "uniqueness_ratio": 0.001,
                       "missing_count": 0, "missing_percentage": 0.0},
    },
}


class TestDatasetAwareProjectIdeas:
    def test_retail_projects_are_structure_specific(self):
        answer = ask("What advanced ML projects can I build with this dataset?")
        low = answer.lower()
        # Transactional structure (repeating invoice id + product code +
        # quantity/price + date) must yield the specific project families.
        assert "market-basket" in low or "market basket" in low, answer
        assert "forecast" in low, answer
        assert "rfm" in low or "segmentation" in low, answer
        # Grounded in actual columns from the profile.
        assert "invoiceno" in low.replace(" ", "") or "stockcode" in low.replace(" ", "")
        assert "quantity" in low or "unitprice" in low
        # Must explain approach and limitation.
        assert "approach" in low
        assert "limitation" in low
        # Must not imply a target exists or a model was trained.
        assert "has been trained" not in low
        assert "target column has been identified" not in low

    def test_market_basket_only_when_relationship_exists(self):
        # No repeating transaction id -> no market-basket suggestion.
        answer = ask("What projects can I build?", findings=[],
                     profile=PATIENT_PROFILE,
                     quality={"overall_score": 90.0},
                     ml={"readiness_percentage": 80})
        low = answer.lower()
        assert "market-basket" not in low and "market basket" not in low
        # A real target enables a supervised idea, honestly caveated.
        assert "readmitted" in low.replace(" ", "").lower() or "classification" in low
        assert "no model has been trained" in low

    def test_projects_work_without_uniqueness_ratio_in_profile(self):
        # The real /api/analyze response omits uniqueness_ratio from
        # column_details; the collector must derive it, not fail silently.
        slim = {"rows": RETAIL_PROFILE["rows"], "columns": RETAIL_PROFILE["columns"],
                "name": "online_retail",
                "column_details": {
                    c: {k: v for k, v in d.items() if k != "uniqueness_ratio"}
                    for c, d in RETAIL_PROFILE["column_details"].items()}}
        assert all("uniqueness_ratio" not in d for d in slim["column_details"].values())
        answer = ask("What advanced ML projects can I build with this dataset?",
                     profile=slim)
        low = answer.lower()
        assert "market-basket" in low or "market basket" in low, answer
        assert "rfm" in low or "segmentation" in low, answer

    def test_project_ideas_do_not_claim_training(self):
        answer = ask("What projects can I build?", findings=[],
                     profile=PATIENT_PROFILE,
                     quality={"overall_score": 90.0},
                     ml={"readiness_percentage": 80})
        low = answer.lower()
        assert "accuracy" not in low and "f1" not in low
        assert "we trained" not in low and "the model achieves" not in low

    def test_give_me_more_yields_fresh_ideas_then_exhaustion(self):
        first = ask("What advanced ML projects can I build with this dataset?")
        history = [
            {"role": "user", "content": "What advanced ML projects can I build with this dataset?"},
            {"role": "assistant", "content": first},
        ]
        second = ask("Give me more", conversation=history)
        assert second != first
        assert "additional" in second.lower()
        history.append({"role": "user", "content": "Give me more"})
        history.append({"role": "assistant", "content": second})
        third = ask("Give me more", conversation=history)
        # Either fresh ideas remain or the assistant says so honestly.
        assert third != second or "already covered" in third.lower()

    def test_which_one_is_best_still_compares(self):
        history = [
            {"role": "user", "content": "What advanced ML projects can I build with this dataset?"},
            {"role": "assistant", "content": ask("What advanced ML projects can I build with this dataset?")},
        ]
        answer = ask("Which one is best?", conversation=history)
        low = answer.lower()
        assert "compar" in low


# ============================================================
# Requirement 6: feature-importance honesty is preserved
# ============================================================

class TestFeatureImportanceHonesty:
    def test_feature_importance_is_not_invented(self):
        answer = ask("Which features influence the target the most?")
        low = answer.lower()
        assert "does not calculate model-based feature importance" in low
        assert "no target column has been identified" in low
        # No fabricated ranking artifacts.
        for banned in ("shap value of", "importance of 0", "ranks first",
                       "most important feature is"):
            assert banned not in low

    def test_resolver_routes_feature_question(self):
        r = resolve("Which features influence the target the most?", None)
        assert r.topic == "feature_importance"
        assert r.intent == "feature_importance"


# ============================================================
# "Give me more" continuation: new information, never a verbatim
# repeat (feature-importance defect + the same principle applied
# to other topics).
# ============================================================

class TestExpandContinuation:
    FI_Q = "Which features influence the target the most?"

    def _history(self, question):
        return [
            {"role": "user", "content": question},
            {"role": "assistant", "content": ask(question)},
        ]

    def test_resolver_marks_give_me_more_as_expand_on_prior_topic(self):
        r = resolve("Give me more.", self._history(self.FI_Q))
        assert r.topic == "feature_importance"
        assert r.action == "expand"
        assert r.uses_context

    def test_feature_importance_give_me_more_is_new_content(self):
        first = ask(self.FI_Q)
        more = ask("Give me more.", conversation=self._history(self.FI_Q))
        # Not a verbatim repeat of the previous answer.
        assert more != first
        low = more.lower()
        # (a) selecting a target from the user's prediction objective.
        assert "prediction objective" in low, more
        assert "classification" in low and "regression" in low, more
        # (b) concrete calculation steps: baseline model, permutation
        # importance / SHAP, cross-validation.
        assert "baseline" in low, more
        assert "permutation importance" in low, more
        assert "shap" in low, more
        assert "cross-validation" in low, more
        # (c) honest: the user must choose; nothing exists until run.
        assert "no target column has been identified" in low, more
        assert "no model has been trained" in low, more
        assert "no importance values" in low, more
        assert "never computed" in low, more

    def test_feature_importance_expansion_never_claims_a_run(self):
        more = ask("Give me more.", conversation=self._history(self.FI_Q)).lower()
        for banned in ("we trained", "the model achieves", "ranks first",
                       "most important feature is", "candidate target column(s)",
                       "model was trained on"):
            assert banned not in more

    def test_project_idea_continuation_excludes_shown_ideas(self):
        q = "What advanced ML projects can I build with this dataset?"
        first = ask(q)
        second = ask("Give me more.", conversation=self._history(q))
        assert second != first
        assert "additional" in second.lower()

        def numbered(text):
            out = set()
            for line in text.splitlines():
                s = line.strip()
                if len(s) > 2 and s[0].isdigit() and s[1] == ".":
                    out.add(s)
            return out

        first_ideas, second_ideas = numbered(first), numbered(second)
        assert first_ideas and second_ideas, (first, second)
        # No idea shown twice between the two batches.
        assert not (first_ideas & second_ideas), first_ideas & second_ideas

    def test_quality_score_give_me_more_shows_category_breakdown(self):
        qc = {"overall_score": 62.0,
              "category_scores": {"missing_values": {"score": 40.0,
                                                     "severity": "HIGH",
                                                     "affected_rows": 135080}}}
        first = ask("What is the quality score?", quality=qc)
        second = ask("Give me more.", quality=qc,
                     conversation=[{"role": "user", "content": "What is the quality score?"},
                                   {"role": "assistant", "content": first}])
        assert second != first
        low = second.lower()
        assert "category breakdown" in low, second
        assert "severity=high" in low, second

    def test_problems_give_me_more_gives_actions_not_a_repeat(self):
        history, first = problems_conversation()
        second = ask("Give me more.", conversation=history)
        assert second != first
        low = second.lower()
        assert "problems detected" not in low  # not the report again
        assert "recommended action for each detected issue" in low, second
        assert "customerid" in low  # real affected columns injected

    def test_guard_replaces_identical_expansion_answer(self):
        prev = "The complete previous answer on this topic."
        history = [
            {"role": "user", "content": "Which model should I use?"},
            {"role": "assistant", "content": prev},
        ]
        expand = Resolution(intent="model_selection", topic="model_selection",
                            action="expand")
        identical = {"answer": prev, "mode": "rules"}
        out = _avoid_expanded_repeat(identical, expand, history)
        assert out["answer"] != prev
        assert "complete answer" in out["answer"]
        # Fresh expansion text passes through untouched.
        fresh = {"answer": "A fresh continuation.", "mode": "rules"}
        assert _avoid_expanded_repeat(fresh, expand, history) == fresh
        # Non-expand turns are never rewritten.
        report = Resolution(intent="problems", topic="problems", action="report")
        assert _avoid_expanded_repeat(identical, report, history) == identical


# ============================================================
# Live LLM guard: partial multi-issue answers must be rejected
# ============================================================

class TestAnswerRelevanceGate:
    ALL = {"missing_values", "statistical_outliers", "invalid_values",
           "duplicate_rows", "possible_sentinel_values"}
    TRUNCATED = ("The dataset has five data quality issues:\n"
                 "1. Missing values: 135,080 missing values in CustomerID.\n"
                 "2. Invalid values: 10,600")
    COMPLETE = ("Problems detected: Missing values, Statistical outliers "
                "(98244), Invalid values, Duplicate rows, and Sentinel / "
                "placeholder values.")

    def test_partial_problem_report_is_rejected(self):
        # A truncated LLM report that omits detected issues must fall
        # back to the deterministic answer (browser case 1 reliability).
        r = resolve("What problems does this dataset have?", None)
        assert r.action == "report"
        assert not answer_is_relevant(self.TRUNCATED, r, self.ALL)

    def test_complete_problem_report_is_accepted(self):
        r = resolve("What problems does this dataset have?", None)
        assert answer_is_relevant(self.COMPLETE, r, self.ALL)

    def test_gate_is_inert_when_issue_set_unknown(self):
        # Backward compatible: without the detected set the legacy
        # behavior (allow plausible answers) is preserved.
        r = resolve("What problems does this dataset have?", None)
        assert answer_is_relevant(self.TRUNCATED, r) is True

    def test_partial_multi_remediation_is_rejected(self):
        history = problems_conversation()[0]
        r = resolve("How can I fix all of them?", history,
                    available_issues=self.ALL)
        assert r.topic is None and r.action == "remediate"
        partial = ("Missing values: impute the gaps. "
                   "Invalid values: correct the entries.")
        complete = (partial +
                    " Outliers: cap at IQR bounds. Duplicates: drop exact "
                    "copies. Sentinel values: replace with NaN.")
        assert not answer_is_relevant(partial, r, self.ALL)
        assert answer_is_relevant(complete, r, self.ALL)

    def test_single_issue_answers_are_not_over_gated(self):
        # Single-topic resolutions keep the per-evidence token check; a
        # missing-values remediation mentioning only missing values passes.
        r = resolve("How do I fix the missing values?", None,
                    available_issues={"missing_values", "statistical_outliers"})
        assert r.topic == "missing_values"
        ans = "Impute the missing values with the column mode."
        assert answer_is_relevant(ans, r, self.ALL)

    def test_remediation_context_carries_business_guidance(self):
        # The LLM prompt context for remediation turns must carry the
        # returns/cancellations/free-items caution and the
        # delete-vs-correct-vs-impute discipline (verified deterministically
        # so the live LLM path cannot silently drop it).
        from app.ai.assistant import build_insights, _build_context_for_llm
        facts = build_insights(RETAIL_FINDINGS, RETAIL_QUALITY, RETAIL_ML,
                               original_findings=RETAIL_FINDINGS)
        history = problems_conversation()[0]
        res = resolve("How can I fix all of them?", history, available_issues=self.ALL)
        ctx = _build_context_for_llm(facts, RETAIL_QUALITY, RETAIL_ML,
                                     RETAIL_PROFILE, res.intent, resolution=res)
        guide = ctx.get("remediation_guidance", "")
        assert "returns" in guide and "cancellations" in guide and "free" in guide
        assert "deletion" in guide and "affected columns" in guide
        # Report turns must NOT carry remediation instructions.
        report_res = resolve("What problems does this dataset have?")
        ctx2 = _build_context_for_llm(facts, RETAIL_QUALITY, RETAIL_ML,
                                      RETAIL_PROFILE, report_res.intent,
                                      resolution=report_res)
        assert "remediation_guidance" not in ctx2


# ============================================================
# Cross-cutting: context still works, explicit override preserved
# ============================================================

class TestContextStillWorks:
    def test_multi_turn_context_is_used(self):
        history = [
            {"role": "user", "content": "What problems does this dataset have?"},
            {"role": "assistant", "content": "Problems detected: Missing Values, Outliers, Duplicates."},
        ]
        r = resolve("How can I fix them?", history,
                    available_issues={"missing_values", "statistical_outliers",
                                      "duplicate_rows"})
        assert r.action == "remediate"
        assert r.uses_context

    def test_explicit_new_question_beats_context(self):
        history = [
            {"role": "user", "content": "What problems does this dataset have?"},
            {"role": "assistant", "content": "Problems detected: Missing Values, Duplicates."},
        ]
        r = resolve("What is the quality score?", history)
        assert r.topic == "quality_score"
        assert r.action == "report"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

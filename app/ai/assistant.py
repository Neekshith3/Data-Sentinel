"""
AI Data Quality Assistant.

Design rule (from the project handoff): detection is done ONLY by the
deterministic Python quality engine. This module receives the engine's
structured findings and turns them into explanations + recommendations.

* Rule-based mode (default, offline, no API key): template explanations.
* LLM mode (optional): if ANTHROPIC_API_KEY is set, the LLM rewrites the
  wording. All numbers/columns always come from the engine, never the LLM.
"""

import json
import os
import urllib.request

from app.ai.chat_resolver import (
    resolve as resolve_chat,
    answer_is_relevant,
    Resolution,
    ISSUE_TOPICS,
    _last_assistant_answer,
)
from app.profiler.column_roles import get_column_tokens

# Simple .env loader (no extra dependency needed)
_ENV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env")
if os.path.exists(_ENV_PATH):
    with open(_ENV_PATH) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

SEVERITY_ORDER = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}

KNOWLEDGE = {
    "duplicate_rows": {
        "title": "Duplicate rows",
        "meaning": "Some rows are exact copies of other rows.",
        "impact": "Duplicates inflate counts and averages and can leak "
                  "identical records into both train and test splits.",
        "actions": [
            "Review the duplicated rows to confirm they are not legitimate repeats.",
            "Drop exact duplicates, keeping the first occurrence.",
            "If a business key exists (e.g. an ID), check for near-duplicates on that key.",
        ],
    },
    "missing_values": {
        "title": "Missing values",
        "meaning": "Some cells are empty, NaN or blank.",
        "impact": "Missing data biases statistics and breaks many ML algorithms.",
        "actions": [
            "Find out whether the values are missing at random or for a systematic reason.",
            "Impute numerical columns (median) and categorical columns (mode or 'Unknown'), or drop columns/rows with very high missing rates.",
            "Consider adding a 'was_missing' indicator feature before imputing.",
        ],
    },
    "categorical_inconsistency": {
        "title": "Categorical inconsistency",
        "meaning": "The same category is written in several ways (case, spacing, spelling).",
        "impact": "Categories fragment, so grouping, counts and one-hot encoding give misleading results.",
        "actions": [
            "Normalize case and trim whitespace.",
            "Map known variants to one canonical value.",
            "Recompute category distributions after cleaning.",
        ],
    },
    "data_type_validation": {
        "title": "Data type problems",
        "meaning": "Some values do not match the expected type of their column.",
        "impact": "Mixed types force numeric columns to be read as text and break calculations.",
        "actions": [
            "Locate the offending values and correct or remove them.",
            "Cast the column to its intended type after cleaning.",
        ],
    },
    "invalid_values": {
        "title": "Invalid values",
        "meaning": "Some values break the validation rules for their column (for example a negative age).",
        "impact": "Impossible values distort statistics and can mislead models.",
        "actions": [
            "Verify the source of the invalid entries.",
            "Correct them if the true value is known, otherwise treat them as missing.",
        ],
    },
    "statistical_outliers": {
        "title": "Statistical outliers",
        "meaning": "Some numerical values fall far outside the typical range (IQR rule).",
        "impact": "Outliers can skew means and standard deviations and dominate some models. "
                  "They are not necessarily errors.",
        "actions": [
            "Investigate whether each outlier is a real rare event or a data-entry error.",
            "Keep real events; for errors, correct or exclude them.",
            "For modeling, consider robust scaling or capping.",
        ],
    },
    "possible_sentinel_values": {
        "title": "Sentinel / placeholder values",
        "meaning": "Values such as -9999 or N/A appear to stand in for 'unavailable' data.",
        "impact": "Treated as real values they badly distort averages, distributions and models.",
        "actions": [
            "Confirm with the data owner what these codes mean.",
            "Replace them with proper missing values (NaN) before analysis.",
            "Exclude them from charts and statistics.",
        ],
    },
    "ml_anomalies": {
        "title": "ML-detected anomalies",
        "meaning": "An Isolation Forest flagged rows whose combination of numerical values is unusual.",
        "impact": "These rows may be errors or rare but real cases; rules alone did not catch them.",
        "actions": [
            "Review the flagged rows against their raw source.",
            "Decide per row whether to keep, fix or remove it.",
        ],
    },
}


def _score_severity(finding, quality_score):
    cat = (quality_score or {}).get("category_scores", {}).get(finding["issue_type"])
    if isinstance(cat, dict) and cat.get("severity"):
        return str(cat["severity"]).upper()
    return "MEDIUM"


def _examples(finding, limit=5):
    """Pull a few concrete examples straight from the engine's details."""
    out = []
    details = finding.get("details") or {}
    if finding["issue_type"] == "categorical_inconsistency":
        for col, d in details.items():
            groups = (d or {}).get("inconsistent_groups") or {}
            for canon, variants in list(groups.items())[:limit]:
                out.append(f"{col}: {canon} <- {variants}")
    elif finding["issue_type"] == "possible_sentinel_values":
        for col, d in (details.get("columns") or {}).items():
            for c in (d.get("sentinel_candidates") or [])[:2]:
                out.append(f"{col}: {c.get('value')} ({c.get('count')} rows)")
            for c in (d.get("placeholder_candidates") or [])[:2]:
                out.append(f"{col}: {c.get('value')} ({c.get('count')} rows)")
    elif finding["issue_type"] == "missing_values":
        mcols = details.get("columns", {})
        for col, c in list(mcols.items())[:limit]:
            if c.get("missing_count", 0) > 0:
                out.append(f"{col}: {c.get('missing_count')} missing")
    elif finding["issue_type"] == "statistical_outliers":
        # Extract actual outlier values from the findings
        for col, d in details.items():
            if isinstance(d, dict):
                outlier_count = d.get("outlier_count", 0)
                outlier_rows = d.get("outlier_rows", [])
                if outlier_count > 0:
                    # Try to get actual values from the outlier rows
                    # The outlier_rows contains row indices - we'd need the actual dataframe to get values
                    # For now, include the bounds and count
                    lower = d.get("lower_bound")
                    upper = d.get("upper_bound")
                    if lower is not None and upper is not None:
                        out.append(f"{col}: {outlier_count} outliers (IQR bounds: {lower:.2f} to {upper:.2f})")
                    else:
                        out.append(f"{col}: {outlier_count} outliers")
    elif finding["issue_type"] == "duplicate_rows":
        count = finding.get("count", 0)
        if count > 0:
            out.append(f"{count} duplicate row occurrences")
    elif finding["issue_type"] == "missing_values":
        mcols = details.get("columns", {})
        for col, c in list(mcols.items())[:limit]:
            if c.get("missing_count", 0) > 0:
                out.append(f"{col}: {c.get('missing_count')} missing")
    return out[:limit]


def build_insights(findings, quality_score=None, ml_readiness=None, original_findings=None):
    insights = []
    for f in findings or []:
        if not f.get("has_issue"):
            continue
        k = KNOWLEDGE.get(f["issue_type"], {
            "title": f["issue_type"].replace("_", " ").title(),
            "meaning": "The quality engine flagged this issue.",
            "impact": "It may reduce data quality.",
            "actions": ["Investigate the affected columns."],
        })
        insights.append({
            "issue_type": f["issue_type"],
            "title": k["title"],
            "severity": _score_severity(f, quality_score),
            "count": f.get("count", 0),
            "affected_columns": f.get("affected_columns", []),
            "examples": _examples(f),
            "what_it_means": k["meaning"],
            "potential_impact": k["impact"],
            "recommended_actions": list(k["actions"]),
        })
    insights.sort(key=lambda i: (SEVERITY_ORDER.get(i["severity"], 1), -int(i["count"] or 0)))

    for n, ins in enumerate(insights, 1):
        ins["priority"] = n

    ml_note = None
    failed = [c.get("check") for c in (ml_readiness or {}).get("checks", []) if not c.get("passed")]
    if failed:
        ml_note = "ML readiness checks not passed: " + ", ".join(failed) + "."

    if insights:
        # Build comprehensive summary with ALL affected columns
        parts = []
        for ins in insights:
            cols = ", ".join(ins["affected_columns"]) if ins["affected_columns"] else "dataset"
            parts.append(f"{ins['count']} {ins['title'].lower()} in {cols}")
        summary = f"{len(insights)} data-quality issue type(s) detected. " + "; ".join(parts) + "."
    else:
        summary = "No data-quality issues were detected by the quality engine."

    result = {"mode": "rules", "summary": summary, "ml_readiness_note": ml_note,
            "insights": insights}
    if original_findings:
        result["_original_findings"] = original_findings
    return result


def _llm_rewrite(result):
    """Optional: let an LLM rephrase text. Numbers stay engine-owned."""
    if not result["insights"]:
        return result

    gemini_key = os.environ.get("GEMINI_API_KEY")
    if gemini_key:
        return _gemini_rewrite(result, gemini_key)

    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return result

    facts = [{k: i[k] for k in ("issue_type", "count", "affected_columns", "examples", "severity")}
             for i in result["insights"]]
    prompt = (
        "You are a data-quality assistant. Below are findings from a deterministic "
        "quality engine. For each, write plain-English text. NEVER invent numbers, "
        "columns or counts; only use what is given. Return ONLY JSON: a list of "
        '{"issue_type","what_it_means","potential_impact","recommended_actions":[...]}.\n\n'
        + json.dumps(facts)
    )
    body = json.dumps({
        "model": os.environ.get("DATA_SENTINEL_LLM_MODEL", "claude-sonnet-4-6"),
        "max_tokens": 1500,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages", data=body,
        headers={"content-type": "application/json", "x-api-key": key,
                 "anthropic-version": "2023-06-01"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            text = json.loads(r.read())["content"][0]["text"]
        _apply_llm_text(result, text)
    except Exception:
        pass  # fall back silently to rule-based text
    return result


def _gemini_rewrite(result, api_key):
    """Google Gemini (free tier) rewrite of the same structured findings."""
    facts = [{k: i[k] for k in ("issue_type", "count", "affected_columns", "examples", "severity")}
             for i in result["insights"]]
    prompt = (
        "You are a data-quality assistant. Below are findings from a deterministic "
        "quality engine. For each, write plain-English text. NEVER invent numbers, "
        "columns or counts; only use what is given. Return ONLY JSON: a list of "
        '{"issue_type","what_it_means","potential_impact","recommended_actions":[...]}.\n\n'
        + json.dumps(facts)
    )
    model = os.environ.get("DATA_SENTINEL_LLM_MODEL", "gemini-3.8-flash")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    body = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": 1500, "temperature": 0.3},
    }).encode()
    req = urllib.request.Request(url, data=body, headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            resp = json.loads(r.read())
        text = resp["candidates"][0]["content"]["parts"][0]["text"]
        _apply_llm_text(result, text)
    except Exception:
        pass  # fall back silently to rule-based text
    return result


def _apply_llm_text(result, text):
    text = text.strip().strip("`")
    if text.startswith("json"):
        text = text[4:]
    by_type = {x["issue_type"]: x for x in json.loads(text)}
    for ins in result["insights"]:
        x = by_type.get(ins["issue_type"])
        if x:
            for field in ("what_it_means", "potential_impact"):
                if isinstance(x.get(field), str):
                    ins[field] = x[field]
            if isinstance(x.get("recommended_actions"), list) and x["recommended_actions"]:
                ins["recommended_actions"] = [str(a) for a in x["recommended_actions"]][:6]
    result["mode"] = "llm"


def explain(findings, quality_score=None, ml_readiness=None):
    return _llm_rewrite(build_insights(findings, quality_score, ml_readiness))


def _column_view(col_details):
    """Normalize column_details for chat consumption.

    The canonical profile historically used 'data_type' and had no 'role'.
    Accept both shapes so chat answers work with any profile variant
    (older fixtures, cached responses, or the enriched profile).
    """
    out = {}
    for col, d in (col_details or {}).items():
        if not isinstance(d, dict):
            continue
        d = dict(d)
        if not d.get("type"):
            d["type"] = d.get("data_type")
        out[col] = d
    return out


def _detected_issues(findings):
    """Set of issue-topic ids actually detected in the current dataset."""
    out = set()
    for f in findings or []:
        if f.get("has_issue"):
            out.add(f.get("issue_type"))
    return out


def _disambiguation_text(options):
    """Human-readable clarification when several issues are in play."""
    labels = {
        "missing_values": "the missing values",
        "duplicate_rows": "the duplicate rows",
        "categorical_inconsistency": "the categorical inconsistencies",
        "data_type_validation": "the data-type problems",
        "invalid_values": "the invalid values",
        "statistical_outliers": "the outliers",
        "possible_sentinel_values": "the sentinel/placeholder values",
        "ml_anomalies": "the ML-detected anomalies",
        "quality_score": "the quality score",
        "ml_readiness": "the ML readiness",
        "column_information": "the columns",
        "project_ideas": "project ideas",
        "dataset_summary": "a dataset summary",
        "problems": "the detected problems",
    }
    nice = [labels.get(o, o.replace("_", " ")) for o in options]
    if not nice:
        return "Could you clarify what you'd like me to focus on?"
    if len(nice) == 1:
        return f"Do you mean {nice[0]}?"
    return "Which do you mean: " + ", ".join(nice[:-1]) + f", or {nice[-1]}?"


def answer_question(question, findings, quality_score=None, ml_readiness=None, profile=None, conversation_history=None):
    """Answer a user question about the dataset.

    Architecture (the fix for the "wrong evidence" bug):

        question -> resolve CONVERSATION -> resolve CURRENT QUESTION
                 -> INTENT + TOPIC + EVIDENCE -> generate answer
                 -> validate RELEVANCE -> return

    The resolver is the single source of truth for intent. Both the LLM
    path and the rule path consume the same ``Resolution``, so they can
    never disagree. The current question always beats prior context; the
    context is consulted only for genuinely elliptical follow-ups.
    """
    facts = build_insights(findings, quality_score, ml_readiness, original_findings=findings)
    available = _detected_issues(findings)

    # Single authoritative resolution for this turn.
    resolution = resolve_chat(question, conversation_history, available)

    # Clarification fast-path: never invent an answer when the intent is
    # genuinely ambiguous. This is deterministic and needs no LLM.
    if resolution.action == "clarify":
        return {
            "answer": _disambiguation_text(resolution.clarification_options),
            "mode": "rules",
        }

    # Build context filtered to the evidence this resolution requires.
    context = _build_context_for_llm(facts, quality_score, ml_readiness, profile,
                                     resolution.intent, resolution=resolution)

    # --------------------------------------------------------
    # LLM path (optional). The prompt carries the resolution so the model
    # explains the *selected* evidence rather than choosing among findings.
    # --------------------------------------------------------
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        try:
            history_str = ""
            if conversation_history:
                history_str = "\nPrevious conversation:\n"
                for msg in conversation_history[-6:]:
                    role = "User" if msg.get("role") == "user" else "Assistant"
                    history_str += f"{role}: {msg.get('content', '')}\n"

            prompt = _build_prompt_for_resolution(resolution, question, context, history_str)

            model = os.environ.get("DATA_SENTINEL_LLM_MODEL", "gemini-3.8-flash")
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
            body = json.dumps({
                "contents": [{"parts": [{"text": prompt}]}],
                # Room for a complete multi-issue report/remediation;
                # partial output is rejected by answer_is_relevant anyway.
                "generationConfig": {"maxOutputTokens": 1400, "temperature": 0.3},
            }).encode()
            req = urllib.request.Request(url, data=body, headers={"content-type": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as r:
                resp = json.loads(r.read())
            text = _strip_markdown(resp["candidates"][0]["content"]["parts"][0]["text"])

            # Phase 14: reject an answer that doesn't address the resolution.
            if answer_is_relevant(text, resolution, available):
                return _avoid_expanded_repeat({"answer": text, "mode": "llm"},
                                              resolution, conversation_history)
            # Irrelevant LLM output -> fall through to the deterministic path.
        except Exception:
            pass

    # Deterministic path (always correct, no network).
    return _avoid_expanded_repeat(
        _rule_based_answer(question, facts, quality_score, ml_readiness, profile,
                           conversation_history, resolution=resolution),
        resolution, conversation_history)


def _avoid_expanded_repeat(answer, resolution, conversation_history):
    """"Give me more" must continue the topic with NEW information.

    Dedicated continuations (feature importance, quality score, ML
    readiness, columns, per-issue details, problems, priority, summary,
    model selection, project ideas) produce fresh text before this
    runs. If a topic has no deeper canonical detail and the generated
    text equals the previous answer, return an honest continuation of
    the SAME topic instead of the response verbatim.
    """
    if not answer or not resolution or resolution.action != "expand":
        return answer
    prev = _last_assistant_answer(conversation_history)
    if not prev:
        return answer

    def _norm(t):
        return " ".join((t or "").lower().split())

    if _norm(answer.get("answer")) != _norm(prev):
        return answer
    label = (resolution.topic or "").replace("_", " ").strip()
    if label in ("", "general"):
        label = "this topic"
    return {"answer": (
        f"My previous reply is the complete answer on {label}; the analysis has "
        "no further measured detail to add on that exact point, so I will not "
        "repeat it. I can continue on a related angle instead: how to fix what "
        "was found, the per-column details behind it, or how it affects the ML "
        "readiness score. Which would you like?"), "mode": "rules"}


def _build_context_for_llm(facts, quality_score, ml_readiness, profile, intent=None, resolution=None):
    """Build structured context for the LLM, filtered to the evidence the
    resolution requires.

    When a ``resolution`` is supplied, only the evidence it names is
    included. This is what stops the model from being handed every finding
    and picking an unrelated one.
    """
    context = {
        "dataset": {
            "rows": (profile or {}).get("rows"),
            "columns": (profile or {}).get("columns"),
            "name": (profile or {}).get("name", "dataset"),
        },
        "quality_score": {
            "overall": (quality_score or {}).get("overall_score"),
            "categories": (quality_score or {}).get("category_scores", {}),
        },
        "ml_readiness": {
            "percentage": (ml_readiness or {}).get("readiness_percentage"),
            "checks": (ml_readiness or {}).get("checks", []),
        },
        "issues": [],
    }

    # Determine which issue types are relevant for the current intent
    relevant_issue_types = _get_relevant_issue_types_for_intent(intent)

    # Phase 10: evidence selection is driven by the resolution. A
    # single-issue resolution narrows to exactly that issue.
    if resolution is not None and resolution.topic in ISSUE_TOPICS:
        relevant_issue_types = {resolution.topic}

    for i in facts["insights"]:
        # Skip issues not relevant to the current intent
        if intent and i["issue_type"] not in relevant_issue_types:
            continue
            
        issue_data = {
            "type": i["issue_type"],
            "title": i["title"],
            "count": i["count"],
            "severity": i["severity"],
            "columns": i["affected_columns"],
            "examples": i["examples"],
            "meaning": i["what_it_means"],
            "impact": i["potential_impact"],
            "actions": i["recommended_actions"],
        }
        
        # Add specific details from the original findings
        # Find the original finding for this issue type
        for f in facts.get("_original_findings", []):
            if f.get("issue_type") == i["issue_type"] and f.get("has_issue"):
                details = f.get("details", {})
                if i["issue_type"] == "statistical_outliers":
                    # Add detailed outlier information
                    issue_data["outlier_details"] = {}
                    for col, d in details.items():
                        if isinstance(d, dict) and d.get("has_issue"):
                            issue_data["outlier_details"][col] = {
                                "count": d.get("outlier_count", 0),
                                "q1": d.get("q1"),
                                "q3": d.get("q3"),
                                "iqr": d.get("iqr"),
                                "lower_bound": d.get("lower_bound"),
                                "upper_bound": d.get("upper_bound"),
                                # Retained evidence: representative values and a
                                # capped index sample (the full list can be huge).
                                "representative_values": [
                                    _fmt_numeric(v)
                                    for v in (d.get("outlier_values") or [])[:5]
                                ],
                                "outlier_row_indices_sample": [
                                    str(x) for x in (d.get("outlier_rows") or [])[:25]
                                ],
                            }
                elif i["issue_type"] == "missing_values":
                    issue_data["missing_details"] = {}
                    for col, d in details.get("columns", {}).items():
                        if d.get("missing_count", 0) > 0:
                            issue_data["missing_details"][col] = {
                                "count": d.get("missing_count"),
                                "percentage": d.get("missing_percentage"),
                            }
                elif i["issue_type"] == "duplicate_rows":
                    issue_data["duplicate_details"] = {
                        "total_count": details.get("percentage", 0),
                        "rows_involved": details.get("duplicate_rows_involved", 0),
                    }
                elif i["issue_type"] == "possible_sentinel_values":
                    issue_data["sentinel_details"] = {}
                    for col, d in details.get("columns", {}).items():
                        if d.get("sentinel_count", 0) > 0 or d.get("placeholder_count", 0) > 0:
                            issue_data["sentinel_details"][col] = {
                                "sentinel_count": d.get("sentinel_count", 0),
                                "placeholder_count": d.get("placeholder_count", 0),
                                "sentinel_candidates": d.get("sentinel_candidates", []),
                                "placeholder_candidates": d.get("placeholder_candidates", []),
                            }
                elif i["issue_type"] == "categorical_inconsistency":
                    issue_data["inconsistency_details"] = {}
                    for col, d in details.items():
                        if d.get("has_issue"):
                            issue_data["inconsistency_details"][col] = {
                                "inconsistent_groups": d.get("inconsistent_groups", {}),
                            }
                elif i["issue_type"] == "duplicate_rows":
                    issue_data["duplicate_details"] = {
                        "duplicate_rows": details.get("duplicate_rows", []),
                        "duplicate_rows_involved": details.get("duplicate_rows_involved", 0),
                    }
                break
        
        context["issues"].append(issue_data)
    
    # Add column-level details from profile if available
    col_details = _column_view((profile or {}).get("column_details", {}))
    if col_details:
        context["column_details"] = {}
        for col, details in col_details.items():
            context["column_details"][col] = {
                "type": details.get("type"),
                "role": details.get("role"),
                "missing_count": details.get("missing_count", 0),
                "missing_pct": details.get("missing_percentage", 0),
                "unique_count": details.get("unique_count", 0),
            }

    # Remediation turns carry explicit handling guidance so the LLM path
    # gives the same method discipline as the rule templates: distinguish
    # deletion / correction / imputation / retention / investigation, and
    # validate business meaning first -- negative quantities or zero
    # prices in transactional data can be legitimate business events
    # (returns, cancellations, free or promotional items).
    if resolution is not None and resolution.action == "remediate":
        context["remediation_guidance"] = (
            "For EACH issue type listed above give a separate actionable "
            "method, grounded in its affected columns, that distinguishes "
            "investigation, correction, imputation, retention, and deletion. "
            "Negative quantities or zero prices in transactional data can be "
            "legitimate business events (returns, cancellations, free or "
            "promotional items): validate the business meaning with the data "
            "owner before treating them as errors. Never recommend deleting "
            "rows blindly -- deletion only for confirmed errors and true "
            "duplicates."
        )

    return context


def _get_relevant_issue_types_for_intent(intent):
    """Return the set of issue types relevant for a given intent."""
    if not intent:
        # Return all issue types if no specific intent
        return {
            "missing_values", "duplicate_rows", "categorical_inconsistency",
            "data_type_validation", "invalid_values", "statistical_outliers",
            "possible_sentinel_values", "ml_anomalies"
        }
    
    intent_to_issues = {
        "overview": {"missing_values", "duplicate_rows", "categorical_inconsistency",
                     "data_type_validation", "invalid_values", "statistical_outliers",
                     "possible_sentinel_values", "ml_anomalies"},
        "problems": {"missing_values", "duplicate_rows", "categorical_inconsistency",
                     "data_type_validation", "invalid_values", "statistical_outliers",
                     "possible_sentinel_values", "ml_anomalies"},
        "priority": {"missing_values", "duplicate_rows", "categorical_inconsistency",
                     "data_type_validation", "invalid_values", "statistical_outliers",
                     "possible_sentinel_values", "ml_anomalies"},
        "remediation": {"missing_values", "duplicate_rows", "categorical_inconsistency",
                        "data_type_validation", "invalid_values", "statistical_outliers",
                        "possible_sentinel_values", "ml_anomalies"},
        "outlier_remediation": {"statistical_outliers"},
        "missing_remediation": {"missing_values"},
        "duplicate_remediation": {"duplicate_rows"},
        "inconsistency_remediation": {"categorical_inconsistency"},
        "sentinel_remediation": {"possible_sentinel_values"},
        "invalid_remediation": {"invalid_values"},
        "type_validation_remediation": {"data_type_validation"},
        "quality_remediation": {"missing_values", "duplicate_rows", "categorical_inconsistency",
                                "data_type_validation", "invalid_values", "statistical_outliers",
                                "possible_sentinel_values", "ml_anomalies"},
        "ml_readiness_remediation": {"missing_values", "duplicate_rows", "categorical_inconsistency",
                                     "data_type_validation", "invalid_values", "statistical_outliers",
                                     "possible_sentinel_values"},
        "ml_readiness": {"missing_values", "duplicate_rows", "categorical_inconsistency",
                         "data_type_validation", "invalid_values", "statistical_outliers",
                         "possible_sentinel_values"},
        "quality_score": {"missing_values", "duplicate_rows", "categorical_inconsistency",
                          "data_type_validation", "invalid_values", "statistical_outliers",
                          "possible_sentinel_values", "ml_anomalies"},
        "missing_values": {"missing_values"},
        "duplicate_rows": {"duplicate_rows"},
        "statistical_outliers": {"statistical_outliers"},
        "possible_sentinel_values": {"possible_sentinel_values"},
        "categorical_inconsistency": {"categorical_inconsistency"},
        "invalid_values": {"invalid_values"},
        "data_type_validation": {"data_type_validation"},
        "project_ideas": set(),  # Use profile data instead
        "feature_importance": {"missing_values", "duplicate_rows", "categorical_inconsistency",
                               "data_type_validation", "invalid_values", "statistical_outliers",
                               "possible_sentinel_values", "ml_anomalies"},
        "general": {"missing_values", "duplicate_rows", "categorical_inconsistency",
                    "data_type_validation", "invalid_values", "statistical_outliers",
                    "possible_sentinel_values", "ml_anomalies"},
    }
    
    return intent_to_issues.get(intent, {
        "missing_values", "duplicate_rows", "categorical_inconsistency",
        "data_type_validation", "invalid_values", "statistical_outliers",
        "possible_sentinel_values", "ml_anomalies"
    })



def _build_prompt_for_resolution(resolution, question, context, history_str):
    """PHASE 27: compact, structured, evidence-first prompt.

    The prompt states the resolved intent, topic and the exact evidence to
    use, and instructs the model to answer ONLY the current question using
    only that evidence. This keeps the LLM in the role of "explainer of
    selected evidence" rather than " chooser among findings".
    """
    topic_label = (resolution.topic or "multiple / dataset-wide").replace("_", " ")
    base = (
        "You are Data Sentinel's data-quality assistant.\n"
        "Answer ONLY the current user question, using ONLY the canonical evidence provided below.\n"
        "Do NOT answer a previous question. Do NOT introduce unrelated findings.\n"
        "Do NOT invent facts, numbers, columns, statistics, feature importance, or ML readiness.\n"
        "If the evidence does not contain what is needed, say so plainly.\n"
        "Use plain text (no markdown symbols). Be concise and practical.\n"
    )
    lines = [base, f"CURRENT DATASET: {(context.get('dataset') or {}).get('name', 'dataset')} "
                   f"({(context.get('dataset') or {}).get('rows')} rows, "
                   f"{(context.get('dataset') or {}).get('columns')} columns)",
             f"CURRENT INTENT: {resolution.intent}",
             f"CURRENT TOPIC: {topic_label}",
             f"CURRENT ACTION: {resolution.action}"]
    if resolution.uses_context:
        lines.append("NOTE: This is a follow-up; resolve references from the previous turn, "
                     "but answer the CURRENT question.")
    lines.append("\nRELEVANT CANONICAL EVIDENCE:\n" + json.dumps(context, default=str))
    if history_str:
        lines.append("\n" + history_str)
    lines.append(f"\nCURRENT USER QUESTION: {question}")
    return "\n".join(lines)


def _strip_markdown(text):
    """Remove markdown formatting from LLM response."""
    if not text:
        return text
    # Remove bold
    text = text.replace("**", "").replace("__", "")
    # Remove italic
    text = text.replace("*", "").replace("_", "")
    # Remove headers
    text = text.replace("# ", "").replace("## ", "").replace("### ", "")
    # Clean up multiple newlines
    import re
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# Helper functions for generating follow-up responses
def _generate_more_outlier_info(original_findings):
    """Generate more detailed outlier information for follow-up."""
    outlier_details = []
    for f in original_findings or []:
        if f.get("issue_type") == "statistical_outliers" and f.get("has_issue"):
            details = f.get("details", {})
            for col, d in details.items():
                if isinstance(d, dict) and d.get("has_issue"):
                    outlier_count = d.get("outlier_count", 0)
                    outlier_values = d.get("outlier_values", [])
                    lower = d.get("lower_bound")
                    upper = d.get("upper_bound")
                    outlier_rows = d.get("outlier_rows", [])
                    outlier_values_list = d.get("outlier_values", [])
                    
                    parts = [f"Column '{col}': {outlier_count} outliers detected"]
                    if outlier_values:
                        vals_str = ", ".join(str(v) for v in outlier_values[:10])
                        parts.append("Values: " + ", ".join(str(v) for v in outlier_values[:20]))
                    if lower is not None and upper is not None:
                        parts.append("IQR bounds: " + f"{lower:.2f} to {upper:.2f}")
                    if outlier_rows:
                        parts.append("Affected row indices (first 10): " + ", ".join(str(r) for r in outlier_rows[:10]))
                    
                    out.append(". ".join(parts))
    
    if not outlier_details:
        return {"answer": "No outlier details available.", "mode": "rules"}
    
    answer = "Detailed outlier information:\n\n" + "\n\n".join(outlier_details)
    return {"answer": answer, "mode": "rules"}


def _generate_more_sentinel_info(original_findings):
    sentinel_details = []
    for f in original_findings or []:
        if f.get("issue_type") == "possible_sentinel_values" and f.get("has_issue"):
            details = f.get("details", {})
            for col, d in details.get("columns", {}).items():
                if d.get("sentinel_count", 0) > 0 or d.get("placeholder_count", 0) > 0:
                    parts = [f"Column '{col}': {d.get('sentinel_count', 0)} sentinels, {d.get('placeholder_count', 0)} placeholders"]
                    for c in d.get("sentinel_candidates", [])[:3]:
                        parts.append(f"  Sentinel: {c.get('value')} ({c.get('count')} rows)")
                    for c in d.get("placeholder_candidates", [])[:3]:
                        parts.append(f"  Placeholder: {c.get('value')} ({c.get('count')} rows)")
                    sentinel_details.append(". ".join(parts))
    
    if not sentinel_details:
        return {"answer": "No sentinel details available.", "mode": "rules"}
    
    answer = "Additional sentinel/placeholder details:\n\n" + "\n\n".join(sentinel_details)
    return {"answer": answer, "mode": "rules"}


def _generate_more_missing_info(original_findings):
    missing_details = []
    for f in original_findings or []:
        if f.get("issue_type") == "missing_values" and f.get("has_issue"):
            details = f.get("details", {})
            for col, d in details.get("columns", {}).items():
                if d.get("missing_count", 0) > 0:
                    pct = d.get("missing_percentage", 0)
                    missing_details.append(f"Column '{col}': {d['missing_count']} missing ({pct:.1f}%)")
    
    if not missing_details:
        return {"answer": "No missing value details available.", "mode": "rules"}
    
    answer = "Additional missing value details:\n\n" + "\n".join(missing_details)
    return {"answer": answer, "mode": "rules"}


def _generate_more_duplicate_info(original_findings):
    dup_details = []
    for f in original_findings or []:
        if f.get("issue_type") == "duplicate_rows" and f.get("has_issue"):
            details = f.get("details", {})
            count = f.get("count", 0)
            involved = details.get("duplicate_rows_involved", 0)
            dups = details.get("duplicate_rows", [])
            parts = [f"Total duplicate occurrences: {count}", f"Rows involved in duplicates: {involved}"]
            if dups:
                parts.append("Example duplicate row indices (first 10): " + ", ".join(str(d) for d in dups[:10]))
            dup_details.append(". ".join(parts))
    
    if not dup_details:
        return {"answer": "No duplicate details available.", "mode": "rules"}
    
    answer = "Additional duplicate details:\n\n" + "\n\n".join(dup_details)
    return {"answer": answer, "mode": "rules"}


def _generate_more_inconsistency_info(original_findings):
    """Generate more detailed inconsistency information for follow-up."""
    details = []
    for f in original_findings or []:
        if f.get("issue_type") == "categorical_inconsistency" and f.get("has_issue"):
            detail = f.get("details", {})
            for col, d in detail.items():
                if isinstance(d, dict) and d.get("has_issue"):
                    groups = d.get("inconsistent_groups", {})
                    count = d.get("inconsistency_count", 0)
                    details.append(f"Column '{col}' ({count} inconsistent values):")
                    for canonical, variants in list(groups.items())[:3]:
                        details.append(f"  Canonical: '{canonical}' -> Variants: {', '.join(variants[:5])}")
    if not details:
        return {"answer": "No additional inconsistency details available.", "mode": "rules"}
    answer = "Additional inconsistency details:\n\n" + "\n".join(details)
    return {"answer": answer, "mode": "rules"}


def _generate_more_quality_info(quality_score):
    cats = (quality_score or {}).get("category_scores", {})
    parts = []
    for cat, d in cats.items():
        sev = d.get("severity", "UNKNOWN")
        score = d.get("score", "N/A")
        affected = d.get("affected_rows", "N/A")
        parts.append(f"{cat}: score={score}, severity={sev}, affected_rows={affected}")
    
    if not parts:
        return {"answer": "No quality score details available.", "mode": "rules"}
    
    answer = "Quality score category breakdown:\n\n" + "\n".join(parts)
    return {"answer": answer, "mode": "rules"}


def _generate_more_ml_info(ml_readiness):
    checks = (ml_readiness or {}).get("checks", [])
    parts = []
    for c in checks:
        status = "PASS" if c.get("passed") else "FAIL"
        parts.append(f"{c.get('check', 'unknown')}: {status} (count: {c.get('count', 'N/A')})")
    
    if not parts:
        return {"answer": "No ML readiness details available.", "mode": "rules"}
    
    answer = "ML Readiness check details:\n\n" + "\n".join(parts)
    return {"answer": answer, "mode": "rules"}


# Remediation generators
def _generate_outlier_remediation(original_findings):
    for f in original_findings or []:
        if f.get("issue_type") == "statistical_outliers" and f.get("has_issue"):
            details = f.get("details", {})
            parts = ["Outlier remediation options:"]
            for col, d in details.items():
                if isinstance(d, dict) and d.get("has_issue"):
                    outlier_count = d.get("outlier_count", 0)
                    parts.append(f"Column '{col}' ({outlier_count} outliers):")
                    parts.append("  1. Inspect the outlier rows to determine if they are genuine rare events or data-entry errors")
                    parts.append("  2. If genuine: keep them but consider robust scaling or capping for modeling")
                    parts.append("  3. If errors: correct the values or replace with NaN/imputed values")
                    parts.append("  4. Consider robust scaling (RobustScaler) or capping at IQR bounds for modeling")
            return {"answer": "\n".join(parts), "mode": "rules"}
    return {"answer": "No outlier remediation details available.", "mode": "rules"}


def _generate_missing_remediation(original_findings):
    for f in original_findings or []:
        if f.get("issue_type") == "missing_values" and f.get("has_issue"):
            details = f.get("details", {})
            parts = ["Missing value remediation options:"]
            for col, d in details.get("columns", {}).items():
                if d.get("missing_count", 0) > 0:
                    pct = d.get("missing_percentage", 0)
                    parts.append(f"Column '{col}' ({d['missing_count']} missing, {pct:.1f}%):")
                    parts.append("  1. If missing at random: impute with median (numerical) or mode/'Unknown' (categorical)")
                    parts.append("  2. If missing not at random: investigate source, consider adding 'was_missing' indicator")
                    parts.append("  3. If missing rate > 50%: consider dropping the column")
            return {"answer": "\n".join(parts), "mode": "rules"}
    return {"answer": "No missing value remediation details available.", "mode": "rules"}


def _generate_duplicate_remediation(original_findings):
    for f in original_findings or []:
        if f.get("issue_type") == "duplicate_rows" and f.get("has_issue"):
            details = f.get("details", {})
            count = f.get("count", 0)
            involved = details.get("duplicate_rows_involved", 0)
            parts = [f"Duplicate row remediation ({count} occurrences, {involved} rows involved):"]
            parts.append("1. Review duplicate rows to confirm they are not legitimate repeated measurements")
            parts.append("2. Drop exact duplicates, keeping the first occurrence (df.drop_duplicates(keep='first'))")
            parts.append("3. If business key exists (e.g., ID), check for near-duplicates on that key")
            return {"answer": "\n".join(parts), "mode": "rules"}
    return {"answer": "No duplicate remediation details available.", "mode": "rules"}


def _generate_inconsistency_remediation(original_findings):
    for f in original_findings or []:
        if f.get("issue_type") == "categorical_inconsistency" and f.get("has_issue"):
            details = f.get("details", {})
            parts = ["Categorical inconsistency remediation:"]
            for col, d in details.items():
                if isinstance(d, dict) and d.get("has_issue"):
                    groups = d.get("inconsistent_groups", {})
                    parts.append(f"Column '{col}':")
                    parts.append("  1. Normalize case (lower/upper) and trim whitespace")
                    parts.append("  2. Map known variants to one canonical value (e.g., 'Credit Card' and 'Credit card' -> 'Credit Card')")
                    parts.append("  3. Recompute category distributions after cleaning")
                    break
            return {"answer": "\n".join(parts), "mode": "rules"}
    return {"answer": "No inconsistency remediation details available.", "mode": "rules"}


def _generate_sentinel_remediation(original_findings):
    for f in original_findings or []:
        if f.get("issue_type") == "possible_sentinel_values" and f.get("has_issue"):
            details = f.get("details", {})
            parts = ["Sentinel/placeholder value remediation:"]
            for col, d in details.get("columns", {}).items():
                if d.get("sentinel_count", 0) > 0 or d.get("placeholder_count", 0) > 0:
                    parts.append(f"Column '{col}' ({d.get('sentinel_count', 0)} sentinels, {d.get('placeholder_count', 0)} placeholders):")
                    parts.append("  1. Confirm with data owner what these codes mean (e.g., -9999 = 'not measured')")
                    parts.append("  2. Replace confirmed sentinels with NaN (pd.NA or np.nan) before analysis")
                    parts.append("  3. Exclude sentinel values from charts, statistics, and model training")
            return {"answer": "\n".join(parts), "mode": "rules"}
    return {"answer": "No sentinel remediation details available.", "mode": "rules"}


def _generate_invalid_remediation(original_findings):
    for f in original_findings or []:
        if f.get("issue_type") == "invalid_values" and f.get("has_issue"):
            details = f.get("details", {})
            parts = ["Invalid value remediation:"]
            for col, d in details.items():
                if isinstance(d, dict) and d.get("has_issue"):
                    count = d.get("invalid_count", 0)
                    parts.append(f"Column '{col}' ({count} invalid values):")
                    parts.append("  1. Investigate first: confirm these values against the business rules and the source system. In transactional data, negative quantities or zero prices can be legitimate events (returns, cancellations, free items).")
                    parts.append("  2. If they are errors and the true value is known: correct them")
                    parts.append("  3. If they are errors and the true value is unknown: treat them as missing (impute or flag)")
                    parts.append("  4. If they are legitimate business events: retain them and relax the validation rule")
            return {"answer": "\n".join(parts), "mode": "rules"}
    return {"answer": "No invalid value remediation details available.", "mode": "rules"}


def _generate_type_validation_remediation(original_findings):
    for f in original_findings or []:
        if f.get("issue_type") == "data_type_validation" and f.get("has_issue"):
            details = f.get("details", {})
            parts = ["Data type validation remediation:"]
            for col, d in details.get("columns", {}).items():
                if d.get("has_issue"):
                    count = d.get("type_mismatch_count", 0)
                    parts.append(f"Column '{col}' ({count} mismatches):")
                    parts.append("  1. Locate the offending values and correct or remove them")
                    parts.append("  2. Cast the column to its intended type after cleaning (df[col].astype(type))")
            return {"answer": "\n".join(parts), "mode": "rules"}
    return {"answer": "No data type validation remediation details available.", "mode": "rules"}


def _collect_project_ideas(profile, limit=6):
    """Dataset-aware project ideas derived only from the canonical profile.

    Ideas emerge from the dataset's actual structure -- column roles,
    cardinality, repeating identifiers, date fields, and measurement
    columns -- never from dataset names or hardcoded columns. Each idea
    states the business objective, the relevant available columns, the
    approach, and the main limitation. No idea assumes a target label
    exists or that any model has been trained.

    This is the single source for project ideas: ``_build_project_ideas``
    and the "give me more" path both consume it, so the two can never
    diverge into duplicate recommendation logic.
    """
    profile = profile or {}
    col_details = _column_view((profile or {}).get("column_details", {}))
    rows = profile.get("rows") or 0

    def tokens(col):
        return set(get_column_tokens(col))

    def d(col):
        return col_details.get(col) or {}

    def ratio(col):
        r = d(col).get("uniqueness_ratio")
        if r is None:
            # The chat API profile may omit uniqueness_ratio; derive it
            # from the canonical cardinality and row count instead.
            uniq = (d(col).get("non_null_unique_count")
                    or d(col).get("unique_count") or 0)
            r = (uniq / rows) if rows else 0
        return r or 0

    numeric_cols = [c for c, v in col_details.items()
                    if v.get("type") in ("numerical", "integer", "float")]
    categorical_cols = [c for c, v in col_details.items()
                        if v.get("type") in ("categorical", "object", "string")]
    datetime_cols = [c for c, v in col_details.items()
                     if v.get("type") in ("datetime", "date") or v.get("role") == "datetime"]
    target_cols = [c for c, v in col_details.items() if v.get("role") in ("target", "label")]
    id_cols = [c for c, v in col_details.items()
               if v.get("role") in ("identifier", "identifier_like")]

    # Generic vocabularies (not dataset-specific): name tokens that mark
    # transaction-like ids, entity ids, product/code columns, and
    # measurement columns. Structure evidence (repetition ratios) is
    # combined with these names -- a project is only suggested when the
    # necessary relationships actually exist in the data.
    TX_TOKENS = {"invoice", "order", "transaction", "receipt", "bill",
                 "ticket", "session", "visit", "trip", "booking"}
    ENTITY_TOKENS = {"customer", "client", "patient", "student", "user",
                     "account", "member", "subscriber"}
    PRODUCT_TOKENS = {"stock", "product", "item", "sku", "article",
                      "commodity", "menu", "service"}
    MEASURE_TOKENS = {"quantity", "qty", "amount", "price", "revenue",
                      "sales", "total", "value", "cost", "spend",
                      "payment", "charge"}

    def with_tokens(cols, vocab):
        return [c for c in cols if tokens(c) & vocab]

    # Repeating identifiers: many rows share one value -- the signature of
    # transactional or entity data (requires enough rows to mean anything).
    repeating_ids = [c for c in id_cols if 0 < ratio(c) <= 0.5] if rows >= 50 else []

    tx_named = with_tokens(repeating_ids, TX_TOKENS)
    entity_named = [c for c in with_tokens(repeating_ids, ENTITY_TOKENS)
                    if c not in tx_named]
    tx_col = tx_named[0] if tx_named else None
    entity_col = entity_named[0] if entity_named else None

    # Cardinality-shape fallbacks: the finest-grained repeating id is most
    # often the transaction id; a second distinct repeating id can serve
    # as the entity id.
    if tx_col is None and repeating_ids:
        tx_col = max(repeating_ids, key=ratio)
    if entity_col is None and tx_col is not None:
        others = [c for c in repeating_ids if c != tx_col]
        if others:
            entity_col = max(others, key=ratio)

    # Product/code column: identifier-role or high-cardinality categorical
    # that looks like a product/item code, distinct from the ids above.
    product_col = None
    for c in with_tokens(id_cols + categorical_cols, PRODUCT_TOKENS):
        if c not in (tx_col, entity_col) and (d(c).get("unique_count", 0) or 0) >= 20:
            product_col = c
            break

    used = {c for c in (tx_col, entity_col, product_col) if c}
    measure_named = with_tokens(numeric_cols, MEASURE_TOKENS)
    measure_cols = ([c for c in measure_named if c not in used]
                    or [c for c in numeric_cols if c not in used])

    ideas = []

    # Supervised ideas ONLY when the profile actually identifies a target.
    if target_cols and numeric_cols:
        target_list = ", ".join(target_cols)
        feats = ", ".join(c for c in numeric_cols if c not in target_cols)[:120]
        target_cats = [c for c in target_cols
                       if d(c).get("type") in ("categorical", "object", "string")]
        kind = "Classification" if target_cats else "Regression"
        ideas.append(
            f"{kind}: predict '{target_list}' from the available features "
            f"(e.g. {feats}). Objective: learn a mapping from the features to the "
            "target. Approach: a simple baseline first (logistic/linear), then "
            "tree-based models with cross-validation. Limitation: the profile only "
            "flags this column as target-like -- verify it is a genuine label, check "
            "class balance and leakage; no model has been trained yet.")

    # Transactional structure -> market-basket analysis.
    if tx_col and product_col:
        n_ids = d(tx_col).get("unique_count", 0) or 0
        ideas.append(
            f"Market-basket analysis: find which '{product_col}' entries are frequently "
            f"purchased together within the same '{tx_col}'. Objective: cross-sell and "
            f"layout insight. Approach: group rows by '{tx_col}', build item sets from "
            f"'{product_col}', and mine association rules (Apriori/FP-growth). "
            f"Limitation: this requires genuinely multi-item '{tx_col}' groups -- the "
            f"data has {n_ids} distinct '{tx_col}' values across {rows} rows; "
            "single-item groups and missing ids must be excluded.")

    # Date + measurement -> demand/sales forecasting.
    if datetime_cols and measure_cols:
        m = measure_cols[0]
        ideas.append(
            f"Demand/quantity forecasting: predict '{m}' over time using "
            f"'{datetime_cols[0]}' as the time index. Objective: plan inventory, "
            f"staffing, or cash flow. Approach: aggregate '{m}' by day/week, check "
            "trend and seasonality, then fit a moving-average/ETS/ARIMA baseline. "
            f"Limitation: needs a sufficiently long, regular time span in "
            f"'{datetime_cols[0]}'; external drivers (promotions, holidays, outages) "
            "are not captured in the data.")

    # Entity + date + measurement -> RFM-style segmentation.
    if entity_col and datetime_cols and measure_cols:
        ideas.append(
            f"Customer/entity segmentation (RFM): group '{entity_col}' by recency, "
            f"frequency, and monetary value using '{datetime_cols[0]}' and "
            f"'{measure_cols[0]}'. Objective: differentiate engagement per segment. "
            "Approach: aggregate per entity, standardize, cluster (k-means) and "
            f"profile the segments. Limitation: rows with missing '{entity_col}' are "
            "excluded, and the segments are descriptive, not causal.")

    # Date + product + measurement -> product sales analysis.
    if datetime_cols and product_col and measure_cols:
        ideas.append(
            f"Product sales analysis: rank the top and bottom '{product_col}' entries "
            f"by '{measure_cols[0]}' and check per-entry seasonality. Objective: "
            "merchandising and stock decisions. Approach: aggregate "
            f"'{measure_cols[0]}' by '{product_col}' and by time period, then compare. "
            f"Limitation: sparse '{product_col}' entries produce noisy trends.")

    # Transaction-level anomaly review.
    if tx_col and measure_cols:
        ideas.append(
            f"Transaction anomaly review: flag unusually large '{measure_cols[0]}' "
            f"'{tx_col}' transactions for manual inspection. Objective: catch errors "
            "or abuse early. Approach: unsupervised scoring (isolation forest or "
            "z-scores) on per-transaction aggregates. Limitation: no labeled anomalies "
            "exist, so flags are leads for review, not confirmed incidents.")

    # Generic fallbacks for other dataset shapes (only when the more
    # specific ideas above do not already cover the same ground).
    if len(numeric_cols) >= 2 and not tx_col:
        ideas.append(
            "Anomaly detection: find unusual combinations of " +
            ", ".join(numeric_cols[:3]) +
            " using isolation forest or similar methods. Objective: flag records that "
            "deviate from the norm. Limitation: unsupervised -- flagged records need "
            "domain review before action.")
    if len(numeric_cols) + len(categorical_cols) >= 3 and not entity_col:
        ideas.append(
            "Record segmentation: group similar rows using " +
            ", ".join(numeric_cols[:2] + categorical_cols[:1]) +
            ". Objective: discover natural groups. Approach: standardize, encode "
            "categoricals, cluster (k-means) and profile the clusters. Limitation: "
            "cluster count is a modeling choice, and groups are descriptive only.")
    high_card = [c for c in categorical_cols if (d(c).get("unique_count", 0) or 0) > 10]
    if high_card:
        ideas.append(
            "Feature engineering: encode high-cardinality categoricals like " +
            ", ".join(high_card[:2]) +
            " (target or frequency encoding) for downstream models. Limitation: "
            "target encoding must be computed inside cross-validation to avoid leakage.")

    if not ideas:
        ideas = [
            "Exploratory data analysis to understand patterns and relationships "
            "(distributions, correlations, category frequencies).",
            "Basic statistical analysis and visualization to track how the "
            "dataset changes over time.",
        ]
    return ideas[:limit]


def _generate_more_project_ideas(profile, insights, exclude_text=None):
    """Generate project ideas based on dataset characteristics.

    ``exclude_text`` is the previous assistant answer; ideas already shown
    in it are skipped so "give me more" actually yields new ideas.
    """
    ideas = _collect_project_ideas(profile)

    # "Give me more": skip ideas already shown in the previous answer.
    if exclude_text:
        fresh = [i for i in ideas if i not in exclude_text]
        if not fresh:
            fresh = [
                "Quality-aware modeling: train models with missing-value indicators "
                "and robust preprocessing built into the pipeline.",
                "Data-quality monitoring: re-run missing-value, duplicate, and "
                "consistency checks automatically whenever new data arrives.",
                "Insight dashboard: visualize key distributions and category "
                "frequencies to track how the dataset changes over time.",
            ]
            fresh = [i for i in fresh if i not in exclude_text]
        if fresh:
            ideas = fresh
        else:
            return {"answer": "We've already covered the main project ideas for this "
                              "dataset. A good next step is to pick one direction and "
                              "start with a small exploratory prototype.", "mode": "rules"}

    lead = ("Based on the dataset's characteristics, here are additional project ideas:"
            if exclude_text else
            "Based on the dataset's characteristics, here are project ideas:")
    answer = lead + "\n\n"
    for idx, idea in enumerate(ideas, 1):
        answer += str(idx) + ". " + idea + "\n"

    return {"answer": answer, "mode": "rules"}


# ============================================================
# NEW HELPER FUNCTIONS FOR RECOMMENDATIONS AND SIMPLE EXPLANATIONS
# ============================================================

def _generate_outlier_recommendation(original_findings):
    """Generate a recommendation for outlier handling based on the data."""
    for f in original_findings or []:
        if f.get("issue_type") == "statistical_outliers" and f.get("has_issue"):
            details = f.get("details", {})
            parts = ["For outlier handling, I recommend:"]
            for col, d in details.items():
                if isinstance(d, dict) and d.get("has_issue"):
                    outlier_count = d.get("outlier_count", 0)
                    outlier_values = d.get("outlier_values", [])
                    lower = d.get("lower_bound")
                    upper = d.get("upper_bound")
                    parts.append(f"\nFor column '{col}' ({outlier_count} outliers):")
                    if outlier_values:
                        vals_str = ", ".join(str(v) for v in outlier_values[:5])
                        parts.append(f"  Outlier values: {vals_str} (IQR bounds: {lower:.2f} to {upper:.2f})")
                    parts.append("  • First, inspect the outlier rows to determine if they are genuine rare events or data-entry errors")
                    parts.append("  • If they are genuine rare events (e.g., legitimate extreme measurements): keep them but use robust scaling or capping for modeling")
                    parts.append("  • If they are data-entry errors: correct the values or replace with NaN/imputed values")
                    parts.append("  • For most ML projects: median replacement or IQR-based capping (Winsorization) are good starting points")
                    parts.append("\nRecommendation: Start with IQR-based capping (Winsorization at 1.5*IQR) as it preserves the data distribution while limiting extreme influence.")
            return {"answer": "".join(parts), "mode": "rules"}
    return {"answer": "No outlier data available for recommendation.", "mode": "rules"}


def _generate_missing_recommendation(original_findings):
    for f in original_findings or []:
        if f.get("issue_type") == "missing_values" and f.get("has_issue"):
            details = f.get("details", {})
            parts = ["For missing value handling, I recommend:"]
            for col, d in details.get("columns", {}).items():
                if d.get("missing_count", 0) > 0:
                    pct = d.get("missing_percentage", 0)
                    parts.append(f"\nColumn '{col}' ({d['missing_count']} missing, {pct:.1f}%):")
                    if pct > 50:
                        parts.append("  • Missing rate > 50%: consider dropping the column")
                    else:
                        parts.append("  • If missing at random: impute with median (numerical) or mode/'Unknown' (categorical)")
                        parts.append("  • If missing not at random: investigate source, consider adding 'was_missing' indicator")
            return {"answer": "".join(parts), "mode": "rules"}
    return {"answer": "No missing value data available for recommendation.", "mode": "rules"}


def _generate_duplicate_recommendation(original_findings):
    for f in original_findings or []:
        if f.get("issue_type") == "duplicate_rows" and f.get("has_issue"):
            details = f.get("details", {})
            count = f.get("count", 0)
            involved = details.get("duplicate_rows_involved", 0)
            parts = [f"For duplicate handling ({count} occurrences, {involved} rows involved):"]
            parts.append("1. Review duplicate rows to confirm they are not legitimate repeated measurements")
            parts.append("2. Drop exact duplicates, keeping the first occurrence (df.drop_duplicates(keep='first'))")
            parts.append("3. If business key exists (e.g., ID), check for near-duplicates on that key")
            return {"answer": "\n".join(parts), "mode": "rules"}
    return {"answer": "No duplicate data available for recommendation.", "mode": "rules"}


def _generate_inconsistency_recommendation(original_findings):
    for f in original_findings or []:
        if f.get("issue_type") == "categorical_inconsistency" and f.get("has_issue"):
            details = f.get("details", {})
            parts = ["For categorical inconsistency handling:"]
            for col, d in details.items():
                if isinstance(d, dict) and d.get("has_issue"):
                    groups = d.get("inconsistent_groups", {})
                    parts.append(f"\nColumn '{col}':")
                    parts.append("  1. Normalize case (lower/upper) and trim whitespace")
                    parts.append("  2. Map known variants to one canonical value (e.g., 'Credit Card' and 'Credit card' -> 'Credit Card')")
                    parts.append("  3. Recompute category distributions after cleaning")
                    break
            return {"answer": "\n".join(parts), "mode": "rules"}
    return {"answer": "No inconsistency data available for recommendation.", "mode": "rules"}


def _generate_sentinel_recommendation(original_findings):
    for f in original_findings or []:
        if f.get("issue_type") == "possible_sentinel_values" and f.get("has_issue"):
            details = f.get("details", {})
            parts = ["For sentinel/placeholder value handling:"]
            for col, d in details.get("columns", {}).items():
                if d.get("sentinel_count", 0) > 0 or d.get("placeholder_count", 0) > 0:
                    parts.append(f"\nColumn '{col}' ({d.get('sentinel_count', 0)} sentinels, {d.get('placeholder_count', 0)} placeholders):")
                    parts.append("  1. Confirm with data owner what these codes mean (e.g., -9999 = 'not measured')")
                    parts.append("  2. Replace confirmed sentinels with NaN (pd.NA or np.nan) before analysis")
                    parts.append("  3. Exclude sentinel values from charts, statistics, and model training")
            return {"answer": "\n".join(parts), "mode": "rules"}
    return {"answer": "No sentinel data available for recommendation.", "mode": "rules"}


def _generate_invalid_recommendation(original_findings):
    for f in original_findings or []:
        if f.get("issue_type") == "invalid_values" and f.get("has_issue"):
            details = f.get("details", {})
            parts = ["For invalid value handling:"]
            for col, d in details.items():
                if isinstance(d, dict) and d.get("has_issue"):
                    count = d.get("invalid_count", 0)
                    parts.append(f"\nColumn '{col}' ({count} invalid values):")
                    parts.append("  1. Verify the source of the invalid entries")
                    parts.append("  2. Correct them if the true value is known, otherwise treat as missing")
            return {"answer": "\n".join(parts), "mode": "rules"}
    return {"answer": "No invalid value data available for recommendation.", "mode": "rules"}


def _generate_type_validation_recommendation(original_findings):
    for f in original_findings or []:
        if f.get("issue_type") == "data_type_validation" and f.get("has_issue"):
            details = f.get("details", {})
            parts = ["For data type validation:"]
            for col, d in details.get("columns", {}).items():
                if d.get("has_issue"):
                    count = d.get("type_mismatch_count", 0)
                    parts.append(f"\nColumn '{col}' ({count} mismatches):")
                    parts.append("  1. Locate the offending values and correct or remove them")
                    parts.append("  2. Cast the column to its intended type after cleaning (df[col].astype(type))")
            return {"answer": "\n".join(parts), "mode": "rules"}
    return {"answer": "No data type validation data available for recommendation.", "mode": "rules"}


def _generate_quality_recommendation(quality_score):
    qs = (quality_score or {}).get("overall_score", "N/A")
    cats = (quality_score or {}).get("category_scores", {})
    low_cats = [(c, d) for c, d in cats.items() if d.get("severity") in ["HIGH", "MEDIUM"]]
    
    if not low_cats:
        return {"answer": f"Quality score is {qs}. All categories are healthy - no specific remediation needed.", "mode": "rules"}
    
    parts = [f"Quality score: {qs}. To improve it, focus on these categories:"]
    for cat, d in low_cats:
        sev = d.get("severity", "UNKNOWN")
        affected = d.get("affected_rows", "N/A")
        parts.append(f"\n{cat} ({sev} severity, {affected} affected rows):")
        if cat == "missing_values":
            parts.append("  - Impute numerical columns with median, categorical with mode/'Unknown'")
            parts.append("  - Or drop columns/rows with very high missing rates (>50%)")
        elif cat == "duplicate_rows":
            parts.append("  - Review and drop exact duplicate rows")
        elif cat == "categorical_inconsistency":
            parts.append("  - Normalize case and trim whitespace in affected columns")
            parts.append("  - Map variant spellings to canonical values")
        elif cat == "statistical_outliers":
            parts.append("  - Investigate outliers: keep genuine rare events, correct errors")
            parts.append("  - Consider capping or robust scaling for modeling")
        elif cat == "possible_sentinel_values":
            parts.append("  - Replace sentinel values (e.g., -9999, N/A) with proper NaN")
        elif cat == "invalid_values":
            parts.append("  - Verify source of invalid entries, correct or treat as missing")
        elif cat == "data_type_validation":
            parts.append("  - Locate and fix type mismatches, then cast to correct type")
    
    return {"answer": "".join(parts), "mode": "rules"}


def _generate_ml_recommendation(ml_readiness):
    ml_pct = (ml_readiness or {}).get("readiness_percentage", 0)
    checks = (ml_readiness or {}).get("checks", [])
    failed = [c for c in checks if not c.get("passed")]
    
    if not failed:
        return {"answer": f"ML readiness: {ml_pct}%. All checks passed - dataset is ready for ML!", "mode": "rules"}
    
    parts = [f"ML readiness: {ml_pct}%. Failed checks to fix:"]
    for c in failed:
        check_name = c.get("check", "unknown")
        count = c.get("count", "N/A")
        parts.append(f"\n{check_name} (FAIL, count: {count}):")
        if check_name == "missing_values":
            parts.append("  - Impute or drop missing values before training")
        elif check_name == "sentinel_values":
            parts.append("  - Replace sentinel values (-9999, N/A) with NaN")
        elif check_name == "constant_columns":
            parts.append("  - Remove constant columns that provide no information")
        elif check_name == "numerical_features":
            parts.append("  - Ensure at least one numerical feature exists")
        elif check_name == "identifier_columns":
            parts.append("  - Review identifier columns - they're flagged but may be OK")
    
    return {"answer": "".join(parts), "mode": "rules"}


def _generate_outlier_simple_explanation(original_findings):
    for f in original_findings or []:
        if f.get("issue_type") == "statistical_outliers" and f.get("has_issue"):
            details = f.get("details", {})
            parts = ["Outliers are values that are unusually far from the rest of the data."]
            for col, d in details.items():
                if isinstance(d, dict) and d.get("has_issue"):
                    outlier_count = d.get("outlier_count", 0)
                    parts.append(f"\nIn column '{col}', {outlier_count} outliers were found.")
                    outlier_values = d.get("outlier_values", [])
                    if outlier_values:
                        vals_str = ", ".join(str(v) for v in outlier_values[:5])
                        parts.append(f"  Values: {vals_str}")
            parts.append("\nWhat to do: Check if these are real extreme values (keep them) or mistakes (fix/remove them).")
            return {"answer": "".join(parts), "mode": "rules"}
    return {"answer": "No outlier details available for simple explanation.", "mode": "rules"}


def _generate_missing_simple_explanation(original_findings):
    for f in original_findings or []:
        if f.get("issue_type") == "missing_values" and f.get("has_issue"):
            details = f.get("details", {})
            parts = ["Missing values are empty cells where data should be."]
            for col, d in details.get("columns", {}).items():
                if d.get("missing_count", 0) > 0:
                    pct = d.get("missing_percentage", 0)
                    parts.append(f"\nColumn '{col}': {d['missing_count']} missing ({pct:.1f}%)")
            parts.append("\nWhat to do: Fill in (impute) the missing values or remove rows/columns with too many missing.")
            return {"answer": "".join(parts), "mode": "rules"}
    return {"answer": "No missing value details available.", "mode": "rules"}


def _generate_duplicate_simple_explanation(original_findings):
    for f in original_findings or []:
        if f.get("issue_type") == "duplicate_rows" and f.get("has_issue"):
            details = f.get("details", {})
            count = f.get("count", 0)
            involved = details.get("duplicate_rows_involved", 0)
            parts = [f"Duplicate rows are exact copies of other rows. Found {count} duplicates involving {involved} rows."]
            parts.append("\nWhat to do: Remove the extra copies, keeping just one of each duplicate.")
            return {"answer": "".join(parts), "mode": "rules"}
    return {"answer": "No duplicate details available.", "mode": "rules"}


def _generate_inconsistency_simple_explanation(original_findings):
    for f in original_findings or []:
        if f.get("issue_type") == "categorical_inconsistency" and f.get("has_issue"):
            details = f.get("details", {})
            parts = ["Categorical inconsistencies are the same category written differently (e.g., 'NY' vs 'New York')."]
            for col, d in details.items():
                if isinstance(d, dict) and d.get("has_issue"):
                    parts.append(f"\nColumn '{col}' has inconsistent spellings.")
            parts.append("\nWhat to do: Standardize the text (same case, trim spaces) and merge similar values.")
            return {"answer": "".join(parts), "mode": "rules"}
    return {"answer": "No inconsistency details available.", "mode": "rules"}


def _generate_sentinel_simple_explanation(original_findings):
    for f in original_findings or []:
        if f.get("issue_type") == "possible_sentinel_values" and f.get("has_issue"):
            details = f.get("details", {})
            parts = ["Sentinel values are fake numbers used to represent missing data (e.g., -9999 for 'not measured')."]
            for col, d in details.get("columns", {}).items():
                if d.get("sentinel_count", 0) > 0 or d.get("placeholder_count", 0) > 0:
                    parts.append(f"\nColumn '{col}': {d.get('sentinel_count', 0)} sentinels, {d.get('placeholder_count', 0)} placeholders")
            parts.append("\nWhat to do: Replace these with proper missing values (NaN) so they don't corrupt your analysis.")
            return {"answer": "".join(parts), "mode": "rules"}
    return {"answer": "No sentinel details available.", "mode": "rules"}


def _generate_invalid_simple_explanation(original_findings):
    for f in original_findings or []:
        if f.get("issue_type") == "invalid_values" and f.get("has_issue"):
            details = f.get("details", {})
            parts = ["Invalid values break the expected rules for a column (e.g., negative age)."]
            for col, d in details.items():
                if isinstance(d, dict) and d.get("has_issue"):
                    count = d.get("invalid_count", 0)
                    parts.append(f"\nColumn '{col}': {count} invalid values.")
            parts.append("\nWhat to do: Fix them if you know the correct value, otherwise treat as missing.")
            return {"answer": "".join(parts), "mode": "rules"}
    return {"answer": "No invalid value details available.", "mode": "rules"}


def _generate_type_validation_simple_explanation(original_findings):
    for f in original_findings or []:
        if f.get("issue_type") == "data_type_validation" and f.get("has_issue"):
            details = f.get("details", {})
            parts = ["Data type issues mean some values don't match the expected type (e.g., text in a number column)."]
            for col, d in details.get("columns", {}).items():
                if d.get("has_issue"):
                    count = d.get("type_mismatch_count", 0)
                    parts.append(f"\nColumn '{col}': {count} mismatches.")
            parts.append("\nWhat to do: Fix the wrong values, then convert the column to the correct type.")
            return {"answer": "".join(parts), "mode": "rules"}
    return {"answer": "No data type details available.", "mode": "rules"}


def _generate_general_recommendation(insights):
    if not insights:
        return {"answer": "No issues detected - no remediation needed.", "mode": "rules"}
    
    parts = ["Recommended remediation priorities:"]
    for i in insights[:3]:
        parts.append(f"\n{i['title']} ({i['severity']}): {i['recommended_actions'][0] if i['recommended_actions'] else 'Investigate and clean'}")
    return {"answer": "".join(parts), "mode": "rules"}


# Rule-based fallback with conversation history
def _rule_based_answer(question, facts, quality_score, ml_readiness, profile,
                       conversation_history=None, resolution=None):
    """Deterministic, evidence-first answer generation.

    This is the single authoritative rule path. It consumes the ``Resolution``
    produced by ``chat_resolver.resolve`` (PHASE 10 / 26):

    * The resolved intent/topic selects WHICH canonical evidence is used.
    * There is no "return the most important finding" fallback. A question
      whose intent cannot be answered from the evidence asks for
      clarification instead of guessing.
    * The current question always wins over prior context; context is only
      consulted for genuinely elliptical follow-ups (already handled upstream
      by the resolver).
    """
    q = (question or "").lower()
    insights = facts["insights"]
    original = facts.get("_original_findings", [])
    available = {i["issue_type"] for i in insights}

    if resolution is None:
        resolution = resolve_chat(question, conversation_history, available)

    def find_insight(issue_type):
        for i in insights:
            if i["issue_type"] == issue_type:
                return i
        return None

    topic = resolution.topic
    action = resolution.action

    # --------------------------------------------------------
    # Clarification (PHASE 6 / 20 / 26): never invent an answer.
    # --------------------------------------------------------
    if resolution.needs_clarification or action == "clarify":
        opts = resolution.clarification_options or sorted(available)
        return {"answer": _disambiguation_text(opts), "mode": "rules"}

    # --------------------------------------------------------
    # "Give me more" continues the PREVIOUS topic with NEW
    # information -- never the previous response again. Route it
    # through the dedicated continuations in _generate_more_for_topic
    # (feature importance, quality score, ML readiness, column
    # information, per-issue details, problems, priority, summary,
    # model selection). Project ideas are the exception: their own
    # branch below excludes already-shown ideas. Topics without a
    # dedicated continuation fall through to normal generation, and
    # _avoid_expanded_repeat guarantees the previous answer is then
    # never returned unchanged.
    # --------------------------------------------------------
    if action == "expand" and topic != "project_ideas":
        more = _generate_more_for_topic(topic, original, quality_score,
                                        ml_readiness, profile, insights)
        if more:
            return more

    # --------------------------------------------------------
    # Project ideas / comparison / implementation (PHASE 16)
    # --------------------------------------------------------
    if topic == "project_ideas":
        if action == "compare":
            return _generate_project_comparison(profile, insights)
        if action == "implement":
            return _generate_project_implementation(profile, insights)
        # "Give me more" should not repeat the ideas already shown.
        exclude = None
        if action == "expand":
            for m in reversed(conversation_history or []):
                is_assistant = (m.get("role") == "assistant") if isinstance(m, dict) \
                    else (getattr(m, "role", "") == "assistant")
                content = (m.get("content", "") if isinstance(m, dict)
                           else getattr(m, "content", ""))
                if is_assistant and content:
                    exclude = content
                    break
        return _generate_more_project_ideas(profile, insights, exclude_text=exclude)

    # --------------------------------------------------------
    # Column / role information (PHASE 17)
    # --------------------------------------------------------
    if topic == "column_information" or topic == "column_role_information":
        return _generate_column_information(profile,
                                            by_role=(topic == "column_role_information"),
                                            filter_tokens=_column_filter_matchers(q))

    # --------------------------------------------------------
    # Quality score (PHASE 12 fast path)
    # --------------------------------------------------------
    if topic == "quality_score":
        if action == "remediate":
            return _generate_quality_recommendation(quality_score)
        return _generate_quality_score_answer(quality_score)

    # --------------------------------------------------------
    # ML readiness (PHASE 8: distinct from ML projects)
    # --------------------------------------------------------
    if topic == "ml_readiness":
        if action == "remediate":
            return _generate_ml_recommendation(ml_readiness)
        return _generate_ml_readiness_answer(ml_readiness)

    # --------------------------------------------------------
    # Feature importance: never invented (PHASE 18)
    # --------------------------------------------------------
    if topic == "feature_importance":
        return _generate_feature_importance_answer(profile)

    # --------------------------------------------------------
    # Model selection (PHASE 7 / 8)
    # --------------------------------------------------------
    if topic == "model_selection":
        return _generate_model_selection_answer(profile)

    # --------------------------------------------------------
    # Multi-issue remediation (PHASE 9 / 19): address every issue
    # actually detected. Never arbitrarily pick one.
    # --------------------------------------------------------
    if topic is None and action == "remediate":
        return _generate_all_remediation(insights)

    # --------------------------------------------------------
    # Problems / priority / overview (PHASE 12 fast paths)
    # --------------------------------------------------------
    if topic == "problems":
        # A remediation action on the problems topic ("how can I fix all
        # of them?") must give remediation guidance, never re-list the
        # problem report.
        if action == "remediate":
            return _generate_all_remediation(insights)
        if not insights:
            return {"answer": "No issues were detected in this dataset.", "mode": "rules"}
        lines = []
        for i in insights:
            cols = ", ".join(i["affected_columns"]) or "dataset"
            lines.append(f"- {i['title']} ({i['severity']}): {i['count']} occurrence(s) in {cols}. {i['what_it_means']}")
        return {"answer": "Problems detected:\n" + "\n".join(lines), "mode": "rules"}

    if topic == "priority":
        if not insights:
            return {"answer": "No issues were detected, so there is nothing to fix first.", "mode": "rules"}
        top = insights[0]
        cols = ", ".join(top["affected_columns"]) or "dataset"
        action0 = top["recommended_actions"][0] if top["recommended_actions"] else "Investigate and clean."
        return {"answer": f"Fix this first: {top['title']} ({top['severity']}, {top['count']} occurrences in {cols}). "
                          f"{top['what_it_means']} Recommended: {action0}", "mode": "rules"}

    if topic == "dataset_summary":
        return _generate_dataset_summary(profile, quality_score, ml_readiness, insights)

    # --------------------------------------------------------
    # Single-issue analysis or remediation (PHASE 9 / 10)
    # --------------------------------------------------------
    if topic in ISSUE_TOPICS:
        # build_insights only emits findings where has_issue is true, so a
        # non-None insight already proves the issue was detected. Do not rely
        # on has_issue being copied into the insight dict (it is not).
        insight = find_insight(topic)
        has = insight is not None

        if action == "expand":
            more = _generate_more_for_topic(topic, original, quality_score, ml_readiness, profile, insights)
            if more:
                return more

        if action == "remediate":
            if not has:
                return {"answer": f"No {topic.replace('_', ' ')} were detected in this dataset, "
                                  f"so there is nothing to remediate.", "mode": "rules"}
            return _generate_remediation_for_topic(topic, original, quality_score, ml_readiness, insights)

        # action == report (analysis)
        if not has:
            return {"answer": _no_issue_message(topic), "mode": "rules"}
        return _generate_analysis_for_topic(topic, insight, original)

    # --------------------------------------------------------
    # General: summarize the dataset. No arbitrary single finding.
    # --------------------------------------------------------
    return _generate_dataset_summary(profile, quality_score, ml_readiness, insights)


# ============================================================
# Topic -> evidence generators (Phase 10: one mapping, one source)
# ============================================================

_NO_ISSUE = {
    "missing_values": "No missing values were detected in this dataset.",
    "duplicate_rows": "No duplicate rows were detected in this dataset.",
    "categorical_inconsistency": "No categorical inconsistencies were detected in this dataset.",
    "data_type_validation": "No data-type problems were detected in this dataset.",
    "invalid_values": "No invalid values were detected in this dataset.",
    "statistical_outliers": "No statistical outliers were detected in this dataset.",
    "possible_sentinel_values": "No sentinel or placeholder values were detected in this dataset.",
    "ml_anomalies": "No ML-detected anomalies were detected in this dataset.",
}


def _no_issue_message(topic):
    return _NO_ISSUE.get(topic, f"No {topic.replace('_', ' ')} were detected in this dataset.")


def _generate_analysis_for_topic(topic, insight, original):
    """Report the canonical evidence for one issue topic."""
    parts = [f"{insight['title']}: {insight['count']} occurrence(s)"]
    if insight.get("affected_columns"):
        parts.append("affecting " + ", ".join(insight["affected_columns"]))
    parts.append(insight.get("what_it_means", ""))
    if insight.get("recommended_actions"):
        parts.append("Recommended: " + "; ".join(insight["recommended_actions"][:2]))
    body = ". ".join(p for p in parts if p) + "."

    # Append concrete details where the engine provides them.
    detail = _detail_lines_for_topic(topic, original)
    if detail:
        body += "\n\n" + "\n".join(detail)
    return {"answer": body, "mode": "rules"}


def _fmt_numeric(v):
    """Format a retained evidence value compactly for answer text."""
    try:
        return f"{float(v):g}"
    except (TypeError, ValueError):
        return str(v)


def _detail_lines_for_topic(topic, original):
    out = []
    for f in original or []:
        if f.get("issue_type") != topic or not f.get("has_issue"):
            continue
        details = f.get("details", {})
        if topic == "missing_values":
            for col, d in (details.get("columns") or {}).items():
                if d.get("missing_count", 0) > 0:
                    out.append(f"- {col}: {d['missing_count']} missing ({d.get('missing_percentage', 0):.1f}%)")
        elif topic == "statistical_outliers":
            for col, d in details.items():
                if isinstance(d, dict) and d.get("has_issue"):
                    lo, hi = d.get("lower_bound"), d.get("upper_bound")
                    line = f"- {col}: {d.get('outlier_count', 0)} outliers"
                    if lo is not None and hi is not None:
                        line += f" (values beyond IQR bounds {lo:.2f} to {hi:.2f})"
                    vals = d.get("outlier_values") or []
                    if vals:
                        rep = ", ".join(_fmt_numeric(v) for v in vals[:5])
                        line += f"; retained representative values: {rep}"
                    idxs = d.get("outlier_rows") or []
                    if idxs:
                        shown = ", ".join(str(x) for x in idxs[:10])
                        line += f"; row indices (first 10 of {len(idxs)}): {shown}"
                    out.append(line)
            if out:
                out.append("Note: these are statistical outliers (values outside the "
                           "1.5xIQR fences) -- a distinct category from invalid values, "
                           "sentinel/placeholder codes, and ML anomalies, which the "
                           "analysis detects and reports separately.")
        elif topic == "possible_sentinel_values":
            # The engine reports two distinct numbers per column:
            #   sentinel_count  = number of DISTINCT sentinel values found
            #   sentinel_occurrences = number of ROWS affected
            # Surface both so the answer is grounded and unambiguous.
            for col, d in (details.get("columns") or {}).items():
                occ = d.get("sentinel_occurrences", 0)
                sc = d.get("sentinel_count", 0)
                pocc = d.get("placeholder_occurrences", 0)
                pc = d.get("placeholder_count", 0)
                if occ or sc or pocc or pc:
                    parts = []
                    if occ:
                        parts.append(f"{occ} affected row(s)")
                    if sc:
                        parts.append(f"{sc} distinct sentinel value(s)")
                    if pocc:
                        parts.append(f"{pocc} placeholder row(s)")
                    if pc:
                        parts.append(f"{pc} distinct placeholder(s)")
                    out.append(f"- {col}: " + ", ".join(parts))
            # Also surface the finding-level totals when present.
            total_s = details.get("total_sentinels")
            total_p = details.get("total_placeholders")
            if total_s or total_p:
                out.append(f"- Total: {total_s or 0} sentinel(s), {total_p or 0} placeholder(s)")
        elif topic == "categorical_inconsistency":
            for col, d in details.items():
                if isinstance(d, dict) and d.get("has_issue"):
                    groups = d.get("inconsistent_groups", {})
                    for canon, variants in list(groups.items())[:3]:
                        out.append(f"- {col}: '{canon}' also written as {variants}")
        elif topic == "duplicate_rows":
            count = f.get("count", 0)
            involved = details.get("duplicate_rows_involved", 0)
            out.append(f"- {count} duplicate occurrence(s), {involved} row(s) involved")
        elif topic == "invalid_values":
            for col, d in details.items():
                if isinstance(d, dict) and d.get("has_issue"):
                    out.append(f"- {col}: {d.get('invalid_count', 0)} invalid value(s)")
        elif topic == "data_type_validation":
            for col, d in (details.get("columns") or {}).items():
                if d.get("has_issue"):
                    out.append(f"- {col}: {d.get('type_mismatch_count', 0)} type mismatch(es)")
    return out


def _generate_remediation_for_topic(topic, original, quality_score, ml_readiness, insights):
    """Remediation guidance for a single, confirmed issue topic."""
    mapping = {
        "missing_values": _generate_missing_remediation,
        "duplicate_rows": _generate_duplicate_remediation,
        "categorical_inconsistency": _generate_inconsistency_remediation,
        "data_type_validation": _generate_type_validation_remediation,
        "invalid_values": _generate_invalid_remediation,
        "statistical_outliers": _generate_outlier_remediation,
        "possible_sentinel_values": _generate_sentinel_remediation,
    }
    fn = mapping.get(topic)
    if fn:
        return fn(original)
    return _generate_all_remediation(insights)


def _generate_more_for_topic(topic, original, quality_score, ml_readiness, profile, insights):
    """'Give me more' continuation for a topic."""
    mapping = {
        "missing_values": lambda: _generate_more_missing_info(original),
        "duplicate_rows": lambda: _generate_more_duplicate_info(original),
        "categorical_inconsistency": lambda: _generate_more_inconsistency_info(original),
        "data_type_validation": lambda: _generate_more_type_info(original),
        "invalid_values": lambda: _generate_more_invalid_info(original),
        "statistical_outliers": lambda: _generate_more_outlier_info(original),
        "possible_sentinel_values": lambda: _generate_more_sentinel_info(original),
        "quality_score": lambda: _generate_more_quality_info(quality_score),
        "ml_readiness": lambda: _generate_more_ml_info(ml_readiness),
        "column_information": lambda: _generate_column_information(profile),
        "project_ideas": lambda: _generate_more_project_ideas(profile, insights),
        "feature_importance": lambda: _generate_feature_importance_more(profile),
        "problems": lambda: _generate_more_problems_info(insights),
        "priority": lambda: _generate_more_priority_info(insights),
        "dataset_summary": lambda: _generate_more_dataset_summary_info(
            profile, quality_score, ml_readiness),
        "model_selection": lambda: _generate_more_model_selection_info(profile),
    }
    fn = mapping.get(topic)
    return fn() if fn else None


# Continuations for "Give me more" on non-issue topics. Each adds
# information the BASE answer does not contain (per-issue actions after
# the problems report, the full ranking after the top priority,
# per-column detail after the summary, the evaluation method after
# model selection) so a continuation never repeats the previous
# response. All content comes from canonical insights/profile data.
def _generate_more_problems_info(insights):
    """The problems report lists title/severity/count/columns/meaning;
    the continuation adds the recommended action for each issue."""
    if not insights:
        return {"answer": "No issues were detected, so there is nothing to add.",
                "mode": "rules"}
    lines = ["Recommended action for each detected issue:", ""]
    for i in insights:
        cols = ", ".join(i["affected_columns"]) or "dataset"
        acts = i.get("recommended_actions") or ["Review the affected values."]
        lines.append(f"- {i['title']} ({cols}): {acts[0]}")
    lines.append("")
    lines.append("Ask \"How can I fix all of them?\" for the full remediation "
                 "method covering every issue at once.")
    return {"answer": "\n".join(lines), "mode": "rules"}


def _generate_more_priority_info(insights):
    """The priority answer gives only the first fix; the continuation
    gives the full severity-ranked list with each first action."""
    if not insights:
        return {"answer": "No issues were detected, so there is nothing to rank.",
                "mode": "rules"}
    lines = ["All detected issues ranked by severity, each with its first "
             "recommended action:", ""]
    for n, i in enumerate(insights, 1):
        cols = ", ".join(i["affected_columns"]) or "dataset"
        act = (i.get("recommended_actions") or ["Review the affected values."])[0]
        lines.append(f"{n}. {i['title']} ({i['severity']}, {i['count']} occurrences "
                     f"in {cols}): {act}")
    lines.append("")
    lines.append("Fix the higher rows first. Ask 'What problems does this dataset "
                 "have?' for the plain problem report.")
    return {"answer": "\n".join(lines), "mode": "rules"}


def _generate_more_dataset_summary_info(profile, quality_score, ml_readiness):
    """The summary is aggregate; the continuation gives per-column detail."""
    col_details = _column_view((profile or {}).get("column_details", {}))
    if not col_details:
        return {"answer": "No column details are available to expand on.",
                "mode": "rules"}
    lines = ["Per-column detail (physical type, semantic role, cardinality, "
             "completeness):", ""]
    for col, d in col_details.items():
        lines.append(
            f"- {col}: type={d.get('type')}, role={d.get('role') or 'unassigned'}, "
            f"unique={d.get('unique_count', '?')}, "
            f"missing={d.get('missing_count', 0)} ({d.get('missing_percentage', 0)}%)")
    lines.append("")
    lines.append(
        f"Overall: quality score {(quality_score or {}).get('overall_score', 'unavailable')}, "
        f"ML readiness {(ml_readiness or {}).get('readiness_percentage', 'unavailable')}%.")
    return {"answer": "\n".join(lines), "mode": "rules"}


def _generate_more_model_selection_info(profile):
    """The model-selection answer names candidates; the continuation
    explains HOW to select and evaluate -- without claiming any run."""
    col_details = _column_view((profile or {}).get("column_details", {}))
    target_cols = [c for c, d in col_details.items()
                   if d.get("role") in ("target", "label")]
    lines = [
        "How to choose and evaluate a model (no model has been trained yet):",
        "",
        "1. State the prediction objective: classification when the target is a "
        "category, regression when it is a quantity. Model choice follows from "
        "the target's type, not from the number of columns.",
    ]
    if target_cols:
        lines.append("2. The profile flags " + ", ".join(target_cols) + " as "
                     "target-like; verify it is a genuine label and check for "
                     "leakage before training.")
    else:
        lines.append("2. No target column has been identified, so choose the "
                     "target first; nothing can be trained or compared until then.")
    lines += [
        "3. Start with a simple baseline (logistic/linear regression or a "
        "tree/forest), evaluate with cross-validation, and use a metric that "
        "matches the objective (accuracy/ROC-AUC/F1 for classification, "
        "MAE/RMSE for regression).",
        "4. Only move to complex models if they beat the baseline; check class "
        "imbalance and time-based splits for temporal data.",
        "",
        "No performance or importance values exist until that analysis is "
        "actually performed.",
    ]
    return {"answer": "\n".join(lines), "mode": "rules"}


# Remediation methods per issue type. Generalized (no dataset or column
# names hardcoded): the affected columns are injected from the canonical
# insights at answer time. Each method distinguishes investigation,
# correction, imputation, retention, and deletion where appropriate --
# deletion is never the default recommendation.
_REMEDIATION_METHODS = {
    "missing_values": lambda cols: [
        f"Check whether the gaps in {cols} are random or systematic (missingness pattern per column).",
        "If plausibly random: impute numerical columns with the median and categorical columns with the mode or an explicit 'Unknown' category.",
        "If not random: investigate the capture process and add a 'was_missing' indicator column instead of silently imputing.",
        "Only drop rows or columns when the missing fraction is too high to impute credibly (commonly >50%) or the field is unusable.",
    ],
    "duplicate_rows": lambda cols: [
        "Review samples of the duplicated rows to confirm they are true duplicates, not legitimate repeated records (e.g. line items sharing attribute values).",
        "If true duplicates: drop them with df.drop_duplicates(keep='first'), or keep='last' when the newest record is authoritative.",
        "If a business key exists (order/invoice/customer identifier), deduplicate on that key with explicit rules instead of on the whole row.",
        "If the repeats are legitimate: retain them and, if useful, derive a repeat_count aggregate instead of deleting rows.",
    ],
    "categorical_inconsistency": lambda cols: [
        f"Normalize formatting first in {cols}: trim whitespace and standardize case.",
        "Map known variants to one canonical label and keep the mapping documented so the change is reversible.",
        "Recompute the category distribution after cleaning and compare it against business expectations.",
        "Add allowed-value validation at ingestion so variants cannot reappear.",
    ],
    "data_type_validation": lambda cols: [
        f"Locate the offending values in {cols} and correct them when the intended value is known.",
        "If a value cannot be corrected, treat it as missing rather than forcing a cast.",
        "After cleaning, cast the column to its intended type (df[col].astype(...)) and verify how many values converted.",
    ],
    "invalid_values": lambda cols: [
        f"Investigate first: confirm the flagged values in {cols} against the business rules and the source system. In transactional data, negative quantities or zero prices can be legitimate events (returns, cancellations, free or promotional items).",
        "If they are errors and the true value is known: correct them at the source and in the dataset.",
        "If they are errors and the true value is unknown: treat them as missing, then impute or flag accordingly.",
        "If they are legitimate business events: retain them and relax the validation rule that assumes all values are positive.",
    ],
    "statistical_outliers": lambda cols: [
        f"Inspect the outlier rows in {cols} (row indices are retained in the analysis) to separate genuine rare events from data-entry errors.",
        "If genuine: retain them; for modeling use robust scaling or capping (winsorization at the IQR bounds) rather than deletion.",
        "If errors: correct the values, or replace them with missing and impute.",
        "Do not delete outliers solely because they are extreme -- extreme records are often the most informative.",
    ],
    "possible_sentinel_values": lambda cols: [
        f"Confirm with the data owner what the sentinel codes in {cols} mean (e.g. -999 or 'NA' used for 'not recorded').",
        "Replace confirmed sentinels with proper missing values (NaN/None) before computing statistics or training models.",
        "Keep the raw values in an archived copy, and exclude sentinels from charts, aggregates, and model inputs.",
    ],
    "ml_anomalies": lambda cols: [
        f"Review the {len(cols.split(', ')) if cols else 'flagged'} record group(s) as a prioritized investigation queue -- anomaly scores are unsupervised signals, not confirmed errors.",
        "Corroborate each flagged record with domain rules before acting; retain legitimate anomalies.",
    ],
}


def _generate_all_remediation(insights):
    """Remediate EVERY detected issue. Never pick one arbitrarily (PHASE 9/19).

    Gives a separate, actionable method per issue type, grounded in the
    canonical insights (actual affected columns and counts). Deletion,
    correction, imputation, retention, and investigation are distinguished
    per issue -- rows are never blindly recommended for deletion.
    """
    # build_insights only emits findings where has_issue is true, so every
    # insight present here IS a detected issue. Do not re-filter on
    # has_issue (it is not copied into the insight dict).
    detected = list(insights)
    if not detected:
        return {"answer": "No data-quality issues were detected, so there is nothing to remediate.",
                "mode": "rules"}
    lines = [f"Here is a remediation method for each of the {len(detected)} "
             "issue type(s) reported in this dataset:", ""]
    for i in detected:
        issue = i.get("issue_type", "")
        cols = ", ".join(i.get("affected_columns") or []) or "the affected columns"
        lines.append(f"{i['title']} ({i.get('count', 0)} occurrence(s); columns: {cols}):")
        builder = _REMEDIATION_METHODS.get(issue)
        if builder:
            for step in builder(cols):
                lines.append(f"  - {step}")
        else:
            for a in (i.get("recommended_actions") or ["Review the affected values."])[:3]:
                lines.append(f"  - {a}")
        lines.append("")
    lines.append("Before deleting anything, validate the business meaning of the "
                 "flagged values: in transactional data some flagged values are "
                 "legitimate events (e.g. negative quantities from returns or "
                 "cancellations, zero prices for free items). Prefer correction or "
                 "imputation when the true value is known, flagging or retention "
                 "with an indicator when it is not, and reserve deletion for "
                 "confirmed errors and true duplicates.")
    return {"answer": "\n".join(lines).strip(), "mode": "rules"}


def _generate_dataset_summary(profile, quality_score, ml_readiness, insights):
    rows = (profile or {}).get("rows", "unknown")
    cols = (profile or {}).get("columns", "unknown")
    qs = (quality_score or {}).get("overall_score", "N/A")
    ml = (ml_readiness or {}).get("readiness_percentage", "N/A")
    detected = [i["title"].lower() for i in insights]
    if detected:
        issue_part = f"{len(detected)} issue type(s) detected: {', '.join(detected)}."
    else:
        issue_part = "No data-quality issues were detected."
    return {"answer": f"This dataset has {rows} rows and {cols} columns. "
                      f"Quality score: {qs}. ML readiness: {ml}%. {issue_part}",
            "mode": "rules"}


def _generate_quality_score_answer(quality_score):
    qs = (quality_score or {}).get("overall_score", "N/A")
    cats = (quality_score or {}).get("category_scores", {})
    low = [c for c, d in cats.items() if d.get("severity") in ("HIGH", "MEDIUM")]
    if low:
        return {"answer": f"Quality score: {qs}. Categories dragging it down: {', '.join(low)}.",
                "mode": "rules"}
    return {"answer": f"Quality score: {qs}. All categories are healthy.", "mode": "rules"}


def _generate_ml_readiness_answer(ml_readiness):
    pct = (ml_readiness or {}).get("readiness_percentage", 0)
    checks = (ml_readiness or {}).get("checks", [])
    failed = [c.get("check") for c in checks if not c.get("passed")]
    if failed:
        return {"answer": f"ML readiness: {pct}%. Failing checks: {', '.join(failed)}. "
                          f"Fix these before training a model.", "mode": "rules"}
    return {"answer": f"ML readiness: {pct}%. All checks passed; the dataset is ready for ML.",
            "mode": "rules"}


def _column_filter_matchers(question):
    """Map type/role words in the question to type/role matchers.

    Generalized: only well-known type/role vocabularies are matched, never
    dataset names, column names, or exact question strings.
    """
    q = (question or "").lower()
    matchers = []
    pairs = [
        (("categorical", "category", "categoricals"), "categor"),
        (("numerical", "numeric", "numbers", "number", "continuous"), "numer"),
        (("identifier", "ids"), "identif"),
        (("target", "label", "labels", "outcome", "dependent"), "target"),
        (("label", "labels"), "label"),
        (("datetime", "date", "time", "timestamps"), "date"),
        (("boolean", "bool", "binary", "flag"), "bool"),
        (("feature", "features", "predictor", "predictors", "independent"), "feature"),
        (("text", "free", "string"), "text"),
        (("object", "objects"), "object"),
    ]
    for needles, matcher in pairs:
        if any(n in q for n in needles) and matcher not in matchers:
            matchers.append(matcher)
    return matchers


def _generate_column_information(profile, by_role=False, filter_tokens=None):
    col_details = _column_view((profile or {}).get("column_details", {}))
    if not col_details:
        return {"answer": "No column details are available in the dataset profile.", "mode": "rules"}

    # Narrow the listing when the question asks about a specific kind of
    # column ("which ones are categorical?"). Fall back to all columns when
    # nothing matches so the user is never shown an empty answer.
    if filter_tokens:
        matches = {
            c: d for c, d in col_details.items()
            if any(t in (str(d.get("type", "")) + " " + str(d.get("role", ""))).lower()
                   for t in filter_tokens)
        }
        if matches:
            col_details = matches
            scope = "Matching columns:"
        else:
            scope = "Columns:"
    else:
        scope = "Columns:"

    rows = (profile or {}).get("rows", "unknown")
    cols = (profile or {}).get("columns", len(col_details))
    parts = [f"Dataset has {cols} columns and {rows} rows.", "", scope]
    for col, d in col_details.items():
        ctype = d.get("type", "unknown")
        role = d.get("role", "unknown")
        line = f"- {col}: type={ctype}, role={role}"
        if d.get("missing_count", 0) > 0:
            line += f", missing={d['missing_count']} ({d.get('missing_percentage', 0):.1f}%)"
        if d.get("unique_count") is not None:
            line += f", unique={d['unique_count']}"
        parts.append(line)
    return {"answer": "\n".join(parts), "mode": "rules"}


def _generate_feature_importance_answer(profile):
    """PHASE 18: never invent feature importance."""
    col_details = _column_view((profile or {}).get("column_details", {}))
    target_cols = [c for c, d in col_details.items() if d.get("role") in ("target", "label")]
    msg = ("The current analysis does not calculate model-based feature importance. "
           "To measure which features influence the target, run a model-based "
           "importance analysis (permutation importance, SHAP, or tree-based importance).")
    if target_cols:
        msg += f" Candidate target column(s): {', '.join(target_cols)}."
    else:
        msg += " No target column has been identified yet."
    return {"answer": msg, "mode": "rules"}


def _generate_feature_importance_more(profile):
    """'Give me more' after feature importance: HOW to choose a target and
    CALCULATE importance -- never implying either already happened."""
    col_details = _column_view((profile or {}).get("column_details", {}))
    target_cols = [c for c, d in col_details.items()
                   if d.get("role") in ("target", "label")]
    numeric = [c for c, d in col_details.items()
               if d.get("type") in ("numerical", "integer", "float")]
    categorical = [c for c, d in col_details.items()
                   if d.get("type") in ("categorical", "object", "string")]
    lines = [
        "How to obtain feature importance (nothing has been calculated yet):",
        "",
        "1. Choose a prediction objective first. Start from the business "
        "question -- what should the model predict? -- and map it to ONE column:",
        "   - Predicting a category (churn, default, class) -> classification "
        "target; predicting a quantity (amount, count, price) -> regression target.",
        "   - Pick a column recorded after the predictors, with no leakage from "
        "the outcome, and enough populated rows to learn from.",
    ]
    if target_cols:
        lines.append("   - The profile flags " + ", ".join(target_cols) +
                     " as target-like; verify that label before using it.")
    else:
        lines.append("   - No target column has been identified in the current "
                     "analysis, so this choice is yours to make. Available "
                     f"predictors: {len(numeric)} numerical and "
                     f"{len(categorical)} categorical columns.")
    lines += [
        "",
        "2. Fit a simple baseline model on that target: split features/target, "
        "impute or flag missing values, encode categoricals, then train with "
        "cross-validation (logistic/linear regression or a tree/forest) so the "
        "estimate is honest.",
        "",
        "3. Measure importance from the fitted model:",
        "   - Permutation importance on held-out data: shuffle one feature and "
        "measure the drop in the score; the bigger the drop, the more important.",
        "   - Or SHAP values for per-row contributions, or tree-based (gain) "
        "importance for tree models.",
        "   - Check that the ranking is stable across cross-validation folds "
        "before trusting it.",
        "",
        "4. Interpret with care: importance is model-dependent and "
        "correlation-aware -- it is not proof of causation.",
        "",
        "Until those steps are actually performed, no model has been trained and "
        "no importance values exist; I cannot report rankings, percentages, or "
        "correlations that were never computed.",
    ]
    return {"answer": "\n".join(lines), "mode": "rules"}


def _generate_model_selection_answer(profile):
    col_details = _column_view((profile or {}).get("column_details", {}))
    target_cols = [c for c, d in col_details.items() if d.get("role") in ("target", "label")]
    numeric = [c for c, d in col_details.items() if d.get("type") in ("numerical", "integer", "float")]
    categorical = [c for c, d in col_details.items() if d.get("type") in ("categorical", "object", "string")]
    parts = ["Model choice depends on the target column's type:"]
    if target_cols:
        t = target_cols[0]
        ttype = col_details.get(t, {}).get("type", "unknown")
        parts.append(f"- Target '{t}' appears {ttype}.")
        if ttype in ("categorical", "object", "string"):
            parts.append("  For a categorical target, start with logistic regression or a "
                         "tree/forest classifier.")
        elif ttype in ("numerical", "integer", "float"):
            parts.append("  For a continuous target, start with linear regression or a "
                         "tree/forest regressor.")
    else:
        parts.append("- No target column has been identified, so pick one before selecting a model.")
    parts.append(f"- Available features: {len(numeric)} numerical, {len(categorical)} categorical.")
    return {"answer": "\n".join(parts), "mode": "rules"}


def _generate_project_comparison(profile, insights):
    ideas = _build_project_ideas(profile)
    if len(ideas) < 2:
        return _generate_more_project_ideas(profile, insights)
    lines = ["Comparing the project directions for this dataset:", ""]
    for idx, idea in enumerate(ideas, 1):
        lines.append(f"{idx}. {idea}")
    lines.append("")
    lines.append("The best one depends on your goal: choose a prediction (classification/"
                 "regression) project if you have a clear target column, or an anomaly/"
                 "segmentation project if you do not.")
    return {"answer": "\n".join(lines), "mode": "rules"}


def _generate_project_implementation(profile, insights):
    ideas = _build_project_ideas(profile)
    chosen = ideas[0] if ideas else "an exploratory analysis project"
    lines = [f"Implementing this project: {chosen}", "",
             "1. Explore the data: distributions, missing values, correlations.",
             "2. Clean: handle missing values, outliers, and inconsistencies.",
             "3. Prepare features: encode categoricals, scale numerics.",
             "4. Split into train and test sets.",
             "5. Train a simple baseline model, then iterate.",
             "6. Evaluate and refine."]
    return {"answer": "\n".join(lines), "mode": "rules"}


def _build_project_ideas(profile):
    """Dataset-aware project ideas derived only from the canonical profile.

    Thin wrapper over the shared collector so comparison, implementation,
    and "give me more" all use one source of ideas.
    """
    return _collect_project_ideas(profile, limit=4)


def _generate_more_type_info(original):
    lines = _detail_lines_for_topic("data_type_validation", original)
    if not lines:
        return {"answer": "No data-type details are available.", "mode": "rules"}
    return {"answer": "Data-type mismatch details:\n" + "\n".join(lines), "mode": "rules"}


def _generate_more_invalid_info(original):
    lines = _detail_lines_for_topic("invalid_values", original)
    if not lines:
        return {"answer": "No invalid-value details are available.", "mode": "rules"}
    return {"answer": "Invalid-value details:\n" + "\n".join(lines), "mode": "rules"}


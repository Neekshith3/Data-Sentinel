"""
Authoritative conversational resolver for the Data Sentinel chat assistant.

Design principle (the fix for the long-standing "wrong evidence" bug):

    USER MESSAGE -> resolve CONVERSATION -> resolve CURRENT QUESTION
                 -> INTENT + TOPIC + ACTION -> RELEVANT EVIDENCE
                 -> (answer generation) -> RELEVANCE CHECK

The chatbot must never ask "which finding can I return?". It must ask
"what is the user asking, what evidence does that need, and where does
that evidence live in the already-computed canonical analysis?".

This module is the single source of truth for that decision. Both the
LLM path and the deterministic (rules) path in ``assistant.py`` consume
the same ``Resolution`` object, so they can never diverge.

Key rules enforced here:

* CURRENT EXPLICIT INTENT beats PREVIOUS CONTEXT. Context is consulted
  only when the current question is genuinely ambiguous or contains a
  reference (pronoun / ellipsis).
* Topic is tracked from structured per-turn records (what the assistant
  actually *answered about*), never by scanning free-text answer words.
* A question that names an issue directly resolves that issue, regardless
  of what was being discussed.
* Genuinely unresolvable questions are flagged for clarification instead
  of silently falling back to "the most important finding".

No dataset names, columns, counts, or exact question strings are hardcoded.
All matching is generalized and semantic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# ============================================================
# Canonical issue / topic identifiers
# ============================================================

ISSUE_TOPICS = {
    "missing_values",
    "duplicate_rows",
    "categorical_inconsistency",
    "data_type_validation",
    "invalid_values",
    "statistical_outliers",
    "possible_sentinel_values",
    "ml_anomalies",
}

NON_ISSUE_TOPICS = {
    "column_information",
    "column_role_information",
    "quality_score",
    "ml_readiness",
    "project_ideas",
    "dataset_summary",
    "problems",
    "priority",
    "feature_importance",
    "model_selection",
    "general",
}

ALL_TOPICS = ISSUE_TOPICS | NON_ISSUE_TOPICS

# Topic -> remediation-intent id. Used when an action word is present.
REMEDIATION_INTENT = {
    "missing_values": "missing_remediation",
    "duplicate_rows": "duplicate_remediation",
    "categorical_inconsistency": "inconsistency_remediation",
    "data_type_validation": "type_validation_remediation",
    "invalid_values": "invalid_remediation",
    "statistical_outliers": "outlier_remediation",
    "possible_sentinel_values": "sentinel_remediation",
    "quality_score": "quality_remediation",
    "ml_readiness": "ml_readiness_remediation",
}

# Which canonical evidence each topic needs. Consumed by assistant.py to
# build a *targeted* prompt / rule answer rather than dumping everything.
TOPIC_EVIDENCE = {
    "missing_values": "missing_values_finding",
    "duplicate_rows": "duplicate_rows_finding",
    "categorical_inconsistency": "categorical_inconsistency_findings",
    "data_type_validation": "data_type_validation_finding",
    "invalid_values": "invalid_values_finding",
    "statistical_outliers": "outlier_evidence",
    "possible_sentinel_values": "sentinel_evidence",
    "ml_anomalies": "ml_anomalies_finding",
    "column_information": "column_metadata",
    "column_role_information": "column_roles",
    "quality_score": "quality_score",
    "ml_readiness": "ml_readiness",
    "project_ideas": "dataset_profile+columns+roles",
    "model_selection": "dataset_profile+column_types",
    "feature_importance": "feature_importance_or_none",
    "problems": "all_detected_findings",
    "priority": "all_detected_findings_ranked",
    "dataset_summary": "dataset_profile+quality_score",
    "general": "dataset_profile+top_findings",
}


# ============================================================
# Resolution object
# ============================================================

@dataclass
class Resolution:
    """The single authoritative decision for one user turn."""

    intent: str
    topic: str | None = None
    action: str = "report"          # report | remediate | expand | compare | implement | clarify
    evidence: str = "dataset_profile+top_findings"
    uses_context: bool = False      # did prior turns materially shape this resolution?
    needs_clarification: bool = False
    clarification_options: list = field(default_factory=list)
    # Structured record of what this turn was about, appended to history.
    note: str = ""

    @property
    def is_issue_remediation(self) -> bool:
        return self.action == "remediate" and self.topic in ISSUE_TOPICS


# ============================================================
# Prior-turn tracking (structured, not text-guessing)
# ============================================================

@dataclass
class _PriorTurn:
    question: str
    intent: str
    topic: str | None
    action: str
    # Issue types the assistant actually enumerated in its answer, in order.
    mentioned_issues: list


def _history_msgs(conversation_history):
    """Normalize the incoming history to a list of {role, content} dicts."""
    out = []
    for msg in conversation_history or []:
        if isinstance(msg, dict):
            out.append(msg)
        else:  # pydantic model or similar
            out.append({"role": getattr(msg, "role", ""),
                        "content": getattr(msg, "content", "")})
    return out


def _last_assistant_answer(conversation_history):
    for msg in reversed(_history_msgs(conversation_history)):
        if msg.get("role") == "assistant":
            return msg.get("content", "") or ""
    return ""


def _issues_mentioned_in(text):
    """Which issue topics an assistant answer actually enumerated.

    Used only to disambiguate an *elliptical* follow-up like "how do I fix
    them?" when the previous turn listed several issue types. We look for
    explicit issue labels in the answer, which is far more reliable than
    guessing a single topic from the first keyword that happens to appear.
    """
    t = (text or "").lower()
    found = []
    checks = [
        ("missing", "missing_values"),
        ("duplicate", "duplicate_rows"),
        ("inconsist", "categorical_inconsistency"),
        ("categorical inconsist", "categorical_inconsistency"),
        ("data type", "data_type_validation"),
        ("type mismatch", "data_type_validation"),
        ("invalid", "invalid_values"),
        ("outlier", "statistical_outliers"),
        ("sentinel", "possible_sentinel_values"),
        ("placeholder", "possible_sentinel_values"),
        ("anomal", "ml_anomalies"),
    ]
    for needle, topic in checks:
        if needle in t and topic not in found:
            found.append(topic)
    return found


# ============================================================
# Lexicons (generalized; no dataset- or column-specific entries)
# ============================================================

# Explicit issue mentions. These always win over context.
_ISSUE_LEXICON = [
    ("missing value", "missing_values"), ("missing values", "missing_values"),
    ("values are missing", "missing_values"), ("values missing", "missing_values"),
    ("are missing", "missing_values"), ("missing data", "missing_values"),
    ("null value", "missing_values"), ("null values", "missing_values"),
    ("empty cell", "missing_values"), ("empty cells", "missing_values"),
    ("na values", "missing_values"), ("missing?", "missing_values"),
    ("imputation", "missing_values"), ("impute", "missing_values"),
    ("missing", "missing_values"),

    ("duplicate row", "duplicate_rows"), ("duplicate rows", "duplicate_rows"),
    ("duplicate", "duplicate_rows"), ("deduplicate", "duplicate_rows"),

    ("categorical inconsistency", "categorical_inconsistency"),
    ("categorical inconsistencies", "categorical_inconsistency"),
    ("inconsistent label", "categorical_inconsistency"),
    ("inconsistent category", "categorical_inconsistency"),
    ("inconsistency", "categorical_inconsistency"),
    ("inconsistencies", "categorical_inconsistency"),
    ("inconsistent", "categorical_inconsistency"),
    ("inconsistent categor", "categorical_inconsistency"),
    ("variant label", "categorical_inconsistency"),

    ("data type", "data_type_validation"), ("data types", "data_type_validation"),
    ("type mismatch", "data_type_validation"), ("type problem", "data_type_validation"),
    ("wrong type", "data_type_validation"),

    ("invalid value", "invalid_values"), ("invalid values", "invalid_values"),
    ("invalid", "invalid_values"), ("wrong value", "invalid_values"),
    ("bad value", "invalid_values"), ("bad values", "invalid_values"),

    ("outlier", "statistical_outliers"), ("outliers", "statistical_outliers"),
    ("extreme value", "statistical_outliers"), ("extreme values", "statistical_outliers"),
    ("unusual value", "statistical_outliers"), ("unusual values", "statistical_outliers"),
    ("iqr", "statistical_outliers"),

    ("sentinel value", "possible_sentinel_values"),
    ("sentinel values", "possible_sentinel_values"),
    ("sentinel", "possible_sentinel_values"),
    ("placeholder value", "possible_sentinel_values"),
    ("placeholder values", "possible_sentinel_values"),
    ("placeholder", "possible_sentinel_values"),
]

# Non-issue topical mentions (checked after issue mentions so that e.g.
# "which columns have outliers" still resolves to outliers).
_TOPIC_LEXICON = [
    ("quality score", "quality_score"), ("quality", "quality_score"),
    ("score", "quality_score"),
    ("ml readiness", "ml_readiness"), ("ml ready", "ml_readiness"),
    ("ready for ml", "ml_readiness"), ("ready for machine learning", "ml_readiness"),
    ("use for ml", "ml_readiness"), ("use for machine learning", "ml_readiness"),
    ("for ml", "ml_readiness"), ("for machine learning", "ml_readiness"),
    ("readiness", "ml_readiness"),
    ("column", "column_information"), ("columns", "column_information"),
    ("field", "column_information"), ("fields", "column_information"),
    ("attribute", "column_information"), ("attributes", "column_information"),
    ("schema", "column_information"), ("structure", "column_information"),
    ("feature", "column_information"), ("features", "column_information"),
    ("datatype", "column_role_information"), ("data type of", "column_role_information"),
    # Generic "what's wrong" asks map to the problems topic. Placed after
    # "quality score" (checked earlier) so a score question still wins, and
    # after issue mentions (checked separately) so naming an issue still wins.
    ("what are the problems", "problems"), ("what are the issues", "problems"),
    ("what problems", "problems"), ("what issues", "problems"),
    ("problems", "problems"), ("issues", "problems"),
]

# Project / ML-project / model / feature-importance asks. These must be
# recognized *before* the bare "ml" readiness lexicon so that
# "what ML projects can I build?" is NOT read as ML readiness.
_PROJECT_PATTERNS = [
    "what project", "what projects", "what kind of project", "what kind of projects",
    "project idea", "project ideas", "projects idea", "projects ideas",
    "what can we build", "what can i build", "what can you build",
    "projects can we build", "projects can i build", "project can we build",
    "project can i build", "what application", "what applications",
    "what kind of application", "what kind of applications",
    "what app", "what apps", "app can we build", "apps can we build",
    "build a", "build an", "build a system", "build a customer",
    "entry system", "customer entry", "ml project", "ml projects",
    "machine learning project", "machine learning projects",
    "data science project", "data science projects",
    "analytics project", "analytics projects", "predictive project",
    "use case", "use cases",
]

_MODEL_PATTERNS = [
    "which model", "what model", "best model", "model should i", "model to use",
    "choose a model", "select a model", "algorithm should i", "which algorithm",
    "what algorithm", "classifier or regressor",
]

_FEATURE_PATTERNS = [
    "feature importance", "feature influence", "feature impact", "feature ranking",
    "which features", "what features", "which features influence",
    "what features influence", "what features matter", "which features matter",
    "most important feature", "most important features",
    "what influences", "what drives", "what affects", "which variables",
    "influence the target", "influence the most", "predictor importance",
]

# Action words -> remediation intent.
_ACTION_LEXICON = [
    "how to fix", "how do i fix", "how can i fix", "how should i fix",
    "how to clean", "how do i clean", "how can i clean",
    "how to handle", "how do i handle", "how can i handle", "how should i handle",
    "how to solve", "how do i solve", "how can i solve",
    "how to remove", "how do i remove", "how can i remove", "how should i remove",
    "how to treat", "how do i treat", "how can i treat",
    "how to replace", "how do i replace", "how can i replace",
    "how to reduce", "how do i reduce",
    "remediation", "remediate", "fix them", "fix it", "fix these", "fix this",
    "clean them", "clean it", "clean this", "clean these",
    "handle them", "handle it", "handle this",
    "solve them", "solve it", "methods to", "method to", "method should i",
    "methods should i", "what method", "what methods", "which method",
    "which methods", "what should i do", "steps to", "step by step",
    "how do i fix them", "how do i remove them", "how can i remove them",
    "how do i clean them", "how do i handle them",
    "improve the", "make it better", "resolve", "repair",
]

# Context-dependent (elliptical) follow-ups. These alone trigger context use.
_CONTEXT_DEPENDENT_PATTERNS = [
    "give me more", "tell me more", "show me more", "more detail", "more info",
    "more example", "another example", "another one", "another option",
    "which one", "which one is", "which of these", "which of them",
    "what about them", "what about it", "what about those", "what about these",
    "how do i fix it", "how do i fix them", "how to fix it", "how to fix them",
    "how do i remove it", "how do i remove them", "how to remove it",
    "how to remove them", "how do i handle it", "how do i handle them",
    "how do i clean it", "how do i clean them", "how do i solve it",
    "how do i solve them", "how can i fix it", "how can i fix them",
    "how can i remove them", "how can i handle them",
    "can i replace them", "can i replace it", "can i drop them",
    "explain that", "explain this", "explain it", "explain them",
    "explain more", "explain simply", "explain in simple", "in simple terms",
    "simply explain", "can you explain", "could you explain",
    "why is that", "why that", "why those", "why them", "why",
    "what do you mean", "what does that mean", "what does it mean",
    "how does that work", "how does it work",
    "go on", "keep going", "continue", "elaborate", "expand on",
    "expand on that", "tell me about it", "tell me about them",
    "show me the value", "show me the values", "list the value",
    "list the values", "what are the actual value", "what are the value",
    "how would i start", "how to start", "how do i start", "how to begin",
    "get started", "how to implement", "how do i implement",
    "how would i implement", "walk me through", "guide me",
]

# A bare "why"/"how"/"what" with no other content is vague. But "why is the
# quality score low?" contains real content, so only pure tokens are vague.
_VAGUE_STRICT = {"what", "what?", "why", "why?", "how", "how?", "huh", "ok",
                 "okay", "hm", "hmm", "and", "then", ".", "?", "!"}


def _is_vague(q):
    """True only when the question carries no real content to resolve on."""
    t = (q or "").strip().lower()
    if not t:
        return True
    if t in _VAGUE_STRICT:
        return True
    # "what do you mean" and variants with no added content
    if t in ("what do you mean", "what do you mean?", "what does that mean",
             "what does that mean?", "what does it mean", "what does it mean?"):
        return True
    return False



def _find(text, lexicon):
    """Return the highest-priority lexicon topic present in text.

    Lexicons are ordered most-specific-first, so the first match wins.
    """
    for needle, topic in lexicon:
        if needle in text:
            return topic
    return None


def _contains_any(text, patterns):
    return any(p in text for p in patterns)


# Standalone reference pronouns / ellipses. If the *whole* question (minus
# an action verb) is essentially just a pronoun, it depends on context.
_REFERENCE_ONLY = re.compile(
    r"^\s*(?:how\s+(?:do|can|should|would)\s+i\s+|how\s+to\s+|can\s+i\s+|should\s+i\s+)?"
    r"(?:fix|clean|handle|solve|remove|treat|reduce|replace|improve|resolve|repair|address|deal\s+with)?"
    r"[\s,]*(?:them|it|these|those|this|that|the\s+above|the\s+previous)?"
    r"[\s,.?!.]*$"
)


def _is_reference_only(q):
    """True when the question is essentially just a pronoun + optional verb."""
    return bool(_REFERENCE_ONLY.match(q or ""))


def _mentions_multiple_issues(text):
    """Count distinct issue topics named in a question."""
    t = text.lower()
    seen = []
    for needle, topic in _ISSUE_LEXICON:
        if needle in t and topic not in seen:
            seen.append(topic)
    return seen


# ============================================================
# The resolver
# ============================================================

def resolve(question, conversation_history=None, available_issues=None):
    """Resolve a user question into an authoritative ``Resolution``.

    ``available_issues`` is the set of issue topics actually detected in the
    current dataset (``{...}`` of topic ids). It is used only to decide
    whether a remediation request is answerable and to build clarification
    options -- never to pick an arbitrary answer.
    """
    raw = (question or "").strip()
    q = raw.lower()
    available = set(available_issues or [])

    # --------------------------------------------------------
    # 1. Empty / trivially vague question -> clarify.
    # --------------------------------------------------------
    if _is_vague(q):
        prev = _last_assistant_answer(conversation_history)
        options = _issues_mentioned_in(prev) or sorted(available)
        return Resolution(
            intent="clarification_required",
            action="clarify",
            needs_clarification=True,
            clarification_options=options,
            uses_context=bool(options),
            note="vague/empty question",
        )

    # --------------------------------------------------------
    # 2. Identify what the current question explicitly names.
    # --------------------------------------------------------
    explicit_issues = _mentions_multiple_issues(q)
    explicit_topic = explicit_issues[0] if explicit_issues else None
    explicit_nonissue = _find(q, _TOPIC_LEXICON)

    is_project = _contains_any(q, _PROJECT_PATTERNS)
    is_model = _contains_any(q, _MODEL_PATTERNS)
    is_feature = _contains_any(q, _FEATURE_PATTERNS)
    is_action = _contains_any(q, _ACTION_LEXICON)
    # "why" alone is context-dependent, but "why is the quality score low?"
    # explicitly names a topic, so it must NOT be treated as a reference.
    has_explicit_subject = bool(explicit_topic or explicit_nonissue or is_project
                                or is_model or is_feature)
    is_context_dep = (not has_explicit_subject) and (
        _contains_any(q, _CONTEXT_DEPENDENT_PATTERNS) or _is_reference_only(q)
    )

    # --------------------------------------------------------
    # 3. Build structured view of the immediately-prior turn.
    # --------------------------------------------------------
    prior = _derive_prior_turn(conversation_history)

    # --------------------------------------------------------
    # 4. Resolve. CURRENT EXPLICIT INTENT always beats CONTEXT.
    # --------------------------------------------------------

    # (a) Feature importance / influence. Must beat bare "ml" readiness.
    if is_feature:
        # A feature question that also names an issue ("which features
        # influence the outliers?") is really a feature question.
        return Resolution(
            intent="feature_importance",
            topic="feature_importance",
            action="report",
            evidence=TOPIC_EVIDENCE["feature_importance"],
            note="explicit feature-influence ask",
        )

    # (b) Model selection.
    if is_model:
        return Resolution(
            intent="model_selection",
            topic="model_selection",
            action="compare",
            evidence=TOPIC_EVIDENCE["model_selection"],
            note="explicit model-choice ask",
        )

    # (c) Project ideas (dataset-aware). Must be checked BEFORE ml_readiness
    #     so "what ML projects can I build?" never becomes ML readiness.
    if is_project:
        # "how do I implement it?" after projects -> implementation of prior.
        if prior and prior.topic == "project_ideas" and _is_implementation(q):
            return Resolution(
                intent="project_implementation",
                topic="project_ideas",
                action="implement",
                evidence=TOPIC_EVIDENCE["project_ideas"],
                uses_context=True,
                note="implementation follow-up on prior project discussion",
            )
        # "which one is best?" after projects -> comparison of prior options.
        if prior and prior.topic == "project_ideas" and _is_comparison(q):
            return Resolution(
                intent="project_comparison",
                topic="project_ideas",
                action="compare",
                evidence=TOPIC_EVIDENCE["project_ideas"],
                uses_context=True,
                note="comparison follow-up on prior project discussion",
            )
        return Resolution(
            intent="project_ideas",
            topic="project_ideas",
            action="report",
            evidence=TOPIC_EVIDENCE["project_ideas"],
            note="explicit project ask",
        )

    # (c2) "How do I implement it?" as a reference-only follow-up (no project
    #      keyword in the question itself) when the prior turn was projects.
    if _is_implementation(q) and prior and prior.topic == "project_ideas":
        return Resolution(
            intent="project_implementation",
            topic="project_ideas",
            action="implement",
            evidence=TOPIC_EVIDENCE["project_ideas"],
            uses_context=True,
            note="implementation follow-up on prior project discussion",
        )

    # (d) An explicit, directly-named non-issue topic (columns, score, ML
    #     readiness, structure). These beat context -- but NOT a directly
    #     named issue ("the outliers in the columns" is an outlier ask,
    #     not a column inventory), and NOT a remediation action applied to
    #     a non-remediable topic ("how can I remove these problems" is a
    #     remediation ask, which (g) resolves from the issues actually
    #     discussed or detected).
    if (explicit_nonissue and not explicit_issues
            and not is_context_dep
            and not (is_action and explicit_nonissue not in REMEDIATION_INTENT)):
        topic = explicit_nonissue
        if topic == "column_role_information" and not _is_role_ask(q):
            topic = "column_information"
        return Resolution(
            intent=topic,
            topic=topic,
            action="remediate" if (is_action and topic in REMEDIATION_INTENT) else "report",
            evidence=TOPIC_EVIDENCE.get(topic, "dataset_profile"),
            note="explicit non-issue topic named in question",
        )

    # (e) An explicit, directly-named issue topic. Always beats context.
    if explicit_topic and not is_context_dep:
        # Multiple issues named -> remediation across all of them.
        if len(explicit_issues) > 1 and is_action:
            return Resolution(
                intent="remediation",
                topic=None,
                action="remediate",
                evidence="all_detected_findings",
                note="multiple issues named with a remediation ask",
            )
        if is_action:
            return Resolution(
                intent=REMEDIATION_INTENT.get(explicit_topic, "remediation"),
                topic=explicit_topic,
                action="remediate",
                evidence=TOPIC_EVIDENCE.get(explicit_topic, "all_detected_findings"),
                note="explicit issue + remediation action",
            )
        return Resolution(
            intent=explicit_topic,
            topic=explicit_topic,
            action="report",
            evidence=TOPIC_EVIDENCE.get(explicit_topic, "all_detected_findings"),
            note="explicit issue named in question",
        )

    # (f) Context-dependent elliptical follow-up (no explicit topic).
    if is_context_dep:
        return _resolve_from_context(q, prior, available, is_action)

    # (g) A remediation action with no resolvable topic: if the previous
    #     turn was about exactly one remediable thing, continue it; if it
    #     enumerated several issues (or the user asks for ALL of them),
    #     remediate all of them (never guess one).
    if is_action:
        mentioned = _issues_mentioned_in(
            _last_assistant_answer(conversation_history)
        )
        plural_ref = _contains_any(q, ["them", "these", "those"])
        wants_all = _contains_any(q, [
            "all of them", "all of these", "all of those",
            "all these", "all those", "all the issues", "all the problems",
            "all problems", "every issue", "every problem",
            "each issue", "each problem",
        ])
        # "Fix ALL of them" after an answer that enumerated several issue
        # types -> remediate every referenced issue, never just one.
        if wants_all and len(mentioned) > 1:
            return Resolution(
                intent="remediation",
                topic=None,
                action="remediate",
                evidence="all_detected_findings",
                uses_context=True,
                note="remediation across all issues referenced in prior answer",
            )
        # A concrete REMEDIABLE prior topic is the authoritative anchor.
        # Non-remediable topics ("problems", "columns", ...) are not:
        # after "What problems does this dataset have?", "how can I fix
        # all of them?" means remediate the issues that were listed --
        # not re-list the problems.
        if (prior and prior.topic in REMEDIATION_INTENT
                and not (plural_ref and prior.topic not in ISSUE_TOPICS)):
            return Resolution(
                intent=REMEDIATION_INTENT[prior.topic],
                topic=prior.topic,
                action="remediate",
                evidence=TOPIC_EVIDENCE.get(prior.topic, "all_detected_findings"),
                uses_context=True,
                note="remediation action resolved from prior topic",
            )
        if len(mentioned) > 1:
            return Resolution(
                intent="remediation",
                topic=None,
                action="remediate",
                evidence="all_detected_findings",
                uses_context=True,
                note="remediation across issues enumerated in prior answer",
            )
        if len(mentioned) == 1:
            return Resolution(
                intent=REMEDIATION_INTENT.get(mentioned[0], "remediation"),
                topic=mentioned[0],
                action="remediate",
                evidence=TOPIC_EVIDENCE.get(mentioned[0], "all_detected_findings"),
                uses_context=True,
                note="remediation resolved from single issue in prior answer",
            )
        # No context. Remediate every detected issue -- never pick one.
        return Resolution(
            intent="remediation",
            topic=None,
            action="remediate",
            evidence="all_detected_findings",
            note="generic remediation with no resolvable single topic",
        )

    # (h) Problems / overview style asks.
    if _contains_any(q, ["what is wrong", "what's wrong", "what issues",
                         "what problems", "main problems", "biggest problems",
                         "problems detected", "issues detected", "any problems",
                         "any issues", "most serious", "worst", "critical issue",
                         "problems with", "issues with",
                         "did you find", "what did you find", "did the analysis find",
                         "what did the analysis", "detected in", "what has been detected",
                         "what was detected"]):
        return Resolution(intent="problems", topic="problems", action="report",
                          evidence=TOPIC_EVIDENCE["problems"],
                          note="explicit problems ask")
    if _contains_any(q, ["what should i fix", "fix first", "priority",
                         "which first", "order of", "fix first"]):
        return Resolution(intent="priority", topic="priority", action="report",
                          evidence=TOPIC_EVIDENCE["priority"],
                          note="explicit priority ask")
    if _contains_any(q, ["how many rows", "row count", "how big", "size of",
                         "dataset size", "number of records"]):
        return Resolution(intent="dataset_summary", topic="dataset_summary",
                          action="report", evidence=TOPIC_EVIDENCE["dataset_summary"],
                          note="explicit dataset-size ask")

    # (i) Last resort: a genuine general/overview question.
    if _contains_any(q, ["overview", "summary", "describe", "about this dataset",
                         "what is this", "what kind of data", "tell me about",
                         "what does this dataset", "general"]):
        return Resolution(intent="dataset_summary", topic="dataset_summary",
                          action="report", evidence=TOPIC_EVIDENCE["dataset_summary"],
                          note="explicit overview ask")

    # (j) Truly unresolvable. Do NOT fall back to an arbitrary finding.
    #     Ask for clarification.
    options = sorted(available) if available else ["dataset_summary"]
    return Resolution(
        intent="general",
        topic=None,
        action="clarify",
        needs_clarification=True,
        clarification_options=options,
        note="no confident intent; asking for clarification instead of guessing",
    )


# ============================================================
# Context resolution helpers
# ============================================================

def _is_implementation(q):
    return _contains_any(q, ["implement", "how do i build", "how to build",
                             "walk me through", "code for", "start building",
                             "how would i code", "how do i code", "give me code"])


def _is_comparison(q):
    return _contains_any(q, ["which one is best", "which is best", "which one is better",
                             "which is better", "compare", "difference between",
                             "which should i pick", "which should i choose",
                             "best one", "best option", "most suitable"])


def _is_role_ask(q):
    return _contains_any(q, ["role", "datatype", "data type of", "type of each",
                             "which are numerical", "which are categorical",
                             "which are identifiers", "which columns are"])


def _derive_prior_turn(conversation_history):
    """Build a structured record of the prior exchange.

    The immediately-previous user message may itself be elliptical
    ("Give me more.", "Which one is best?"), which resolves to no topic in
    isolation. In that case walk back through earlier user turns to find the
    last turn that resolves to a concrete topic, so the conversation thread
    (e.g. project ideas) is not lost.
    """
    msgs = _history_msgs(conversation_history)
    if not msgs:
        return None

    user_turns = [m.get("content", "") for m in msgs if m.get("role") == "user"]
    last_assistant = _last_assistant_answer(conversation_history)
    if not user_turns and not last_assistant:
        return None

    # Resolve user turns newest-first (bounded walk-back) until a concrete
    # topic is found. Resolving with an empty history avoids recursion.
    topic = None
    prev_resolution = None
    for turn in reversed(user_turns[-6:]):
        if not turn:
            continue
        try:
            r = resolve(turn, conversation_history=None)
        except Exception:
            r = None
        if prev_resolution is None and r is not None:
            prev_resolution = r
        if r is not None and r.topic and r.intent != "clarification_required":
            topic = r.topic
            prev_resolution = r
            break

    # Fall back to what the assistant actually enumerated in its answer.
    if not topic and last_assistant:
        mentioned = _issues_mentioned_in(last_assistant)
        if len(mentioned) == 1:
            topic = mentioned[0]

    last_user = user_turns[-1] if user_turns else ""
    return _PriorTurn(
        question=last_user,
        intent=prev_resolution.intent if prev_resolution else "general",
        topic=topic,
        action=prev_resolution.action if prev_resolution else "report",
        mentioned_issues=_issues_mentioned_in(last_assistant),
    )


def _resolve_from_context(q, prior, available, is_action):
    """Resolve an elliptical follow-up using the structured prior turn."""
    if not prior:
        return Resolution(
            intent="clarification_required",
            action="clarify",
            needs_clarification=True,
            clarification_options=sorted(available),
            note="context-dependent question but no prior turn to anchor to",
        )

    # "Which one?" / "what about them?" style -> enumerate the prior options.
    if _contains_any(q, ["which one", "which of these", "which of them",
                         "what about them", "what about it", "what about those",
                         "what about these", "best one", "best option"]):
        return Resolution(
            intent="project_comparison" if prior.topic == "project_ideas"
                   else (prior.topic or "general"),
            topic=prior.topic,
            action="compare",
            evidence=TOPIC_EVIDENCE.get(prior.topic, "all_detected_findings"),
            uses_context=True,
            note="selection/comparison follow-up on prior options",
        )

    # "Give me more" / "tell me more" -> continue the prior topic.
    if _contains_any(q, ["give me more", "tell me more", "show me more",
                         "more detail", "more info", "more example",
                         "another example", "another one", "go on",
                         "keep going", "continue", "elaborate", "expand on"]):
        if prior.topic:
            return Resolution(
                intent=prior.topic,
                topic=prior.topic,
                action="expand",
                evidence=TOPIC_EVIDENCE.get(prior.topic, "all_detected_findings"),
                uses_context=True,
                note="expand previous topic",
            )
        return Resolution(
            intent="clarification_required",
            action="clarify",
            needs_clarification=True,
            clarification_options=prior.mentioned_issues or sorted(available),
            uses_context=True,
            note="'more' requested but prior topic was not a single subject",
        )

    # An explicit remediation action on a reference ("fix them", "remove it").
    if is_action or _contains_any(q, ["fix", "clean", "handle", "remove", "solve",
                                      "replace", "treat", "reduce"]):
        # A concrete prior topic is the authoritative anchor: after "What is
        # the quality score?", "how do I fix it?" means improve the score --
        # not whatever issue words happen to appear in the answer text.
        # Exception: a plural reference ("fix them") after a NON-issue topic
        # points at the issues mentioned in that answer, not at the topic.
        plural_ref = _contains_any(q, ["them", "these", "those"])
        if prior.topic in REMEDIATION_INTENT and not (plural_ref and prior.topic not in ISSUE_TOPICS):
            topic = prior.topic
            return Resolution(
                intent=REMEDIATION_INTENT[topic],
                topic=topic,
                action="remediate",
                evidence=TOPIC_EVIDENCE.get(topic, "all_detected_findings"),
                uses_context=True,
                note="remediation anchored to the concrete prior topic",
            )
        mentioned = prior.mentioned_issues
        # Single issue in the prior answer -> unambiguous.
        if len(mentioned) == 1:
            topic = mentioned[0]
            return Resolution(
                intent=REMEDIATION_INTENT.get(topic, "remediation"),
                topic=topic,
                action="remediate",
                evidence=TOPIC_EVIDENCE.get(topic, "all_detected_findings"),
                uses_context=True,
                note="remediation of the single issue discussed previously",
            )
        # Prior topic was a single non-issue subject -> continue it.
        if prior.topic and prior.topic not in ISSUE_TOPICS and not mentioned:
            return Resolution(
                intent=REMEDIATION_INTENT.get(prior.topic, "remediation"),
                topic=prior.topic,
                action="remediate",
                evidence=TOPIC_EVIDENCE.get(prior.topic, "all_detected_findings"),
                uses_context=True,
                note="remediation of the prior non-issue topic",
            )
        # Several issues were discussed -> never guess; address all of them.
        if len(mentioned) > 1 or (prior.topic in ISSUE_TOPICS and len(mentioned) != 1):
            return Resolution(
                intent="remediation",
                topic=None,
                action="remediate",
                evidence="all_detected_findings",
                uses_context=True,
                clarification_options=mentioned or sorted(available),
                note="multiple issues in play; remediate all or clarify",
            )
        if prior.topic:
            return Resolution(
                intent=REMEDIATION_INTENT.get(prior.topic, "remediation"),
                topic=prior.topic,
                action="remediate",
                evidence=TOPIC_EVIDENCE.get(prior.topic, "all_detected_findings"),
                uses_context=True,
                note="remediation of prior topic",
            )

    # "Why?" / "what do you mean?" -> continue the prior topic.
    if prior.topic:
        return Resolution(
            intent=prior.topic,
            topic=prior.topic,
            action="expand",
            evidence=TOPIC_EVIDENCE.get(prior.topic, "all_detected_findings"),
            uses_context=True,
            note="explanation follow-up on prior topic",
        )

    # Nothing usable.
    return Resolution(
        intent="clarification_required",
        action="clarify",
        needs_clarification=True,
        clarification_options=prior.mentioned_issues or sorted(available),
        uses_context=True,
        note="could not resolve reference from prior turn",
    )


# ============================================================
# Answer-relevance validation (Phase 14)
# ============================================================

# Evidence kind -> the issue/topic tokens that must appear (or be implied)
# in a correct answer. Used as a guard against returning unrelated findings.
_EVIDENCE_TOKENS = {
    "missing_values_finding": ["missing", "empty", "null", "nan", "no missing"],
    "duplicate_rows_finding": ["duplicate", "no duplicate"],
    "categorical_inconsistency_findings": ["inconsist", "categor", "label", "no inconsist"],
    "data_type_validation_finding": ["data type", "type mismatch", "type problem", "no data type"],
    "invalid_values_finding": ["invalid", "wrong value", "no invalid"],
    "outlier_evidence": ["outlier", "extreme", "iqr", "no outlier"],
    "sentinel_evidence": ["sentinel", "placeholder", "no sentinel"],
    "ml_anomalies_finding": ["anomal", "unusual", "isolation"],
    "column_metadata": ["column", "field", "feature", "type", "role"],
    "column_roles": ["role", "identifier", "target", "feature", "categorical", "numerical"],
    "quality_score": ["quality", "score"],
    "ml_readiness": ["readiness", "ready", "ml", "machine learning", "check"],
}


def answer_is_relevant(answer, resolution, available_issues=None):
    """Guard: does the generated answer plausibly address the resolution?

    This is a heuristic safety net, not a hard filter. It catches the
    classic failure where an unrelated finding is returned just because it
    exists. When it can't judge (e.g. general/project answers), it allows.

    When ``available_issues`` (the issue types actually detected) is
    supplied, answers whose evidence IS the full issue set -- problem
    reports, priority lists, and multi-issue remediations -- must
    enumerate EVERY detected issue. A truncated or partial listing is
    rejected so the deterministic answer (which always covers them all)
    is used instead.
    """
    if resolution is None or resolution.action == "clarify":
        return True
    text = (answer or "").lower()
    if not text.strip():
        return False

    # Full-issue-set answers must cover the full issue set.
    if (available_issues
            and resolution.action in ("report", "remediate")
            and str(resolution.evidence).startswith("all_detected_findings")):
        mentioned = set(_issues_mentioned_in(text))
        if not set(available_issues) <= mentioned:
            return False

    # Project / comparison / implementation answers are free-form.
    if resolution.topic in ("project_ideas", "model_selection", "feature_importance"):
        return True
    if resolution.intent in ("dataset_summary", "problems", "priority", "general"):
        return True
    if resolution.topic is None:
        # Multi-issue remediation: should mention at least one issue.
        return any(tok in text for toks in _EVIDENCE_TOKENS.values() for tok in toks)

    tokens = _EVIDENCE_TOKENS.get(resolution.evidence)
    if not tokens:
        return True
    return any(tok in text for tok in tokens)

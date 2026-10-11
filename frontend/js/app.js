/* Data Sentinel — frontend application */

const $ = (id) => document.getElementById(id);

let currentFile = null;

/* ---------- helpers ---------- */
function fmtNum(n) {
    if (n === null || n === undefined || isNaN(n)) return "—";
    if (n >= 1e6) return (n / 1e6).toFixed(1).replace(/\.0$/, "") + "M";
    if (n >= 1e3) return (n / 1e3).toFixed(1).replace(/\.0$/, "") + "K";
    return String(n);
}
function fmtPct(v) { return (v === null || v === undefined || isNaN(v)) ? "—" : v.toFixed(1) + "%"; }
function esc(s) { return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }

function severityBadge(sev) {
    const s = String(sev || "CLEAN").toUpperCase();
    const cls = s === "CLEAN" ? "badge-clean" : s === "LOW" ? "badge-low" : s === "MEDIUM" ? "badge-medium" : "badge-critical";
    return `<span class="badge ${cls}">${s}</span>`;
}

/* Friendly names for technical finding types */
const FRIENDLY = {
    duplicate_rows: "Duplicate rows",
    missing_values: "Missing / empty cells",
    categorical_inconsistency: "Inconsistent labels",
    data_type_validation: "Wrong data types",
    invalid_values: "Wrong / invalid values",
    statistical_outliers: "Unusual values (outliers)",
    possible_sentinel_values: "Suspicious placeholder values",
    ml_anomalies: "Unusual record combinations",
};
function friendlyName(t) { return FRIENDLY[t] || String(t).replace(/_/g, " "); }

/* Plotly shared theme */
const PLOT_FONT = { family: "Inter, sans-serif", color: "#9aa7bd", size: 12 };
function plotLayout(title, extra = {}) {
    return Object.assign({
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        font: PLOT_FONT,
        margin: { l: 48, r: 20, t: 12, b: 44 },
        xaxis: { gridcolor: "#1a2133", zerolinecolor: "#1a2133", tickfont: { color: "#9aa7bd" }, titlefont: { color: "#9aa7bd" } },
        yaxis: { gridcolor: "#1a2133", zerolinecolor: "#1a2133", tickfont: { color: "#c7d2e4" }, titlefont: { color: "#c7d2e4" } },
        hoverlabel: { bgcolor: "#151b2b", bordercolor: "#2a354d", font: { color: "#e8edf6", size: 12 } },
        bargap: 0.25,
    }, extra);
}
const PLOT_CONFIG = { responsive: true, displayModeBar: false };
const TRACE_COLORS = ["#38bdf8", "#a78bfa", "#34d399", "#fbbf24", "#f87171", "#60a5fa"];

/* ---------- upload ---------- */
const dropzone = $("dropzone"), fileInput = $("fileInput");

dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") fileInput.click(); });
$("browseBtn").addEventListener("click", (e) => { e.stopPropagation(); fileInput.click(); });
["dragover", "dragenter"].forEach(ev => dropzone.addEventListener(ev, e => { e.preventDefault(); dropzone.classList.add("drag"); }));
["dragleave", "drop"].forEach(ev => dropzone.addEventListener(ev, e => { e.preventDefault(); dropzone.classList.remove("drag"); }));
dropzone.addEventListener("drop", e => { const f = e.dataTransfer.files[0]; if (f) selectFile(f); });
fileInput.addEventListener("change", () => { if (fileInput.files[0]) selectFile(fileInput.files[0]); });

function selectFile(f) {
    currentFile = f;
    $("fileName").textContent = f.name;
    $("fileMeta").textContent = `${(f.size / 1024 / 1024).toFixed(2)} MB · ${f.name.split(".").pop().toUpperCase()}`;
    $("fileCard").hidden = false;
}

$("analyzeBtn").addEventListener("click", runAnalysis);
$("retryBtn").addEventListener("click", () => { $("errorState").hidden = true; $("landing").hidden = false; });
$("brandHome").addEventListener("click", (e) => { e.preventDefault(); location.reload(); });

async function runAnalysis() {
    if (!currentFile) return;
    $("landing").hidden = true;
    $("errorState").hidden = true;
    $("processing").hidden = false;

    const stages = [
        ["Uploading file", 15],
        ["Profiling dataset", 35],
        ["Detecting data-quality issues", 60],
        ["Running ML readiness checks", 80],
        ["Generating visualizations", 92],
    ];
    let si = 0;
    const timer = setInterval(() => {
        if (si < stages.length) {
            $("processingStage").textContent = stages[si][0];
            $("progressBar").style.width = stages[si][1] + "%";
            si++;
        }
    }, 700);

    try {
        const fd = new FormData();
        fd.append("file", currentFile);
        const resp = await fetch("/api/analyze", { method: "POST", body: fd });
        const text = await resp.text();
        let data;
        try {
            data = JSON.parse(text);
        } catch {
            throw new Error(`Server error (HTTP ${resp.status}). The file may be too large or the analysis failed on the server.`);
        }
        clearInterval(timer);
        $("progressBar").style.width = "100%";
        if (!resp.ok || data.status !== "success") throw new Error(data.message || `HTTP ${resp.status}`);
        await new Promise(r => setTimeout(r, 350));
        $("processing").hidden = true;
        renderDashboard(data);
    } catch (err) {
        clearInterval(timer);
        $("processing").hidden = true;
        $("errorState").hidden = false;
        $("errorMessage").textContent = err.message;
    }
}

/* ---------- derived helpers ---------- */
function issueByType(data, type) {
    return (data.findings || []).find(f => f.issue_type === type && f.has_issue);
}
function findingAffected(data, type) {
    const f = (data.findings || []).find(f => f.issue_type === type);
    return f && f.has_issue ? (f.affected_columns || []) : [];
}

/* ---------- dashboard render ---------- */
function renderDashboard(data) {
    $("dashboard").hidden = false;
    document.body.classList.add("has-dataset");   // reveals sidebar shell + nav
    const profile = data.profile || {};
    const score = data.quality_score || {};
    const ml = data.ml_readiness || {};
    const issues = (data.findings || []).filter(f => f.has_issue);

    $("topbarMeta").textContent = data.filename || "";

    $("dsName").textContent = data.filename || "dataset";
    const chunked = profile.processing_mode === "chunked" || (profile.rows || 0) > 100000;
    $("dsMeta").textContent =
        `${fmtNum(profile.rows)} rows · ${profile.columns} columns · ${chunked ? "Chunked processing" : "Standard processing"}`;

    /* KPI cards */
    const qs = score.overall_score;
    const kpiIC = {
        gauge: '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>',
        chip: '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><rect x="4" y="4" width="16" height="16" rx="3"/><path d="M9 9h6v6H9z"/><path d="M9 1v3M15 1v3M1 9h3M1 15h3M20 9h3M20 15h3"/></svg>',
        rows: '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><rect x="3" y="4" width="18" height="16" rx="2"/><path d="M3 9h18M3 14h18"/></svg>',
        cols: '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><rect x="3" y="4" width="18" height="16" rx="2"/><path d="M9 4v16M15 4v16"/></svg>',
        alert: '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M12 9v4M12 17h.01"/><path d="M10.3 3.9L1.8 18a2 2 0 001.7 3h17a2 2 0 001.7-3L13.7 3.9a2 2 0 00-3.4 0z"/></svg>',
    };
    $("kpiRow").innerHTML = `
        <div class="kpi kpi-accent-blue">
            <div class="kpi-top"><span class="kpi-label">DATA QUALITY</span><span class="kpi-ic">${kpiIC.gauge}</span></div>
            <div class="kpi-value">${qs != null ? qs.toFixed(1) + "%" : "—"}</div>
            <div class="kpi-foot">Overall score</div>
        </div>
        <div class="kpi kpi-accent-violet">
            <div class="kpi-top"><span class="kpi-label">ML READINESS</span><span class="kpi-ic">${kpiIC.chip}</span></div>
            <div class="kpi-value">${ml.readiness_percentage != null ? ml.readiness_percentage.toFixed(1) + "%" : "—"}</div>
            <div class="kpi-foot">${ml.passed_checks ?? "—"}/${ml.total_checks ?? "—"} checks passed</div>
        </div>
        <div class="kpi kpi-accent-green">
            <div class="kpi-top"><span class="kpi-label">ROWS</span><span class="kpi-ic">${kpiIC.rows}</span></div>
            <div class="kpi-value">${fmtNum(profile.rows)}</div>
            <div class="kpi-foot">Records analyzed</div>
        </div>
        <div class="kpi kpi-accent-green">
            <div class="kpi-top"><span class="kpi-label">COLUMNS</span><span class="kpi-ic">${kpiIC.cols}</span></div>
            <div class="kpi-value">${profile.columns ?? "—"}</div>
            <div class="kpi-foot">Features detected</div>
        </div>
        <div class="kpi ${issues.length ? "kpi-accent-red" : "kpi-accent-green"}">
            <div class="kpi-top"><span class="kpi-label">ISSUES</span><span class="kpi-ic">${kpiIC.alert}</span></div>
            <div class="kpi-value">${issues.length}</div>
            <div class="kpi-foot">${issues.length ? "Detected" : "None found"}</div>
        </div>`;

    renderScoreBar($("qualityGauge"), qs, "DATA QUALITY");
    renderScoreBar($("mlGauge"), ml.readiness_percentage, "ML READINESS");

    /* category scores */
    const cats = score.category_scores || {};
    $("qualityBreakdown").innerHTML = Object.entries(cats).map(([k, v]) =>
        `<div class="cat-row"><span class="cat-name">${friendlyName(k)}</span><span class="cat-score">${v.score.toFixed(0)} ${severityBadge(v.severity)}</span></div>`
    ).join("") || '<p class="empty-state">No category scores.</p>';

    /* quality issues chart */
    const issuesViz = (data.visualizations || []).find(v => v.chart_id === "qualityIssues" || (v.type === "bar" && /quality/i.test(v.title || "")));
    let qLabels, qCounts;
    if (issuesViz && issuesViz.data && issuesViz.data.length) {
        qLabels = issuesViz.data.map(d => FRIENDLY[d.category] || d.category);
        qCounts = issuesViz.data.map(d => d.count);
    } else {
        qLabels = issues.map(f => friendlyName(f.issue_type));
        qCounts = issues.map(f => f.count);
    }
    if (qLabels.length) {
        Plotly.newPlot($("qualityIssuesChart"), [{
            type: "bar", x: qCounts, y: qLabels, orientation: "h",
            marker: { color: TRACE_COLORS, borderradius: 6 },
            hovertemplate: "%{y}: %{x}<extra></extra>",
        }], plotLayout("", { margin: { l: 190, r: 20, t: 12, b: 40 }, yaxis: { autorange: "reversed", gridcolor: "rgba(0,0,0,0)" } }), PLOT_CONFIG);
    } else {
        $("qualityIssuesChart").innerHTML = '<p class="empty-state">No quality issues detected. 🎉</p>';
    }

    renderSummary(data);
    renderHealthMap(data);
    renderColumns(data);
    renderPreview(data);
    renderCharts(data);
    renderMLChecks(ml);
    renderSentinels(data);
    renderAI(data);
}

/* ---------- score gauge (SVG radial, same inputs as before) ---------- */
function renderScoreBar(el, value, label) {
    const v = value == null || isNaN(value) ? 0 : Math.max(0, Math.min(100, value));
    const color = v >= 85 ? "#34d399" : v >= 60 ? "#fbbf24" : "#f87171";
    const state = v >= 85 ? "Healthy" : v >= 60 ? "Needs attention" : "Critical";
    const R = 78, C = 2 * Math.PI * R, SWEEP = 0.75;      // 270° arc
    const track = C * SWEEP;
    const fill = C * SWEEP * (v / 100);
    el.innerHTML = `
        <div class="gauge">
            <svg viewBox="0 0 200 168" role="img" aria-label="${esc(label)}: ${value == null ? "not available" : v.toFixed(0) + " percent, " + state}">
                <circle class="gauge-track" cx="100" cy="100" r="${R}"
                        stroke-dasharray="${track} ${C}" transform="rotate(135 100 100)"/>
                <circle class="gauge-fill" cx="100" cy="100" r="${R}" stroke="${color}"
                        stroke-dasharray="${fill} ${C}" transform="rotate(135 100 100)"/>
                <text class="gauge-value" x="100" y="97">${value == null ? "—" : v.toFixed(0) + "%"}</text>
                <text class="gauge-label" x="100" y="122">${esc(label)}</text>
            </svg>
            <div class="gauge-state" style="color:${color}">${state}</div>
        </div>`;
}

/* ---------- dataset summary ---------- */
function renderSummary(data) {
    const profile = data.profile || {};
    const score = data.quality_score || {};
    const ml = data.ml_readiness || {};
    const issues = (data.findings || []).filter(f => f.has_issue);
    const dups = profile.duplicate_count || 0;

    let verdict;
    const q = score.overall_score;
    if (q == null) verdict = "We could not compute a quality score.";
    else if (q >= 85) verdict = `Your dataset is in <strong style="color:var(--green)">good health</strong> (score ${q.toFixed(1)}%).`;
    else if (q >= 60) verdict = `Your dataset <strong style="color:var(--amber)">needs attention</strong> (score ${q.toFixed(1)}%).`;
    else verdict = `Your dataset has <strong style="color:var(--red)">serious quality problems</strong> (score ${q.toFixed(1)}%).`;

    const issueLines = issues.map(f => {
        const name = friendlyName(f.issue_type).toLowerCase();
        const cols = (f.affected_columns || []).join(", ");
        let examples = "";
        const det = f.details || {};
        if (f.issue_type === "categorical_inconsistency") {
            const groups = det.inconsistent_groups || det.columns || {};
            const ex = [];
            for (const [col, g] of Object.entries(groups)) {
                const gg = typeof g === "object" && g !== null ? (g.inconsistent_groups || g) : g;
                if (gg && typeof gg === "object") {
                    for (const [canon, variants] of Object.entries(gg).slice(0, 2)) {
                        ex.push(`<em>${esc(col)}</em>: "${esc(canon)}" also written as ${Array.isArray(variants) ? variants.map(v => `"${esc(v)}"`).join(", ") : esc(variants)}`);
                    }
                }
            }
            if (ex.length) examples = `<br><span class="ex">What we found: ${ex.join(" · ")}</span>`;
        } else if (f.issue_type === "possible_sentinel_values") {
            const sCols = det.columns || {};
            const ex = [];
            for (const [col, info] of Object.entries(sCols)) {
                for (const c of (info.sentinel_candidates || []).slice(0, 2)) ex.push(`<em>${esc(col)}</em>: <code>${esc(c.value)}</code> × ${c.count}`);
                for (const c of (info.placeholder_candidates || []).slice(0, 2)) ex.push(`<em>${esc(col)}</em>: <code>${esc(c.value)}</code> × ${c.count}`);
            }
            if (ex.length) examples = `<br><span class="ex">What we found: ${ex.join(" · ")}</span>`;
        } else if (f.issue_type === "missing_values") {
            const mcols = det.columns || {};
            const ex = Object.entries(mcols).filter(([, c]) => (c.missing_count || 0) > 0)
                .map(([c, d]) => `<em>${esc(c)}</em>: ${d.missing_count} empty`).join(" · ");
            if (ex) examples = `<br><span class="ex">What we found: ${ex}</span>`;
        }
        return `<li><strong>${f.count}</strong> case(s) of ${name}${cols ? ` — affecting <em>${esc(cols)}</em>` : ""}.${examples}</li>`;
    }).join("");

    $("datasetSummary").innerHTML = `
        <p class="summary-verdict">${verdict} It has <strong>${fmtNum(profile.rows)}</strong> rows across <strong>${profile.columns}</strong> columns,
        <strong>${dups}</strong> duplicate row(s), and ML readiness of <strong>${ml.readiness_percentage != null ? ml.readiness_percentage.toFixed(0) + "%" : "—"}</strong>.</p>
        ${issues.length ? `<ul class="summary-list">${issueLines}</ul>` : `<p class="summary-ok">✓ No data-quality issues were detected by the engine.</p>`}`;
}

/* ---------- health map ---------- */
function renderHealthMap(data) {
    const profile = data.profile || {};
    const details = profile.column_details || {};
    const cols = Object.keys(details).length ? Object.keys(details) : (profile.column_names || []);
    const invalid = findingAffected(data, "invalid_values");
    const outliers = findingAffected(data, "statistical_outliers");
    const incons = findingAffected(data, "categorical_inconsistency");
    const sentinelF = (data.findings || []).find(f => f.issue_type === "possible_sentinel_values");
    const sCols = sentinelF && sentinelF.details && sentinelF.details.columns ? sentinelF.details.columns : {};

    const dims = [
        { key: "missing", label: "Missing / Empty", get: c => { const m = details[c]?.missing_percentage || 0; return m > 5 ? 2 : m > 0 ? 1 : 0; } },
        { key: "invalid", label: "Wrong Values", get: c => invalid.includes(c) ? 2 : 0 },
        { key: "outlier", label: "Outliers", get: c => outliers.includes(c) ? 1 : 0 },
        { key: "sentinel", label: "Suspicious Values", get: c => (sCols[c]?.sentinel_count || 0) > 0 ? 2 : 0 },
        { key: "inconsistent", label: "Inconsistent Labels", get: c => incons.includes(c) ? 1 : 0 },
    ];

    $("healthMap").innerHTML = `
    <table class="heatmap">
        <tr><th scope="col"></th>${dims.map(d => `<th scope="col">${d.label}</th>`).join("")}</tr>
        ${cols.map(c => `<tr><th scope="row" class="col-label">${esc(c)}</th>${dims.map(d => {
            const v = d.get(c);
            return `<td class="${v === 2 ? "hm-bad" : v === 1 ? "hm-warn" : "hm-clean"}">${v === 2 ? "● Issue" : v === 1 ? "● Review" : "● OK"}</td>`;
        }).join("")}</tr>`).join("")}
    </table>`;
}

/* ---------- column explorer ---------- */
/* role → accent dot color for the column list */
const ROLE_DOT = {
    identifier: "#a78bfa", identifier_like: "#a78bfa",
    numerical_measurement: "#38bdf8", numerical: "#38bdf8",
    categorical: "#34d399", boolean: "#34d399",
    datetime: "#fbbf24", email: "#38bdf8", postal_code: "#f87171",
    unknown: "#6d7a91",
};

function renderColumns(data) {
    const analyses = (data.understanding && data.understanding.column_analysis) || [];
    const container = $("columnExplorer");
    if (!analyses.length) { container.innerHTML = '<p class="empty-state">No column analysis available.</p>'; return; }

    const sentinelF = (data.findings || []).find(f => f.issue_type === "possible_sentinel_values");
    const sCols = sentinelF?.details?.columns || {};

    container.innerHTML = analyses.map((a, i) => {
        const issues = [];
        if (a.missing_count > 0) issues.push(`<span class="chip chip-amber">${a.missing_count} missing</span>`);
        if (sCols[a.column]?.sentinel_count > 0) issues.push(`<span class="chip chip-red">${sCols[a.column].sentinel_count} sentinel</span>`);
        if (sCols[a.column]?.placeholder_count > 0) issues.push(`<span class="chip chip-amber">${sCols[a.column].placeholder_count} placeholder</span>`);
        if (!issues.length) issues.push('<span class="chip chip-green">✓ clean</span>');
        const [primaryRole, semanticType] = getRoleDisplay(a.role);
        return `<div class="column-row" data-idx="${i}" role="button" tabindex="0">
            <span class="cname"><span class="type-dot" style="background:${ROLE_DOT[a.role] || ROLE_DOT.unknown}"></span>${esc(a.column)}</span>
            <span class="ctype">${esc(primaryRole)} · ${esc(semanticType)}</span>
            <span class="cmeta">${a.unique_count ?? "—"} unique · ${fmtPct(a.missing_percentage)} missing</span>
            <span class="cmeta">${issues.join(" ")}</span>
            <span class="arrow" aria-hidden="true">→</span>
        </div>`;
    }).join("");

    container.querySelectorAll(".column-row").forEach(row => {
        const open = () => {
            container.querySelectorAll(".column-row").forEach(r => r.classList.remove("selected"));
            row.classList.add("selected");
            showColumnDetail(data, analyses[Number(row.dataset.idx)]);
        };
        row.addEventListener("click", open);
        row.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); open(); } });
    });
}

function showColumnDetail(data, a) {
    const el = $("columnDetail");
    el.hidden = false;
    const isId = /identifier|id(_|$)|^id$/i.test(a.role || "") || /(_id|^id_|^id$)/i.test(a.column);
    const stats = a.clean_statistics || a.raw_statistics || {};
    const sentinelF = (data.findings || []).find(f => f.issue_type === "possible_sentinel_values");
    const colDetail = sentinelF?.details?.columns?.[a.column] || {};
    const sentinels = (a.possible_sentinels || []).concat(colDetail.sentinel_candidates || []);
    
    // Get primary role and semantic type
    const [primaryRole, semanticType] = getRoleDisplay(a.role);

    el.innerHTML = `
        <h4>${esc(a.column)} <span class="type-tag">${esc(a.type || "")}</span></h4>
        <p class="card-sub" style="margin-top:6px">Primary Role: <strong>${esc(primaryRole)}</strong> | Semantic Type: <strong>${esc(semanticType)}</strong> · ${fmtPct(a.missing_percentage)} missing · ${a.unique_count ?? "—"} unique values${isId ? " · <em>identifier — statistics not examined</em>" : ""}</p>
        ${isId ? "" : `<div class="stat-grid">
            ${["mean", "median", "min", "max", "standard_deviation"].filter(k => stats[k] !== undefined).map(k =>
                `<div class="stat"><div class="k">${k.replace(/_/g, " ").toUpperCase()}</div><div class="v">${typeof stats[k] === "number" ? stats[k].toFixed(2) : stats[k]}</div></div>`).join("")}
        </div>`}
        ${sentinels.length ? `<p style="font-size:13px"><strong>Sentinel candidates:</strong> ${sentinels.map(s => `<span class="chip chip-red">${esc(s.value)} × ${s.count}</span>`).join(" ")}</p>` : ""}
        <div id="columnChart" class="chart-md"></div>`;

    const unique = a.unique_count || 0;
    if (isId) {
        $("columnChart").innerHTML = '<p class="empty-state">Identifier column — no distribution chart.</p>';
    } else if (a.type === "categorical" && unique > 20) {
        $("columnChart").innerHTML = `<p class="empty-state">${unique} unique values — too many to chart clearly. See the preview or filter the column.</p>`;
    } else {
        const viz = (data.visualizations || []).find(v => v.column === a.column || (v.chart_id || "").includes(a.column));
        if (viz) renderViz($("columnChart"), viz);
        else $("columnChart").innerHTML = '<p class="empty-state">No distribution chart available for this column.</p>';
    }
    el.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

// Get role display names
function getRoleDisplay(role) {
    const mapping = {
        "identifier": ["Identifier", "Numeric ID"],
        "identifier_like": ["Identifier", "High-cardinality ID"],
        "numerical_measurement": ["Numerical", "Measurement"],
        "numerical": ["Numerical", "Measurement"],
        "categorical": ["Categorical", "Category"],
        "datetime": ["Date/Time", "Timestamp"],
        "email": ["Contact", "Email"],
        "postal_code": ["Postal Code", "Postal Identifier"],
        "boolean": ["Boolean", "True/False"],
        "unknown": ["Unknown", "Unknown"],
    };
    if (mapping[role]) return mapping[role];
    const label = String(role || "unknown").replace(/_/g, " ")
        .replace(/\b\w/g, c => c.toUpperCase());
    return [label, label];
}

/* ---------- charts ---------- */
function renderViz(el, viz) {
    const data = viz.data || [];
    if (!data.length) { el.innerHTML = '<p class="empty-state">No data.</p>'; return; }
    if (viz.type === "bar") {
        Plotly.newPlot(el, [{ type: "bar", x: data.map(d => d.category), y: data.map(d => d.count), marker: { color: TRACE_COLORS[0], borderradius: 6 } }], plotLayout(), PLOT_CONFIG);
    } else if (viz.type === "histogram") {
        Plotly.newPlot(el, [{ type: "histogram", x: data.map(d => d.value), marker: { color: TRACE_COLORS[0] }, nbinsx: 30 }], plotLayout(), PLOT_CONFIG);
    } else if (viz.type === "scatter") {
        Plotly.newPlot(el, [{ type: "scatter", mode: "markers", x: data.map(d => d.x), y: data.map(d => d.y), marker: { color: TRACE_COLORS[1], size: 7, opacity: .65 } }],
            plotLayout("", { xaxis: { title: viz.x_column || "x", gridcolor: "#1a2436" }, yaxis: { title: viz.y_column || "y", gridcolor: "#1a2436" } }), PLOT_CONFIG);
    } else if (viz.type === "grouped") {
        Plotly.newPlot(el, [{ type: "bar", x: data.map(d => d.category), y: data.map(d => d.value), marker: { color: TRACE_COLORS[2], borderradius: 6 } }], plotLayout(), PLOT_CONFIG);
    } else {
        el.innerHTML = '<p class="empty-state">Unsupported chart type.</p>';
    }
}

/* ---------- inconsistency charts ---------- */
function renderCharts(data) {
    const grid = $("chartGrid");
    const profile = data.profile || {};
    const details = profile.column_details || {};
    const charts = [];

    const missingCols = Object.entries(details).filter(([, d]) => (d.missing_percentage || 0) > 0)
        .map(([c, d]) => ({ category: c, count: d.missing_count }));
    if (missingCols.length) charts.push({ title: "Missing Values by Column", desc: "Columns with empty/NaN cells", data: missingCols, color: TRACE_COLORS[3] });

    const sentinelF = (data.findings || []).find(f => f.issue_type === "possible_sentinel_values" && f.has_issue);
    if (sentinelF && sentinelF.details && sentinelF.details.columns) {
        const sCols = Object.entries(sentinelF.details.columns)
            .map(([c, d]) => ({ category: c, count: (d.sentinel_count || 0) + (d.placeholder_count || 0) }))
            .filter(d => d.count > 0);
        if (sCols.length) charts.push({ title: "Sentinels & Placeholders by Column", desc: "Columns with placeholder-like values", data: sCols, color: TRACE_COLORS[4] });
    }

    const invalid = (data.findings || []).find(f => f.issue_type === "invalid_values" && f.has_issue);
    if (invalid && invalid.affected_columns && invalid.affected_columns.length) {
        charts.push({ title: "Invalid Values by Column", desc: "Values violating validation rules", data: invalid.affected_columns.map(c => ({ category: c, count: invalid.count })), color: TRACE_COLORS[4] });
    }

    const outliers = (data.findings || []).find(f => f.issue_type === "statistical_outliers" && f.has_issue);
    if (outliers && outliers.affected_columns && outliers.affected_columns.length) {
        charts.push({ title: "Outliers by Column", desc: "Columns with statistical outliers", data: outliers.affected_columns.map(c => ({ category: c, count: outliers.count })), color: TRACE_COLORS[1] });
    }

    const incons = (data.findings || []).find(f => f.issue_type === "categorical_inconsistency" && f.has_issue);
    if (incons && incons.affected_columns && incons.affected_columns.length) {
        charts.push({ title: "Categorical Inconsistencies", desc: "Columns with variant labels", data: incons.affected_columns.map(c => ({ category: c, count: incons.count })), color: TRACE_COLORS[2] });
    }

    if (!charts.length) {
        grid.innerHTML = '<div class="chart-card" style="grid-column:1/-1"><h4 style="color:var(--green)">✓ Dataset looks clean</h4><p>No column-level inconsistencies were detected by the quality engine.</p></div>';
    } else {
        grid.innerHTML = charts.map((c, i) => `
            <div class="chart-card">
                <h4>${esc(c.title)}</h4>
                <p>${esc(c.desc)}</p>
                <div id="inc_${i}" class="chart-md"></div>
            </div>`).join("");
        charts.forEach((c, i) => {
            Plotly.newPlot($(`inc_${i}`), [{
                type: "bar", x: c.data.map(d => d.category), y: c.data.map(d => d.count),
                marker: { color: c.color, borderradius: 6 },
            }], plotLayout(), PLOT_CONFIG);
        });
    }

    /* Issue details */
    const issues = (data.findings || []).filter(f => f.has_issue);
    const hints = {
        duplicate_rows: "Drop exact duplicate rows, keeping the first occurrence.",
        missing_values: "Fill with a sensible default (e.g. median for numbers, 'Unknown' for categories) or remove the rows.",
        categorical_inconsistency: "Standardize case and trim whitespace; map variants to one canonical label.",
        data_type_validation: "Fix or re-cast the offending values to the expected type.",
        invalid_values: "Replace with a valid value or treat as missing.",
        statistical_outliers: "Review the values; keep if legitimate, correct or cap if erroneous.",
        possible_sentinel_values: "Replace sentinel/placeholder values (e.g. -9999, N/A) with proper missing values (NaN).",
        ml_anomalies: "Review flagged rows against the raw source data.",
    };
    $("issueDetailList").innerHTML = issues.length ? `<div class="issue-list">${issues.map(f => `
        <div class="issue-item">
            <span class="issue-ic" aria-hidden="true">
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M12 9v4M12 17h.01"/><path d="M10.3 3.9L1.8 18a2 2 0 001.7 3h17a2 2 0 001.7-3L13.7 3.9a2 2 0 00-3.4 0z"/></svg>
            </span>
            <div>
                <h4>${esc(friendlyName(f.issue_type))}</h4>
                <p class="issue-count"><strong>${f.count}</strong> occurrence(s)${(f.affected_columns || []).length ? ` in <em>${esc(f.affected_columns.join(", "))}</em>` : ""}</p>
                <p class="issue-hint">${esc(hints[f.issue_type] || "Review and address this issue.")}</p>
            </div>
        </div>`).join("")}</div>` : '<p class="empty-state">✓ No issues detected — the dataset passed all quality checks.</p>';
}

function renderPreview(data) {
    const rows = data.preview || [];
    const el = $("datasetPreview");
    if (!rows.length) { el.innerHTML = '<p class="empty-state">Row-level preview is not included in this analysis response — see the Column Explorer for per-column statistics.</p>'; return; }
    const cols = Object.keys(rows[0]);
    el.innerHTML = `<div class="table-scroll"><table class="data-table">
        <thead><tr>${cols.map(c => `<th>${esc(c)}</th>`).join("")}</tr></thead>
        <tbody>${rows.slice(0, 100).map(r =>
            `<tr>${cols.map(c => `<td>${esc(r[c] ?? "")}</td>`).join("")}</tr>`).join("")}</tbody>
    </table></div>`;
}

/* ---------- ML readiness ---------- */
function renderMLChecks(ml) {
    const checks = ml.checks || [];
    $("mlChecks").innerHTML = checks.length ? checks.map(c =>
        `<div class="check-row"><span>${esc(String(c.check).replace(/_/g, " "))}</span><span class="${c.passed ? "check-pass" : "check-fail"}">${c.passed ? "✓ PASS" : "⚠ REVIEW"}</span></div>`
    ).join("") : '<p class="empty-state">No readiness checks available.</p>';
}

/* ---------- sentinels ---------- */
function renderSentinels(data) {
    const sentinelF = (data.findings || []).find(f => f.issue_type === "possible_sentinel_values");
    const el = $("sentinelSummary");
    if (!sentinelF || !sentinelF.details) { el.innerHTML = '<p class="empty-state">No sentinel or placeholder analysis available.</p>'; return; }
    const d = sentinelF.details;
    const explainer = `<p class="card-sub" style="margin-bottom:14px"><strong>What are these values?</strong> Source systems often store "no data" using fake numbers instead of blank cells — for example <code>-9999</code> for a temperature that was never measured, or <code>N/A</code> written into a city field. Our engine flags these automatically so they don't corrupt your analysis.</p>`;
    const chips = [];
    chips.push(`<span class="chip chip-blue">Total candidates: ${d.total_candidates ?? 0}</span>`);
    chips.push(`<span class="chip chip-red">Sentinels: ${d.total_sentinels ?? 0}</span>`);
    chips.push(`<span class="chip chip-amber">Placeholders: ${d.total_placeholders ?? 0}</span>`);
    Object.entries(d.columns || {}).forEach(([col, info]) => {
        const cls = (info.sentinel_count || 0) > 0 ? "chip-red" : "chip-amber";
        chips.push(`<span class="chip ${cls}">${esc(col)}: ${info.sentinel_count || 0} sentinels · ${info.placeholder_count || 0} placeholders</span>`);
    });
    el.innerHTML = explainer + chips.join("") || '<p class="empty-state">None detected.</p>';
}

/* ---------- AI assistant ---------- */
let lastAnalysis = null;
function renderAI(data) {
    lastAnalysis = data;
    setupChat();
    // A completed analysis is a NEW dataset context: reset the conversation
    // so turns about the previous dataset never shape answers here.
    chatHistory = [];
    $("chatLog").innerHTML = chatWelcome();
    const el = $("aiInsight");
    const issues = (data.findings || []).filter(f => f.has_issue);
    const sentinelF = issues.find(f => f.issue_type === "possible_sentinel_values");

    const recommendations = {
        duplicate_rows: "Remove or deduplicate repeated records before training.",
        missing_values: "Impute or drop missing values; document the strategy per column.",
        categorical_inconsistency: "Standardize category labels (case, whitespace, synonyms) before grouping.",
        data_type_validation: "Cast or fix columns with unexpected data types.",
        invalid_values: "Review records violating validation rules and correct or exclude them.",
        statistical_outliers: "Investigate extreme values — confirm whether they are legitimate or erroneous.",
        possible_sentinel_values: "Treat sentinel/placeholder values as missing and review the source system.",
        ml_anomalies: "Review anomalous rows flagged by the ML layer before modeling.",
    };

    if (!issues.length) {
        el.innerHTML = `<h4>✦ AI INSIGHT</h4><p>No significant issues were detected. This dataset looks well-formed for analysis and machine-learning workflows. Consider validating it against your domain rules before production use.</p>`;
        return;
    }

    /* Prefer server-generated AI insights (Gemini/rules from the backend) */
    const serverInsights = data.ai_insights && Array.isArray(data.ai_insights.insights) ? data.ai_insights.insights : null;
    if (serverInsights && serverInsights.length) {
        const mode = data.ai_insights.mode === "llm" ? "Powered by Gemini" : "Rule-based analysis";
        el.innerHTML = `
            <h4>✦ AI INSIGHT · <span class="mode">${esc(mode)}</span></h4>
            <p>${esc(data.ai_insights.summary || "")}</p>
            ${serverInsights.map(ins => `
                <div>
                    <p><strong>${esc(ins.title || ins.issue_type)}</strong>
                       <span class="badge ${String(ins.severity).toUpperCase() === "HIGH" ? "badge-critical" : String(ins.severity).toUpperCase() === "MEDIUM" ? "badge-medium" : "badge-low"}">${esc(ins.severity || "")}</span>
                       — ${ins.count ?? 0} occurrence(s)${(ins.affected_columns || []).length ? ` in <em>${esc(ins.affected_columns.slice(0, 3).join(", "))}</em>` : ""}</p>
                    <p style="color:var(--text-dim);font-size:13px">${esc(ins.what_it_means || "")} ${esc(ins.potential_impact || "")}</p>
                    <ul>${(ins.recommended_actions || []).map(a => `<li>${esc(a)}</li>`).join("")}</ul>
                </div>`).join("")}
            ${data.ai_insights.ml_readiness_note ? `<p class="ai-note">⚠ ${esc(data.ai_insights.ml_readiness_note)}</p>` : ""}`;
        return;
    }

    const lines = issues.map(f => {
        const cols = (f.affected_columns || []).slice(0, 3).join(", ");
        return `<strong>${esc(friendlyName(f.issue_type))}</strong> — ${f.count} occurrence(s)${cols ? ` in <em>${esc(cols)}</em>` : ""}.`;
    });
    const actions = issues.map(f => `<li>${esc(recommendations[f.issue_type] || "Review and address this issue.")}</li>`);

    el.innerHTML = `
        <h4>✦ AI INSIGHT</h4>
        <p>Your dataset shows <strong>${issues.length}</strong> issue type(s) that deserve attention:</p>
        <ul>${lines.map(l => `<li>${l}</li>`).join("")}</ul>
        <h4 class="h4-gap">RECOMMENDED ACTIONS</h4>
        <ul>${actions.join("")}</ul>
        ${sentinelF ? `<p class="ai-note" style="color:var(--text-dim)">Note: sentinel values (e.g. -9999) were excluded from the distribution charts above, so the plotted distributions reflect cleaned data.</p>` : ""}`;
}

/* ---------- chat ---------- */
let chatBound = false;
let chatHistory = [];

/* Avatar + row builders (legacy classes kept for compatibility). */
const AV_AI = '<span class="chat-av" aria-hidden="true">✦</span>';
const AV_USER = '<span class="chat-av" aria-hidden="true"><svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="8" r="4"/><path d="M4 21c0-4 3.6-6 8-6s8 2 8 6"/></svg></span>';

function chatRow(role, inner, bubbleCls) {
    const legacy = role === "user" ? "chat-msg-user" : "chat-msg-ai";
    const av = role === "user" ? AV_USER : AV_AI;
    const cls = bubbleCls ? " " + bubbleCls : "";
    return `<div class="chat-row ${role} ${legacy}">${av}<div class="chat-bubble${cls}">${inner}</div></div>`;
}

function chatWelcome() {
    return chatRow("ai", '<p class="muted">Ask me anything about your dataset.</p>');
}

/* Format an assistant answer into readable structure. Content is escaped
   FIRST (untrusted), then only our own tags are added. Numbered lines
   keep their original numbering via <ol start="n">. */
function inlineMd(s) {
    return s
        .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
        .replace(/`([^`]+)`/g, "<code>$1</code>");
}
function formatAnswer(text) {
    const lines = esc(text).split(/\r?\n/);
    let html = "", list = null;
    const close = () => { if (list) { html += `</${list}>`; list = null; } };
    for (const raw of lines) {
        const line = raw.trim();
        if (!line) { close(); continue; }
        let m;
        if ((m = line.match(/^[-*•]\s+(.*)$/))) {
            if (list !== "ul") { close(); html += "<ul>"; list = "ul"; }
            html += `<li>${inlineMd(m[1])}</li>`;
        } else if ((m = line.match(/^(\d+)[.)]\s+(.*)$/))) {
            if (list !== "ol") { close(); html += `<ol start="${Number(m[1])}">`; list = "ol"; }
            html += `<li>${inlineMd(m[2])}</li>`;
        } else if (/^[^.!?]{2,90}:$/.test(line)) {
            close();
            html += `<div class="cb-head">${inlineMd(line.slice(0, -1).trim())}</div>`;
        } else {
            close();
            html += `<p>${inlineMd(line)}</p>`;
        }
    }
    close();
    return html;
}

function setupChat() {
    if (chatBound) return;
    chatBound = true;
    $("chatSend").addEventListener("click", sendChat);
    $("chatClear").addEventListener("click", clearChat);
    $("chatInput").addEventListener("keydown", e => { if (e.key === "Enter") sendChat(); });
    document.querySelectorAll(".chat-chip").forEach(chip => {
        chip.addEventListener("click", () => { $("chatInput").value = chip.dataset.q; sendChat(); });
    });
    // Initialize chat log with welcome message
    chatHistory = [];
    $("chatLog").innerHTML = chatWelcome();
}

function clearChat() {
    chatHistory = [];
    $("chatLog").innerHTML = chatWelcome();
}
async function sendChat() {
    const q = $("chatInput").value.trim();
    if (!q || !lastAnalysis) return;
    $("chatInput").value = "";
    const log = $("chatLog");
    
    // Add user message to history
    const userMsg = { role: "user", content: q };
    chatHistory.push(userMsg);
    
    log.innerHTML += chatRow("user", `<p>${esc(q)}</p>`) +
        `<div class="chat-row ai chat-msg-ai" id="chatPending">${AV_AI}` +
        `<div class="chat-bubble"><span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span></div></div>`;
    log.scrollTop = log.scrollHeight;
    try {
        const resp = await fetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                question: q,
                conversation: chatHistory,  // Send full conversation history
                findings: lastAnalysis.findings || [],
                quality_score: lastAnalysis.quality_score || {},
                ml_readiness: lastAnalysis.ml_readiness || {},
                profile: lastAnalysis.profile || {},
                dataset_name: lastAnalysis.filename || "",
            }),
        });
        const data = await resp.json();
        
        // Add assistant response to history
        const aiMsg = { role: "assistant", content: data.response || data.answer || "No answer." };
        chatHistory.push(aiMsg);
        
        document.getElementById("chatPending").outerHTML = chatRow("ai", formatAnswer(aiMsg.content));
    } catch {
        document.getElementById("chatPending").outerHTML = chatRow("ai", "<p>Could not reach the AI service.</p>", "err");
    }
    log.scrollTop = log.scrollHeight;
}

/* ---------- sidebar navigation ---------- */
function setupNav() {
    const toggle = $("navToggle"), scrim = $("scrim");
    const closeNav = () => document.body.classList.remove("nav-open");
    toggle.addEventListener("click", () => document.body.classList.toggle("nav-open"));
    scrim.addEventListener("click", closeNav);

    const links = Array.from(document.querySelectorAll("#sideNav a"));
    links.forEach(a => a.addEventListener("click", closeNav));

    // Highlight the section currently in view.
    if ("IntersectionObserver" in window) {
        const obs = new IntersectionObserver(entries => {
            entries.forEach(en => {
                if (en.isIntersecting) {
                    links.forEach(l => l.classList.toggle(
                        "active", l.getAttribute("href") === "#" + en.target.id));
                }
            });
        }, { rootMargin: "-25% 0px -65% 0px" });
        links.forEach(l => {
            const target = document.querySelector(l.getAttribute("href"));
            if (target) obs.observe(target);
        });
    }
}
setupNav();


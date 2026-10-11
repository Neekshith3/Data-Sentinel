# Data Sentinel — AI Data Quality & ML Readiness Platform

Upload a CSV/Excel dataset -> profile -> deterministic quality engine -> quality score + ML readiness -> Plotly dashboard -> AI assistant explains findings and recommends actions.

## Run
```
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload          # API on http://127.0.0.1:8000
```
Then open `frontend/index.html` in a browser (or serve it: `python -m http.server 5500 -d frontend`).

## AI assistant
Works offline with rule-based explanations. For LLM-written wording:
```
export ANTHROPIC_API_KEY=...        # Windows: set ANTHROPIC_API_KEY=...
```
The LLM only rewords findings; all counts/columns come from the Python engine.

## Endpoints
- `GET /api/health`
- `POST /api/analyze` — CSV, XLSX, XLS (files >= 100 MB use chunked CSV processing)
- `POST /api/explain` — structured findings -> explanations + recommendations

## Notes
- ML anomaly detection (Isolation Forest) now runs automatically on numerical, non-identifier columns (datasets with >= 20 rows, regular path).
- Large (chunked) Excel files are converted to CSV first; very large Excel files may be slow.
- Not yet built: auth, DB persistence, PDF/Excel reports, Docker, automatic cleaning/export.

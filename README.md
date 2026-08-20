# CardioSense — Backend

A FastAPI backend for an AI-powered cardiovascular/hypertension clinical decision-support assistant. It combines a RAG pipeline over WHO clinical guidelines (pgvector) with a LangGraph state machine that interviews a patient case, retrieves relevant guideline evidence, and produces a cited, confidence-scored cardiovascular risk assessment.

## Architecture

```
FastAPI (app/main.py)
├── /api/cases        — patient case CRUD (app/api/endpoints/cases.py)
├── /api/guidelines    — ingest & list clinical guideline PDFs/TXT (app/api/endpoints/guidelines.py)
└── /api/agent/sessions — chat sessions with the LangGraph clinical assistant (app/api/endpoints/agent.py)

app/services/graph.py — LangGraph state machine:
  topic_guard → check_missing → interview | (rag → evaluate)
                                              ↓
                                    pgvector similarity search
                                    (app/services/vector_search.py)

app/services/rag.py — PDF/TXT extraction, chunking, embeddings (Gemini or OpenAI)
app/models/          — SQLAlchemy models (Patient, PatientCase, Guideline, GuidelineChunk, AgentSession, AgentMessage)
```

### Graph flow

1. **Topic guard** — refuses (deterministically, no LLM call) any message unrelated to cardiovascular/hypertension assessment, before it can reach the RAG/LLM pipeline.
2. **Check missing fields** — inspects the case for required clinical variables (age, sex, BP, smoking, diabetes, kidney disease, prior CVD, cholesterol, HDL).
3. **Interview** — if fields are missing, asks the user for the next one or two (LLM-generated, or a canned fallback in mock/no-key mode).
4. **RAG retrieval** — once the case is complete (or the user says "evaluate"), embeds a query from the patient's profile and retrieves the top-k nearest guideline chunks via pgvector cosine similarity, computing a calibrated retrieval-confidence score per chunk.
5. **Evaluate** — if retrieval confidence is below a calibrated threshold, refuses to fabricate a risk category ("Insufficient Evidence" — the clinical-safety gate). Otherwise, produces a risk category, recommendations, and a **deterministic, code-generated Sources list** (never LLM-written, so it can't hallucinate a citation).

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12 (pinned in `.python-version`; see note below on Vercel deploys).

```bash
uv sync
cp .env.example .env   # then fill in real values, see below
uv run fastapi dev     # http://localhost:8000, docs at /docs
```

### Environment variables (`.env`)

| Variable | Purpose |
|---|---|
| `DATABASE_URL` | Async (asyncpg) Postgres connection string — must have the `pgvector` extension available |
| `DATABASE_URL_SYNC` | Sync (psycopg2) connection string, used only for schema init/migration on startup |
| `GEMINI_API_KEY` | Google AI Studio key. If set, Gemini is used for both embeddings and chat |
| `GEMINI_EMBEDDING_MODEL` | e.g. `models/gemini-embedding-001` — must stay consistent with the DB's embedding column dimension (768, enforced automatically at startup) |
| `GEMINI_LLM_MODEL` | e.g. `models/gemini-3.6-flash` — check `client.models.list()` if you get a 404, Google deprecates model IDs |
| `OPENAI_API_KEY`, `EMBEDDING_MODEL`, `LLM_MODEL` | Fallback provider, used only if no Gemini key is configured |
| `LANGCHAIN_API_KEY`, `LANGCHAIN_TRACING_V2`, `LANGCHAIN_PROJECT` | Optional LangSmith tracing |

**No API key configured** → the app runs in a fully deterministic mock mode (rule-based interview questions and risk classification), useful for local frontend development without burning API quota.

## Ingesting guidelines

Either via the API:
```bash
curl -X POST http://localhost:8000/api/guidelines/ingest \
  -F "title=Hypertension Guideline" -F "version=1" -F "author=WHO" \
  -F "file=@/path/to/guideline.pdf"
```
or the CLI script:
```bash
uv run python scripts/ingest.py --file guideline.pdf --title "Hypertension Guideline" --author WHO
```

Both extract text page-by-page, chunk it (token-based sliding window), embed each chunk, and store it in `guideline_chunks` with page-range metadata used later for citations.

⚠️ **Free-tier embedding API quotas are easy to exhaust** on a large document (hundreds of chunks). `get_embeddings()` in `app/services/rag.py` batches requests and retries on rate limits, but on total failure it **raises** rather than silently storing zero-vectors — if ingestion fails, check the response; a `201 Created` with garbage embeddings is worse than a `500`.

## Scripts

| Script | Purpose |
|---|---|
| `scripts/ingest.py` | CLI guideline ingestion (see above) |
| `scripts/check_db.py` | Prints guideline/chunk counts and a content sample — sanity-check that ingestion actually populated real (non-zero) embeddings |
| `scripts/eval_retrieval.py` | Retrieval evaluation harness: labeled in-domain queries (hit rate, cross-guideline routing accuracy) + deliberately out-of-domain queries (confidence-threshold calibration). Re-run this after re-ingesting guidelines or changing the embedding model, and use its output to recalibrate `RELEVANCE_DISTANCE_THRESHOLD` / `CONFIDENCE_DISTANCE_FLOOR`/`CEIL` in `app/services/graph.py` |
| `scripts/test_agent.py` | Smoke-test the full LangGraph flow against a synthetic patient case |

## Notable design decisions

- **Split DB sessions around the graph call.** `create_session`/`send_message` in `agent.py` deliberately don't hold one DB connection open for the whole request — `graph.ainvoke()` can take anywhere from a few seconds to over a minute (Gemini calls + rate-limit backoff), and connection poolers (e.g. Supabase's) kill idle-in-transaction connections across a wait that long. DB access is split into short-lived sessions bracketing the slow graph call.
- **Confidence scores are calibrated, not guessed.** `distance_to_confidence()` in `graph.py` maps cosine distance to a 0–100% score via linear interpolation between two measured points from `eval_retrieval.py` (75th percentile of confirmed true-positive distances → 100%, 25th percentile of confirmed true-negative distances → 0%) — not an arbitrary formula.
- **Citations are code-generated, not LLM-generated.** The Sources list appended to every risk assessment is built directly from retrieved-chunk metadata, so it cannot hallucinate a source that wasn't actually retrieved.
- **Topic guard is heuristic, not an LLM call.** Given how often this app's embedding/LLM quota gets exhausted during testing, an off-topic message is rejected via keyword/pattern matching before it can reach any external API — trading a little recall for zero added latency/cost/quota risk.

## Deploying (Vercel)

`.python-version` is pinned to `3.12` — Vercel's `uv sync --frozen` build step needs a Python version it can actually resolve/download; `3.10` failed with "No interpreter found in managed installations." If you hit a similar error, check what versions Vercel's build image reports as available and re-pin accordingly, then run `uv lock` locally before redeploying.

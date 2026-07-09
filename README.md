# Event Extraction from Heterogeneous Websites

**Honours BSc Capstone Project — University of Ottawa (2025–2026)**

Automatic extraction of structured events (**name, date, time, location**) from
heterogeneous college fair web pages — no site-specific XPath rules, no
hardcoded selectors. A LightGBM node classifier (validated with strict
Leave-One-Site-Out cross-validation on 16 real-world sites) detects event
content, a structure-aware clustering groups nodes into events, and a
**confidence-gated LLM fallback** (GPT-4o-mini / GPT-4o, cost-aware) repairs
the clusters the ML layer distrusts. Shipped as a FastAPI service with
monitoring, Docker packaging, CI/CD and a Streamlit demo.

## Architecture

```
                 ┌──────────────────────────────────────────────────────────┐
                 │                    FastAPI  /extract                     │
                 └──────────────────────────────────────────────────────────┘
   raw HTML ──▶  1. html_parser      BeautifulSoup → DOM-node frame
                 │                   (same feature schema as training CSVs)
                 ▼
                 2. scoring          LightGBM (27 features) → raw score
                 │                   Platt calibrator → P(event node)
                 ▼
                 3. selection        adaptive top-K (30–100) + noise filter
                 │
                 ▼
                 4. clustering       PERIODIC: parent-atomic units →
                 │                   marker family (regular core) →
                 │                   recursive segmentation + rotation
                 │                   fallback: gap-based chain
                 ▼
                 5. assembly         cluster → {name,date,time,location}
                 │                   + confidence (method-aware)
                 ▼
              confidence ≥ 0.40 ──────────────▶ event  (method: "ml")
                 │
                 └─ confidence < 0.40 (suspect cluster)
                        │
                        ▼
                 6. LLM fallback     ≤40 nodes → gpt-4o-mini
                    (cost-capped)    >40 nodes → gpt-4o
                        │            JSON out, never invents values
                        ▼
                                     events (method: "llm_fallback")

   every request ──▶ monitoring (JSONL) ──▶ GET /metrics
```

## Strict Metric — Perfect Reconstitution

An event counts as "perfectly reconstituted" only if **all four** conditions hold:

1. **Detection** — at least one of its ground-truth nodes appears in the top-K candidates
2. **Same cluster** — all its detected nodes land in the same predicted cluster
3. **Exclusivity** — that cluster contains nodes from *only* this event (no mixing)
4. **Field coverage** — ≥33% of its Date, Time, and Location nodes are present in the cluster

## Metrics (measured, reproducible)

Leave-One-Site-Out over 16 sites (~200 events). Reproduce with
`python scripts/eval_fallback.py` (per-site table in `runs/fallback_eval.csv`)
and `python scripts/tune_threshold.py`.

| Metric | Value | Notes |
|--------|-------|-------|
| Event detection (≥1 node in top-K) | **99.5 %** | training benchmark (`src/dateclf/train.py`) |
| Perfect reconstitution — research pipeline | 62.2 % | used label-guided expansion (leaky, unavailable in production) |
| Perfect reconstitution — ML only, before periodic clustering | 49.8 % | leakage-free baseline |
| **Perfect reconstitution — ML only, current** | **67.3 %** | periodic clustering; beats the leaky research pipeline |
| LLM fallback uplift (measured at the 49.8 % baseline) | 49.8 % → 67.7 % | full-benchmark cost a few cents; escalated 72.7 % of events |
| Confidence gate @ 0.40 (current default) | escalates **11 %** of events | catches 26 % of remaining ML failures for 4 % waste |

Per-site wins from periodic clustering: wacac 0 → 100 %, neacac_1 20 → 100 %,
sacac_column 0 → 77 %, neacac_fall 60 → 80 %, iacac stays 100 % with 42→1
escalations. Remaining failures are structural: `members.sacac.org_pattern`
is the same physical page as `_column` with a **contradictory labeling**
(the date chips are attached to the previous event in one file, the next in
the other — no single rotation can satisfy both); gpacac has one DOM parent
containing several events (breaks parent-atomicity); outofstate and
neacac_spring are genuinely aperiodic (variable event length) — exactly the
cases the LLM fallback exists for. Re-run `eval_fallback.py` with
`OPENAI_API_KEY` to measure the post-optimization after-rate.

## Why this architecture

- **Detection is a solved problem here** (99.5 % of events have ≥1 node in
  top-K); **clustering was the bottleneck**. The LLM is therefore applied
  *only* to the clusters the ML layer distrusts, instead of sending whole
  pages to an LLM — cost scales with clustering failures, not with traffic.
- **Periodic clustering fixed the dense sites.** Diagnosis on the failing
  sites (sacac, wacac, neacac): event listings are strictly periodic — the
  same DOM-unit sequence repeats once per event — and the rendering gap
  *between* events equals the gap *inside* one, so gap heuristics cannot
  separate them; meanwhile parent groups almost never mix two events
  (6 impure / ~700 across 16 sites). The clustering therefore builds
  parent-atomic units, finds a once-per-event marker family (regular-core
  spacing, robust to interleaved noise), segments recursively
  (page ⊃ sections ⊃ events), disambiguates the rotation with a semantic
  prior (events start with their heading/name; the date sits in the first
  half), and validates that segments carry temporal signals (nav menus
  repeat too). Aperiodic pages fall back to the validated gap chain.
  Result: **49.8 % → 67.3 %** perfect reconstitution, pure ML.
- **Calibration makes the gate meaningful.** Raw LightGBM scores are distorted
  by undersampling + `scale_pos_weight`; Platt scaling is fitted on
  **out-of-fold LOSO scores**, so the confidence reflects performance on
  *unseen* sites — the production condition. Periodic clusters carry a
  high structural confidence instead (the grouping was proven, not scored).
- **Cost-aware orchestration.** Small batches go to `gpt-4o-mini`; dense pages
  (>40 candidate nodes) go to `gpt-4o` — exactly where both the heuristics and
  the small model degrade. A hard USD cap per request refuses escalation
  rather than overspending. Costs are computed from API-reported token usage.
- **Graceful degradation.** No API key / API error / cost cap hit → the
  service returns the ML events with their (low) confidence instead of failing.

## API

| Endpoint   | Method | Description |
|------------|--------|-------------|
| `/extract` | POST   | `{html, site?, use_llm_fallback?}` → structured events |
| `/health`  | GET    | liveness + model/LLM status (used by Docker & Render) |
| `/metrics` | GET    | aggregated monitoring stats |

```bash
curl -X POST http://localhost:8000/extract \
  -H "Content-Type: application/json" \
  -d '{"html": "<html>...</html>", "site": "wacac.org"}'
```

```json
{
  "events": [
    {
      "name": "Sacramento College Fair",
      "date": "March 14, 2025",
      "time": "6:00 pm - 8:30 pm",
      "location": "Sacramento Convention Center",
      "confidence_score": 0.87,
      "extraction_method": "ml"
    }
  ],
  "extraction_method": "ml",
  "n_nodes": 214, "n_candidates": 42,
  "llm_cost_usd": 0.0, "latency_ms": 145.2
}
```

`extraction_method` per event and per response: `"ml"`, `"llm_fallback"`, or
`"hybrid"` (response-level, when both paths produced events) — full
traceability of what produced each field.

## Monitoring

Every request is appended to a JSONL log (`MONITORING_PATH`): timestamp, site,
node count, extraction method, LLM model + cost, latency, event count, errors.
`GET /metrics` aggregates it (totals, method breakdown, per-site counts,
average latency). JSONL over SQLite: append-only immutable facts, single
writer, trivially shippable to any log stack.

## Project structure

```
app/                     production layer
  html_parser.py         raw HTML → node frame (training schema)
  pipeline.py            score → top-K → cluster → assemble → gate
  clustering.py          periodic clustering + gap-chain fallback
  calibration.py         Platt scaling (fitted on OOF LOSO scores)
  llm_fallback.py        cost-aware GPT-4o(-mini) orchestration
  monitoring.py          JSONL request log + aggregation
  main.py                FastAPI app (/extract /health /metrics)
  config.py, schemas.py  env-driven settings, Pydantic contracts
src/dateclf/             research pipeline (LOSO training, EDA)
  data.py                loading, label normalization, feature matrices
  split.py               LOSO cross-validation folds
  features.py            DOM-neighborhood feature engineering
  sampling.py            negative undersampling
  metrics.py             node-level and event-level metrics
  train.py               research benchmark: train, cluster, evaluate
scripts/
  train_production.py    trains model.txt + calibrator + feature importance
  eval_fallback.py       before/after LLM-fallback benchmark
  tune_threshold.py      offline sweep of the confidence gate (zero LLM cost)
models/                  committed artifacts served by the API
data/raw/                16 site CSVs (not tracked in git)
tests/                   41 tests: parser, clustering, fallback, pipeline, API
streamlit_app.py         demo UI (local pipeline or deployed API)
Dockerfile, render.yaml, .github/workflows/ci.yml
```

## Data format

Each site is a CSV where every row represents one DOM node from the parsed HTML page:

| Column | Description |
|--------|-------------|
| `rendering_order` | DFS traversal order of the node in the DOM |
| `tag` | HTML tag (Span, Div, A, H5, P, etc.) |
| `attributes` | Raw CSS classes, IDs, and other attributes |
| `text_context` | Text content of the node |
| `depth` | Depth in the DOM tree |
| `parent_index` | `rendering_order` of the parent node |
| `parent_tag` | HTML tag of the parent |
| `text_length`, `word_count` | Text statistics |
| `letter_ratio`, `digit_ratio`, `whitespace_ratio` | Character composition |
| `contains_date`, `contains_time` | Boolean date/time pattern detection |
| `has_class`, `has_id` | Whether node has CSS class/ID |
| `attr_has_word_date/time/location` | Keyword detection in attributes |
| `label` | Ground-truth label (Date, Time, StartEndTime, Location, Name, Description, Other) |
| `event_id` | Ground-truth event assignment (NaN for non-event nodes) |

The production `app/html_parser.py` reproduces this schema from raw HTML so
the same model scores live pages.

## Features (27 total)

**Structural** — `depth`, `tag`, `parent_tag`, `num_siblings`, `has_class`, `has_id`

**Textual** — `text_length`, `word_count`, `letter_ratio`, `digit_ratio`, `whitespace_ratio`, `contains_date`, `contains_time`, `starts_with_digit`, `ends_with_digit`

**Contextual** — `attr_has_word_date/time/location`, `text_has_word_date/time/location`, `prev_contains_date`, `next_contains_date`, `prev_contains_time`, `next_contains_time`, `same_parent_as_prev`, `same_parent_as_next`

## Key findings

### Two types of sites

Exploratory analysis revealed the 16 sites fall into two fundamentally different HTML patterns:

| Type | Sites | Structure | Clustering signal |
|------|-------|-----------|------------------|
| **Row-per-event** | iacac, outofstate, pnacac, rmacac | Each `<tr>` = 1 event | Clear parent_index gap separation |
| **Card-based** | sacac, nacacnet, neacac, wacac, gpacac, hawaii | Nested div cards | Overlapping gaps, needs different strategy |

This structural heterogeneity is the core challenge — no single gap threshold
works for all sites. Periodic clustering resolves it by detecting each page's
own repetition structure instead of applying a global threshold.

### The rotation (phase) problem

A periodic sequence [ABCDE][ABCDE]… is structurally identical under rotation;
only semantics pin the event start (heading/name first, date in the first
half). The two sacac labelings of the *same page* disagree on the rotation —
proof the residual ambiguity is in the ground truth, not the algorithm.

### BIO sequence labeling (explored)

An alternative approach treating DOM nodes as a sequence (B-Event / I-Event / O tags) was validated via EDA:
- 0 event overlaps across all tested sites
- 0 revisits (strictly sequential)
- 95% of intra-event gaps ≤ 3 nodes

BIO achieved **100% on neacac sites** but struggled with boundary detection
(B label = only 6.2% of nodes). The approach is promising as an auxiliary
signal combined with the main pipeline.

## Run locally

```bash
pip install -r requirements.txt -r requirements-api.txt -r requirements-dev.txt

# research benchmark (16-fold LOSO)
cd src && python -m dateclf.train

# (re)train production artifacts — requires data/raw/*.csv
python scripts/train_production.py

# API
uvicorn app.main:app --reload          # http://localhost:8000/docs

# demo UI
streamlit run streamlit_app.py         # local pipeline
API_URL=https://<service>.onrender.com streamlit run streamlit_app.py  # via API

# tests
python -m pytest tests/ -v
```

Set `OPENAI_API_KEY` (see `.env.example`) to enable the LLM fallback; without
it the service runs pure-ML.

## Deployment (Docker + Render + GitHub Actions)

```bash
docker build -t event-extraction-api .
docker run -p 8000:8000 -e OPENAI_API_KEY=sk-... event-extraction-api
```

CI/CD (`.github/workflows/ci.yml`), on every push to `main`:
1. **test** — pytest against the committed model artifacts
2. **docker** — build the image and smoke-test `/health` inside the container
3. **deploy** — trigger the Render deploy hook, then poll `/health` until green

One-time setup:
- create the service on Render from `render.yaml` (Blueprint) or point a
  Docker web service at this repo
- set `OPENAI_API_KEY` in the Render dashboard
- add the repo secret `RENDER_DEPLOY_HOOK_URL` (and optionally the repo
  variable `RENDER_SERVICE_URL` for the post-deploy health poll)

## Configuration

| Variable | Default | Effect |
|----------|---------|--------|
| `OPENAI_API_KEY` | — | empty → LLM fallback disabled |
| `LLM_CONFIDENCE_THRESHOLD` | `0.40` | gate below which clusters escalate (benchmark knee) |
| `LLM_COMPLEXITY_NODE_THRESHOLD` | `40` | nodes above which gpt-4o is used |
| `LLM_MAX_COST_PER_REQUEST` | `0.05` | hard USD cap; refuses escalation past it |
| `MONITORING_PATH` | `data/monitoring/requests.jsonl` | request log location |
| `MODEL_DIR` | `models/` | artifact directory |

## Interview crib notes

- **Why not send everything to the LLM?** 99.5 % detection means the ML layer
  already finds the content; the LLM only arbitrates grouping on the clusters
  the heuristics distrust. Cost scales with failures, not traffic.
- **Why Platt and not isotonic?** ~3 k nodes of OOF scores; isotonic overfits
  small samples, Platt is 2 parameters.
- **Why one batched LLM call, not per-cluster calls?** One call carrying all
  suspect nodes lets the LLM re-decide grouping globally (its whole job) and
  pays one prompt overhead instead of N.
- **Why is `label` never used at inference?** The research clustering expanded
  clusters using ground-truth labels; `app/clustering.py` replaces that with
  pattern features and segment-range expansion. The leakage-free pipeline now
  *beats* the leaky research one (67.3 % vs 62.2 %).
- **How does periodic clustering avoid false positives?** Four guards: a
  marker needs ≥4 evenly-spaced occurrences (3 happen by chance), spacing is
  measured on the family's regular core (noise-robust), median (not mean)
  cut gaps resist pool-sampling outliers, and segments must carry temporal
  signals — nav menus are periodic too, but dateless.
- **Confidence is method-aware**: periodic clusters carry high structural
  confidence (their grouping was validated); gap-chain clusters fall back to
  calibrated node probability × penalties (≥3 dates, mega-cluster, missing
  fields). This cut gate escalation from 81 % of events to 11 % at equal
  failure coverage.

## Authors

**Jad Kebal** — Computer Science Honours BSc, University of Ottawa.

**Mohamed Amine Ablouh** — Computer Science Honours BSc, University of Ottawa.

**Darine Fourar Laidi** — Computer Science Honours BSc, University of Ottawa.

Supervised capstone project — Applied ML/AI specialization with emphasis on NLP and web information extraction.

## License

Academic project — University of Ottawa, 2025–2026.

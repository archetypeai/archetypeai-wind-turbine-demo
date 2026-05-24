# Penmanshiel · live turbine anomaly monitor

A side-by-side live demo of [Archetype AI's Newton Machine State Lens](https://www.archetypeai.io/) classifying real wind-turbine SCADA telemetry. Three months of data from the Penmanshiel wind farm are replayed at hourly cadence in the browser; Newton classifies each ~21-hour window as `healthy` or `fault` against a pair of n-shot reference windows; the verdicts drive the per-turbine state, the anomaly feed, and the SVG blade colours in real time.

**All inference — every state badge, every anomaly feed entry, every blade colour — comes from Newton.** There is no local heuristic; the Flask backend is a thin replay-streamer plus a Newton multiplexer.

The dataset features a documented frequency-converter outage on **WT01** in early November 2019. Its healthy peer **WT09** runs the same hardware on the same hill in the same minute-by-minute wind. The demo invites you to watch Newton find the difference.

![demo overview](docs/screenshot.png) <!-- optional; safe if missing -->

## What the app does

- **Replays** 2,184 hourly SCADA ticks (2019-09-01 → 2019-12-01) for WT01 + WT09 over an SSE stream, paced by a Slow/Normal/Fast control.
- **Pushes channel-first 128-sample windows** to a single Newton Machine State Lens session, multiplexed across both turbines (see [Architecture](#architecture-newton-lens-orchestration)).
- **Streams Newton verdicts back** as `newton_prediction` events on the same SSE stream. ~204 windows total drain through Newton at ~1/s; predictions arrive over ~3.5 minutes of wall-clock time regardless of replay speed.
- **Holds the timeline** until Newton produces its first non-unknown verdict for at least one turbine (~30s warm-up), so the user never sees empty panels.
- **Emits debounced anomaly events** on strong-majority verdict transitions (`fault` → `healthy` and back) — see [Anomaly logic](#anomaly-logic) for the why.
- **Renders the dashboard** in the Archetype AI design system (Geist sans + Geist Mono, OKLCH palette, 2 px radii, FlatLogItem-style feed). Following [`archetypeai-agent-skills/DESIGN.md`](https://github.com/archetypeai/archetypeai-agent-skills/blob/main/DESIGN.md) — see [`templates/index.html`](templates/index.html) and [`static/style.css`](static/style.css).

## Dataset

Loaded from `./data/`. The full Penmanshiel release (~2 GB, not committed) is published by Cubico Sustainable Investments on Zenodo via Greenbyte — 14 turbines on the Scottish-Borders site, 2019 calendar-year SCADA at 10-minute cadence, status logs, grid-meter time-series, and PMU readings.

The demo only needs:
- `data/Penmanshiel_WT_static.csv` — turbine metadata (Senvion MM82, 2,050 kW rated, 82 m rotor)
- `data/Penmanshiel_SCADA_2019_WT01-10_3112/Turbine_Data_Penmanshiel_01_*.csv` — WT01 full-year 10-minute SCADA
- `data/Penmanshiel_SCADA_2019_WT01-10_3112/Turbine_Data_Penmanshiel_09_*.csv` — WT09 full-year 10-minute SCADA

The replay window (Sept 1 → Dec 1, ~13 k rows per turbine) is a slice of those CSVs; the n-shot reference windows live elsewhere in the same year:

| Class | Source | Window | Rows |
|---|---|---|---|
| `healthy` | WT09 | 2019-07-01 → 2019-07-22 | ~3,024 |
| `fault` | WT01 | 2019-10-27 → 2019-11-13 | ~2,448 |

Both reference windows are single contiguous blocks (per the [`newton-machine-state` "contiguous + z-scored" requirement](https://github.com/archetypeai/archetypeai-agent-skills/blob/main/skills/newton-machine-state/SKILL.md)), z-scored per channel with a global scaler fit on the union, then uploaded once per process.

### Fetching the data

The dataset lives on Zenodo — either [record 16807304](https://zenodo.org/records/16807304) (newer; also mirrored on [HLRS WindLab](https://windlab.hlrs.de/dataset/zenodo-16807304/resource/b16ea689-f8ca-4873-bf19-81110daf191c)) or [record 5946808](https://zenodo.org/records/5946808) (the original release). Both contain the same WT01–15 Penmanshiel SCADA at 10-minute cadence; the filenames and directory layout below are identical between the two — substitute either record number into the URLs. License: CC-BY-4.0.

You need **two files** for this demo — the SCADA zip that contains WT01 and WT09, plus the static metadata:

```bash
mkdir -p data && cd data

# WT01-10 2019 SCADA (~1.9 GB zipped, includes WT01 + WT09)
wget https://zenodo.org/api/records/16807304/files/Penmanshiel_SCADA_2019_WT01-10_3112.zip/content \
    -O Penmanshiel_SCADA_2019_WT01-10_3112.zip
unzip Penmanshiel_SCADA_2019_WT01-10_3112.zip
rm Penmanshiel_SCADA_2019_WT01-10_3112.zip

# Static metadata (rated power, hub height, lat/long, etc.)
wget https://zenodo.org/api/records/16807304/files/Penmanshiel_WT_static.csv/content \
    -O Penmanshiel_WT_static.csv

cd ..
```

After extraction your `data/` directory should look like:

```
data/
├── Penmanshiel_WT_static.csv
└── Penmanshiel_SCADA_2019_WT01-10_3112/
    ├── Turbine_Data_Penmanshiel_01_2019-01-01_-_2020-01-01_1075.csv  # WT01 (faulty)
    ├── Turbine_Data_Penmanshiel_09_2019-01-01_-_2020-01-01_1075.csv  # WT09 (healthy peer)
    ├── Turbine_Data_Penmanshiel_02_2019-01-01_-_2020-01-01_1075.csv
    ├── Status_Penmanshiel_01_…csv
    └── … (other turbines and status logs, not used by the demo)
```

The demo only reads the **WT01** and **WT09** turbine-data CSVs plus the static metadata. The other files in the zip (WT02, WT04–08, WT10, status logs) are harmless extras — `data_loader.discover_turbines()` enumerates whatever is present.

**Optional extras**, not needed for the demo:
- `Penmanshiel_SCADA_2019_WT11-15_3117.zip` — WT11–15 2019 SCADA (drop in to enable other turbines via the picker, if you wire one up — see Future improvements).
- `Penmanshiel_WT_dataSignalMapping.xlsx` — column-name reference for the SCADA exports.

## Setup

Requires Python 3.11+ (Archetype AI SDK needs ≥ 3.10), Archetype AI API credentials for staging or prod, and the Penmanshiel dataset.

```bash
# Clone
git clone https://github.com/archetypeai/archetypeai-wind-turbine-demo.git
cd archetypeai-wind-turbine-demo

# Fetch the dataset (see above) into ./data/

# Credentials
cp .env.example .env
# Edit .env with your ATAI_API_KEY and ATAI_API_ENDPOINT

# Virtual env
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Run
.venv/bin/python app.py
open http://127.0.0.1:5050/
```

The page auto-connects on load. Expect ~30 s of warm-up (Newton lens registration + focus CSV upload + first inference) before the timeline starts.

## Using the UI

- **Top bar**: Archetype AI wordmark · title · live `Newton starting…` / `Newton analysing` / `Newton complete` status pill · streaming-state dot · **Slow / Normal / Fast** speed picker.
- **Timeline strip**: simulated date range with a cursor that advances per tick; `t-now` mono readout on the right.
- **Per-turbine panel** (left and right of the feed):
  - `ANALYSING` / `HEALTHY` / `FAULT` state badge (driven by Newton's *debounced* state).
  - `NEWTON {class} {H/F · X/Y}` verdict badge — updates the vote counter on every prediction, but the colour/pulse only flips on strong-majority verdicts (filters out close 3-2 KNN ties).
  - Animated SVG turbine — rotor spin tracks measured RPM; blade colour mirrors the panel state.
  - Numeric stat row (Power / Wind / Rotor RPM / Pitch / Gear oil) in mono with current-window values.
  - Rolling 96-tick power + wind sparkline.
- **Anomaly feed** (centre column): strong-majority state transitions ("Detected: fault classification", "Recovered: now healthy") with the window range and timestamp. Single 3-2 KNN flickers are intentionally suppressed.

## Architecture: Newton lens orchestration

The demo started life as a textbook single-session-per-turbine setup and converged through three working architectures before settling on the current one. The journey produced four PRs against [`archetypeai-agent-skills`](https://github.com/archetypeai/archetypeai-agent-skills) — see [Lessons learned](#lessons-learned-skill-prs).

### Current shape: single shared session with multiplexed pushes

```
                ┌──────────────────────────────────────────────────┐
                │             Archetype AI Newton                  │
                │     ┌────────────────────────────────────────┐   │
                │     │  Child lens: penmanshiel-turbines-…    │   │
                │     │  model: omega_embeddings_1_4           │   │
                │     │  n-shot focus: {healthy, fault}        │   │
                │     │  window=128, step=128, knn=5           │   │
                │     └────────────────────────────────────────┘   │
                │     ┌────────────────────────────────────────┐   │
                │     │  Session (one runner slot)             │   │
                │     │  rate: ~1 inference/s                  │   │
                │     └────────────┬───────────────────────────┘   │
                │                  │  SSE                          │
                └──────────────────┼───────────────────────────────┘
                                   │
   push WT01,w0 ──┐                │  inference.result (fault)
   push WT09,w0 ──┤  session.update │  inference.result (healthy)
   push WT01,w1 ──┤  channel-first  │  inference.result (healthy)
   push WT09,w1 ──┤  events 1/s     │  inference.result (fault)
   …             ─┘                │  …
                                   │
   ┌───────────────────────────────┴────────────────────────────────┐
   │  Flask backend (newton_client.MultiplexNewtonSession)          │
   │  FIFO push-tag queue routes each result back to its turbine    │
   │  by the order in which it was pushed.                          │
   └────────────────────────────────────────────────────────────────┘
```

### Why one lens, one session, multiplexed

Three things bit us in sequence; the architecture is the residual after fixing each:

1. **Using the platform-mounted Machine State Lens directly produced silent runs.** The hosted lens (`lns-1d519091822706e2-…`) is pinned to `omega_embeddings_01`. Sessions accepted every config event with `is_valid: true`, reached `SESSION_STATUS_RUNNING`, and then emitted zero `inference.result` events. Fix: register a child lens with `model_parameters.model_version = "OmegaEncoder::omega_embeddings_1_4"` and all the n-shot / csv / knn / output-streams config inline, not split between `lens/register` and `session.modify`.
2. **One lens, two sessions silently dropped one of them.** Both sessions reached `RUNNING`, both consumed pushes at the same pace, but only one ever emitted results. Switching to *one lens per turbine* eliminated the silence (each session got its own runner). See the "One lens per stream — even when the column set is identical" section in [`newton-machine-state` SKILL.md](https://github.com/archetypeai/archetypeai-agent-skills/blob/main/skills/newton-machine-state/SKILL.md).
3. **The account's runner pool is quota=1.** Two separate lenses didn't help: the second `POST /lens/sessions/create` returns `"Failed to allocate lens runner — try stopping an older session!"` regardless of waiting or cooldown. Newton's runner-pool quota for this account is one concurrent session. Fix: collapse back to one lens / one session and *multiplex* — interleave both turbines' window pushes into a single push queue, tag each push with `(stream_id, window_index)`, and FIFO-route incoming `inference.result` events back to their stream.

The full code shape lives in [`newton_client.py`](newton_client.py) → `MultiplexNewtonSession`.

### Other production-shape choices

- **Push pacing: `MIN_PUSH_INTERVAL_SEC = 1.0` per session.** Pushing 102 windows in one burst yields exactly 20 predictions then silence — Newton's lens-runner has a ~20-window buffer depth. Pacing at ≥1 s per push keeps the buffer drained and predictions stream cleanly.
- **Warm-up gate: hold ticks until first non-unknown verdict.** Documented in `app.py:_stream_body`. The user never sees an empty timeline + stale panels — predictions land first, then the timeline starts. After the first verdict lands per turbine, ticks stream at the chosen `tps`.
- **Per-request orphan-session cleanup at `/api/replay` entry.** Killed Python processes, abandoned curl `--max-time` connections, and browser tab closes all leave platform-side sessions running. We call `cleanup_orphan_sessions()` at every request start to reap them; the SSE generator body is wrapped in `try/finally` so our own auto-destroy always fires.
- **Channel-first `session.update` push, never `csv_file_reader`.** File-reader mode is silently broken on this lens — see [`newton-machine-state` Step 6](https://github.com/archetypeai/archetypeai-agent-skills/blob/main/skills/newton-machine-state/SKILL.md).

## Anomaly logic

Newton's KNN on a small n-shot library (~15–20 reference windows per class) returns **3-2 votes ~60 % of the time** on this dataset — the lens genuinely can't separate borderline windows. The demo treats those as noise.

The rule that survived experimentation:

1. **Drop weak verdicts.** A verdict is "strong" when the winning class has ≥ 3 more votes than the runner-up (so with `n_neighbors = 5`: 4-1 or 5-0 splits qualify; 3-2 doesn't). Weak verdicts don't change the panel state, don't fire anomaly entries, and don't repaint the per-panel verdict badge — only the vote counter updates so users can see total throughput.
2. **Commit on the first strong verdict in a new direction.** No further consecutive requirement; strong verdicts are scarce enough on this dataset (~30 % of predictions) that requiring two in a row reintroduces false negatives.

Trade-off accepted: a single strong-but-wrong verdict on the healthy turbine can produce a detect/recover pair. The honest fix is upstream — larger n-shot library, better encoder separation — not in the debounce logic. The per-panel verdict badge surfaces vote counts so users can see *why* each decision fired.

The two rules we tried and rejected (with measurements) are documented in the [`newton-machine-state` SKILL.md](https://github.com/archetypeai/archetypeai-agent-skills/blob/main/skills/newton-machine-state/SKILL.md) — Anomaly debounce when multiplexing.

## API surface

| Endpoint | What it returns |
|---|---|
| `GET /` | The dashboard UI |
| `GET /api/scada/<wt_id>` | Downsampled 3-month SCADA series for a single turbine (JSON, debug aid) |
| `GET /api/replay?tps=N` | SSE stream: `meta`, `newton_status`, `tick`, `newton_prediction`, `anomaly`, `newton_done`, `complete` |

`tps` clamps to `[1, 200]`. The replay duration scales linearly: tps = 15 → 145 s replay, tps = 40 → 55 s, tps = 120 → 18 s. Newton's prediction stream takes ~204 s regardless of `tps`, so faster replays leave a longer tail of predictions arriving after the visible timeline completes.

## Project layout

```
app.py                         # Flask + SSE; tick loop; debounce + anomaly synth
newton_client.py               # ArchetypeAI SDK wrapper:
                               #   build_client, ensure_scaler, ensure_focus_files,
                               #   ensure_child_lens (per-stream cache),
                               #   cleanup_orphan_sessions,
                               #   NewtonSession (one-stream),
                               #   MultiplexNewtonSession (current demo's path),
                               #   replay_events (pure SCADA tick streamer)
data_loader.py                 # CSV reader, window slicing, hourly downsampling
templates/index.html           # Topbar + timeline + 3-column grid
templates/_turbine_card.html   # Reusable panel: state badge, SVG, stats, spark
templates/_wordmark.svg        # Archetype AI wordmark (inlined for currentColor)
static/style.css               # OKLCH tokens, Geist fonts, sharp radii, FlatLogItem feed
static/app.js                  # SSE consumer, panel updates, rotor RAF loop, Chart.js spark
requirements.txt
.env.example
LICENSE                        # Apache-2.0
```

## Lessons learned (skill PRs)

Each architectural pivot landed back into the [`archetypeai-agent-skills`](https://github.com/archetypeai/archetypeai-agent-skills) repo as a documentation PR so the next person rebuilding this app doesn't rediscover them:

- [PR #22](https://github.com/archetypeai/archetypeai-agent-skills/pull/22) — Push mode + the four silent-inference failure modes (model_version, csv_file_reader, push-rate, config-location).
- [PR #23](https://github.com/archetypeai/archetypeai-agent-skills/pull/23) — Warm-up priming requirements; per-request orphan-session cleanup.
- [PR #24](https://github.com/archetypeai/archetypeai-agent-skills/pull/24) — One lens per stream, even when the column set is identical.
- [PR #25](https://github.com/archetypeai/archetypeai-agent-skills/pull/25) — Account-quota multiplex pattern; framework-agnostic wordmark inlining; anomaly debounce when multiplexing.

## Credits

- **SCADA data**: Cubico Sustainable Investments, Penmanshiel wind farm. Zenodo records [16807304](https://zenodo.org/records/16807304) (newer) and [5946808](https://zenodo.org/records/5946808) (original); HLRS WindLab [mirror](https://windlab.hlrs.de/dataset/zenodo-16807304/resource/b16ea689-f8ca-4873-bf19-81110daf191c). CC-BY-4.0.
- **Inference**: Archetype AI Newton Machine State Lens (`lens_timeseries_state_processor` + Omega 1.4 encoder).
- **Visual design**: [Archetype AI design system](https://github.com/archetypeai/archetypeai-agent-skills/blob/main/DESIGN.md) — Geist + Geist Mono, OKLCH palette, sharp 2 px radii.

## License

Apache-2.0 — see [LICENSE](LICENSE).

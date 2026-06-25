"""Penmanshiel turbine anomaly detection via Newton's Direct Query API.

No lenses, no sessions, no SSE plumbing: each window is classified with the
`atai-newton-omega-model` pattern — per-channel Omega embeddings over `/query`,
scored by a local KNN against an n-shot library of reference windows.

- `replay_events()` streams raw SCADA ticks (no inference).
- `BackgroundClassifier` classifies the demo turbines' windows in a background
  thread and exposes the results as `newton_prediction` events; `app.py` drains
  them and paces them onto the replay timeline.

Leakage-free split (training references are disjoint from what we classify):
  - healthy references = WT09 Jul 2019 (summer) + WT06 Oct 2019 (autumn), so the
                         healthy class matches both seasons of the live window
  - fault   reference  = WT05's sustained ~36h Mar 2019 frequency-converter
                         outage — a *different* turbine, so WT01's November fault
                         is never in the library (genuine cross-turbine detection)
  - live / classified  = WT01 + WT09, Sep-Dec 2019
`_build_library()` asserts the references share no (turbine, time-range) with the
live playback before building, so the split can't silently regress.
"""
from __future__ import annotations

import json
import logging
import os
import queue
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Generator

import pandas as pd
from archetypeai.api_client import ArchetypeAI

from data_loader import discover_turbines, load_turbine_window

logger = logging.getLogger(__name__)

OMEGA_MODEL = "OmegaEncoder::omega_embeddings_1_4"

# Leakage-free n-shot references (see module docstring). Multiple windows per
# class. The `fault` class is a DIFFERENT turbine (WT05) than the ones we
# classify (WT01/WT09), so WT01's November event is unseen — detecting it is
# genuine cross-turbine generalization. `healthy` spans both summer (WT09 Jul)
# and autumn (WT06 Oct) so it matches the autumn live window's season; without
# the autumn reference, healthy-autumn windows mis-classify as fault.
REFERENCE_WINDOWS: dict[str, list[dict]] = {
    "healthy": [
        {"wt": "09", "start": "2019-07-01 00:00:00", "end": "2019-07-22 00:00:00"},
        {"wt": "06", "start": "2019-10-01 00:00:00", "end": "2019-10-22 00:00:00"},
    ],
    # The WT05 frequency-converter outage is intermittent across March; this is
    # the single sustained ~36h stopped-in-wind run, so the windows are
    # fault-dominated (not contaminated with normal operation).
    "fault": [
        {"wt": "05", "start": "2019-03-08 06:00:00", "end": "2019-03-10 00:00:00"},
    ],
}

# Hardcoded scenario for the demo.
DEMO_WT_A = "01"  # frequency converter fault on 2019-11-02
DEMO_WT_B = "09"  # healthy peer over the same period
DEMO_START = "2019-09-01 00:00:00"
DEMO_END = "2019-12-01 00:00:00"  # 3 months

# Four signal columns we embed.
FEATURE_COLUMNS: list[tuple[str, str]] = [
    ("a1", "Wind speed (m/s)"),
    ("a2", "Power (kW)"),
    ("a3", "Blade angle (pitch position) A (°)"),
    ("a4", "Gear oil temperature (°C)"),
]

# Extra columns for the live UI (rotor speed drives blade animation, nacelle
# position drives the yaw indicator). Not part of the Newton payload.
RICH_COLUMNS: list[tuple[str, str]] = [
    *FEATURE_COLUMNS,
    ("rpm", "Rotor speed (RPM)"),
    ("yaw", "Nacelle position (°)"),
    ("wind_dir", "Wind direction (°)"),
]

# Direct Query Omega + local KNN windowing. One window length for the n-shot
# library and the turbines under test so embeddings are comparable.
EMBED_WINDOW = 128    # rows per window (~21h at 10-min cadence)
HEALTHY_STEP = 96     # library: overlapping healthy windows
FAULT_STEP = 16       # library: dense windows over the short sustained outage
HEALTHY_CAP = 18      # cap healthy windows to balance against the (few) fault windows
DATA_STEP = 256       # live: one verdict per ~1.8 days (~50 per 3-month turbine)
KNN_K = 5
OMEGA_MAX_CONCURRENCY = 6   # bounded per-channel fan-out (the skill's thread-pool pattern)
OMEGA_RETRIES = 3
OMEGA_TIMEOUT = 30

# Precomputed n-shot library (atai-newton-omega-model-data-prep pattern): the
# reference set is static, so embed it offline once with `python build_library.py`
# and load the (scaler + vectors) from disk at runtime — no /query calls or
# "Building reference library…" wait on cold start. Falls back to a live build
# if the file is missing or its config fingerprint no longer matches.
LIBRARY_PATH = os.path.join(os.path.dirname(__file__), "library.json")


@dataclass
class TurbineRequest:
    wt_id: str
    start: str = DEMO_START
    end: str = DEMO_END


def _resolve_endpoint(raw: str) -> str:
    base = (raw or "").rstrip("/")
    if not base:
        raise RuntimeError("ATAI_API_ENDPOINT is not set")
    last_seg = base.rsplit("/", 1)[-1]
    if not last_seg.startswith("v"):
        base = base + "/v0.5"
    return base


def window_to_records(wt_id: str, start: str = DEMO_START, end: str = DEMO_END) -> list[dict]:
    """Sample the turbine window down to a UI-friendly list of dicts (timestamp + signals)."""
    turbines = {t.wt_id: t for t in discover_turbines()}
    if wt_id not in turbines:
        return []
    window = load_turbine_window(turbines[wt_id], start, end, max_rows=2200)
    ts = pd.to_datetime(window["Date and time"])
    rows: list[dict] = []
    for i in range(len(window)):
        rec: dict = {"ts": ts.iloc[i].strftime("%Y-%m-%d %H:%M")}
        for short, full in RICH_COLUMNS:
            v = window[full].iloc[i]
            rec[short] = None if pd.isna(v) else round(float(v), 2)
        rows.append(rec)
    return rows


def _load_hourly_series(start: str, end: str) -> tuple[list[str], dict[str, list[dict]], dict[str, float]]:
    """Load both demo turbines at hourly cadence and return aligned tick lists."""
    turbines = {t.wt_id: t for t in discover_turbines() if t.wt_id in (DEMO_WT_A, DEMO_WT_B)}
    rated: dict[str, float] = {wt: float(t.rated_power_kw) for wt, t in turbines.items()}

    # Hourly = every 6th 10-minute row.
    full = {wt: load_turbine_window(t, start, end, max_rows=20_000) for wt, t in turbines.items()}
    hourly = {wt: df.iloc[::6].reset_index(drop=True) for wt, df in full.items()}

    n = min(len(df) for df in hourly.values())
    timestamps = [
        pd.to_datetime(hourly[DEMO_WT_A]["Date and time"]).iloc[i].strftime("%Y-%m-%d %H:%M")
        for i in range(n)
    ]
    per_wt: dict[str, list[dict]] = {}
    for wt, df in hourly.items():
        rows: list[dict] = []
        for i in range(n):
            rec: dict = {}
            for short, full_col in RICH_COLUMNS:
                v = df[full_col].iloc[i]
                rec[short] = None if pd.isna(v) else round(float(v), 2)
            rows.append(rec)
        per_wt[wt] = rows
    return timestamps, per_wt, rated


def replay_events(
    start: str = DEMO_START,
    end: str = DEMO_END,
) -> Generator[dict, None, None]:
    """Yield raw replay ticks (meta + ticks + done). No classification.

    All anomaly/state inference comes from Newton via the BackgroundClassifier.
    """
    timestamps, per_wt, rated = _load_hourly_series(start, end)
    n = len(timestamps)
    yield {
        "kind": "meta",
        "start": timestamps[0] if n else start,
        "end": timestamps[-1] if n else end,
        "total_ticks": n,
        "turbines": {wt: {"rated_kw": rated[wt]} for wt in per_wt},
    }
    for i, ts in enumerate(timestamps):
        tick: dict = {"kind": "tick", "i": i, "ts": ts, "progress": (i + 1) / n if n else 1.0, "wts": {}}
        for wt, rows in per_wt.items():
            row = rows[i]
            tick["wts"][wt] = {
                "wind": row.get("a1") or 0.0,
                "power": row.get("a2") or 0.0,
                "pitch": row.get("a3") or 0.0,
                "gear_c": row.get("a4"),
                "rpm": row.get("rpm") or 0.0,
                "yaw": row.get("yaw"),
                "wind_dir": row.get("wind_dir"),
            }
        yield tick
    yield {"kind": "done"}


# ---------- Data prep: feature frame + per-channel scaler ----------

def _feature_frame(wt_info, start: str, end: str) -> pd.DataFrame:
    """Return a DataFrame with timestamp + columns a1..a4 (NaN-filled), full cadence."""
    window = load_turbine_window(wt_info, start, end, max_rows=20_000)
    out = pd.DataFrame({"timestamp": pd.to_datetime(window["Date and time"]).astype("int64") // 10**9})
    for short, full in FEATURE_COLUMNS:
        out[short] = pd.to_numeric(window[full], errors="coerce").fillna(0.0)
    return out


# Per-process scaler fit on the n-shot reference pool, applied to every window.
# Pre-normalizing with fixed stats and passing the raw (unnormalized) window to
# Omega preserves cross-window amplitude — per-window normalization would erase
# it (a low-power and a high-power window would look identical). This is the
# atai-newton-omega-model skill's recommended downstream pattern.
_scaler_lock = threading.Lock()
_scaler: dict[str, dict[str, float]] | None = None


def ensure_scaler() -> dict[str, dict[str, float]]:
    global _scaler
    with _scaler_lock:
        if _scaler is not None:
            return _scaler
        turbines = {t.wt_id: t for t in discover_turbines()}
        frames: list[pd.DataFrame] = []
        for cls, specs in REFERENCE_WINDOWS.items():
            for spec in specs:
                wt = turbines.get(spec["wt"])
                if wt is None:
                    raise RuntimeError(f"Reference turbine WT{spec['wt']} not in dataset")
                frames.append(_feature_frame(wt, spec["start"], spec["end"]))
        pool = pd.concat(frames, ignore_index=True)
        mean = {c: float(pool[c].mean()) for c, _ in FEATURE_COLUMNS}
        std = {c: max(float(pool[c].std(ddof=0)), 1e-6) for c, _ in FEATURE_COLUMNS}
        _scaler = {"mean": mean, "std": std}
        logger.info("Built scaler: mean=%s std=%s", mean, std)
        return _scaler


def _scale_frame(df: pd.DataFrame) -> pd.DataFrame:
    sc = ensure_scaler()
    out = df.copy()
    for c, _ in FEATURE_COLUMNS:
        out[c] = (out[c] - sc["mean"][c]) / sc["std"][c]
    return out


# ---------- Direct Query Omega embedding + local KNN ----------

# Official Archetype AI Python client, per the atai-newton-omega-model skill's
# references/_common.py (the /query POST goes through the client's retrying
# transport). Built once, lazily.
_client_lock = threading.Lock()
_client: ArchetypeAI | None = None


def _get_client() -> ArchetypeAI:
    global _client
    with _client_lock:
        if _client is None:
            key = os.environ.get("ATAI_API_KEY", "")
            if not key:
                raise RuntimeError("ATAI_API_KEY is not set")
            endpoint = _resolve_endpoint(os.environ.get("ATAI_API_ENDPOINT", ""))
            _client = ArchetypeAI(key, api_endpoint=endpoint)
        return _client


def _embed_channel(channel: list[float]) -> list[float]:
    """One /query per channel → flat 768-d vector, via the official client. Retries transient failures."""
    client = _get_client()
    body = json.dumps({
        "query": "",
        "model": OMEGA_MODEL,
        "normalize_input": False,
        "events": [{"type": "data.numeric_array", "event_data": {"contents": [channel]}}],
    })
    last: Exception | None = None
    for attempt in range(OMEGA_RETRIES):
        try:
            payload = client.requests_post(
                f"{client.api_endpoint}/query",
                data_payload=body,
                additional_headers={"Content-Type": "application/json"},
            )
            vec = (payload.get("response") or {}).get("response")
            if not isinstance(vec, list) or not isinstance(vec[0], (int, float)):
                raise RuntimeError(f"unexpected Omega response shape: {str(vec)[:120]}")
            return vec
        except Exception as exc:  # noqa: BLE001
            last = exc
            if attempt < OMEGA_RETRIES - 1:
                time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"Omega embed failed after {OMEGA_RETRIES} tries: {last}")


def _channels_from(scaled_slice: pd.DataFrame) -> list[list[float]]:
    return [scaled_slice[c].tolist() for c, _ in FEATURE_COLUMNS]


def _embed_window(scaled_channels: list[list[float]]) -> list[float]:
    """Per-channel embeds (bounded parallel fan-out) concatenated into one vector."""
    with ThreadPoolExecutor(max_workers=OMEGA_MAX_CONCURRENCY) as ex:
        per_channel = list(ex.map(_embed_channel, scaled_channels))
    out: list[float] = []
    for v in per_channel:
        out.extend(v)
    return out


def _knn(vec: list[float], library: list[tuple[list[float], str]], k: int = KNN_K):
    dists = sorted(
        ((sum((a - b) ** 2 for a, b in zip(vec, lvec)), label) for lvec, label in library),
        key=lambda x: x[0],
    )[:k]
    votes: dict[str, int] = {}
    for _, label in dists:
        votes[label] = votes.get(label, 0) + 1
    winner = max(votes.items(), key=lambda x: x[1])[0]
    return winner, votes


def _assert_disjoint() -> None:
    """Guard: references must share no (turbine, time-range) with the live playback."""
    live = {DEMO_WT_A, DEMO_WT_B}
    live_start, live_end = pd.to_datetime(DEMO_START), pd.to_datetime(DEMO_END)
    for cls, specs in REFERENCE_WINDOWS.items():
        for spec in specs:
            if spec["wt"] in live:
                rs, re = pd.to_datetime(spec["start"]), pd.to_datetime(spec["end"])
                if rs < live_end and re > live_start:
                    raise RuntimeError(
                        f"Leakage: {cls} reference (WT{spec['wt']} {spec['start']}..{spec['end']}) "
                        f"overlaps the live playback window for the same turbine"
                    )
            # different turbine than the live ones → disjoint by turbine


_library_lock = threading.Lock()
_library_cache: list[tuple[list[float], str]] | None = None


def _collect_slices(specs: list[dict], step: int, turbines: dict) -> list[pd.DataFrame]:
    """Scaled EMBED_WINDOW-row slices over each reference window at the given step."""
    out: list[pd.DataFrame] = []
    for spec in specs:
        frame = _scale_frame(_feature_frame(turbines[spec["wt"]], spec["start"], spec["end"]))
        for s in range(0, len(frame) - EMBED_WINDOW + 1, step):
            out.append(frame.iloc[s:s + EMBED_WINDOW])
    return out


def _library_fingerprint() -> dict:
    """Config that determines the embeddings — a disk library only loads if it matches."""
    return {
        "model": OMEGA_MODEL,
        "embed_window": EMBED_WINDOW,
        "feature_columns": [c for c, _ in FEATURE_COLUMNS],
        "reference_windows": REFERENCE_WINDOWS,
        "steps": {"healthy": HEALTHY_STEP, "fault": FAULT_STEP, "healthy_cap": HEALTHY_CAP},
    }


def _build_library_live() -> list[tuple[list[float], str]]:
    """Embed the n-shot reference windows via Omega `/query` → KNN library of (vec, label).

    Healthy windows are abundant; fault windows come from one short sustained
    outage. We cap healthy to HEALTHY_CAP so the (few) fault windows aren't
    swamped in the KNN vote.
    """
    _assert_disjoint()
    turbines = {t.wt_id: t for t in discover_turbines()}
    healthy = _collect_slices(REFERENCE_WINDOWS["healthy"], HEALTHY_STEP, turbines)
    random.Random(0).shuffle(healthy)
    healthy = healthy[:HEALTHY_CAP]
    fault = _collect_slices(REFERENCE_WINDOWS["fault"], FAULT_STEP, turbines)
    library: list[tuple[list[float], str]] = (
        [(_embed_window(_channels_from(s)), "healthy") for s in healthy]
        + [(_embed_window(_channels_from(s)), "fault") for s in fault]
    )
    logger.info("Library (live): %d healthy + %d fault windows", len(healthy), len(fault))
    return library


def save_library(path: str = LIBRARY_PATH) -> str:
    """Build the library live and persist (scaler + vectors + fingerprint) to disk.

    Run offline once: `python build_library.py`. Runtime then loads instantly.
    """
    scaler = ensure_scaler()
    library = _build_library_live()
    blob = {
        "fingerprint": _library_fingerprint(),
        "scaler": scaler,
        "library": [{"vec": vec, "label": label} for vec, label in library],
    }
    with open(path, "w") as f:
        json.dump(blob, f)
    logger.info("Saved library to %s (%d windows)", path, len(library))
    return path


def _load_library(path: str = LIBRARY_PATH) -> list[tuple[list[float], str]] | None:
    """Load a persisted library if present and its config fingerprint still matches."""
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            blob = json.load(f)
        if blob.get("fingerprint") != _library_fingerprint():
            logger.warning("Library %s fingerprint stale — rebuilding live", path)
            return None
        global _scaler
        with _scaler_lock:
            _scaler = blob["scaler"]  # reuse the exact scaler the vectors were built with
        library = [(item["vec"], item["label"]) for item in blob["library"]]
        logger.info("Loaded library from %s (%d windows)", path, len(library))
        return library
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load library %s (%s) — rebuilding live", path, exc)
        return None


def _build_library() -> list[tuple[list[float], str]]:
    """Return the KNN library: in-memory cache → disk (precomputed) → live build."""
    global _library_cache
    with _library_lock:
        if _library_cache is not None:
            return _library_cache
        library = _load_library()
        if library is None:
            library = _build_library_live()
        _library_cache = library
        return library


class BackgroundClassifier:
    """Classify the demo turbines' windows via Direct Query + KNN in a thread.

    Emits newton_status / newton_prediction / newton_error events onto a queue;
    app.py drains them (`drain()`) and paces predictions onto the replay timeline
    via each prediction's `tick_index` (window_index * total_ticks / n_windows).
    """

    def __init__(self, turbines: list[str], total_ticks: int, start: str = DEMO_START, end: str = DEMO_END):
        self.turbines = list(turbines)
        self.total_ticks = max(int(total_ticks), 1)
        self._start = start
        self._end = end
        self._q: "queue.Queue[dict]" = queue.Queue()
        self.done = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def drain(self) -> list[dict]:
        out: list[dict] = []
        while True:
            try:
                out.append(self._q.get_nowait())
            except queue.Empty:
                break
        return out

    def close(self) -> None:
        # Nothing to tear down (no sessions/sockets) — the thread is a daemon.
        self.done = True

    def _run(self) -> None:
        # "starting" while the reference library is built (per turbine, so the
        # warm-up badge — which aggregates per-turbine state — lights up).
        for wt in self.turbines:
            self._q.put({"kind": "newton_status", "turbine": wt, "stage": "starting"})
        try:
            library = _build_library()
        except Exception as exc:  # noqa: BLE001
            logger.exception("Library build failed")
            for wt in self.turbines:
                self._q.put({"kind": "newton_error", "turbine": wt, "message": f"Library build: {exc}"})
            self.done = True
            return
        for wt in self.turbines:
            self._q.put({"kind": "newton_status", "turbine": wt, "stage": "running"})

        # Plan each turbine's windows up front, then classify round-robin across
        # turbines so both streams' verdicts arrive together (sequential per-turbine
        # processing would finish WT01 entirely before WT09's first window).
        turbines = {t.wt_id: t for t in discover_turbines()}
        start_unix = int(pd.to_datetime(self._start).timestamp())
        plans: list[dict] = []
        for wt in self.turbines:
            info = turbines.get(wt)
            if info is None:
                self._q.put({"kind": "newton_error", "turbine": wt, "message": f"Unknown WT{wt}"})
                self._q.put({"kind": "newton_status", "turbine": wt, "stage": "done"})
                continue
            frame = _scale_frame(_feature_frame(info, self._start, self._end))
            starts = list(range(0, len(frame) - EMBED_WINDOW + 1, DATA_STEP))
            plans.append({"wt": wt, "frame": frame, "starts": starts, "n_w": max(len(starts), 1)})

        max_windows = max((len(p["starts"]) for p in plans), default=0)
        for k in range(max_windows):
            for p in plans:
                if k >= len(p["starts"]):
                    if k == len(p["starts"]):  # just exhausted this turbine
                        self._q.put({"kind": "newton_status", "turbine": p["wt"], "stage": "done"})
                    continue
                wt, frame, s = p["wt"], p["frame"], p["starts"][k]
                try:
                    vec = _embed_window(_channels_from(frame.iloc[s:s + EMBED_WINDOW]))
                    label, votes = _knn(vec, library)
                except Exception as exc:  # noqa: BLE001
                    self._q.put({"kind": "newton_error", "turbine": wt, "message": f"window {k}: {exc}"})
                    continue
                win_start = start_unix + s * 600  # 10-minute cadence → seconds
                win_end = win_start + EMBED_WINDOW * 600
                self._q.put({
                    "kind": "newton_prediction",
                    "turbine": wt,
                    "window_index": k,
                    "tick_index": int(k * self.total_ticks / p["n_w"]),
                    "window_start": pd.to_datetime(win_start, unit="s").strftime("%Y-%m-%d %H:%M"),
                    "window_end": pd.to_datetime(win_end, unit="s").strftime("%Y-%m-%d %H:%M"),
                    "class": label,
                    "votes": votes,
                    "model": OMEGA_MODEL,
                })
        # Final done for any turbine whose last window == max_windows-1.
        for p in plans:
            if len(p["starts"]) == max_windows:
                self._q.put({"kind": "newton_status", "turbine": p["wt"], "stage": "done"})
        self.done = True

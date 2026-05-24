"""Archetype AI Newton lens integration + replay tick streamer.

`newton_classify_turbine()` drives the Machine State Lens with two reference
classes built from the dataset — `healthy` (WT09, July 2019) and `fault`
(WT01, late Oct-Nov 2019 frequency-converter event). The setup follows the
canonical pattern from the archetypeai-swat-demo:

  - Register a child lens with the FULL model_parameters block at register
    time (model_version=omega_embeddings_1_4, normalize_input=false,
    input_n_shot, csv_configs, knn_configs, output_streams).
  - Pre-normalize every CSV / window with a per-channel StandardScaler fit
    on the n-shot reference pool — keeps cross-window amplitude signal
    intact, which `normalize_input=true` would erase per-window.
  - Wait for SESSION_STATUS_RUNNING before pushing data.
  - Stream channel-first windows via `session.update` events. The
    `csv_file_reader` input stream silently skips inference.result events
    on this lens (reproduced in smoke tests); push is what works.

`replay_events()` is the live SSE tick streamer — pure data, no inference.
Newton drives the anomaly verdicts; this generator just paces the timeline.
"""
from __future__ import annotations

import logging
import os
import queue
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import Generator

import pandas as pd

from archetypeai.api_client import ArchetypeAI

from data_loader import discover_turbines, load_turbine_window

logger = logging.getLogger(__name__)

# Platform-mounted Machine State Lens — pinned to omega_embeddings_01. We
# register a child lens at startup pinned to omega_embeddings_1_4 so we run
# on the current default encoder and not the older _01 checkpoint.
PLATFORM_LENS_ID = "lns-1d519091822706e2-bc108andqxf8b4os"
OMEGA_MODEL_VERSION = "OmegaEncoder::omega_embeddings_1_4"
CHILD_LENS_NAME_PREFIX = "penmanshiel-turbines"

# Reference windows we upload once and pin as the n-shot classes. WT01's
# frequency-converter outage is 5 days (720 rows < 1024 window), so we widen
# to Oct 27 – Nov 13 (~2400 rows) to get two full fault windows.
REFERENCE_WINDOWS: dict[str, dict] = {
    "healthy": {"wt": "09", "start": "2019-07-01 00:00:00", "end": "2019-07-22 00:00:00"},
    "fault":   {"wt": "01", "start": "2019-10-27 00:00:00", "end": "2019-11-13 00:00:00"},
}

# Hardcoded scenario for the demo.
DEMO_WT_A = "01"  # frequency converter fault on 2019-11-02
DEMO_WT_B = "09"  # healthy peer over the same period
DEMO_START = "2019-09-01 00:00:00"
DEMO_END = "2019-12-01 00:00:00"  # 3 months

# Four signal columns we send to the lens.
FEATURE_COLUMNS: list[tuple[str, str]] = [
    ("a1", "Wind speed (m/s)"),
    ("a2", "Power (kW)"),
    ("a3", "Blade angle (pitch position) A (°)"),
    ("a4", "Gear oil temperature (°C)"),
]

# Extra columns we pull for the live UI (rotor speed drives blade animation,
# nacelle position drives the yaw indicator). Not part of the Newton payload.
RICH_COLUMNS: list[tuple[str, str]] = [
    *FEATURE_COLUMNS,
    ("rpm", "Rotor speed (RPM)"),
    ("yaw", "Nacelle position (°)"),
    ("wind_dir", "Wind direction (°)"),
]

# Window matches the swat-demo's known-working config. The newton-machine-state
# skill's staging gotcha and swat-demo's newton.js both report that smaller
# windows trigger silent inference.result skipping.
WINDOW_SIZE = 128
STEP_SIZE = 128
# Minimum wall-clock interval between session.update pushes per session.
# Newton processes ~1 inference/s/session on staging; pushing faster fills
# the lens buffer faster than it drains and the runner goes silent after
# ~20 predictions (the buffer depth). Keep our push rate at or below the
# processing rate so we never queue more than 1-2 windows ahead.
MIN_PUSH_INTERVAL_SEC = 1.0


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


def build_client() -> ArchetypeAI:
    api_key = os.environ.get("ATAI_API_KEY", "")
    endpoint = _resolve_endpoint(os.environ.get("ATAI_API_ENDPOINT", ""))
    if not api_key:
        raise RuntimeError("ATAI_API_KEY is not set")
    return ArchetypeAI(api_key, api_endpoint=endpoint)


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

    All anomaly/state inference comes from Newton via newton_classify_turbine.
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


# ---------- Newton Machine State Lens ----------


def _feature_frame(wt_info, start: str, end: str) -> pd.DataFrame:
    """Return a DataFrame indexed by timestamp_unix with columns a1..a4 (NaN-filled)."""
    window = load_turbine_window(wt_info, start, end, max_rows=20_000)
    out = pd.DataFrame({"timestamp": pd.to_datetime(window["Date and time"]).astype("int64") // 10**9})
    for short, full in FEATURE_COLUMNS:
        out[short] = pd.to_numeric(window[full], errors="coerce").fillna(0.0)
    return out


# Per-process scaler computed from the n-shot reference pool. Applied to both
# focus CSVs (pre-upload) and pushed windows so the encoder sees a consistent
# amplitude reference. `normalize_input=false` on the lens lets cross-window
# amplitude signal through; per-window normalize_input=true would erase it.
_scaler_lock = threading.Lock()
_scaler: dict[str, dict[str, float]] | None = None


def ensure_scaler() -> dict[str, dict[str, float]]:
    global _scaler
    with _scaler_lock:
        if _scaler is not None:
            return _scaler
        turbines = {t.wt_id: t for t in discover_turbines()}
        frames: list[pd.DataFrame] = []
        for cls, spec in REFERENCE_WINDOWS.items():
            wt = turbines.get(spec["wt"])
            if wt is None:
                raise RuntimeError(f"Reference turbine WT{spec['wt']} not in dataset")
            frames.append(_feature_frame(wt, spec["start"], spec["end"]))
        pool = pd.concat(frames, ignore_index=True)
        mean = {c: float(pool[c].mean()) for c, _ in FEATURE_COLUMNS}
        # Guard against zero std (constant column) → divide-by-zero blows up.
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


def _upload_bytes(client: ArchetypeAI, content: bytes, suffix: str) -> dict:
    with tempfile.NamedTemporaryFile("wb", suffix=suffix, delete=False) as fh:
        fh.write(content)
        tmp_path = fh.name
    try:
        return client.files.local.upload(tmp_path)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


_focus_lock = threading.Lock()
_focus_cache: dict[str, str] = {}  # class_name -> platform file_id


def ensure_focus_files(client: ArchetypeAI) -> dict[str, str]:
    """Upload (once per process) pre-normalized n-shot focus CSVs."""
    with _focus_lock:
        if _focus_cache:
            return dict(_focus_cache)
        turbines = {t.wt_id: t for t in discover_turbines()}
        for cls, spec in REFERENCE_WINDOWS.items():
            wt = turbines.get(spec["wt"])
            if wt is None:
                raise RuntimeError(f"Reference turbine WT{spec['wt']} not in dataset")
            frame = _scale_frame(_feature_frame(wt, spec["start"], spec["end"]))
            if len(frame) < WINDOW_SIZE:
                raise RuntimeError(f"Reference {cls} window has only {len(frame)} rows; needs >= {WINDOW_SIZE}")
            resp = _upload_bytes(client, frame.to_csv(index=False).encode(), f"_focus_{cls}.csv")
            file_id = resp.get("file_id") or resp.get("file_uid")
            _focus_cache[cls] = file_id
            logger.info("Uploaded focus %s (%d rows) → %s", cls, len(frame), file_id)
        return dict(_focus_cache)


_lens_lock = threading.Lock()
_child_lens_ids: dict[str, str] = {}  # turbine_id -> lens_id
_stale_cleanup_done = False


def cleanup_orphan_sessions() -> int:
    """Public wrapper: destroy any active sessions matching our child-lens prefix.

    Call at the top of each replay request so abandoned sessions from prior
    runs (browser tab closed without auto_destroy firing, curl --max-time
    cutting the SSE, Flask process killed mid-run) don't starve the lens
    runner pool.
    """
    try:
        client = build_client()
    except Exception as exc:  # noqa: BLE001
        logger.warning("cleanup_orphan_sessions: client build failed: %s", exc)
        return 0
    return _clean_stale_sessions(client)


def _clean_stale_sessions(client: ArchetypeAI) -> int:
    """Destroy any active lens sessions tied to our child-lens prefix.

    Orphaned sessions accumulate when the Flask process is killed without
    triggering auto_destroy. Newton allocates a limited pool of "lens
    runners" — if all are claimed by orphans, the next create_session call
    fails with "Failed to allocate lens runner".
    """
    try:
        sessions = client.lens.sessions.get_metadata()
    except Exception as exc:  # noqa: BLE001
        logger.warning("sessions.get_metadata failed: %s", exc)
        return 0
    if not isinstance(sessions, list):
        return 0
    # Our child lens_id encodes the first 16 chars of CHILD_LENS_NAME_PREFIX
    # ("penmanshiel-turb") in the lens_id slug, so we can match without
    # cross-referencing lens metadata.
    lens_prefix = "lns-" + CHILD_LENS_NAME_PREFIX[:16]
    killed = 0
    for entry in sessions:
        if not isinstance(entry, dict):
            continue
        if (entry.get("lens_id") or "").startswith(lens_prefix):
            sid = entry.get("session_id")
            if not sid:
                continue
            try:
                client.lens.sessions.destroy(sid)
                killed += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not destroy stale session %s: %s", sid, exc)
    if killed:
        logger.info("Destroyed %d stale session(s)", killed)
    return killed


def _clean_stale_lenses(client: ArchetypeAI) -> None:
    """Delete any child lenses left over from previous runs (and their sessions)."""
    _clean_stale_sessions(client)
    try:
        meta = client.lens.get_metadata()
    except Exception:  # noqa: BLE001
        return
    entries = meta if isinstance(meta, list) else meta.get("lenses") or meta.get("entries") or []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        name = entry.get("lens_name") or ""
        lens_id = entry.get("lens_id")
        if name.startswith(CHILD_LENS_NAME_PREFIX) and lens_id:
            try:
                client.lens.delete(lens_id)
                logger.info("Deleted stale lens %s (%s)", lens_id, name)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not delete stale lens %s: %s", lens_id, exc)


def ensure_child_lens(client: ArchetypeAI, wt_id: str = "shared") -> str:
    """Register (once per process per turbine) a child lens.

    Running parallel sessions on the same lens has produced silent-runner
    behavior — only one of the two sessions emits inference.result events
    even though both reach SESSION_STATUS_RUNNING. Registering a dedicated
    lens per turbine eliminates the contention (matches the per-stage
    pattern in archetypeai-swat-demo).
    """
    global _stale_cleanup_done
    with _lens_lock:
        if wt_id in _child_lens_ids:
            return _child_lens_ids[wt_id]
        if not _stale_cleanup_done:
            _clean_stale_lenses(client)
            _stale_cleanup_done = True
        focus = ensure_focus_files(client)
        lens_payload = {
            "lens_name": f"{CHILD_LENS_NAME_PREFIX}-wt{wt_id}-{int(time.time())}",
            "lens_config": {
                "model_pipeline": [
                    {"processor_name": "lens_timeseries_state_processor", "processor_config": {}}
                ],
                "model_parameters": {
                    "model_name": "OmegaEncoder",
                    "model_version": OMEGA_MODEL_VERSION,
                    "normalize_input": False,
                    "buffer_size": WINDOW_SIZE,
                    "input_n_shot": focus,
                    "csv_configs": {
                        "timestamp_column": "timestamp",
                        "data_columns": [c for c, _ in FEATURE_COLUMNS],
                        "window_size": WINDOW_SIZE,
                        "step_size": STEP_SIZE,
                    },
                    "knn_configs": {
                        "n_neighbors": 5,
                        "metric": "euclidean",
                        "weights": "uniform",
                        "algorithm": "ball_tree",
                        "normalize_embeddings": False,
                    },
                },
                "output_streams": [{"stream_type": "server_sent_events_writer"}],
            },
        }
        resp = client.lens.register(lens_payload)
        lens_id = resp["lens_id"]
        _child_lens_ids[wt_id] = lens_id
        logger.info("Registered child lens %s for wt%s (%s)", lens_id, wt_id, OMEGA_MODEL_VERSION)
        return lens_id


def _wait_for_session(client: ArchetypeAI, session_id: str, max_wait_sec: float = 60.0) -> bool:
    """Poll session.status until RUNNING (or FAILED, or timeout)."""
    deadline = time.time() + max_wait_sec
    while time.time() < deadline:
        try:
            resp = client.lens.sessions.process_event(session_id, {"type": "session.status"})
        except Exception as exc:  # noqa: BLE001
            logger.warning("session.status raised: %s", exc)
            time.sleep(1.0)
            continue
        status = resp.get("session_status") or (resp.get("event_data") or {}).get("session_status") or ""
        status_str = str(status)
        if "RUNNING" in status_str or status_str == "3":
            return True
        if "FAILED" in status_str or status_str == "6":
            return False
        time.sleep(1.0)
    return False


class MultiplexNewtonSession:
    """One lens, one session, multiple data streams multiplexed by push order.

    Built for accounts where the runner-pool quota is 1 concurrent session.
    Instead of one session per turbine, we share one session and interleave
    pushes from N turbines through it. The Machine State Lens emits one
    `inference.result` per `session.update` push; we route results back to
    their source by maintaining a FIFO queue of (turbine_id, window_index)
    tags in the order we pushed.

    Order-based routing is reliable because:
      - Newton processes pushes serially (~1 inference/s).
      - SSE events arrive in processing order.
      - We never push window N+1 of any stream before window N of that
        stream lands (per-stream FIFO is preserved as long as we keep
        push_next_window's per-turbine order).
    """

    def __init__(self, streams: list[tuple[str, str]], start: str = DEMO_START, end: str = DEMO_END, max_run_time_sec: float = 600.0):
        """`streams` is a list of (stream_id, wt_id) — typically the wt_id is reused as stream_id."""
        self.stream_ids = [s[0] for s in streams]
        self.max_run_time_sec = max_run_time_sec
        self.start_ts = start
        self.end_ts = end

        turbines = {t.wt_id: t for t in discover_turbines()}
        self._frames: dict[str, pd.DataFrame] = {}
        self._n_windows: dict[str, int] = {}
        for stream_id, wt_id in streams:
            if wt_id not in turbines:
                raise ValueError(f"Unknown turbine WT{wt_id}")
            frame = _scale_frame(_feature_frame(turbines[wt_id], start, end))
            self._frames[stream_id] = frame
            self._n_windows[stream_id] = max(0, (len(frame) - WINDOW_SIZE) // STEP_SIZE + 1)

        self._window_seconds = WINDOW_SIZE * 600
        self._start_unix = int(pd.to_datetime(start).timestamp())

        self._events: "queue.Queue[dict]" = queue.Queue()
        # Each item: (stream_id, window_index). FIFO so SSE results can be tagged in order.
        self._pushes: "queue.Queue[tuple[str, int] | None]" = queue.Queue()
        self._pending_tags: "queue.Queue[tuple[str, int]]" = queue.Queue()
        self._stop = threading.Event()
        self._main: threading.Thread | None = None
        self._pushed_per_stream: dict[str, int] = {sid: 0 for sid in self.stream_ids}
        self._predicted_per_stream: dict[str, int] = {sid: 0 for sid in self.stream_ids}

    def n_windows_for(self, stream_id: str) -> int:
        return self._n_windows.get(stream_id, 0)

    def pushed_for(self, stream_id: str) -> int:
        return self._pushed_per_stream.get(stream_id, 0)

    def predicted_for(self, stream_id: str) -> int:
        return self._predicted_per_stream.get(stream_id, 0)

    def start(self) -> None:
        if self._main is not None:
            return
        for sid in self.stream_ids:
            self._emit({"kind": "newton_status", "turbine": sid, "stage": "starting"})
        self._main = threading.Thread(target=self._run, daemon=True)
        self._main.start()

    def push_next_window(self, stream_id: str) -> bool:
        if stream_id not in self.stream_ids:
            return False
        idx = self._pushed_per_stream[stream_id]
        if idx >= self._n_windows[stream_id]:
            return False
        self._pushes.put((stream_id, idx))
        self._pushed_per_stream[stream_id] = idx + 1
        return True

    def flush_remaining(self) -> int:
        n = 0
        # Interleave remaining pushes so neither stream sits behind a long queue.
        more = True
        while more:
            more = False
            for sid in self.stream_ids:
                if self.push_next_window(sid):
                    n += 1
                    more = True
        return n

    def drain_events(self) -> list[dict]:
        out: list[dict] = []
        while True:
            try:
                out.append(self._events.get_nowait())
            except queue.Empty:
                break
        return out

    def close(self) -> None:
        self._stop.set()
        self._pushes.put(None)

    def _emit(self, ev: dict) -> None:
        self._events.put(ev)

    def _run(self) -> None:
        try:
            client = build_client()
        except Exception as exc:  # noqa: BLE001
            for sid in self.stream_ids:
                self._emit({"kind": "newton_error", "turbine": sid, "message": f"Client setup: {exc}"})
                self._emit({"kind": "newton_done", "turbine": sid})
            return

        try:
            lens_id = ensure_child_lens(client, "shared")
        except Exception as exc:  # noqa: BLE001
            logger.exception("Lens setup failed")
            for sid in self.stream_ids:
                self._emit({"kind": "newton_error", "turbine": sid, "message": f"Lens setup: {exc}"})
                self._emit({"kind": "newton_done", "turbine": sid})
            return
        for sid in self.stream_ids:
            self._emit({"kind": "newton_status", "turbine": sid, "stage": "focus_ready", "lens_id": lens_id})
            self._emit({"kind": "newton_status", "turbine": sid, "stage": "uploaded",
                        "rows": len(self._frames[sid]), "expected_predictions": self._n_windows[sid]})

        def _session_body(session_id: str, session_endpoint: str, **_):
            ready = _wait_for_session(client, session_id)
            if not ready:
                for sid in self.stream_ids:
                    self._emit({"kind": "newton_error", "turbine": sid, "message": "Session did not reach RUNNING state"})
                return
            for sid in self.stream_ids:
                self._emit({"kind": "newton_status", "turbine": sid, "stage": "running", "session_id": session_id})

            sse_stop = threading.Event()
            sse_thread = threading.Thread(target=self._consume_sse, args=(client, session_id, sse_stop), daemon=True)
            sse_thread.start()
            time.sleep(0.5)

            channels = [c for c, _ in FEATURE_COLUMNS]
            last_push = 0.0
            while not self._stop.is_set():
                try:
                    item = self._pushes.get(timeout=1.0)
                except queue.Empty:
                    continue
                if item is None:
                    break
                wait = MIN_PUSH_INTERVAL_SEC - (time.time() - last_push)
                if wait > 0:
                    time.sleep(wait)
                stream_id, w_idx = item
                frame = self._frames[stream_id]
                window = frame.iloc[w_idx * STEP_SIZE : w_idx * STEP_SIZE + WINDOW_SIZE]
                sensor_data = [window[c].astype(float).tolist() for c in channels]
                event = {
                    "type": "session.update",
                    "event_data": {
                        "type": "data.json",
                        "event_data": {
                            "sensor_data": sensor_data,
                            "sensor_metadata": {
                                "sensor_timestamp": time.time(),
                                "sensor_id": f"wt{stream_id}_{w_idx}",
                            },
                        },
                    },
                }
                try:
                    self._pending_tags.put((stream_id, w_idx))
                    client.lens.sessions.process_event(session_id, event)
                    last_push = time.time()
                except Exception as exc:  # noqa: BLE001
                    self._emit({"kind": "newton_error", "turbine": stream_id, "message": f"Push {w_idx}: {exc}"})
                    break

            wait_deadline = time.time() + self.max_run_time_sec
            while time.time() < wait_deadline and not self._stop.is_set():
                if all(self._predicted_per_stream[sid] >= self._pushed_per_stream[sid] for sid in self.stream_ids):
                    break
                time.sleep(0.5)
            sse_stop.set()
            sse_thread.join(timeout=2.0)

        try:
            client.lens.create_and_run_session(lens_id, _session_body, auto_destroy=True, client=client)
        except Exception as exc:  # noqa: BLE001
            for sid in self.stream_ids:
                self._emit({"kind": "newton_error", "turbine": sid, "message": f"{type(exc).__name__}: {exc}"})
        finally:
            for sid in self.stream_ids:
                self._emit({"kind": "newton_done", "turbine": sid})

    def _consume_sse(self, client: ArchetypeAI, session_id: str, stop_event: threading.Event) -> None:
        try:
            sse = client.lens.sessions.create_sse_consumer(session_id, max_read_time_sec=self.max_run_time_sec)
        except Exception as exc:  # noqa: BLE001
            for sid in self.stream_ids:
                self._emit({"kind": "newton_error", "turbine": sid, "message": f"SSE open: {exc}"})
            return
        try:
            for event in sse.read(block=True):
                if stop_event.is_set():
                    break
                etype = event.get("type")
                if etype == "inference.result":
                    ed = event.get("event_data") or {}
                    response = ed.get("response")
                    predicted_class, votes = None, {}
                    if isinstance(response, list) and response:
                        predicted_class = response[0]
                        if len(response) > 1 and isinstance(response[1], dict):
                            votes = response[1]
                    elif isinstance(response, dict):
                        predicted_class = response.get("class_name") or response.get("label") or response.get("prediction")
                    # Route by FIFO push-tag order.
                    try:
                        stream_id, w_idx = self._pending_tags.get_nowait()
                    except queue.Empty:
                        # Unexpected — got a result with no pending tag. Skip rather than misroute.
                        continue
                    self._predicted_per_stream[stream_id] = self._predicted_per_stream.get(stream_id, 0) + 1
                    win_start = self._start_unix + w_idx * self._window_seconds
                    win_end = win_start + self._window_seconds
                    self._emit({
                        "kind": "newton_prediction",
                        "turbine": stream_id,
                        "window_index": w_idx,
                        "window_start": pd.to_datetime(win_start, unit="s").strftime("%Y-%m-%d %H:%M"),
                        "window_end": pd.to_datetime(win_end, unit="s").strftime("%Y-%m-%d %H:%M"),
                        "class": predicted_class,
                        "votes": votes,
                    })
                elif etype in ("error_message", "inference.error"):
                    ed = event.get("event_data") or {}
                    msg = ed.get("message") or "; ".join(ed.get("error_messages") or []) or str(ed)
                    for sid in self.stream_ids:
                        self._emit({"kind": "newton_error", "turbine": sid, "message": msg})
                elif etype == "sse.stream.end":
                    break
        except Exception as exc:  # noqa: BLE001
            for sid in self.stream_ids:
                self._emit({"kind": "newton_error", "turbine": sid, "message": f"SSE read: {exc}"})
        finally:
            try:
                sse.close()
            except Exception:  # noqa: BLE001
                pass


class NewtonSession:
    """Long-running per-turbine lens session with externally paced push.

    Lifecycle:
      s = NewtonSession(wt_id); s.start()             # spawns setup thread
      while replay running:
          if it's time, s.push_next_window()           # non-blocking
          for ev in s.drain_events(): ...              # newton_* events
      s.flush_remaining(); s.close()                   # tear down

    Newton's inference cadence on staging is ~1 prediction/s/session, so
    pushing faster than that overflows the lens buffer and Newton goes silent
    after ~20 windows. Pacing pushes to the replay tick keeps the rate sane.
    """

    def __init__(self, wt_id: str, start: str = DEMO_START, end: str = DEMO_END, max_run_time_sec: float = 600.0):
        self.wt_id = wt_id
        self.start_ts = start
        self.end_ts = end
        self.max_run_time_sec = max_run_time_sec
        self._events: "queue.Queue[dict]" = queue.Queue()
        self._pushes: "queue.Queue[int | None]" = queue.Queue()
        self._stop = threading.Event()
        self._main: threading.Thread | None = None
        self._pushed_count = 0
        self._pred_count = 0
        self._pred_lock = threading.Lock()

        turbines = {t.wt_id: t for t in discover_turbines()}
        if wt_id not in turbines:
            raise ValueError(f"Unknown turbine WT{wt_id}")
        self._wt_info = turbines[wt_id]
        self._frame = _scale_frame(_feature_frame(self._wt_info, start, end))
        self.n_windows = max(0, (len(self._frame) - WINDOW_SIZE) // STEP_SIZE + 1)
        self._window_seconds = WINDOW_SIZE * 600
        self._start_unix = int(pd.to_datetime(start).timestamp())

    @property
    def pushed(self) -> int:
        return self._pushed_count

    @property
    def predicted(self) -> int:
        with self._pred_lock:
            return self._pred_count

    def start(self) -> None:
        if self._main is not None:
            return
        self._emit({"kind": "newton_status", "turbine": self.wt_id, "stage": "starting"})
        self._main = threading.Thread(target=self._run, daemon=True)
        self._main.start()

    def push_next_window(self) -> bool:
        if self._pushed_count >= self.n_windows:
            return False
        self._pushes.put(self._pushed_count)
        self._pushed_count += 1
        return True

    def flush_remaining(self) -> int:
        n = 0
        while self.push_next_window():
            n += 1
        return n

    def drain_events(self) -> list[dict]:
        out: list[dict] = []
        while True:
            try:
                out.append(self._events.get_nowait())
            except queue.Empty:
                break
        return out

    def close(self) -> None:
        self._stop.set()
        self._pushes.put(None)

    def _emit(self, ev: dict) -> None:
        self._events.put(ev)

    def _run(self) -> None:
        try:
            client = build_client()
        except Exception as exc:  # noqa: BLE001
            self._emit({"kind": "newton_error", "turbine": self.wt_id, "message": f"Client setup: {exc}"})
            self._emit({"kind": "newton_done", "turbine": self.wt_id})
            return

        try:
            lens_id = ensure_child_lens(client, self.wt_id)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Lens setup failed")
            self._emit({"kind": "newton_error", "turbine": self.wt_id, "message": f"Lens setup: {exc}"})
            self._emit({"kind": "newton_done", "turbine": self.wt_id})
            return
        self._emit({"kind": "newton_status", "turbine": self.wt_id, "stage": "focus_ready", "lens_id": lens_id})
        self._emit({"kind": "newton_status", "turbine": self.wt_id, "stage": "uploaded",
                    "rows": len(self._frame), "expected_predictions": self.n_windows})

        def _session_body(session_id: str, session_endpoint: str, **_):
            ready = _wait_for_session(client, session_id)
            if not ready:
                self._emit({"kind": "newton_error", "turbine": self.wt_id, "message": "Session did not reach RUNNING state"})
                return
            self._emit({"kind": "newton_status", "turbine": self.wt_id, "stage": "running", "session_id": session_id})

            sse_stop = threading.Event()
            sse_thread = threading.Thread(target=self._consume_sse, args=(client, session_id, sse_stop), daemon=True)
            sse_thread.start()
            time.sleep(0.5)  # let the SSE handshake land

            channels = [c for c, _ in FEATURE_COLUMNS]
            last_push = 0.0
            while not self._stop.is_set():
                try:
                    item = self._pushes.get(timeout=1.0)
                except queue.Empty:
                    continue
                if item is None:
                    break
                # Pace pushes so we never run ahead of Newton's drain rate.
                wait = MIN_PUSH_INTERVAL_SEC - (time.time() - last_push)
                if wait > 0:
                    time.sleep(wait)
                w_idx = item
                window = self._frame.iloc[w_idx * STEP_SIZE : w_idx * STEP_SIZE + WINDOW_SIZE]
                sensor_data = [window[c].astype(float).tolist() for c in channels]
                event = {
                    "type": "session.update",
                    "event_data": {
                        "type": "data.json",
                        "event_data": {
                            "sensor_data": sensor_data,
                            "sensor_metadata": {
                                "sensor_timestamp": time.time(),
                                "sensor_id": f"wt{self.wt_id}_{w_idx}",
                            },
                        },
                    },
                }
                try:
                    client.lens.sessions.process_event(session_id, event)
                    last_push = time.time()
                except Exception as exc:  # noqa: BLE001
                    self._emit({"kind": "newton_error", "turbine": self.wt_id, "message": f"Push {w_idx}: {exc}"})
                    break

            # Wait for outstanding predictions (up to max_run_time_sec).
            wait_deadline = time.time() + self.max_run_time_sec
            while time.time() < wait_deadline and not self._stop.is_set():
                if self.predicted >= self._pushed_count:
                    break
                time.sleep(0.5)
            sse_stop.set()
            sse_thread.join(timeout=2.0)

        try:
            client.lens.create_and_run_session(lens_id, _session_body, auto_destroy=True, client=client)
        except Exception as exc:  # noqa: BLE001
            self._emit({"kind": "newton_error", "turbine": self.wt_id, "message": f"{type(exc).__name__}: {exc}"})
        finally:
            self._emit({"kind": "newton_done", "turbine": self.wt_id})

    def _consume_sse(self, client: ArchetypeAI, session_id: str, stop_event: threading.Event) -> None:
        try:
            sse = client.lens.sessions.create_sse_consumer(session_id, max_read_time_sec=self.max_run_time_sec)
        except Exception as exc:  # noqa: BLE001
            self._emit({"kind": "newton_error", "turbine": self.wt_id, "message": f"SSE open: {exc}"})
            return
        try:
            for event in sse.read(block=True):
                if stop_event.is_set():
                    break
                etype = event.get("type")
                if etype == "inference.result":
                    ed = event.get("event_data") or {}
                    response = ed.get("response")
                    predicted_class, votes = None, {}
                    if isinstance(response, list) and response:
                        predicted_class = response[0]
                        if len(response) > 1 and isinstance(response[1], dict):
                            votes = response[1]
                    elif isinstance(response, dict):
                        predicted_class = response.get("class_name") or response.get("label") or response.get("prediction")
                    with self._pred_lock:
                        idx = self._pred_count
                        self._pred_count += 1
                    win_start = self._start_unix + idx * self._window_seconds
                    win_end = win_start + self._window_seconds
                    self._emit({
                        "kind": "newton_prediction",
                        "turbine": self.wt_id,
                        "window_index": idx,
                        "window_start": pd.to_datetime(win_start, unit="s").strftime("%Y-%m-%d %H:%M"),
                        "window_end": pd.to_datetime(win_end, unit="s").strftime("%Y-%m-%d %H:%M"),
                        "class": predicted_class,
                        "votes": votes,
                    })
                elif etype in ("error_message", "inference.error"):
                    ed = event.get("event_data") or {}
                    msg = ed.get("message") or "; ".join(ed.get("error_messages") or []) or str(ed)
                    self._emit({"kind": "newton_error", "turbine": self.wt_id, "message": msg})
                elif etype == "sse.stream.end":
                    break
        except Exception as exc:  # noqa: BLE001
            self._emit({"kind": "newton_error", "turbine": self.wt_id, "message": f"SSE read: {exc}"})
        finally:
            try:
                sse.close()
            except Exception:  # noqa: BLE001
                pass


def newton_classify_turbine(
    wt_id: str,
    start: str = DEMO_START,
    end: str = DEMO_END,
    max_run_time_sec: float = 300.0,
) -> Generator[dict, None, None]:
    """Stream a turbine's data through the Machine State Lens.

    Setup (cached per process): scaler, n-shot focus uploads, child lens with
    full model_parameters. Per call: create a session, wait for RUNNING, push
    each WINDOW_SIZE-row window via session.update events while concurrently
    reading the SSE consumer for inference.result events.
    """
    yield {"kind": "newton_status", "turbine": wt_id, "stage": "starting"}

    turbines = {t.wt_id: t for t in discover_turbines()}
    if wt_id not in turbines:
        yield {"kind": "newton_error", "turbine": wt_id, "message": f"Unknown turbine WT{wt_id}"}
        yield {"kind": "newton_done", "turbine": wt_id}
        return

    try:
        client = build_client()
    except Exception as exc:  # noqa: BLE001
        yield {"kind": "newton_error", "turbine": wt_id, "message": f"Client setup: {exc}"}
        yield {"kind": "newton_done", "turbine": wt_id}
        return

    try:
        lens_id = ensure_child_lens(client, wt_id)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Lens setup failed")
        yield {"kind": "newton_error", "turbine": wt_id, "message": f"Lens setup: {exc}"}
        yield {"kind": "newton_done", "turbine": wt_id}
        return
    yield {"kind": "newton_status", "turbine": wt_id, "stage": "focus_ready", "lens_id": lens_id}

    frame = _scale_frame(_feature_frame(turbines[wt_id], start, end))
    n_rows = len(frame)
    if n_rows < WINDOW_SIZE:
        yield {"kind": "newton_error", "turbine": wt_id, "message": f"Data too short ({n_rows} < {WINDOW_SIZE})"}
        yield {"kind": "newton_done", "turbine": wt_id}
        return

    n_windows = max(0, (n_rows - WINDOW_SIZE) // STEP_SIZE + 1)
    window_seconds = WINDOW_SIZE * 600  # 10-minute cadence
    start_unix = int(pd.to_datetime(start).timestamp())
    yield {
        "kind": "newton_status",
        "turbine": wt_id,
        "stage": "uploaded",
        "rows": n_rows,
        "expected_predictions": n_windows,
    }

    bucket: dict = {"events": [], "done": False, "pred_index": 0}

    def _session(session_id: str, session_endpoint: str, **_) -> None:
        ready = _wait_for_session(client, session_id)
        if not ready:
            bucket["events"].append({"kind": "newton_error", "turbine": wt_id, "message": "Session did not reach RUNNING state"})
            return
        bucket["events"].append({"kind": "newton_status", "turbine": wt_id, "stage": "running", "session_id": session_id})

        # SSE consumer thread — must be reading before we push any data.
        stop_sse = threading.Event()

        def _consume_sse() -> None:
            try:
                sse = client.lens.sessions.create_sse_consumer(session_id, max_read_time_sec=max_run_time_sec)
            except Exception as exc:  # noqa: BLE001
                bucket["events"].append({"kind": "newton_error", "turbine": wt_id, "message": f"SSE open: {exc}"})
                return
            try:
                for event in sse.read(block=True):
                    if stop_sse.is_set():
                        break
                    etype = event.get("type")
                    if etype == "inference.result":
                        ed = event.get("event_data") or {}
                        response = ed.get("response")
                        predicted_class, votes = None, {}
                        if isinstance(response, list) and response:
                            predicted_class = response[0]
                            if len(response) > 1 and isinstance(response[1], dict):
                                votes = response[1]
                        elif isinstance(response, dict):
                            predicted_class = response.get("class_name") or response.get("label") or response.get("prediction")
                        idx = bucket["pred_index"]
                        bucket["pred_index"] += 1
                        win_start = start_unix + idx * window_seconds
                        win_end = win_start + window_seconds
                        bucket["events"].append({
                            "kind": "newton_prediction",
                            "turbine": wt_id,
                            "window_index": idx,
                            "window_start": pd.to_datetime(win_start, unit="s").strftime("%Y-%m-%d %H:%M"),
                            "window_end": pd.to_datetime(win_end, unit="s").strftime("%Y-%m-%d %H:%M"),
                            "class": predicted_class,
                            "votes": votes,
                        })
                    elif etype in ("error_message", "inference.error"):
                        ed = event.get("event_data") or {}
                        msg = ed.get("message") or "; ".join(ed.get("error_messages") or []) or str(ed)
                        bucket["events"].append({"kind": "newton_error", "turbine": wt_id, "message": msg})
                    elif etype == "sse.stream.end":
                        break
            except Exception as exc:  # noqa: BLE001
                bucket["events"].append({"kind": "newton_error", "turbine": wt_id, "message": f"SSE read: {exc}"})
            finally:
                try:
                    sse.close()
                except Exception:  # noqa: BLE001
                    pass

        sse_thread = threading.Thread(target=_consume_sse, daemon=True)
        sse_thread.start()
        time.sleep(0.5)  # give the SSE handshake a moment to land

        # Push each window via session.update. Channel-first: outer = channels.
        channels = [c for c, _ in FEATURE_COLUMNS]
        for w_idx in range(n_windows):
            window = frame.iloc[w_idx * STEP_SIZE : w_idx * STEP_SIZE + WINDOW_SIZE]
            sensor_data = [window[c].astype(float).tolist() for c in channels]
            event = {
                "type": "session.update",
                "event_data": {
                    "type": "data.json",
                    "event_data": {
                        "sensor_data": sensor_data,
                        "sensor_metadata": {
                            "sensor_timestamp": time.time(),
                            "sensor_id": f"wt{wt_id}_{w_idx}",
                        },
                    },
                },
            }
            try:
                client.lens.sessions.process_event(session_id, event)
            except Exception as exc:  # noqa: BLE001
                bucket["events"].append({"kind": "newton_error", "turbine": wt_id, "message": f"Push {w_idx}: {exc}"})
                break

        # Let any outstanding inference.result events land before we tear down.
        wait_deadline = time.time() + min(max_run_time_sec, 90.0)
        while time.time() < wait_deadline and bucket["pred_index"] < n_windows:
            time.sleep(0.5)
        stop_sse.set()
        sse_thread.join(timeout=2.0)

    def _runner():
        try:
            client.lens.create_and_run_session(lens_id, _session, auto_destroy=True, client=client)
        except Exception as exc:  # noqa: BLE001
            bucket["events"].append({"kind": "newton_error", "turbine": wt_id, "message": f"{type(exc).__name__}: {exc}"})
        finally:
            bucket["done"] = True

    threading.Thread(target=_runner, daemon=True).start()

    cursor = 0
    while True:
        evs = bucket["events"]
        while cursor < len(evs):
            yield evs[cursor]
            cursor += 1
        if bucket["done"]:
            break
        time.sleep(0.1)
    evs = bucket["events"]
    while cursor < len(evs):
        yield evs[cursor]
        cursor += 1
    yield {"kind": "newton_done", "turbine": wt_id}

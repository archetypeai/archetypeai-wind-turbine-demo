"""Penmanshiel turbine side-by-side anomaly demo.

All inference and anomaly detection runs through Newton's Machine State Lens
(`lens_timeseries_state_processor`). Per turbine, a NewtonSession registers a
child lens (Omega 1.4 + pre-scaled n-shot focus CSVs), opens an SSE consumer,
and accepts window pushes paced from the replay tick loop. Newton verdicts
arrive asynchronously and merge into the same SSE stream the browser reads.

Endpoints:
    GET /                       UI
    GET /api/scada/<wt_id>      Downsampled rows for initial chart
    GET /api/replay?tps=N
"""
from __future__ import annotations

import json
import time
from typing import Iterator

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template, request, stream_with_context

import newton_client as nc

load_dotenv()

app = Flask(__name__)


@app.route("/")
def index() -> str:
    return render_template(
        "index.html",
        wt_a=nc.DEMO_WT_A,
        wt_b=nc.DEMO_WT_B,
        start=nc.DEMO_START,
        end=nc.DEMO_END,
    )


@app.route("/api/scada/<wt_id>")
def scada(wt_id: str):
    return jsonify({"wt_id": wt_id, "rows": nc.window_to_records(wt_id)})


@app.route("/api/replay")
def replay():
    """SSE: live tick stream paced by tps + Newton lens predictions."""
    try:
        tps = float(request.args.get("tps", "20"))
    except ValueError:
        tps = 20.0
    tps = max(1.0, min(200.0, tps))
    tick_interval = 1.0 / tps

    @stream_with_context
    def _stream() -> Iterator[str]:
        # Reap orphan sessions left behind by abandoned SSE connections
        # (browser tab closed, curl --max-time cutting the stream, ...).
        # Newton's lens runner pool is finite; orphans hold slots until
        # explicitly destroyed.
        nc.cleanup_orphan_sessions()

        # Single multiplexed session — staging account is capped at 1
        # concurrent lens-runner slot, so we share one session across both
        # turbines and interleave pushes. Results are routed by FIFO
        # push-tag order. See newton_client.MultiplexNewtonSession.
        try:
            mx = nc.MultiplexNewtonSession([
                (nc.DEMO_WT_A, nc.DEMO_WT_A),
                (nc.DEMO_WT_B, nc.DEMO_WT_B),
            ])
            mx.start()
        except Exception as exc:  # noqa: BLE001
            yield "data: " + json.dumps({"kind": "newton_error", "message": f"Session init: {exc}"}) + "\n\n"
            mx = None

        sessions = {nc.DEMO_WT_A: mx, nc.DEMO_WT_B: mx} if mx else {}
        stream_ids = [nc.DEMO_WT_A, nc.DEMO_WT_B] if mx else []

        # Anomaly logic on this dataset:
        #   - Filter out WEAK verdicts (3-2 KNN ties) — those are pure noise.
        #     Only consider verdicts where the winner has ≥ STRONG_MARGIN
        #     more votes than the runner-up.
        #   - Commit on the FIRST strong verdict in a new direction. We don't
        #     debounce further because strong verdicts are already rare on
        #     this n-shot library (~30% of predictions), and requiring two
        #     in a row reliably suppresses real signal too.
        #
        # Trade-off accepted: WT09 (healthy turbine) will occasionally show
        # a fault → recovery pair from a single strong false-positive verdict.
        # The fix would be a larger n-shot library or a better-separated
        # encoder; both are out of scope for a visualization demo.
        STRONG_MARGIN = 3
        committed: dict[str, str | None] = {wt: None for wt in stream_ids}

        def drain_all() -> list[dict]:
            out: list[dict] = []
            if not mx:
                return out
            for ev in mx.drain_events():
                out.append(ev)
                if ev.get("kind") != "newton_prediction":
                    continue
                wt = ev.get("turbine")
                if wt not in stream_ids:
                    continue
                new_class = ev.get("class")
                if not new_class or new_class == "unknown":
                    continue
                votes = ev.get("votes") or {}
                if not isinstance(votes, dict) or not votes:
                    continue
                winner_n = max(votes.values())
                others_n = max((v for k, v in votes.items() if k != new_class), default=0)
                if winner_n - others_n < STRONG_MARGIN:
                    continue  # weak — skip
                prev = committed[wt]
                committed[wt] = new_class
                if prev is None or prev == new_class:
                    continue
                severity = "critical" if new_class == "fault" else "info"
                msg = "Detected: fault classification" if new_class == "fault" else f"Recovered: now {new_class}"
                out.append({
                    "kind": "anomaly",
                    "turbine": wt,
                    "ts": ev.get("window_start"),
                    "from": prev,
                    "to": new_class,
                    "severity": severity,
                    "message": msg,
                    "window_start": ev.get("window_start"),
                    "window_end": ev.get("window_end"),
                })
            return out

        try:
            yield from _stream_body(mx, stream_ids, drain_all, tick_interval)
        finally:
            # Always close the session, even if the client disconnects mid-stream.
            if mx:
                mx.close()

    def _stream_body(mx, stream_ids, drain_all, tick_interval):
        # ---- Phase 1: meta + warm-up ----------------------------------
        # Emit meta immediately so the UI knows total_ticks and turbine
        # metadata. Then hold the tick stream until Newton has produced a
        # first real (non-"unknown") verdict per turbine. During this
        # ~25-30s window the browser only sees newton_status events; the
        # top-bar warm-up indicator covers it, the rotors stay still
        # (no ticks → rpm=0), and the panels stay "analysing".
        ev_iter = iter(nc.replay_events())
        meta = next(ev_iter)
        yield "data: " + json.dumps(meta) + "\n\n"

        total_ticks = meta.get("total_ticks") or 0
        push_thresholds: dict[str, list[int]] = {}
        if mx:
            for wt in stream_ids:
                n_w = mx.n_windows_for(wt)
                push_thresholds[wt] = (
                    [(k * total_ticks) // max(n_w, 1) for k in range(n_w)] if n_w else []
                )

        # Push every window up front, interleaved across turbines. Reproduced
        # empirically: with a partial push (5 windows), Newton stays silent.
        # Newton's session internally rate-limits to 1 push/s, so flushing all
        # ~200 combined windows takes ~200s wall time — predictions stream
        # back as the pushes drain.
        if mx:
            mx.flush_remaining()

        # Release ticks once at least one turbine has produced a real verdict
        # (or has errored out). Waiting for all turbines is too brittle —
        # a single allocation failure would freeze the entire dashboard.
        warmed = {wt: False for wt in stream_ids}
        errored = {wt: False for wt in stream_ids}
        WARMUP_TIMEOUT_SEC = 120
        warmup_deadline = time.time() + WARMUP_TIMEOUT_SEC

        def warmup_ready() -> bool:
            if not stream_ids:
                return True
            done = all(warmed[wt] or errored[wt] for wt in stream_ids)
            any_warmed = any(warmed.values())
            return done or any_warmed

        while stream_ids and not warmup_ready() and time.time() < warmup_deadline:
            for nev in drain_all():
                yield "data: " + json.dumps(nev) + "\n\n"
                wt = nev.get("turbine")
                if wt in warmed and nev.get("kind") == "newton_prediction" and nev.get("class") not in (None, "unknown"):
                    warmed[wt] = True
                elif wt in errored and nev.get("kind") in ("newton_error", "newton_done"):
                    errored[wt] = True
            time.sleep(0.3)
            yield ": keepalive\n\n"

        # ---- Phase 2: tick loop ---------------------------------------
        # Now that Newton is warm, stream the replay. Push window k for
        # turbine wt when the replay tick crosses k * total_ticks / n_windows.
        last_tick_walltime = time.time()
        for ev in ev_iter:
            yield "data: " + json.dumps(ev) + "\n\n"
            for nev in drain_all():
                yield "data: " + json.dumps(nev) + "\n\n"
            if ev.get("kind") == "tick" and mx:
                i = ev.get("i", 0)
                for wt in stream_ids:
                    thresholds = push_thresholds.get(wt, [])
                    while mx.pushed_for(wt) < len(thresholds) and i >= thresholds[mx.pushed_for(wt)]:
                        mx.push_next_window(wt)
                now = time.time()
                elapsed = now - last_tick_walltime
                if elapsed < tick_interval:
                    time.sleep(tick_interval - elapsed)
                last_tick_walltime = time.time()

        # Replay is done. Flush any remaining pushes, then keep the
        # connection open while Newton drains outstanding predictions.
        if mx:
            mx.flush_remaining()

        deadline = time.time() + 240
        while time.time() < deadline:
            evs = drain_all()
            for nev in evs:
                yield "data: " + json.dumps(nev) + "\n\n"
            if not mx or not stream_ids:
                break
            all_done = all(mx.predicted_for(wt) >= mx.n_windows_for(wt) for wt in stream_ids)
            if all_done:
                break
            if not evs:
                time.sleep(0.3)
                yield ": keepalive\n\n"

        # Final drain.
        for nev in drain_all():
            yield "data: " + json.dumps(nev) + "\n\n"

        yield "event: complete\ndata: {}\n\n"

    return Response(
        _stream(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=False, threaded=True)

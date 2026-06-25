"""Penmanshiel turbine side-by-side anomaly demo.

All inference runs through Newton's **Direct Query API** — no lenses, no
sessions, no SSE plumbing. A background classifier (newton_client) embeds each
window with the Omega encoder (one /query per channel) and scores it with a
local KNN against an n-shot library. Verdicts stream into the same SSE stream
the browser reads, paced onto the replay timeline.

Endpoints:
    GET /                       UI
    GET /api/scada/<wt_id>      Downsampled rows for initial chart
    GET /api/replay?tps=N       SSE: meta, tick, newton_status, newton_prediction, anomaly, complete
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

# A verdict only commits if the KNN winner leads the runner-up by this many
# votes — filters out weak (e.g. 3-2) ties that are noise on an n-shot library.
STRONG_MARGIN = 3


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


def _anomaly_from(pred: dict, committed: dict[str, str | None]) -> dict | None:
    """Turn a strong, direction-changing prediction into an `anomaly` event."""
    wt = pred.get("turbine")
    new_class = pred.get("class")
    if wt not in committed or not new_class or new_class == "unknown":
        return None
    votes = pred.get("votes") or {}
    if not isinstance(votes, dict) or not votes:
        return None
    winner_n = max(votes.values())
    others_n = max((v for k, v in votes.items() if k != new_class), default=0)
    if winner_n - others_n < STRONG_MARGIN:
        return None  # weak verdict — skip
    prev = committed[wt]
    committed[wt] = new_class
    if prev == new_class:
        return None
    initial = prev is None  # first confident read — flips the panel, but not an alarm
    severity = "critical" if new_class == "fault" else "info"
    if new_class == "fault":
        msg = "Detected: fault classification"
    elif initial:
        msg = "Baseline: healthy"
    else:
        msg = f"Recovered: now {new_class}"
    return {
        "kind": "anomaly", "turbine": wt, "ts": pred.get("window_start"),
        "from": prev, "to": new_class, "severity": severity, "message": msg,
        # silent → flip the panel state but don't add a feed row (baseline healthy).
        "initial": initial, "silent": initial and new_class != "fault",
        "window_start": pred.get("window_start"), "window_end": pred.get("window_end"),
    }


@app.route("/api/replay")
def replay():
    """SSE: live tick stream paced by tps, with Direct-Query Newton verdicts overlaid."""
    try:
        tps = float(request.args.get("tps", "20"))
    except ValueError:
        tps = 20.0
    tps = max(1.0, min(200.0, tps))
    tick_interval = 1.0 / tps

    @stream_with_context
    def _stream() -> Iterator[str]:
        def sse(obj: dict) -> str:
            return "data: " + json.dumps(obj) + "\n\n"

        ev_iter = iter(nc.replay_events())
        meta = next(ev_iter)
        yield sse(meta)
        total_ticks = meta.get("total_ticks") or 0

        clf = nc.BackgroundClassifier([nc.DEMO_WT_A, nc.DEMO_WT_B], total_ticks)
        clf.start()

        committed: dict[str, str | None] = {nc.DEMO_WT_A: None, nc.DEMO_WT_B: None}
        pending: list[dict] = []  # predictions whose tick_index hasn't been reached yet

        def absorb() -> list[dict]:
            """Pull classifier events; buffer predictions, pass status/errors through."""
            passthrough: list[dict] = []
            for ev in clf.drain():
                if ev.get("kind") == "newton_prediction":
                    pending.append(ev)
                else:
                    passthrough.append(ev)
            return passthrough

        def release(upto_tick: int) -> list[dict]:
            """Emit buffered predictions due by `upto_tick`, plus derived anomalies."""
            out: list[dict] = []
            due = [p for p in pending if p.get("tick_index", 0) <= upto_tick]
            for p in sorted(due, key=lambda x: (x.get("tick_index", 0), x.get("window_index", 0))):
                pending.remove(p)
                out.append(p)
                anom = _anomaly_from(p, committed)
                if anom:
                    out.append(anom)
            return out

        try:
            # Phase 1: stream ticks, overlaying verdicts as the playhead reaches them.
            last = time.time()
            for ev in ev_iter:
                yield sse(ev)
                for e in absorb():
                    yield sse(e)
                if ev.get("kind") == "tick":
                    for e in release(ev.get("i", 0)):
                        yield sse(e)
                    elapsed = time.time() - last
                    if elapsed < tick_interval:
                        time.sleep(tick_interval - elapsed)
                    last = time.time()

            # Phase 2: replay done — drain any verdicts still computing.
            deadline = time.time() + 240
            while time.time() < deadline:
                for e in absorb():
                    yield sse(e)
                for e in release(total_ticks):
                    yield sse(e)
                if clf.done and not pending:
                    break
                time.sleep(0.2)
                yield ": keepalive\n\n"
        finally:
            clf.close()

        yield "event: complete\ndata: {}\n\n"

    return Response(
        _stream(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=False, threaded=True)

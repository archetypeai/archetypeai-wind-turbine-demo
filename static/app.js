// Live turbine monitor — consumes /api/replay SSE.
// Newton drives state and anomalies; ticks carry only raw measurements.

const panels = {};
document.querySelectorAll(".panel").forEach((el) => {
    const wt = el.dataset.wt;
    el.querySelector(".wt-id").textContent = wt;
    panels[wt] = {
        el,
        rotor: el.querySelector(".rotor"),
        angle: 0,
        rpm: 0,
        state: "analysing",
        stateLabel: el.querySelector(".state-label"),
        fields: {
            wind: el.querySelector('[data-field="wind"]'),
            power: el.querySelector('[data-field="power"]'),
            rpm: el.querySelector('[data-field="rpm"]'),
            pitch: el.querySelector('[data-field="pitch"]'),
            gear_c: el.querySelector('[data-field="gear_c"]'),
        },
        powerBar: el.querySelector(".bar-fill"),
        ratedKw: 2050,
        spark: makeSpark(el.querySelector(".spark canvas")),
        newtonVerdict: el.querySelector(".newton-verdict"),
        newtonCount: 0,
        newtonFaultCount: 0,
    };
    el.dataset.state = "analysing";
    panels[wt].stateLabel.textContent = labelOf("analysing");
});

// Newton lifecycle tracker (per turbine).
const newtonState = { "01": "idle", "09": "idle" };
function setNewtonBadge() {
    const badge = document.getElementById("newton-status");
    const states = Object.values(newtonState);
    let stage = "idle";
    if (states.some((s) => s === "error")) stage = "error";
    else if (states.every((s) => s === "done")) stage = "done";
    else if (states.some((s) => s === "running")) stage = "running";
    else if (states.some((s) => s === "starting")) stage = "starting";
    badge.className = "newton-status " + stage;
    const text = {
        idle: "Newton idle",
        starting: "Building reference library…",
        running: "Newton analysing",
        done: "Newton complete",
        error: "Newton error",
    }[stage] || stage;
    badge.querySelector(".newton-text").textContent = text;
}

function makeSpark(canvas) {
    return new Chart(canvas, {
        type: "line",
        data: {
            labels: [],
            datasets: [
                {
                    label: "Power (kW)",
                    data: [],
                    borderColor: "#6aa0ff",
                    borderWidth: 1.4,
                    pointRadius: 0,
                    tension: 0.25,
                    fill: { target: "origin", above: "rgba(106,160,255,0.10)" },
                    yAxisID: "y",
                },
                {
                    label: "Wind (m/s)",
                    data: [],
                    borderColor: "#9aa5b8",
                    borderWidth: 0.9,
                    pointRadius: 0,
                    tension: 0.25,
                    borderDash: [3, 3],
                    yAxisID: "y2",
                },
            ],
        },
        options: {
            animation: false,
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: "nearest", intersect: false },
            plugins: {
                legend: {
                    display: true,
                    position: "top",
                    align: "end",
                    labels: { color: "#8a93a1", font: { size: 10 }, boxWidth: 14 },
                },
                tooltip: { enabled: false },
            },
            scales: {
                x: {
                    ticks: { color: "#8a93a1", maxTicksLimit: 4, font: { size: 9 } },
                    grid: { color: "rgba(255,255,255,0.04)" },
                },
                y: {
                    position: "left",
                    ticks: { color: "#8a93a1", font: { size: 9 } },
                    grid: { color: "rgba(255,255,255,0.04)" },
                    min: 0,
                },
                y2: {
                    position: "right",
                    ticks: { color: "#5e6675", font: { size: 9 } },
                    grid: { display: false },
                    min: 0,
                },
            },
        },
    });
}

// ---------- rotor animation loop ----------
// Rotor spin tracks measured RPM (with a 3× visual multiplier so a healthy
// ~14 rpm reads as motion); when RPM is 0 the rotor naturally stops.
const VISUAL_RPM_MULT = 3;
let lastFrame = performance.now();
function frame(now) {
    const dt = Math.min(0.1, (now - lastFrame) / 1000);
    lastFrame = now;
    for (const wt in panels) {
        const p = panels[wt];
        const effRpm = p.rpm * VISUAL_RPM_MULT;
        if (effRpm > 0.1) {
            p.angle = (p.angle + effRpm * 6 * dt) % 360;
            p.rotor.setAttribute("transform", `rotate(${p.angle.toFixed(2)} 0 0)`);
        }
    }
    requestAnimationFrame(frame);
}
requestAnimationFrame(frame);

// ---------- SSE handling ----------
let sse = null;
function connect(tps) {
    if (sse) sse.close();
    document.getElementById("anomaly-list").innerHTML = "";
    setStatus("connecting…");
    // Reset panel state.
    for (const wt in panels) {
        const p = panels[wt];
        p.state = "analysing";
        p.el.dataset.state = "analysing";
        p.stateLabel.textContent = labelOf("analysing");
        p.newtonCount = 0;
        p.newtonFaultCount = 0;
        const v = p.newtonVerdict;
        v.classList.remove("healthy", "fault", "pulse");
        v.querySelector(".nv-class").textContent = "—";
        v.querySelector(".nv-votes").textContent = "";
    }
    sse = new EventSource(`/api/replay?tps=${tps}`);
    sse.onmessage = (e) => {
        try { handle(JSON.parse(e.data)); } catch (err) { console.warn("parse", err, e.data); }
    };
    sse.addEventListener("complete", () => { sse.close(); sse = null; setStatus("ready"); setPlaying(false); });
    sse.onerror = () => { setStatus("disconnected"); };
    setPlaying(true);
}

function stop() {
    if (sse) { sse.close(); sse = null; }
    setStatus("stopped");
    setPlaying(false);
}

const REPLAY_TPS = 15; // fixed "slow" pace — see /api/replay?tps=N

function setPlaying(on) {
    const btn = document.getElementById("play-toggle");
    btn.textContent = on ? "Stop" : "Start";
    btn.setAttribute("aria-label", on ? "Stop replay" : "Start replay");
    btn.classList.toggle("playing", on);
}

function setStatus(text) {
    document.getElementById("status-dot").textContent = "● " + text;
}

function handle(ev) {
    switch (ev.kind) {
        case "meta":
            for (const [wt, info] of Object.entries(ev.turbines || {})) {
                if (panels[wt]) panels[wt].ratedKw = info.rated_kw;
            }
            setStatus("streaming");
            break;
        case "tick":
            applyTick(ev);
            break;
        case "anomaly":
            pushAnomaly(ev);
            break;
        case "newton_status":
            newtonState[ev.turbine] = ev.stage;
            setNewtonBadge();
            break;
        case "newton_prediction":
            applyNewtonPrediction(ev);
            break;
        case "newton_error":
            newtonState[ev.turbine] = "error";
            setNewtonBadge();
            console.warn("newton_error", ev);
            break;
        case "newton_done":
            newtonState[ev.turbine] = "done";
            setNewtonBadge();
            break;
        case "done":
            setStatus("ready");
            break;
    }
}

function applyTick(ev) {
    document.getElementById("t-cursor").style.width = (ev.progress * 100).toFixed(2) + "%";
    document.getElementById("t-now").textContent = ev.ts;
    for (const [wt, row] of Object.entries(ev.wts || {})) {
        const p = panels[wt];
        if (!p) continue;
        p.rpm = row.rpm || 0;
        // Format fields.
        setField(p.fields.wind, fmt(row.wind, 1));
        setField(p.fields.power, Math.round(row.power));
        setField(p.fields.rpm, fmt(row.rpm, 1));
        setField(p.fields.pitch, fmt(row.pitch, 1));
        setField(p.fields.gear_c, fmt(row.gear_c, 0));
        // Power bar — clamp to 0..rated.
        const pct = Math.max(0, Math.min(100, (row.power / p.ratedKw) * 100));
        p.powerBar.style.width = pct.toFixed(1) + "%";
        // Sparkline — rolling 96-tick window (~4 days at hourly cadence).
        const labels = p.spark.data.labels;
        const power = p.spark.data.datasets[0].data;
        const wind = p.spark.data.datasets[1].data;
        labels.push(ev.ts.slice(5, 16));
        power.push(Math.max(0, row.power));
        wind.push(row.wind);
        const cap = 96;
        if (labels.length > cap) {
            labels.splice(0, labels.length - cap);
            power.splice(0, power.length - cap);
            wind.splice(0, wind.length - cap);
        }
        p.spark.update("none");
    }
}

function pushAnomaly(ev) {
    // `silent` = the first baseline-healthy read: flip the panel, no feed row.
    if (!ev.silent) {
        const recovery = ev.to === "healthy";
        const cls = recovery ? "recovery" : ev.severity || "info";
        const ctx = ev.window_start && ev.window_end
            ? `window ${ev.window_start.slice(5, 16)} → ${ev.window_end.slice(5, 16)}`
            : "";
        appendFeedItem(ev.turbine, cls, ev.message, ctx, ev.ts);
    }

    // Flip panel state on debounced transition (incl. the first commit).
    const p = panels[ev.turbine];
    if (!p || !ev.to) return;
    const newState = (ev.to === "fault" || ev.to === "healthy") ? ev.to : "analysing";
    if (newState === p.state) return;
    const intoFault = newState === "fault";
    p.state = newState;
    p.el.dataset.state = newState;
    p.stateLabel.textContent = labelOf(newState);
    const flashClass = intoFault ? "flash-fault" : "flash-recover";
    p.el.classList.remove("flash-fault", "flash-recover");
    void p.el.offsetWidth;
    p.el.classList.add(flashClass);
}

function appendFeedItem(wt, classes, msg, ctx, ts) {
    const list = document.getElementById("anomaly-list");
    const li = document.createElement("li");
    li.className = classes;
    li.innerHTML = `
        <span class="wt-tag">WT${wt}</span>
        <span class="msg">${escapeHtml(msg)}</span>
        <span class="ctx">${escapeHtml(ctx)}</span>
        <span class="ts">${escapeHtml(ts || "")}</span>
    `;
    list.prepend(li);
    while (list.children.length > 60) list.removeChild(list.lastChild);
}

// Match the server-side STRONG_MARGIN filter: only verdicts with the winning
// class ahead by ≥3 votes change the badge class / trigger the pulse. Weak
// 3-2 ties still update the counter (so users see total throughput) but don't
// churn the visible badge.
const STRONG_MARGIN = 3;

function applyNewtonPrediction(ev) {
    const p = panels[ev.turbine];
    if (!p) return;
    p.newtonCount += 1;
    if (ev.class === "fault") p.newtonFaultCount += 1;

    // Panel state is driven by server-side debounced `anomaly` events
    // (pushAnomaly), not per-prediction — single 2-1 KNN flips would
    // otherwise churn the UI.
    const v = p.newtonVerdict;
    const votes = ev.votes || {};
    const f = votes.fault ?? 0;
    const h = votes.healthy ?? 0;

    // Vote counter always updates so users can see total predictions processed.
    v.querySelector(".nv-votes").textContent = `${h}H/${f}F · ${p.newtonFaultCount}/${p.newtonCount}`;

    // Badge class + pulse only on strong verdicts. This stops the
    // post-replay flicker where every 3-2 tie was repainting the badge.
    const cls = ev.class;
    if (!cls || cls === "unknown") return;
    const winner = Math.max(h, f);
    const loser = Math.min(h, f);
    if (winner - loser < STRONG_MARGIN) return;

    v.classList.remove("healthy", "fault", "pulse");
    v.classList.add(cls);
    void v.offsetWidth;
    v.classList.add("pulse");
    v.querySelector(".nv-class").textContent = cls;
}

function labelOf(state) {
    return ({
        analysing: "analysing",
        healthy: "healthy",
        fault: "fault",
        unknown: "unknown",
    })[state] || state;
}

function fmt(v, digits) {
    if (v === null || v === undefined) return "—";
    return Number(v).toFixed(digits);
}
function setField(node, val) { if (node) node.textContent = val; }
function escapeHtml(s) {
    return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

// ---------- start / stop ----------
document.getElementById("play-toggle").addEventListener("click", () => {
    if (sse) stop();
    else connect(REPLAY_TPS);
});

setStatus("idle");

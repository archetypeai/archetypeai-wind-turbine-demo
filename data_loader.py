"""Load and sample Penmanshiel SCADA data.

The turbine CSVs have leading `# ...` comment lines, then a header row, then 10-minute
rows for the whole year. The status CSVs are a flat table of events.
"""
from __future__ import annotations

import csv
import glob
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# Signals we send to Newton for anomaly analysis. Chosen for signal density and
# fault visibility — power, rotation, temps and pitch are the usual SCADA tells.
SIGNAL_COLUMNS: list[str] = [
    "Date and time",
    "Wind speed (m/s)",
    "Power (kW)",
    "Rotor speed (RPM)",
    "Generator RPM (RPM)",
    "Gear oil temperature (°C)",
    "Generator bearing front temperature (°C)",
    "Generator bearing rear temperature (°C)",
    "Front bearing temperature (°C)",
    "Nacelle ambient temperature (°C)",
    "Grid frequency (Hz)",
    "Blade angle (pitch position) A (°)",
    "Capacity factor",
    "Nacelle position (°)",
    "Wind direction (°)",
]

# Short, JSON-friendly keys for the Newton payload.
SIGNAL_ALIASES: dict[str, str] = {
    "Date and time": "ts",
    "Wind speed (m/s)": "wind_mps",
    "Power (kW)": "power_kw",
    "Rotor speed (RPM)": "rotor_rpm",
    "Generator RPM (RPM)": "gen_rpm",
    "Gear oil temperature (°C)": "gear_oil_c",
    "Generator bearing front temperature (°C)": "gen_brg_front_c",
    "Generator bearing rear temperature (°C)": "gen_brg_rear_c",
    "Front bearing temperature (°C)": "main_brg_c",
    "Nacelle ambient temperature (°C)": "nacelle_c",
    "Grid frequency (Hz)": "grid_hz",
    "Blade angle (pitch position) A (°)": "pitch_deg",
    "Capacity factor": "capacity_factor",
}


@dataclass
class TurbineInfo:
    wt_id: str  # "01" .. "15"
    title: str  # "Penmanshiel 01"
    turbine_csv: str
    status_csv: str
    rated_power_kw: int


def _strip_comments_and_read(path: str) -> pd.DataFrame:
    """Read a Greenbyte-style CSV that has `# ...` comment lines before the header.

    The header line itself begins with `# ` (e.g. `# Date and time,Wind speed...`).
    We locate it as the *last* contiguous `#` line, strip the leading `# `, then read
    the rest via pandas.
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        lines = fh.readlines()
    header_idx = 0
    for i, line in enumerate(lines):
        if line.startswith("#"):
            header_idx = i
        else:
            break
    header_line = lines[header_idx].lstrip("#").lstrip()
    data = [header_line] + lines[header_idx + 1:]
    from io import StringIO
    return pd.read_csv(StringIO("".join(data)), low_memory=False)


def discover_turbines() -> list[TurbineInfo]:
    """Enumerate Penmanshiel turbines from the SCADA folders + static metadata."""
    static = pd.read_csv(os.path.join(DATA_DIR, "Penmanshiel_WT_static.csv"))
    static = static.dropna(subset=["Title"])
    rated_by_title = {
        row["Title"]: int(row["Rated power (kW)"])
        for _, row in static.iterrows()
        if pd.notna(row.get("Rated power (kW)"))
    }

    turbines: list[TurbineInfo] = []
    for folder in ("Penmanshiel_SCADA_2019_WT01-10_3112", "Penmanshiel_SCADA_2019_WT11-15_3117"):
        for turbine_path in sorted(glob.glob(os.path.join(DATA_DIR, folder, "Turbine_Data_*.csv"))):
            fname = os.path.basename(turbine_path)
            # Turbine_Data_Penmanshiel_01_2019-01-01_..._1042.csv
            wt_id = fname.split("Penmanshiel_")[1].split("_")[0]
            title = f"Penmanshiel {wt_id}"
            status_path = turbine_path.replace("Turbine_Data_", "Status_")
            turbines.append(
                TurbineInfo(
                    wt_id=wt_id,
                    title=title,
                    turbine_csv=turbine_path,
                    status_csv=status_path if os.path.exists(status_path) else "",
                    rated_power_kw=rated_by_title.get(title, 2050),
                )
            )
    return turbines


def load_turbine_window(
    info: TurbineInfo,
    start: str,
    end: str,
    max_rows: int = 144,
) -> pd.DataFrame:
    """Return rows in [start, end), downsampled to ≤ max_rows. Times are ISO strings."""
    df = _strip_comments_and_read(info.turbine_csv)
    ts_col = "Date and time"
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.dropna(subset=[ts_col])

    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)
    mask = (df[ts_col] >= start_ts) & (df[ts_col] < end_ts)
    window = df.loc[mask, SIGNAL_COLUMNS].copy()

    if len(window) > max_rows:
        # Even-step downsample, rounding up so the full window is covered.
        step = (len(window) + max_rows - 1) // max_rows
        window = window.iloc[::step]
    return window.reset_index(drop=True)


def to_compact_events(window: pd.DataFrame, wt_id: str) -> list[dict]:
    """Convert sampled SCADA rows to compact JSON events for the Newton sensor lens."""
    events: list[dict] = []
    for _, row in window.iterrows():
        event: dict = {"turbine": wt_id}
        for col, alias in SIGNAL_ALIASES.items():
            val = row.get(col)
            if pd.isna(val):
                continue
            if isinstance(val, pd.Timestamp):
                event[alias] = val.strftime("%Y-%m-%d %H:%M:%S")
            elif isinstance(val, float):
                # Round to keep payloads small.
                event[alias] = round(float(val), 2)
            else:
                event[alias] = val
        events.append(event)
    return events


def load_status_window(info: TurbineInfo, start: str, end: str) -> list[dict]:
    """Return status events overlapping the window."""
    if not info.status_csv:
        return []
    df = _strip_comments_and_read(info.status_csv)
    if "Timestamp start" not in df.columns:
        return []
    df["Timestamp start"] = pd.to_datetime(df["Timestamp start"], errors="coerce")
    df = df.dropna(subset=["Timestamp start"])
    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)
    mask = (df["Timestamp start"] >= start_ts) & (df["Timestamp start"] < end_ts)
    sub = df.loc[mask, ["Timestamp start", "Duration", "Status", "Code", "Message"]].copy()
    sub["Timestamp start"] = sub["Timestamp start"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return sub.fillna("").to_dict(orient="records")


# A few curated demo presets so the user can launch the side-by-side comparison
# without hunting for a fault timestamp. Times are UTC.
DEMO_SCENARIOS: list[dict] = [
    {
        "id": "freq_converter_wt01",
        "label": "Frequency converter error (WT01, Nov 2-7 2019)",
        "wt_faulty": "01",
        "wt_healthy": "09",
        "start": "2019-11-02 12:00:00",
        "end":   "2019-11-08 00:00:00",
    },
    {
        "id": "freq_converter_wt05",
        "label": "Frequency converter error (WT05, Mar 7-10 2019)",
        "wt_faulty": "05",
        "wt_healthy": "09",
        "start": "2019-03-07 18:00:00",
        "end":   "2019-03-10 12:00:00",
    },
    {
        "id": "set_point_axis_wt14",
        "label": "Set-point/actual mismatch axis 1 (WT14, Sep 7-9 2019)",
        "wt_faulty": "14",
        "wt_healthy": "11",
        "start": "2019-09-07 18:00:00",
        "end":   "2019-09-09 12:00:00",
    },
    {
        "id": "grid_event_2019_08_01",
        "label": "Grid event Aug 1 2019 (WT06 vs WT10)",
        "wt_faulty": "06",
        "wt_healthy": "10",
        "start": "2019-08-01 06:00:00",
        "end":   "2019-08-02 12:00:00",
    },
]

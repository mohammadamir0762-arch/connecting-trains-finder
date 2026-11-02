"""Past train runs, as one table of halts with everything the delay model needs.

Sources (all real records of when trains actually arrived):
- 2025-02-08 .. 2026-02-07: ~1.09M runs (Hugging Face pjay-in/indian-railways-delays,
  converted from a Kaggle set; its delays track an independent IIT Kharagpur
  set train by train, and it shows the real fog and festival seasons).
- 2026-10 onwards: the same Hugging Face dataset's own daily NTES crawl, with
  actual departure times and whether each time was reported by the station.
- September 2024: the RSTGCN (IIT Kharagpur) records the timetable comes from.

Every source ends up with the same columns, so features are computed one way.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .seasons import is_festival_season, is_winter_fog_season
from .timetable import DATA_DIR, DELAYS_PATH, MINUTES_PER_DAY, Timetable

YEAR_DIR = DATA_DIR / "raw" / "ntes" / "2025-02-08_2026-02-07"
RECENT_DIR = DATA_DIR / "raw" / "ntes" / "data"
RUN = ["train", "start_date"]
MAX_DELAY = 24 * 60

_HF_COLUMNS = ["train_number", "train_name", "start_date", "status", "seq", "station_code",
               "scheduled_arrival", "scheduled_departure", "arrival_delay_min", "departure_delay_min",
               "arrival_reported", "departure_reported"]


def _hf_file(path: Path, percent: int, completed_only: bool) -> pd.DataFrame:
    t = pq.read_table(path, columns=_HF_COLUMNS).to_pandas()
    if completed_only:
        t = t[t["status"] == "completed"]
    if percent < 100:  # keep whole runs, chosen by a stable hash
        key = t["train_number"].astype(str) + t["start_date"].astype(str)
        t = t[pd.util.hash_pandas_object(key, index=False).to_numpy() % 100 < percent]
    t = t.sort_values(["train_number", "start_date", "seq"])
    start = pd.to_datetime(t["start_date"]).dt.tz_localize("Asia/Kolkata")
    arr = (t["scheduled_arrival"] - start).dt.total_seconds() / 60
    dep = (t["scheduled_departure"] - start).dt.total_seconds() / 60
    timed = (arr.notna() | dep.notna()).to_numpy()  # a few halts of disrupted runs have no times
    t, arr, dep = t[timed], arr[timed], dep[timed]
    return pd.DataFrame({
        "train": t["train_number"].astype(str).to_numpy(),
        "train_name": t["train_name"].astype(str).to_numpy(),
        "start_date": pd.to_datetime(t["start_date"]).to_numpy(),
        "station": t["station_code"].astype(str).to_numpy(),
        "arr_el": arr.fillna(dep).to_numpy(np.float32),  # minutes since the start day's midnight
        "dep_el": dep.fillna(arr).to_numpy(np.float32),
        "arr_delay": t["arrival_delay_min"].to_numpy(np.float32),
        "dep_delay": t["departure_delay_min"].to_numpy(np.float32),
        "arr_reported": t["arrival_reported"].to_numpy(),
        "dep_reported": t["departure_reported"].to_numpy(),
    })


def load_hf(directory: Path, percent: int = 100, completed_only: bool = False) -> pd.DataFrame:
    """Every Parquet day-file in `directory`, keeping `percent`% of runs."""
    parts = [_hf_file(path, percent, completed_only) for path in sorted(directory.rglob("*.parquet"))]
    halts = pd.concat(parts, ignore_index=True)
    # Files hold runs by the day they finished, so one run never spans two files.
    return _finish(halts)


def load_rstgcn(timetable: Timetable, path: Path = DELAYS_PATH) -> pd.DataFrame:
    """September 2024 runs, timed by the timetable they were recorded against."""
    records = pd.read_csv(path, dtype={"train": str}, usecols=["train", "date", "station", "arr_delay", "dep_delay"])
    stops = pd.DataFrame(
        [(n, s.station, s.arr, s.dep, t.name) for n, t in timetable.trains.items() for s in t.stops],
        columns=["train", "station", "arr_el", "dep_el", "train_name"],
    ).drop_duplicates(["train", "station"])
    halts = records.merge(stops, on=["train", "station"])
    halts["start_date"] = pd.to_datetime(halts["date"])
    halts = halts.sort_values(RUN + ["arr_el"]).drop(columns="date")
    halts["arr_reported"] = halts["dep_reported"] = True
    return _finish(halts.astype({"arr_el": np.float32, "dep_el": np.float32,
                                 "arr_delay": np.float32, "dep_delay": np.float32}))


def _finish(halts: pd.DataFrame) -> pd.DataFrame:
    """Route position of each halt, and the delay the model predicts for it."""
    halts = halts.reset_index(drop=True)
    run = halts.groupby(RUN, sort=False)
    halts["stop_index"] = run.cumcount().astype(np.int16)
    halts["n_stops"] = run["station"].transform("size").astype(np.int16)
    start = run["dep_el"].transform("first")
    total = (run["arr_el"].transform("last") - start).clip(lower=1)
    halts["arr_hours"] = ((halts["arr_el"] - start) / 60).astype(np.float32)
    halts["dep_hours"] = ((halts["dep_el"] - start) / 60).astype(np.float32)
    halts["route_share"] = ((halts["arr_el"] - start) / total).astype(np.float32)
    halts["arr_clock"] = (halts["arr_el"] % MINUTES_PER_DAY / 60).astype(np.float32)
    halts["dep_clock"] = (halts["dep_el"] % MINUTES_PER_DAY / 60).astype(np.float32)
    halts["halt_minutes"] = (halts["dep_el"] - halts["arr_el"]).astype(np.float32)
    halts["weekday"] = halts["start_date"].dt.weekday.astype(np.int8)
    halts["halt_date"] = halts["start_date"] + pd.to_timedelta(halts["arr_el"] // MINUTES_PER_DAY, unit="D")
    # How late the train is at this halt: leaving the first station, arriving everywhere else.
    # At intermediate halts trains leave about as late as they arrive (NTES 2026: the mean
    # gap is 2.4 min), so this one number serves as both.
    delay = np.where(halts["stop_index"] == 0, halts["dep_delay"], halts["arr_delay"])
    halts["delay"] = np.clip(delay, 0, MAX_DELAY - 1).astype(np.float32)
    halts.loc[np.isnan(delay), "delay"] = np.nan
    return halts


def add_conditions(halts: pd.DataFrame, rain_mm: np.ndarray) -> pd.DataFrame:
    """Rain at the halt and along the route so far that day, festival season, fog season."""
    halts["rain_mm"] = rain_mm.astype(np.float32)
    keys = [halts[c] for c in RUN]
    so_far = halts["rain_mm"].fillna(0).groupby(keys, sort=False).cumsum()
    counted = halts["rain_mm"].notna().astype(np.float32).groupby(keys, sort=False).cumsum()
    halts["route_rain_mm"] = (so_far / counted.replace(0, np.nan)).astype(np.float32)
    days = halts["halt_date"].dt.date
    for column, in_season in (("festival", is_festival_season), ("winter", is_winter_fog_season)):
        flags = {d: in_season(d) for d in days.unique()}
        halts[column] = days.map(flags).astype(np.int8)
    return halts


def routes(halts: pd.DataFrame) -> list[list[str]]:
    """One station sequence per train (from its first run), for placing stations on a map."""
    first = halts.drop_duplicates("train")[RUN]
    runs = halts.merge(first, on=RUN)
    return runs.groupby("train", sort=False)["station"].agg(list).tolist()

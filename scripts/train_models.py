"""Train and score the delay and waitlist models, then save them to models/.

Delay model (railroute/delay_model.py), on real runs:
  2025-02-08 .. 2026-02-07  ~30% of runs, split by whole weeks spread over the year:
                            4 weeks in 6 to learn, 1 to choose/stop (validation), 1 held out (test)
  September 2024 (RSTGCN)   calibrate miss risk on connections that really happened
  October 2026 (NTES)       a later year from a different collection: final connection test

Every halt carries the day's rain (IMD; Open-Meteo for 2026), festival season and
winter fog season, so the model learns what each does to delays.

Run:  .venv/bin/python scripts/train_models.py
"""

import json
import sys
import time
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from railroute import delay_model, history, waitlist_model, weather  # noqa: E402
from railroute.deep_model import DeepBackend  # noqa: E402
from railroute.timetable import DATA_DIR, load_timetable  # noqa: E402

METRICS_PATH = delay_model.MODEL_PATH.parent / "metrics.json"
RECENT_RAIN_PATH = DATA_DIR / "processed" / "rain_recent.json"
YEAR_PERCENT = 30


def with_rain(halts: pd.DataFrame, rain: weather.RainArchive, coords: dict, label: str) -> pd.DataFrame:
    """Rain from IMD where it has published the year, Open-Meteo (cached) otherwise."""
    mm = rain.rain(halts["station"].to_numpy(), halts["halt_date"].to_numpy())
    later = ~np.array([rain.covers(d) for d in halts["halt_date"].dt.date])
    if later.any() and label == "recent":
        days = halts.loc[later, "halt_date"].dt.date
        cache = json.loads(RECENT_RAIN_PATH.read_text()) if RECENT_RAIN_PATH.exists() else {}
        wanted = {s for s in halts.loc[later, "station"].unique() if s in coords}
        key = f"{days.min()}_{days.max()}"
        cached = cache.setdefault(key, {})
        missing = {s for s in wanted if f"{s}|{days.min()}" not in cached}
        if missing:
            print(f"  fetching {len(missing):,} stations' rain for {key} from Open-Meteo…", flush=True)
            fetched = weather.open_meteo_rain({s: coords[s] for s in missing}, days.min(), days.max())
            cached.update({f"{s}|{d}": v for (s, d), v in fetched.items()})
            RECENT_RAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
            RECENT_RAIN_PATH.write_text(json.dumps(cache))
        lookup = cache[key]
        mm[later] = [lookup.get(f"{s}|{d}", np.nan) for s, d in zip(halts.loc[later, "station"], days)]
    covered = np.isfinite(mm).mean()
    print(f"  {label}: rain known for {covered:.0%} of halts", flush=True)
    return history.add_conditions(halts, np.asarray(mm, dtype=np.float32))


def train_delay_model() -> dict:
    started = time.time()
    timetable = load_timetable()
    year = history.load_hf(history.YEAR_DIR, percent=YEAR_PERCENT)
    recent = history.load_hf(history.RECENT_DIR, completed_only=True)
    september_2024 = history.load_rstgcn(timetable)
    print(f"Halts: {len(year):,} from 2025-26, {len(recent):,} from Oct 2026, "
          f"{len(september_2024):,} from Sep 2024", flush=True)

    routes = history.routes(year) + history.routes(recent) + [[s.station for s in t.stops]
                                                              for t in timetable.trains.values()]
    coords = weather.station_coordinates(routes)
    rain = weather.RainArchive(coords)
    year = with_rain(year, rain, coords, "2025-26").dropna(subset=["delay"])
    recent = with_rain(recent, rain, coords, "recent")
    september_2024 = with_rain(september_2024, rain, coords, "Sep 2024").dropna(subset=["delay"])

    week = delay_model.week_of(year["start_date"]) % 6
    training, validation, test = year[~np.isin(week, (2, 5))], year[week == 5], year[week == 2]
    print(f"Weeks: {len(training):,} training / {len(validation):,} validation / {len(test):,} test halts",
          flush=True)

    model = delay_model.fit(training, validation, delay_model.load_zones(),
                            {"trees": delay_model.TreeBackend(), "deep": DeepBackend()})
    choice = delay_model.choose_backends(model, validation)
    print(f"Validation log loss: {choice} -> using {'+'.join(model.use)}", flush=True)
    delay_model.fit_calibration(model, september_2024)

    model.metrics = {
        "validation_log_loss": choice,
        "using": "+".join(model.use),
        "test_halts": delay_model.halt_metrics(model, test),
        "conditions": delay_model.condition_effects(model, test),
        "observed_by_season": delay_model.observed_by_season(year, model.zones),
        "connections": delay_model.connection_metrics(model, recent),
        "trained_on": {"halts": int(len(training)), "runs_percent": YEAR_PERCENT,
                       "from": str(year["start_date"].min().date()), "to": str(year["start_date"].max().date())},
        "trained_seconds": round(time.time() - started),
    }
    joblib.dump(model, delay_model.MODEL_PATH)
    print(json.dumps(model.metrics, indent=2), flush=True)
    return model.metrics


def train_waitlist_model() -> dict:
    model = waitlist_model.fit_and_evaluate(waitlist_model.load_tickets())
    joblib.dump(model, waitlist_model.MODEL_PATH)
    print("Waitlist model:", json.dumps(model.metrics, indent=2), flush=True)
    return model.metrics


if __name__ == "__main__":
    delay_model.MODEL_PATH.parent.mkdir(exist_ok=True)
    if "--waitlist-only" in sys.argv:  # the delay model takes ~10 min; this takes seconds
        metrics = json.loads(METRICS_PATH.read_text())
        metrics["waitlist"] = train_waitlist_model()
        METRICS_PATH.write_text(json.dumps(metrics, indent=2))
        sys.exit()
    metrics = {"waitlist": train_waitlist_model(), "delay": train_delay_model(),
               "trained": date.today().isoformat()}
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"Saved models and {METRICS_PATH}")

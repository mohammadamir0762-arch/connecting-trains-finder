"""Load the Indian Railways timetable into a structure that is quick to search.

Source: RSTGCN dataset (IIT Kharagpur), September 2024: every train's route,
plus the observed delay at every halt of every run that month. The runs tell
us which weekdays each train actually operates on.
"""

from __future__ import annotations

import pickle
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RSTGCN_DIR = DATA_DIR / "raw" / "rstgcn" / "Indian-Railway-Network-and-Delays"
ROUTES_PATH = RSTGCN_DIR / "train_routes_Sep2024.csv"
DELAYS_PATH = RSTGCN_DIR / "train_routes_delays_Sep2024.csv"
ZONES_PATH = RSTGCN_DIR / "stations_zones_mapping.json"
CACHE_PATH = DATA_DIR / "processed" / "timetable.pkl"

MINUTES_PER_DAY = 24 * 60
WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")

# A weekday counts as a running day when the train ran on at least this share
# of the count of its busiest weekday; this drops one-off extra or rescheduled runs.
RUNNING_DAY_SHARE = 0.4

# The source abbreviates or uses new official names travellers won't search for.
STATION_NAMES = {
    "MAS": "Chennai Central",
    "LKO": "Lucknow Charbagh",
    "LJN": "Lucknow Jn",
    "CSMT": "Mumbai CSMT",
    "LTT": "Mumbai Lokmanya Tilak T",
    "MMCT": "Mumbai Central",
    "VGLJ": "Jhansi Jn",
    "DDU": "Pt Deen Dayal Upadhyaya Jn (Mughalsarai)",
    "YPR": "Bengaluru Yesvantpur",
    "SBC": "Bengaluru KSR City",
    "SMVB": "Bengaluru SMVT",
    "RKMP": "Bhopal Rani Kamalapati",
    "KOAA": "Kolkata Terminal",
    "KCVL": "Thiruvananthapuram North",
    "PRYJ": "Prayagraj Jn",
    "BSBS": "Banaras",
}


@dataclass(frozen=True)
class Stop:
    station: str
    # Minutes since 00:00 on the day the train leaves its first station,
    # so a halt on the 2nd day at 06:30 is 1440 + 390.
    arr: int
    dep: int
    km: int


@dataclass(frozen=True)
class Train:
    number: str
    name: str
    stops: tuple[Stop, ...]
    running_days: frozenset[int]  # weekdays (Mon=0) the train leaves its first station

    def runs_on(self, start: date) -> bool:
        return start.weekday() in self.running_days


@dataclass
class Timetable:
    trains: dict[str, Train]
    station_names: dict[str, str]
    # station code -> [(train number, index of the station in train.stops)]
    station_index: dict[str, list[tuple[str, int]]]


def _clock_minutes(value: str) -> int:
    clock = datetime.strptime(value, "%I:%M %p")
    return clock.hour * 60 + clock.minute


def _elapsed(clock_times: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """(arrival, departure) clock minutes -> minutes since the first day's midnight.

    Consecutive halts are never 24h apart, so each step forward is the clock
    difference modulo a day.
    """
    elapsed = []
    previous = None
    for arr, dep in clock_times:
        arr_elapsed = arr if previous is None else previous + (arr - previous % MINUTES_PER_DAY) % MINUTES_PER_DAY
        dep_elapsed = arr_elapsed + (dep - arr) % MINUTES_PER_DAY
        elapsed.append((arr_elapsed, dep_elapsed))
        previous = dep_elapsed
    return elapsed


def _is_plausible(stops: list[Stop]) -> bool:
    """Reject timings that can't be real: no time passing, or over 180 km/h between halts."""
    if len(stops) < 2 or stops[-1].arr <= stops[0].dep:
        return False
    for before, after in zip(stops, stops[1:]):
        km, minutes = after.km - before.km, after.arr - before.dep
        if km < 0 or (km > 30 and (minutes <= 0 or km / minutes * 60 > 180)):
            return False
    return True


def running_days(run_dates) -> frozenset[int]:
    counts = Counter(d.weekday() for d in run_dates)
    if not counts:
        return frozenset()
    busiest = max(counts.values())
    return frozenset(day for day, n in counts.items() if n >= RUNNING_DAY_SHARE * busiest)


def build_timetable(routes_path: Path = ROUTES_PATH, delays_path: Path = DELAYS_PATH) -> Timetable:
    routes = pd.read_csv(routes_path, dtype={"trainNumber": str}).sort_values(["trainNumber", "stnSerialNumber"])
    delays = pd.read_csv(delays_path, dtype={"train": str}, usecols=["train", "date", "station", "sch_arr", "sch_dep"])

    run_dates = {
        number: [date.fromisoformat(d) for d in dates]
        for number, dates in delays.groupby("train")["date"].unique().items()
    }
    # ~9% of routes list the same clock time at every halt. The per-run records
    # carry the real schedule, so use each halt's most common scheduled time.
    usual_times = (
        delays.groupby(["train", "station", "sch_arr", "sch_dep"]).size()
        .sort_values().groupby(level=["train", "station"]).tail(1)
        .reset_index(level=["sch_arr", "sch_dep"])[["sch_arr", "sch_dep"]]
    )

    def make_stops(number: str, rows: pd.DataFrame, use_runs: bool) -> list[Stop]:
        clock = []
        for row in rows.itertuples():
            arr, dep = row.arrivalTime, row.departureTime
            if use_runs and (number, row.station_code) in usual_times.index:
                arr, dep = usual_times.loc[(number, row.station_code)]
            clock.append((_clock_minutes(arr), _clock_minutes(dep)))
        return [
            Stop(code, arr, dep, int(km))
            for code, km, (arr, dep) in zip(rows.station_code, rows.distance, _elapsed(clock))
        ]

    trains = {}
    station_names = {}
    station_index: dict[str, list[tuple[str, int]]] = {}
    for number, rows in routes.groupby("trainNumber"):
        rows = rows.drop_duplicates("station_code")
        stops = make_stops(number, rows, use_runs=False)
        if not _is_plausible(stops):
            stops = make_stops(number, rows, use_runs=True)
            if not _is_plausible(stops):
                continue
        days = running_days(run_dates.get(number, []))
        if not days:
            continue

        trains[number] = Train(number, rows.trainName.iloc[0].strip(), tuple(stops), days)
        for i, stop in enumerate(stops):
            station_index.setdefault(stop.station, []).append((number, i))
        for code, name in zip(rows.station_code, rows.station_name):
            if code not in station_names or code in STATION_NAMES:
                fallback = name.strip().title() if isinstance(name, str) else code
                station_names[code] = STATION_NAMES.get(code, fallback)

    return Timetable(trains, station_names, station_index)


def load_timetable() -> Timetable:
    """Build the timetable from the raw CSVs once, then reuse a pickled copy."""
    # A deployed copy ships only the cache, without the raw CSVs it was built from.
    sources = [path for path in (ROUTES_PATH, DELAYS_PATH) if path.exists()]
    if CACHE_PATH.exists() and all(CACHE_PATH.stat().st_mtime >= path.stat().st_mtime for path in sources):
        with open(CACHE_PATH, "rb") as f:
            return pickle.load(f)

    timetable = build_timetable()
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(timetable, f)
    return timetable

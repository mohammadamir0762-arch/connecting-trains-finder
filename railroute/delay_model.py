"""Will I make the connection? Delay distributions learned from a year of real runs.

For a train at a halt the model predicts how late it will be, as probabilities
over delay ranges (0-5 min, 5-15, ... 12-24 h), from what is known when
booking plus the conditions on the day:

- the train's, the station's and the train-at-station delay history;
- the kind of train, where the halt sits on its route, time of day, weekday;
- rain at the halt and along the route that day (IMD gridded rainfall),
  festival travel season and winter fog season.

Because the conditions are features learned from real days, a scenario
("heavy rain", "Diwali rush", "winter fog") is answered by setting them.

Two model families learn the same task and are compared on the same held-out
weeks: gradient-boosted trees and a neural network with learned embeddings
for every train and station (deep_model.py). The app uses whichever does
better on validation weeks, or their average.

A connection is missed when train 1 arrives later than train 2 leaves, minus
the time to change: P(A - D > layover - change time), with A and D from the
two predicted distributions, then calibrated on connections that really
happened.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression

from .search import Journey
from .seasons import Scenario
from .timetable import MINUTES_PER_DAY, ZONES_PATH, Timetable

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "delay_model.joblib"

BUCKET_EDGES = np.array([0, 5, 15, 30, 60, 90, 120, 180, 240, 360, 480, 720, 1440])
N_BUCKETS = len(BUCKET_EDGES) - 1
# Each bucket's probability is spread evenly over a few minute values inside it.
POINTS_PER_BUCKET = 4
SUPPORT = (
    BUCKET_EDGES[:-1, None]
    + (np.arange(POINTS_PER_BUCKET) + 0.5) / POINTS_PER_BUCKET * np.diff(BUCKET_EDGES)[:, None]
).ravel()

PLATFORM_CHANGE_MINUTES = 15  # walk to another platform
OTHER_STATION_MINUTES = 60  # get across the city to another station
LATE_MINUTES = 30
SMOOTHING = 10  # runs of evidence before a train's own history outweighs the average
N_FOLDS = 3

TRAIN_TYPES = [
    "vande_bharat", "rajdhani", "shatabdi", "jan_shatabdi", "duronto", "tejas", "humsafar",
    "garib_rath", "superfast", "express", "passenger", "memu_demu", "special",
]
ENCODED_KEYS = {"train": ["train"], "station": ["station"], "train_station": ["train", "station"]}
ROUTE = ["stop_index", "n_stops", "arr_hours", "dep_hours", "route_share", "arr_clock", "dep_clock", "halt_minutes"]
CONDITIONS = ["rain_mm", "route_rain_mm", "festival", "winter"]
# How late each train and station usually runs in festival and fog seasons: the rush
# and the fog hit some places (Bihar, the north) far harder than others.
SEASONS = ("festival", "winter")
SEASONAL_KEYS = ("train", "station")
HISTORY = ["train_mean", "train_late", "station_mean", "station_late",
           "train_station_mean", "train_station_late", "train_station_runs"] + [
           f"{key}_{season}_mean" for season in SEASONS for key in SEASONAL_KEYS]
CATEGORICAL = ["weekday", "train_type", "zone"]
FEATURES = ROUTE + CONDITIONS + HISTORY + CATEGORICAL


def train_type(number: str, name: str) -> str:
    name = name.lower().replace(" ", "")
    for key, kind in [
        ("vande", "vande_bharat"), ("rajdhani", "rajdhani"), ("janshatabdi", "jan_shatabdi"),
        ("shatabdi", "shatabdi"), ("duronto", "duronto"), ("tejas", "tejas"),
        ("humsafar", "humsafar"), ("garib", "garib_rath"),
    ]:
        if key in name:
            return kind
    if number.startswith("0"):
        return "special"
    if number[0] == "5":
        return "passenger"
    if number[0] in "67":
        return "memu_demu"
    if number[:2] in ("12", "20", "22"):
        return "superfast"
    return "express"


def bucket_of(delay_minutes) -> np.ndarray:
    clipped = np.clip(np.asarray(delay_minutes, dtype=float), 0, BUCKET_EDGES[-1] - 1)
    return np.searchsorted(BUCKET_EDGES, clipped, side="right") - 1


def miss_probability(arrival_probs: np.ndarray, departure_probs: np.ndarray, slack: np.ndarray) -> np.ndarray:
    """P(A - D > slack) per row, for bucket probabilities of arrival delay A and departure delay D."""
    mass_a = np.repeat(arrival_probs, POINTS_PER_BUCKET, axis=1) / POINTS_PER_BUCKET
    mass_d = np.repeat(departure_probs, POINTS_PER_BUCKET, axis=1) / POINTS_PER_BUCKET
    cdf_a = np.hstack([np.zeros((len(mass_a), 1)), np.cumsum(mass_a, axis=1)])
    rows = np.arange(len(mass_a))
    missed = np.zeros(len(mass_a))
    for j, d in enumerate(SUPPORT):
        at_most = np.searchsorted(SUPPORT, slack + d, side="right")  # support points <= slack + d
        missed += mass_d[:, j] * (1 - cdf_a[rows, at_most])
    return np.clip(missed, 0, 1)


def median_delay(probs: np.ndarray) -> np.ndarray:
    cdf = np.cumsum(probs, axis=1)
    bucket = (cdf < 0.5).sum(axis=1).clip(max=N_BUCKETS - 1)
    below = np.where(bucket > 0, cdf[np.arange(len(cdf)), bucket - 1], 0.0)
    inside = probs[np.arange(len(probs)), bucket]
    share = np.divide(0.5 - below, inside, out=np.zeros_like(inside), where=inside > 0)
    return BUCKET_EDGES[bucket] + share * np.diff(BUCKET_EDGES)[bucket]


def week_of(dates: pd.Series) -> np.ndarray:
    return ((dates - pd.Timestamp("2025-02-08")).dt.days // 7).to_numpy()


@dataclass
class DelayHistory:
    """Smoothed mean delay and share of runs over 30 min late, per train, station and both,
    plus each train's and station's mean delay in festival and fog seasons."""

    stats: dict[str, pd.DataFrame] = field(default_factory=dict)
    seasonal: dict[tuple[str, str], pd.DataFrame] = field(default_factory=dict)
    prior_mean: float = 0.0
    prior_late: float = 0.0
    bucket_prior: np.ndarray | None = None  # the delay distribution ignoring every feature

    def fit(self, halts: pd.DataFrame) -> DelayHistory:
        self.prior_mean = float(halts["delay"].mean())
        self.bucket_prior = np.bincount(bucket_of(halts["delay"]), minlength=N_BUCKETS) / len(halts)
        self.prior_late = float((halts["delay"] > LATE_MINUTES).mean())
        late = (halts["delay"] > LATE_MINUTES).rename("late")
        values = pd.concat([halts["delay"], late], axis=1)
        self.stats = {
            name: values.groupby([halts[k] for k in keys]).agg(
                total=("delay", "sum"), late=("late", "sum"), runs=("delay", "size"))
            for name, keys in ENCODED_KEYS.items()
        }
        self.seasonal = {
            (key, season): halts.loc[halts[season] == 1].groupby(key)["delay"].agg(total="sum", runs="size")
            for season in SEASONS for key in SEASONAL_KEYS
        }
        return self

    def encode(self, halts: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=halts.index)
        for name, keys in ENCODED_KEYS.items():
            joined = halts[keys].merge(self.stats[name], left_on=keys, right_index=True, how="left")
            runs = joined["runs"].fillna(0)
            out[f"{name}_mean"] = ((joined["total"].fillna(0) + SMOOTHING * self.prior_mean)
                                   / (runs + SMOOTHING)).astype(np.float32)
            out[f"{name}_late"] = ((joined["late"].fillna(0) + SMOOTHING * self.prior_late)
                                   / (runs + SMOOTHING)).astype(np.float32)
            if name == "train_station":
                out["train_station_runs"] = runs.astype(np.float32)
        for (key, season), stats in self.seasonal.items():
            joined = halts[[key]].merge(stats, left_on=key, right_index=True, how="left")
            # Pulled towards the train's / station's usual delay when the season left little evidence.
            usual = out[f"{key}_mean"]
            out[f"{key}_{season}_mean"] = ((joined["total"].fillna(0) + SMOOTHING * usual)
                                           / (joined["runs"].fillna(0) + SMOOTHING)).astype(np.float32)
        return out

    @staticmethod
    def out_of_fold(history: pd.DataFrame, halts: pd.DataFrame) -> pd.DataFrame:
        """Encodings for `halts` (drawn from `history`) built without their own weeks' delays.

        Folds are whole weeks, so a run never informs its own features.
        """
        history_folds = week_of(history["start_date"]) % N_FOLDS
        halt_folds = week_of(halts["start_date"]) % N_FOLDS
        parts = [DelayHistory().fit(history[history_folds != k]).encode(halts[halt_folds == k])
                 for k in range(N_FOLDS)]
        return pd.concat(parts).loc[halts.index]


@dataclass
class TreeBackend:
    """Gradient-boosted trees over the feature table."""

    model: HistGradientBoostingClassifier | None = None

    def fit(self, X: pd.DataFrame, y: np.ndarray, X_val: pd.DataFrame, y_val: np.ndarray, seed: int = 0):
        self.model = HistGradientBoostingClassifier(
            max_iter=500, learning_rate=0.08, max_leaf_nodes=63, min_samples_leaf=500, l2_regularization=1.0,
            early_stopping=True, n_iter_no_change=20, categorical_features=[f in CATEGORICAL for f in FEATURES],
            random_state=seed,
        )
        self.model.fit(X[FEATURES].to_numpy(np.float32), y,
                       X_val=X_val[FEATURES].to_numpy(np.float32), y_val=y_val)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        probs = np.zeros((len(X), N_BUCKETS))
        if len(X):
            probs[:, self.model.classes_] = self.model.predict_proba(X[FEATURES].to_numpy(np.float32))
        return probs


@dataclass
class DelayModel:
    history: DelayHistory
    zones: dict[str, str]
    backends: dict[str, object] = field(default_factory=dict)  # "trees", "deep"
    use: tuple[str, ...] = ("trees",)  # averaged when more than one
    calibrator: IsotonicRegression | None = None
    metrics: dict = field(default_factory=dict)

    # ---- features

    def features(self, halts: pd.DataFrame, encoded: pd.DataFrame | None = None) -> pd.DataFrame:
        """The model's input table; history encodings are computed unless given (out-of-fold)."""
        encoded = self.history.encode(halts) if encoded is None else encoded
        frame = pd.concat([halts[ROUTE + CONDITIONS + ["weekday", "train", "station"]], encoded], axis=1)
        names = halts["train_name"] if "train_name" in halts else pd.Series("", index=halts.index)
        kinds = [train_type(n, m) for n, m in zip(halts["train"], names)]
        frame["train_type"] = pd.Categorical(kinds, categories=TRAIN_TYPES).codes
        zones = halts["station"].map(self.zones).fillna("unknown")
        frame["zone"] = pd.Categorical(zones, categories=sorted(set(self.zones.values()))).codes
        for column in ("train_type", "zone"):
            frame[column] = frame[column].where(frame[column] >= 0, np.nan)
        return frame

    def probs(self, frame: pd.DataFrame, use: tuple[str, ...] | None = None) -> np.ndarray:
        use = use or self.use
        return np.mean([self.backends[name].predict_proba(frame) for name in use], axis=0)

    def calibrate(self, raw_risk: np.ndarray) -> np.ndarray:
        return raw_risk if self.calibrator is None else self.calibrator.predict(raw_risk)

    # ---- prediction for trips in the timetable

    def timetable_halts(self, timetable: Timetable, halts: list[tuple[str, int, date]],
                        scenario: Scenario) -> pd.DataFrame:
        rows = []
        for number, index, start in halts:
            train = timetable.trains[number]
            stop, first = train.stops[index], train.stops[0]
            total = max(train.stops[-1].arr - first.dep, 1)
            day = start + timedelta(days=stop.arr // MINUTES_PER_DAY)
            rows.append({
                "train": number, "train_name": train.name, "station": stop.station, "start_date": start,
                "stop_index": index, "n_stops": len(train.stops),
                "arr_hours": (stop.arr - first.dep) / 60, "dep_hours": (stop.dep - first.dep) / 60,
                "route_share": (stop.arr - first.dep) / total,
                "arr_clock": stop.arr % MINUTES_PER_DAY / 60, "dep_clock": stop.dep % MINUTES_PER_DAY / 60,
                "halt_minutes": stop.dep - stop.arr, "weekday": start.weekday(),
                "rain_mm": scenario.rain_mm, "route_rain_mm": scenario.rain_mm,
                "festival": int(scenario.festival_on(day)), "winter": int(scenario.winter_on(day)),
            })
        return pd.DataFrame(rows)

    def miss_risks(self, timetable: Timetable, connections: list[Journey], scenario: Scenario) -> np.ndarray:
        if not connections:
            return np.zeros(0)
        arrivals = self.timetable_halts(
            timetable, [(j.legs[0].train_number, j.legs[0].alight_index, j.legs[0].start_date)
                        for j in connections], scenario)
        departures = self.timetable_halts(
            timetable, [(j.legs[1].train_number, j.legs[1].board_index, j.legs[1].start_date)
                        for j in connections], scenario)
        slack = np.array([
            j.layover.total_seconds() / 60
            - (PLATFORM_CHANGE_MINUTES if j.legs[0].to_station == j.legs[1].from_station else OTHER_STATION_MINUTES)
            for j in connections
        ])
        raw = miss_probability(self.probs(self.features(arrivals)), self.probs(self.features(departures)), slack)
        return self.calibrate(raw)

    def typical_arrival_delays(self, timetable: Timetable, journeys: list[Journey], scenario: Scenario) -> np.ndarray:
        """Median predicted delay (minutes) at the final station."""
        if not journeys:
            return np.zeros(0)
        halts = self.timetable_halts(
            timetable, [(j.legs[-1].train_number, j.legs[-1].alight_index, j.legs[-1].start_date)
                        for j in journeys], scenario)
        return median_delay(self.probs(self.features(halts)))


# ---------------------------------------------------------------- training


def fit(training: pd.DataFrame, validation: pd.DataFrame, zones: dict[str, str], backends: dict[str, object],
        max_rows: int = 1_500_000, seed: int = 0) -> DelayModel:
    """Fit every backend on `training` halts, stopping early on `validation` weeks."""
    model = DelayModel(DelayHistory().fit(training), zones)
    sample = training.sample(min(max_rows, len(training)), random_state=seed)
    X = model.features(sample, DelayHistory.out_of_fold(training, sample))
    X_val = model.features(validation)
    for name, backend in backends.items():
        print(f"  fitting {name} on {len(X):,} halts…", flush=True)
        model.backends[name] = backend.fit(X, bucket_of(sample["delay"]), X_val, bucket_of(validation["delay"]),
                                           seed=seed)
    return model


def choose_backends(model: DelayModel, validation: pd.DataFrame) -> dict[str, float]:
    """Pick the backend (or average) with the lowest log loss on validation weeks."""
    from sklearn.metrics import log_loss

    X_val, y_val = model.features(validation), bucket_of(validation["delay"])
    options = [(name,) for name in model.backends] + ([tuple(model.backends)] if len(model.backends) > 1 else [])
    losses = {"+".join(o): log_loss(y_val, model.probs(X_val, o), labels=range(N_BUCKETS)) for o in options}
    model.use = min(options, key=lambda o: losses["+".join(o)])
    return {k: round(float(v), 4) for k, v in losses.items()}


def halt_metrics(model: DelayModel, halts: pd.DataFrame) -> dict:
    """Per-halt delay forecasts on held-out weeks, for every backend and the average."""
    from sklearn.metrics import log_loss, roc_auc_score

    X, y = model.features(halts), bucket_of(halts["delay"])
    no_features = np.tile(model.history.bucket_prior, (len(y), 1))
    out = {"halts": int(len(halts)),
           "log_loss_no_features": round(float(log_loss(y, no_features, labels=range(N_BUCKETS))), 4)}
    options = [(name,) for name in model.backends] + ([tuple(model.backends)] if len(model.backends) > 1 else [])
    for option in options:
        p = model.probs(X, option)
        out["+".join(option)] = {
            "log_loss": round(float(log_loss(y, p, labels=range(N_BUCKETS))), 4),
            "auc_over_1h_late": round(float(roc_auc_score(halts["delay"] > 60, p[:, BUCKET_EDGES[:-1] >= 60].sum(1))), 4),
            "median_abs_error_min": round(float(np.median(np.abs(median_delay(p) - halts["delay"]))), 1),
        }
    return out


# Zones where the festival rush is felt most: East Central (Bihar) and North Eastern (east UP).
FESTIVAL_ZONES = ("ECR", "NER")


def _late_share(model: DelayModel, frame: pd.DataFrame) -> np.ndarray:
    return model.probs(frame)[:, BUCKET_EDGES[:-1] >= 60].sum(1)


def _effect(model: DelayModel, sample: pd.DataFrame, condition: str, in_condition: np.ndarray) -> dict:
    X = model.features(sample)
    base = X.copy()
    base[["rain_mm", "route_rain_mm"]] = 0.0
    base[["festival", "winter"]] = 0
    what_if = base.copy()
    if condition == "heavy_rain":
        what_if[["rain_mm", "route_rain_mm"]] = 64.5
    else:
        what_if[condition] = 1
    over_hour = (sample["delay"] > 60).to_numpy()
    predicted = _late_share(model, X)
    return {
        "halts": int(len(sample)),
        "observed_over_1h": round(float(over_hour[in_condition].mean()), 3) if in_condition.any() else None,
        "observed_over_1h_otherwise": round(float(over_hour[~in_condition].mean()), 3),
        "predicted_over_1h": round(float(predicted[in_condition].mean()), 3) if in_condition.any() else None,
        "predicted_over_1h_otherwise": round(float(predicted[~in_condition].mean()), 3),
        "what_if_over_1h": round(float(_late_share(model, what_if).mean()), 3),
        "what_if_baseline_over_1h": round(float(_late_share(model, base).mean()), 3),
    }


def condition_effects(model: DelayModel, halts: pd.DataFrame, n: int = 50_000, seed: int = 0) -> dict:
    """What actually happened in each condition, what the model predicted, and the what-if effect.

    Observed: share of held-out halts over 1 h late in that condition vs not.
    Predicted: the model's average P(over 1 h late) for those same halts.
    What-if: the same halts re-predicted dry, off-season (baseline), then with only
    that condition switched on.
    """
    sample = halts.sample(min(n, len(halts)), random_state=seed)
    out = {
        "heavy_rain": _effect(model, sample, "heavy_rain", sample["route_rain_mm"].fillna(0).to_numpy() >= 32),
        "festival": _effect(model, sample, "festival", sample["festival"].to_numpy() == 1),
        "winter": _effect(model, sample, "winter", sample["winter"].to_numpy() == 1),
    }
    east = halts[halts["station"].map(model.zones).isin(FESTIVAL_ZONES)]
    if len(east):
        east = east.sample(min(n, len(east)), random_state=seed)
        out["festival_bihar_east_up"] = _effect(model, east, "festival", east["festival"].to_numpy() == 1)
    return out


def observed_by_season(halts: pd.DataFrame, zones: dict[str, str]) -> dict:
    """What the data itself shows, no model: share of halts over 1 h late by season and region."""
    arriving = halts[halts["stop_index"] > 0]
    east = arriving["station"].map(zones).isin(FESTIVAL_ZONES).to_numpy()
    month = arriving["start_date"].dt.month.to_numpy()
    festival, winter = arriving["festival"].to_numpy() == 1, arriving["winter"].to_numpy() == 1
    late = (arriving["delay"] > 60).to_numpy()
    periods = {
        "September, before the festivals": (month == 9) & ~festival,
        "Festival season (Durga Puja to Chhath, Holi)": festival,
        "Late Nov to early Dec (after the festivals, before fog season)": ((month == 11) | (month == 12))
        & ~festival & ~winter,
        "Winter fog season": winter,
    }
    return {name: {"bihar_east_up": round(float(late[mask & east].mean()), 3),
                   "rest_of_india": round(float(late[mask & ~east].mean()), 3)}
            for name, mask in periods.items()}


def connection_pairs(model: DelayModel, halts: pd.DataFrame, n_pairs: int, seed: int = 0,
                     use: tuple[str, ...] | None = None) -> pd.DataFrame:
    """Real connections: train 1 arrives and train 2 leaves the same station 30 min - 6 h later.

    Only times the stations actually reported count. For each pair: predicted
    miss risk (raw and calibrated), the risk from the layover alone, and
    whether it was really missed.
    """
    arrivals = halts[(halts["stop_index"] > 0) & halts["arr_reported"].astype(bool) & halts["arr_delay"].notna()]
    departures = halts[(halts["stop_index"] < halts["n_stops"] - 1) & halts["dep_reported"].astype(bool)
                       & halts["dep_delay"].notna()]
    arr_time = (arrivals["start_date"] + pd.to_timedelta(arrivals["arr_el"], unit="min")).to_numpy()
    dep_time = (departures["start_date"] + pd.to_timedelta(departures["dep_el"], unit="min")).to_numpy()
    dep_groups = {station: idx.to_numpy() for station, idx in
                  pd.Series(np.arange(len(departures))).groupby(departures["station"].to_numpy())}
    first_idx, second_idx = [], []
    for station, arr_idx in pd.Series(np.arange(len(arrivals))).groupby(arrivals["station"].to_numpy()):
        candidates = dep_groups.get(station)
        if candidates is None:
            continue
        order = candidates[np.argsort(dep_time[candidates])]
        a = arr_idx.to_numpy()
        lo = np.searchsorted(dep_time[order], arr_time[a] + np.timedelta64(30, "m"))
        hi = np.searchsorted(dep_time[order], arr_time[a] + np.timedelta64(360, "m"), side="right")
        first_idx.append(np.repeat(a, hi - lo))
        second_idx.append(order[np.concatenate([np.arange(l, h) for l, h in zip(lo, hi)] + [np.zeros(0, int)])])
    first_idx, second_idx = np.concatenate(first_idx), np.concatenate(second_idx)
    keep = arrivals["train"].to_numpy()[first_idx] != departures["train"].to_numpy()[second_idx]
    first_idx, second_idx = first_idx[keep], second_idx[keep]
    pick = np.random.default_rng(seed).choice(len(first_idx), size=min(n_pairs, len(first_idx)), replace=False)
    first_idx, second_idx = first_idx[pick], second_idx[pick]

    slack = (dep_time[second_idx] - arr_time[first_idx]) / np.timedelta64(1, "m") - PLATFORM_CHANGE_MINUTES
    missed = (arrivals["arr_delay"].to_numpy()[first_idx] - departures["dep_delay"].to_numpy()[second_idx]) > slack
    first, second = arrivals.iloc[first_idx], departures.iloc[second_idx]
    raw = miss_probability(model.probs(model.features(first), use), model.probs(model.features(second), use), slack)
    prior = np.tile(model.history.bucket_prior, (len(slack), 1))
    return pd.DataFrame({
        "raw": raw, "risk": model.calibrate(raw), "layover_only": miss_probability(prior, prior, slack),
        "missed": missed, "day": first["start_date"].to_numpy(),
    })


def fit_calibration(model: DelayModel, halts: pd.DataFrame, n_pairs: int = 200_000) -> None:
    """Map raw miss probabilities onto how often such connections were really missed."""
    pairs = connection_pairs(model, halts, n_pairs)
    model.calibrator = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(pairs["raw"], pairs["missed"])


def connection_metrics(model: DelayModel, halts: pd.DataFrame, n_pairs: int = 200_000) -> dict:
    from sklearn.metrics import brier_score_loss, roc_auc_score

    pairs = connection_pairs(model, halts, n_pairs)
    out = {
        "pairs": int(len(pairs)),
        "days": f"{pd.Timestamp(pairs['day'].min()).date()} .. {pd.Timestamp(pairs['day'].max()).date()}",
        "missed_share": round(float(pairs["missed"].mean()), 4),
        "auc": round(float(roc_auc_score(pairs["missed"], pairs["risk"])), 4),
        "brier": round(float(brier_score_loss(pairs["missed"], pairs["risk"])), 4),
        "auc_layover_only": round(float(roc_auc_score(pairs["missed"], pairs["layover_only"])), 4),
        "brier_layover_only": round(float(brier_score_loss(pairs["missed"], pairs["layover_only"])), 4),
        "calibration": [],
    }
    for name in model.backends:  # ranking quality of each backend on its own
        raw = connection_pairs(model, halts, n_pairs, use=(name,))
        out[f"auc_{name}"] = round(float(roc_auc_score(raw["missed"], raw["raw"])), 4)
    for lo, hi in [(0, 0.05), (0.05, 0.15), (0.15, 0.3), (0.3, 0.5), (0.5, 1.01)]:
        band = pairs[(pairs["risk"] >= lo) & (pairs["risk"] < hi)]
        if len(band):
            out["calibration"].append({"predicted": f"{lo:.0%}-{min(hi, 1):.0%}", "connections": int(len(band)),
                                       "actually_missed": round(float(band["missed"].mean()), 3)})
    return out


def load_zones(path: Path = ZONES_PATH) -> dict[str, str]:
    return json.loads(path.read_text())


def load() -> DelayModel | None:
    return joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None


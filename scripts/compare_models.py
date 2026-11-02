"""Compare the app's models with methods used in earlier studies, on the same data.

Delay model. Every method learns from the same 1.5M training halts and is scored on
the same held-out weeks and the same real connections from October 2026:
- historical average: each train-at-station's past delay distribution (what delay
  apps typically show);
- multinomial logistic regression (regression models, e.g. Tiong et al. 2023);
- random forest (widely used in earlier delay studies);
- the app's gradient-boosted trees, neural network and their average (already trained).

Waitlist model: logistic regression, random forest, gradient-boosted trees, neural network.

Also times the app: training, prediction, and a full search with risk scoring.
Writes models/comparison.json. Run after train_models.py:
    .venv/bin/python scripts/compare_models.py
"""

import json
import sys
import time
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import train_models  # noqa: E402
from railroute import delay_model, history, waitlist_model, weather  # noqa: E402
from railroute.cities import build_places  # noqa: E402
from railroute.delay_model import BUCKET_EDGES, CATEGORICAL, FEATURES, N_BUCKETS, bucket_of, median_delay  # noqa: E402
from railroute.search import find_journeys  # noqa: E402
from railroute.seasons import SCENARIOS  # noqa: E402
from railroute.timetable import load_timetable  # noqa: E402

OUT_PATH = delay_model.MODEL_PATH.parent / "comparison.json"
NUMERIC = [f for f in FEATURES if f not in CATEGORICAL]


class HistoricalAverage:
    """Each train-at-station's past delay distribution, falling back to the station's, then everyone's."""

    def fit(self, halts: pd.DataFrame, k: float = 10.0):
        buckets = pd.Series(bucket_of(halts["delay"]), index=halts.index)
        overall = np.bincount(buckets, minlength=N_BUCKETS) / len(buckets)
        counts = lambda keys: pd.crosstab([halts[k] for k in keys], buckets).reindex(columns=range(N_BUCKETS),
                                                                                       fill_value=0)
        station = counts(["station"])
        station = (station.to_numpy() + k * overall) / (station.sum(axis=1).to_numpy()[:, None] + k)
        self.station = pd.DataFrame(station, index=counts(["station"]).index)
        pair = counts(["train", "station"])
        self.pair, self.pair_runs = pair, pair.sum(axis=1)
        self.overall, self.k = overall, k
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        station = self.station.reindex(X["station"]).to_numpy()
        station = np.where(np.isnan(station), self.overall, station)
        index = pd.MultiIndex.from_arrays([X["train"], X["station"]])
        pair = self.pair.reindex(index).fillna(0).to_numpy()
        return (pair + self.k * station) / (pair.sum(axis=1, keepdims=True) + self.k)


def logistic_regression():
    return make_pipeline(
        ColumnTransformer([
            ("numbers", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), NUMERIC),
            ("groups", make_pipeline(SimpleImputer(strategy="constant", fill_value=-1),
                                     OneHotEncoder(handle_unknown="ignore")), CATEGORICAL),
        ]),
        LogisticRegression(max_iter=300, tol=1e-3),
    )


def random_forest():
    return RandomForestClassifier(n_estimators=100, min_samples_leaf=100, max_features="sqrt", n_jobs=-1,
                                  random_state=0)


def full_probs(model, X: pd.DataFrame) -> np.ndarray:
    out = np.zeros((len(X), N_BUCKETS))
    out[:, model.classes_] = model.predict_proba(X)
    return out


def scores(probs: np.ndarray, halts: pd.DataFrame) -> dict:
    y = bucket_of(halts["delay"])
    return {
        "log_loss": round(float(log_loss(y, np.clip(probs, 1e-12, 1), labels=range(N_BUCKETS))), 4),
        "auc_over_1h_late": round(float(roc_auc_score(halts["delay"] > 60,
                                                      probs[:, BUCKET_EDGES[:-1] >= 60].sum(1))), 4),
        "median_abs_error_min": round(float(np.median(np.abs(median_delay(probs) - halts["delay"]))), 1),
    }


class _Wrapped:
    """Lets a fitted baseline plug into DelayModel.probs / connection_pairs."""

    def __init__(self, predict):
        self.predict = predict

    def predict_proba(self, X):
        return self.predict(X)


def compare_delay_models() -> dict:
    timetable = load_timetable()
    year = history.load_hf(history.YEAR_DIR, percent=train_models.YEAR_PERCENT)
    recent = history.load_hf(history.RECENT_DIR, completed_only=True)
    routes = history.routes(year) + history.routes(recent) + [[s.station for s in t.stops]
                                                              for t in timetable.trains.values()]
    coords = weather.station_coordinates(routes)
    rain = weather.RainArchive(coords)
    year = train_models.with_rain(year, rain, coords, "2025-26").dropna(subset=["delay"])
    recent = train_models.with_rain(recent, rain, coords, "recent")
    week = delay_model.week_of(year["start_date"]) % 6
    training, test = year[~np.isin(week, (2, 5))], year[week == 2]

    model = delay_model.load()
    sample = training.sample(min(1_500_000, len(training)), random_state=0)  # the same rows the app's models saw
    X = model.features(sample, delay_model.DelayHistory.out_of_fold(training, sample))
    y = bucket_of(sample["delay"])
    X_test = model.features(test)

    results = {}

    def record(name, label, probs_fn, seconds):
        started = time.time()
        probs = probs_fn(X_test)
        per_10k = (time.time() - started) / len(X_test) * 10_000
        model.backends[name] = _Wrapped(probs_fn)
        pairs = delay_model.connection_pairs(model, recent, 200_000, use=(name,))
        results[name] = {"label": label, **scores(probs, test),
                         "connection_auc": round(float(roc_auc_score(pairs["missed"], pairs["raw"])), 4),
                         "train_seconds": round(seconds, 1), "predict_seconds_per_10k": round(per_10k, 3)}
        print(name, results[name], flush=True)

    started = time.time()
    hist = HistoricalAverage().fit(training)
    record("history", "Historical average (train at station)", hist.predict_proba, time.time() - started)

    for name, label, make in (("logistic", "Logistic regression", logistic_regression),
                              ("forest", "Random forest", random_forest)):
        print(f"fitting {label}…", flush=True)
        started = time.time()
        fitted = make().fit(X[NUMERIC + CATEGORICAL] if name == "logistic" else X[FEATURES].to_numpy(np.float32), y)
        seconds = time.time() - started
        if name == "logistic":
            record(name, label, lambda F, m=fitted: full_probs(m, F[NUMERIC + CATEGORICAL]), seconds)
        else:
            record(name, label, lambda F, m=fitted: full_probs(m, F[FEATURES].to_numpy(np.float32)), seconds)

    trained = json.loads(train_models.METRICS_PATH.read_text())["delay"]
    for name, label, use in (("trees", "Gradient-boosted trees (ours)", ("trees",)),
                             ("deep", "Neural network with embeddings (ours)", ("deep",)),
                             ("trees+deep", "Average of both (ours, used)", ("trees", "deep"))):
        started = time.time()
        probs = model.probs(X_test, use)
        per_10k = (time.time() - started) / len(X_test) * 10_000
        pairs = delay_model.connection_pairs(model, recent, 200_000, use=use)
        results[name] = {"label": label, **scores(probs, test),
                         "connection_auc": round(float(roc_auc_score(pairs["missed"], pairs["raw"])), 4),
                         "predict_seconds_per_10k": round(per_10k, 3)}
        print(name, results[name], flush=True)
    results["trees+deep"]["train_seconds"] = trained["trained_seconds"]
    results["layover_only"] = {"label": "Waiting time only", "connection_auc": trained["connections"]["auc_layover_only"]}
    return {"test_halts": int(len(test)), "train_halts": int(len(sample)), "methods": results}


def compare_waitlist_models() -> dict:
    tickets = waitlist_model.load_tickets()
    X, y = tickets[waitlist_model.FEATURES].to_numpy(float), tickets["confirmed"].to_numpy()
    train, test = train_test_split(np.arange(len(y)), test_size=0.25, random_state=0, stratify=y)
    one_hot = lambda: ColumnTransformer([("class", OneHotEncoder(handle_unknown="ignore"), [0]),
                                         ("number", StandardScaler(), [1])])
    candidates = {
        "logistic": ("Logistic regression", make_pipeline(one_hot(), LogisticRegression(max_iter=500))),
        "forest": ("Random forest", RandomForestClassifier(n_estimators=300, min_samples_leaf=20, random_state=0)),
        "trees": ("Gradient-boosted trees (ours, used)", HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, categorical_features=[True, False], monotonic_cst=[0, -1],
            random_state=0)),
        "neural_net": ("Neural network", make_pipeline(one_hot(), MLPClassifier(
            hidden_layer_sizes=(64, 32), early_stopping=True, max_iter=300, random_state=0))),
    }
    results = {}
    for name, (label, model) in candidates.items():
        started = time.time()
        model.fit(X[train], y[train])
        p = model.predict_proba(X[test])[:, 1]
        results[name] = {"label": label, "auc": round(float(roc_auc_score(y[test], p)), 4),
                         "brier": round(float(brier_score_loss(y[test], p)), 4),
                         "train_seconds": round(time.time() - started, 2)}
        print(name, results[name], flush=True)
    results["position_only"] = {"label": "Waitlist number only",
                                "auc": round(float(roc_auc_score(y[test], -X[test, 1])), 4)}
    return {"tickets": int(len(y)), "test_tickets": int(len(test)), "methods": results}


def time_searches() -> list[dict]:
    timetable = load_timetable()
    places = build_places(timetable)
    model = delay_model.load()
    rows = []
    for origin, destination in (("Chennai", "Lucknow"), ("Mumbai", "Patna"), ("Delhi", "Bengaluru"),
                                ("Kolkata", "Jaipur")):
        a, b = places[f"{origin} (all stations)"], places[f"{destination} (all stations)"]
        started = time.time()
        result = find_journeys(timetable, a, b, date(2026, 11, 2))
        searched = time.time() - started
        model.miss_risks(timetable, result.connections, SCENARIOS["Auto (from date)"])
        scored = time.time() - started - searched
        rows.append({"route": f"{origin} → {destination}", "direct": len(result.direct),
                     "connections": len(result.connections), "search_seconds": round(searched, 3),
                     "risk_seconds": round(scored, 3)})
        print(rows[-1], flush=True)
    return rows


if __name__ == "__main__":
    out = {"waitlist": compare_waitlist_models(), "search_timing": time_searches(), "delay": compare_delay_models()}
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"Saved {OUT_PATH}")

"""Will my waitlisted ticket confirm? Learned from real waitlisted tickets.

Data: Kaggle "Railway Waitinglist Dataset", the real set used in the irctc
project: ~53k waitlisted tickets with the class, the waitlist number when the
ticket was booked, and whether it was confirmed by chart preparation.

The app's question is "IRCTC shows WL n right now; if I book, will it
confirm?", so the model learns exactly that: class and waitlist number at
booking -> confirmed. (An earlier version used the number seen 1 week or 2
days before travel. Fed a just-booked WL 1, it read it as a ticket stuck at
WL 1 and predicted far too low.) The chance is constrained to never rise as
the waitlist number rises.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .timetable import DATA_DIR

WAITLIST_PATH = DATA_DIR / "raw" / "waitlist" / "Railway Ticket WaitingList Data.csv"
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "waitlist_model.joblib"

TRAVEL_CLASSES = ["SL", "3A", "2A", "CC", "1A", "2S"]
FEATURES = ["travel_class", "waitlist_number"]
# The irctc project's bands, checked against real outcomes there: 20%+ is a "good chance".
GOOD_CHANCE = 0.20


def load_tickets(path: Path = WAITLIST_PATH) -> pd.DataFrame:
    tickets = pd.read_csv(path, index_col=0)
    tickets = tickets[tickets["travelClass"].isin(TRAVEL_CLASSES) & (tickets["bookingStatus"] > 0)]
    return pd.DataFrame({
        "travel_class": tickets["travelClass"].map(TRAVEL_CLASSES.index),
        "waitlist_number": tickets["bookingStatus"],
        "confirmed": tickets["labels"].astype(bool),
    })


def chance_label(chance: float) -> str:
    """The irctc project's wording for a confirmation chance."""
    if chance >= GOOD_CHANCE:
        return "good chance"
    return "some chance" if chance >= 0.10 else "unlikely" if chance >= 0.05 else "very unlikely"


@dataclass
class WaitlistModel:
    model: object  # trees or neural net, whichever validated better
    metrics: dict = field(default_factory=dict)

    def confirm_chance(self, travel_class: str, waitlist_number: int) -> float:
        """Chance a ticket booked now at WL `waitlist_number` confirms; 0 means seats are available."""
        if waitlist_number <= 0:
            return 1.0
        row = np.array([[TRAVEL_CLASSES.index(travel_class), waitlist_number]], dtype=float)
        return float(self.model.predict_proba(row)[0, 1])


def _candidates(seed: int) -> dict:
    return {
        # Monotonic: a higher waitlist number never raises the chance.
        "trees": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, categorical_features=[True, False], monotonic_cst=[0, -1],
            random_state=seed),
        # A small neural network; class one-hot encoded, waitlist number scaled.
        "neural_net": make_pipeline(
            ColumnTransformer([("class", OneHotEncoder(handle_unknown="ignore"), [0]),
                               ("number", StandardScaler(), [1])]),
            MLPClassifier(hidden_layer_sizes=(64, 32), early_stopping=True, max_iter=300, random_state=seed)),
    }


def fit_and_evaluate(tickets: pd.DataFrame, seed: int = 0) -> WaitlistModel:
    """Trees vs a neural net: choose on a validation split, score both on held-out tickets."""
    X = tickets[FEATURES].to_numpy(dtype=float)
    y = tickets["confirmed"].to_numpy()
    train, test = train_test_split(np.arange(len(y)), test_size=0.25, random_state=seed, stratify=y)
    fit_idx, val_idx = train_test_split(train, test_size=0.25, random_state=seed, stratify=y[train])

    validation_auc = {}
    for name, model in _candidates(seed).items():
        model.fit(X[fit_idx], y[fit_idx])
        validation_auc[name] = round(float(roc_auc_score(y[val_idx], model.predict_proba(X[val_idx])[:, 1])), 4)
    chosen = max(validation_auc, key=validation_auc.get)

    test_scores, final = {}, None
    for name, model in _candidates(seed).items():
        model.fit(X[train], y[train])
        p = model.predict_proba(X[test])[:, 1]
        test_scores[name] = {"auc": round(float(roc_auc_score(y[test], p)), 4),
                             "brier": round(float(brier_score_loss(y[test], p)), 4)}
        if name == chosen:
            final, p_final = model, p

    bands = []
    for lo, hi in [(0, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 1.01)]:
        band = (p_final >= lo) & (p_final < hi)
        if band.sum():
            bands.append({"predicted": f"{lo:.0%}-{min(hi, 1):.0%}", "tickets": int(band.sum()),
                          "actually_confirmed": round(float(y[test][band].mean()), 3)})
    metrics = {
        "tickets": int(len(tickets)),
        "confirmed_share": round(float(y.mean()), 4),
        "validation_auc": validation_auc,
        "using": chosen,
        "test": test_scores,
        "test_auc": test_scores[chosen]["auc"],
        "test_brier": test_scores[chosen]["brier"],
        "calibration": bands,
    }
    return WaitlistModel(final, metrics)


def load() -> WaitlistModel | None:
    return joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None

"""Which option is best? A ranking learned from the options people pick.

Each time someone picks an option in the app, the full list they were shown
is logged with their pick (data/choices.jsonl). A pairwise logistic model
learns weights that score the picked option above each one passed over (a
Bradley-Terry model). Until enough real picks exist, options stay in the plain
order: fastest first.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from .search import Journey
from .timetable import DATA_DIR

CHOICES_PATH = DATA_DIR / "choices.jsonl"
MIN_CHOICES = 10
FEATURES = [
    "total_hours",
    "changes",
    "wait_hours",
    "miss_risk",
    "confirm_chance",
    "night_arrival",
    "night_departure",
]


def _at_night(moment: datetime) -> float:
    return float(moment.hour >= 23 or moment.hour < 5)


def option_features(journey: Journey, miss_risk: float, confirm_chance: float) -> dict[str, float]:
    return {
        "total_hours": journey.duration.total_seconds() / 3600,
        "changes": len(journey.legs) - 1,
        "wait_hours": journey.layover.total_seconds() / 3600 if journey.layover else 0.0,
        "miss_risk": miss_risk,
        "confirm_chance": confirm_chance,
        "night_arrival": _at_night(journey.arrives),
        "night_departure": _at_night(journey.departs),
    }


def log_choice(options: list[dict], chosen: int, context: dict, path: Path = CHOICES_PATH) -> None:
    """Append one pick: the features of every option shown, and which one was taken."""
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"logged_at": datetime.now().isoformat(timespec="seconds"), **context,
              "options": options, "chosen": chosen}
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def load_choices(path: Path = CHOICES_PATH) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [c for c in map(json.loads, f) if len(c["options"]) > 1]


def _matrix(options: list[dict]) -> np.ndarray:
    return np.array([[option[name] for name in FEATURES] for option in options], dtype=float)


@dataclass
class Ranker:
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    n_choices: int

    def scores(self, options: list[dict]) -> np.ndarray:
        return (_matrix(options) - self.mean) / self.scale @ self.weights

    def weight_table(self) -> dict[str, float]:
        """Per feature: how much one typical-size step in it moves an option's score."""
        return {name: round(float(w), 3) for name, w in zip(FEATURES, self.weights)}

    @classmethod
    def fit(cls, choices: list[dict]) -> Ranker:
        everything = np.vstack([_matrix(c["options"]) for c in choices])
        mean, scale = everything.mean(axis=0), everything.std(axis=0)
        scale[scale == 0] = 1.0
        differences = []
        for choice in choices:
            x = (_matrix(choice["options"]) - mean) / scale
            picked = x[choice["chosen"]]
            differences.extend(picked - other for i, other in enumerate(x) if i != choice["chosen"])
        differences = np.array(differences)
        # Each pair both ways round, so the classes are balanced and no intercept is needed.
        X = np.vstack([differences, -differences])
        y = np.r_[np.ones(len(differences)), np.zeros(len(differences))]
        model = LogisticRegression(fit_intercept=False, C=1.0).fit(X, y)
        return cls(mean, scale, model.coef_[0], len(choices))


def evaluate(choices: list[dict], folds: int = 5) -> dict:
    """Cross-validated: how often the top-ranked option is the one actually picked."""
    order = np.arange(len(choices))
    learned_hits = fastest_hits = 0
    for k in range(folds):
        held_out = [choices[i] for i in order[k::folds]]
        training = [choices[i] for i in order if i % folds != k]
        if not held_out or len(training) < 2:
            continue
        ranker = Ranker.fit(training)
        for choice in held_out:
            learned_hits += int(np.argmax(ranker.scores(choice["options"])) == choice["chosen"])
            fastest_hits += int(np.argmin(_matrix(choice["options"])[:, 0]) == choice["chosen"])
    return {
        "choices": len(choices),
        "top1_learned": round(learned_hits / len(choices), 3),
        "top1_fastest_first": round(fastest_hits / len(choices), 3),
    }


def load_ranker(path: Path = CHOICES_PATH) -> Ranker | None:
    choices = load_choices(path)
    return Ranker.fit(choices) if len(choices) >= MIN_CHOICES else None

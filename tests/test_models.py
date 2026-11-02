from datetime import date

import numpy as np
import pytest

from railroute import delay_model, ranker, waitlist_model
from railroute.cities import build_places
from railroute.delay_model import BUCKET_EDGES, N_BUCKETS, POINTS_PER_BUCKET, SUPPORT
from railroute.search import find_journeys
from railroute.seasons import SCENARIOS
from railroute.timetable import load_timetable


def test_bucket_of_uses_lower_edges():
    assert list(delay_model.bucket_of([0, 4, 5, 59, 60, 5000])) == [0, 0, 1, 3, 4, N_BUCKETS - 1]


def test_miss_probability_matches_brute_force():
    rng = np.random.default_rng(0)
    arrival = rng.dirichlet(np.ones(N_BUCKETS), size=20)
    departure = rng.dirichlet(np.ones(N_BUCKETS), size=20)
    slack = rng.uniform(-30, 600, size=20)

    got = delay_model.miss_probability(arrival, departure, slack)

    mass_a = np.repeat(arrival, POINTS_PER_BUCKET, axis=1) / POINTS_PER_BUCKET
    mass_d = np.repeat(departure, POINTS_PER_BUCKET, axis=1) / POINTS_PER_BUCKET
    late = SUPPORT[None, :, None] - SUPPORT[None, None, :] > slack[:, None, None]  # (row, a, d)
    expected = (mass_a[:, :, None] * mass_d[:, None, :] * late).sum(axis=(1, 2))
    np.testing.assert_allclose(got, expected)


def test_miss_probability_extremes():
    one_hour_late = np.zeros((1, N_BUCKETS))
    one_hour_late[0, 4] = 1  # arrives 60-90 min late
    on_time = np.zeros((1, N_BUCKETS))
    on_time[0, 0] = 1
    assert delay_model.miss_probability(one_hour_late, on_time, np.array([30.0]))[0] == 1
    assert delay_model.miss_probability(one_hour_late, on_time, np.array([120.0]))[0] == 0


def test_median_inside_bucket():
    probs = np.zeros((1, N_BUCKETS))
    probs[0, 4] = 1
    assert delay_model.median_delay(probs)[0] == pytest.approx((BUCKET_EDGES[4] + BUCKET_EDGES[5]) / 2)


def test_train_types():
    assert delay_model.train_type("12433", "Mas Nzm Rajdhani") == "rajdhani"
    assert delay_model.train_type("12611", "Nzm Garibrath") == "garib_rath"
    assert delay_model.train_type("12023", "Howrah Patna Jan Shatabdi") == "jan_shatabdi"
    assert delay_model.train_type("01099", "Csmt-jp One Way Special") == "special"
    assert delay_model.train_type("16093", "Mas Ljn Express") == "express"


def make_choices(n, rng):
    """Simulated users who always take the option least likely to miss its connection."""
    choices = []
    for _ in range(n):
        options = [{
            "total_hours": float(rng.uniform(20, 40)), "changes": 1, "wait_hours": float(rng.uniform(1, 8)),
            "miss_risk": float(rng.uniform(0, 0.6)), "confirm_chance": 1.0,
            "night_arrival": float(rng.integers(0, 2)), "night_departure": float(rng.integers(0, 2)),
        } for _ in range(6)]
        choices.append({"options": options, "chosen": int(np.argmin([o["miss_risk"] for o in options]))})
    return choices


def test_ranker_learns_what_people_pick():
    choices = make_choices(60, np.random.default_rng(1))
    learned = ranker.Ranker.fit(choices)

    assert learned.weight_table()["miss_risk"] < 0
    scores = ranker.evaluate(choices)
    assert scores["top1_learned"] > 0.8 > scores["top1_fastest_first"]


def test_ranker_waits_for_enough_choices(tmp_path):
    path = tmp_path / "choices.jsonl"
    for choice in make_choices(ranker.MIN_CHOICES - 1, np.random.default_rng(2)):
        ranker.log_choice(choice["options"], choice["chosen"], {}, path)
    assert ranker.load_ranker(path) is None
    ranker.log_choice(choice["options"], choice["chosen"], {}, path)
    assert ranker.load_ranker(path) is not None


@pytest.mark.skipif(not waitlist_model.MODEL_PATH.exists(), reason="run scripts/train_models.py first")
def test_waitlist_lower_number_more_likely_to_confirm():
    model = waitlist_model.load()
    assert model.confirm_chance("SL", 0) == 1.0
    chances = [model.confirm_chance("SL", wl) for wl in (1, 2, 3, 5, 10, 20, 45, 100, 300)]
    assert chances == sorted(chances, reverse=True)  # never rises with the waitlist number
    assert chances[0] > 0.3  # WL 1 confirmed 37% of the time in the data


@pytest.mark.skipif(not delay_model.MODEL_PATH.exists(), reason="run scripts/train_models.py first")
def test_delay_model_risks_on_real_search():
    timetable = load_timetable()
    places = build_places(timetable)
    connections = find_journeys(timetable, places["Chennai (all stations)"], places["Lucknow (all stations)"],
                                date(2026, 10, 12)).connections
    model = delay_model.load()
    normal = model.miss_risks(timetable, connections, SCENARIOS["Normal"])
    fog = model.miss_risks(timetable, connections, SCENARIOS["Winter fog"])
    assert len(normal) == len(connections)
    assert ((normal >= 0) & (normal <= 1)).all()
    assert fog.mean() > normal.mean()  # fog is the strongest effect in the data

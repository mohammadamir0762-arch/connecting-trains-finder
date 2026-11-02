import json
from datetime import date

import numpy as np
import pandas as pd
import pytest
from scipy.io import netcdf_file

from railroute import history, weather
from railroute.delay_model import N_BUCKETS


def run(train, start, stations, times, delays):
    """One run's halts: (arrival, departure) minutes after the start day's midnight."""
    return pd.DataFrame({
        "train": train, "train_name": f"Train {train}", "start_date": pd.Timestamp(start), "station": stations,
        "arr_el": np.array([t[0] for t in times], np.float32), "dep_el": np.array([t[1] for t in times], np.float32),
        "arr_delay": np.array([np.nan] + delays[1:], np.float32),
        "dep_delay": np.array([delays[0]] + [np.nan] * (len(delays) - 1), np.float32),
        "arr_reported": True, "dep_reported": True,
    })


def test_route_features_and_conditions():
    halts = history._finish(run("1", "2025-10-20", ["A", "B", "C"], [(1380, 1380), (1440 + 60, 1440 + 70),
                                                                     (1440 + 180, 1440 + 180)], [5, 20, 30]))
    assert list(halts["stop_index"]) == [0, 1, 2] and set(halts["n_stops"]) == {3}
    assert list(halts["delay"]) == [5, 20, 30]  # departure delay at the origin, arrival delay after
    assert list(halts["arr_hours"]) == [0, 2, 4]
    assert halts["halt_minutes"].tolist() == [0, 10, 0]
    assert halts["halt_date"].dt.day.tolist() == [20, 21, 21]  # crosses midnight

    halts = history.add_conditions(halts, np.array([0, 30, np.nan], np.float32))
    assert halts["route_rain_mm"].tolist() == [0, 15, 15]  # mean of the rain seen so far
    assert halts["festival"].tolist() == [1, 1, 1]  # Diwali 2025
    assert halts["winter"].tolist() == [0, 0, 0]


def test_station_between_neighbours_gets_coordinates(tmp_path):
    path = tmp_path / "stations.json"
    path.write_text(json.dumps({"features": [
        {"properties": {"code": "A"}, "geometry": {"coordinates": [80.0, 20.0]}},
        {"properties": {"code": "C"}, "geometry": {"coordinates": [82.0, 22.0]}},
    ]}))
    coords = weather.station_coordinates([["A", "NEW", "C"]], path)
    assert coords["NEW"] == pytest.approx((21.0, 81.0))


def test_rain_lookup_uses_nearest_cell_with_data(tmp_path):
    grid = np.full((2, 4, 4), weather.NO_DATA, np.float32)  # 2 days, 4x4 cells from 6.5N 66.5E
    grid[:, 2, 2] = [10.0, 70.0]  # the only land cell
    with netcdf_file(tmp_path / "rain_2025.grd", "w") as nc:
        nc.createDimension("TIME", 2)
        nc.createDimension("LATITUDE", 4)
        nc.createDimension("LONGITUDE", 4)
        nc.createVariable("RAINFALL", "f4", ("TIME", "LATITUDE", "LONGITUDE"))[:] = grid
    coords = {"COAST": (6.5 + 0.25 * 1, 66.5 + 0.25 * 2), "FAR": (30.0, 90.0)}  # one cell off land; far away

    archive = weather.RainArchive(coords, tmp_path)
    mm = archive.rain(np.array(["COAST", "COAST", "FAR"]),
                      np.array(["2025-01-01", "2025-01-02", "2025-01-01"], dtype="datetime64[D]"))
    assert mm[:2].tolist() == [10.0, 70.0]
    assert np.isnan(mm[2])
    assert archive.covers(date(2025, 6, 1)) and not archive.covers(date(2026, 6, 1))


def test_neural_network_learns_and_outputs_probabilities():
    from railroute.deep_model import DeepBackend, NUMERIC

    rng = np.random.default_rng(0)
    n = 4000
    X = pd.DataFrame({name: rng.normal(size=n).astype(np.float32) for name in NUMERIC})
    X["train"] = rng.choice(["1", "2"], n)
    X["station"] = "S"
    X["zone"], X["train_type"], X["weekday"] = 1.0, 2.0, 3
    y = np.where(X["train"] == "1", 0, 5)  # train 1 always on time, train 2 always 90+ min late

    model = DeepBackend(epochs=3, batch_size=256).fit(X, y, X, y)
    p = model.predict_proba(X)

    assert p.shape == (n, N_BUCKETS)
    np.testing.assert_allclose(p.sum(axis=1), 1, atol=1e-6)
    assert (p.argmax(axis=1) == y).mean() > 0.95

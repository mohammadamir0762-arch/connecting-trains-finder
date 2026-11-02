import csv
from datetime import date, datetime, timedelta

import pytest

from railroute.cities import build_places
from railroute.search import SearchOptions, find_journeys
from railroute.timetable import ROUTES_PATH, build_timetable, load_timetable, running_days

DAY = date(2026, 10, 15)  # a Thursday
SEPTEMBER = [date(2024, 9, 1) + timedelta(days=i) for i in range(30)]
DAILY = SEPTEMBER


def on(*weekdays):
    return [d for d in SEPTEMBER if d.weekday() in weekdays]


def clock(hhmm: str) -> str:
    return datetime.strptime(hhmm, "%H:%M").strftime("%I:%M %p")


def build(tmp_path, trains, runs=None, recorded_times=None):
    """trains: number -> [(station, "HH:MM" arrival, "HH:MM" departure)]; runs: number -> run dates."""
    routes = tmp_path / "routes.csv"
    with open(routes, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["stnSerialNumber", "trainNumber", "trainName", "station_code", "station_name",
                         "distance", "arrivalTime", "departureTime"])
        for number, stops in trains.items():
            for i, (station, arr, dep) in enumerate(stops):
                writer.writerow([i + 1, number, f"Train {number}", station, station, i * 100,
                                 clock(arr or dep), clock(dep or arr)])
    delays = tmp_path / "delays.csv"
    with open(delays, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["train", "date", "station", "sch_arr", "sch_dep"])
        for number, stops in trains.items():
            times = (recorded_times or {}).get(number, stops)
            for run in (runs or {}).get(number, DAILY):
                for station, arr, dep in times:
                    writer.writerow([number, run.isoformat(), station, clock(arr or dep), clock(dep or arr)])
    return build_timetable(routes, delays)


@pytest.fixture
def timetable(tmp_path):
    return build(tmp_path, {
        "1": [("A", None, "22:00"), ("X", "05:00", "05:10"), ("Y", "08:00", None)],  # overnight A -> X
        "2": [("X", "06:55", "07:00"), ("B", "12:00", None)],
        "4": [("A", None, "08:00"), ("B", "20:00", None)],
        "5": [("X", None, "05:30"), ("B", "09:00", None)],
        "6": [("A", None, "09:00"), ("B", "21:00", None)],  # Fridays only
    }, runs={"6": on(4)})


def test_direct_trains_only_on_running_days(timetable):
    result = find_journeys(timetable, frozenset({"A"}), frozenset({"B"}), DAY)

    assert [j.legs[0].train_number for j in result.direct] == ["4"]
    assert result.direct[0].departs == datetime(2026, 10, 15, 8, 0)
    assert result.direct[0].arrives == datetime(2026, 10, 15, 20, 0)
    assert result.direct_not_running == ["6"]

    friday = find_journeys(timetable, frozenset({"A"}), frozenset({"B"}), DAY + timedelta(days=1))
    assert [j.legs[0].train_number for j in friday.direct] == ["4", "6"]


def test_connection_crosses_midnight_with_dates(timetable):
    connections = find_journeys(timetable, frozenset({"A"}), frozenset({"B"}), DAY).connections

    assert len(connections) == 1  # train 5 leaves X 30 min after arrival: below the 60 min default
    first, second = connections[0].legs
    assert (first.train_number, first.to_station) == ("1", "X")
    assert first.departs == datetime(2026, 10, 15, 22, 0)
    assert first.arrives == datetime(2026, 10, 16, 5, 0)
    assert (second.train_number, second.from_station) == ("2", "X")
    assert second.departs == datetime(2026, 10, 16, 7, 0)
    assert second.arrives == datetime(2026, 10, 16, 12, 0)
    assert connections[0].layover == timedelta(hours=2)


def test_next_train_if_the_change_is_missed(timetable):
    connection = find_journeys(timetable, frozenset({"A"}), frozenset({"B"}), DAY).connections[0]

    # Missing train 2 (07:00 on the 16th), the next way on is train 5 at 05:30 the day after.
    assert connection.fallback_train == "5"
    assert connection.fallback_arrives == datetime(2026, 10, 17, 9, 0)


def test_shorter_change_time_allows_tighter_connection(timetable):
    options = SearchOptions(min_transfer_same_station=15)
    fastest = find_journeys(timetable, frozenset({"A"}), frozenset({"B"}), DAY, options).connections[0]

    assert fastest.legs[1].train_number == "5"
    assert fastest.arrives == datetime(2026, 10, 16, 9, 0)


def test_second_leg_waits_for_a_day_it_runs(tmp_path):
    timetable = build(tmp_path, {
        "1": [("A", None, "08:00"), ("X", "10:00", None)],
        "2": [("X", None, "12:00"), ("B", "14:00", None)],  # Saturdays only
    }, runs={"2": on(5)})

    friday = DAY + timedelta(days=1)
    connection = find_journeys(timetable, frozenset({"A"}), frozenset({"B"}), friday,
                               SearchOptions(max_layover=30 * 60)).connections[0]
    assert connection.legs[1].departs == datetime(2026, 10, 17, 12, 0)  # Saturday, not Friday
    assert connection.legs[1].start_date.weekday() == 5
    assert not find_journeys(timetable, frozenset({"A"}), frozenset({"B"}), friday).connections  # 26h > 12h


def test_first_leg_must_run_that_day(tmp_path):
    timetable = build(tmp_path, {
        "1": [("A", None, "08:00"), ("X", "10:00", None)],  # Fridays only
        "2": [("X", None, "12:00"), ("B", "14:00", None)],
    }, runs={"1": on(4)})

    assert not find_journeys(timetable, frozenset({"A"}), frozenset({"B"}), DAY).connections
    assert find_journeys(timetable, frozenset({"A"}), frozenset({"B"}), DAY + timedelta(days=1)).connections


def test_broken_route_times_are_repaired_from_run_records(tmp_path):
    timetable = build(
        tmp_path,
        {"1": [("A", None, "07:25"), ("X", "07:25", "07:25"), ("B", "07:25", None)]},
        recorded_times={"1": [("A", None, "11:30"), ("X", "13:50", "13:55"), ("B", "16:50", None)]},
    )

    stops = timetable.trains["1"].stops
    assert [(s.arr, s.dep) for s in stops] == [(690, 690), (830, 835), (1010, 1010)]


def test_one_off_runs_are_not_running_days():
    weekly_plus_stray = on(5) + [date(2024, 9, 2)]  # every Saturday, plus one Monday
    assert running_days(weekly_plus_stray) == frozenset({5})
    assert running_days(DAILY) == frozenset(range(7))


@pytest.mark.skipif(not ROUTES_PATH.exists(), reason="run scripts/download_data.py first")
def test_chennai_to_lucknow_real_data():
    timetable = load_timetable()
    places = build_places(timetable)
    chennai, lucknow = places["Chennai (all stations)"], places["Lucknow (all stations)"]

    monday = find_journeys(timetable, chennai, lucknow, date(2026, 10, 12))
    assert not monday.direct
    assert "16093" in monday.direct_not_running  # runs Tue and Sat
    tuesday = find_journeys(timetable, chennai, lucknow, date(2026, 10, 13))
    assert "16093" in [j.legs[0].train_number for j in tuesday.direct]
    assert any(j.legs[0].to_station == "VGLJ" for j in monday.connections)  # change at Jhansi
    for journey in monday.connections:
        for leg in journey.legs:
            assert timetable.trains[leg.train_number].runs_on(leg.start_date)

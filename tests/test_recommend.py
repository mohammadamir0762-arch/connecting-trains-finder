from datetime import date, datetime, timedelta

from railroute.recommend import assess, recommend
from railroute.search import Journey, Leg
from railroute.seasons import SCENARIOS, is_festival_season, is_winter_fog_season

DAY = date(2026, 10, 20)


def leg(train, start, end, departs_h, arrives_h):
    base = datetime(2026, 10, 20)
    return Leg(train, f"Train {train}", start, end, base + timedelta(hours=departs_h),
               base + timedelta(hours=arrives_h), DAY, 0, 1)


def connection(first, second, change_arrives_h, leaves_h, arrives_h, rescue_h):
    return Journey(
        (leg(first, "A", "X", 0, change_arrives_h), leg(second, "X", "B", leaves_h, arrives_h)),
        fallback_arrives=datetime(2026, 10, 20) + timedelta(hours=rescue_h),
        fallback_train="9",
    )


def test_tight_connection_loses_to_a_safe_one():
    tight = connection("1", "2", 10, 10.5, 20, rescue_h=40)  # 20 h on paper, next train +20 h
    safe = connection("3", "4", 10, 13, 22, rescue_h=30)  # 22 h on paper
    options = [assess(tight, miss_risk=0.5, confirm_chance=None), assess(safe, miss_risk=0.05, confirm_chance=None)]

    best, reasons = recommend(options)

    assert best.journey is safe
    assert options[0].expected_duration == timedelta(hours=30)  # 20 h + 50% of the extra 20 h
    assert any("fastest option (1 → X → 2)" in r and "50% miss risk" in r for r in reasons)


def test_low_risk_fast_connection_wins():
    fast = connection("1", "2", 10, 11, 20, rescue_h=30)
    slow = connection("3", "4", 10, 13, 26, rescue_h=40)
    best, _ = recommend([assess(fast, 0.02, None), assess(slow, 0.0, None)])
    assert best.journey is fast


def test_a_missed_change_never_makes_you_earlier():
    # The next train arrives before the planned one: missing it costs nothing, but gains nothing either.
    j = connection("1", "2", 10, 11, 20, rescue_h=19)
    assert assess(j, 0.6, None).expected_arrival == j.arrives


def test_unlikely_seat_counts_against_an_option():
    quick = Journey((leg("1", "A", "B", 0, 20),))
    slower = Journey((leg("2", "A", "B", 0, 24),))
    best, reasons = recommend([assess(quick, 0.0, confirm_chance=0.1), assess(slower, 0.0, confirm_chance=None)])
    assert best.journey is slower
    assert any("only a 10% chance its waitlisted ticket confirms" in r for r in reasons)


def test_low_waitlist_with_a_good_chance_keeps_the_faster_train():
    # WL 1-2 confirms ~40% of the time: a good chance, so it shouldn't push you onto a slower train.
    quick = Journey((leg("1", "A", "B", 0, 20),))
    slower = Journey((leg("2", "A", "B", 0, 24),))
    best, reasons = recommend([assess(quick, 0.0, confirm_chance=0.43), assess(slower, 0.0, confirm_chance=None)])
    assert best.journey is quick
    assert any("43% chance your waitlisted ticket(s) confirm (good chance)" in r for r in reasons)


def test_seasons():
    assert is_festival_season(date(2025, 10, 20))  # Diwali 2025
    assert is_festival_season(date(2026, 11, 16))  # Chhath 2026
    assert not is_festival_season(date(2025, 8, 1))
    assert is_winter_fog_season(date(2026, 1, 5)) and not is_winter_fog_season(date(2026, 4, 5))

    auto = SCENARIOS["Auto (from date)"]
    assert auto.festival_on(date(2026, 11, 8)) and not auto.festival_on(date(2026, 7, 1))
    assert not SCENARIOS["Normal"].festival_on(date(2026, 11, 8))
    assert SCENARIOS["Heavy rain"].rain_mm == 64.5

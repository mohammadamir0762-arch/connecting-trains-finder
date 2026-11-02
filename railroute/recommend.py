"""Which option should I take? Weigh speed against the risk of a missed change.

A connection that looks fast on paper is a bad bet if train 1 often arrives
too late for train 2. For every option we work out when you'd *expect* to
arrive: on time with probability 1 - miss risk, otherwise on the next train
from the change station (found in the timetable). The option that gets you
there soonest on average is recommended, skipping options whose waitlisted
ticket is unlikely to confirm (below a 20% chance) while a better one exists.
The reasons are spelled out.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta

from .search import Journey
from .waitlist_model import GOOD_CHANCE, chance_label

# If train 2 is missed and no other train is found within the search horizon.
NO_RESCUE = timedelta(days=1)


@dataclass(frozen=True)
class Assessment:
    journey: Journey
    miss_risk: float  # 0 for direct trains
    confirm_chance: float | None  # None when no leg is waitlisted
    expected_arrival: datetime

    @property
    def expected_duration(self) -> timedelta:
        return self.expected_arrival - self.journey.departs

    @property
    def seat_unlikely(self) -> bool:
        return self.confirm_chance is not None and self.confirm_chance < GOOD_CHANCE

    @property
    def rank_key(self) -> tuple:
        """Options likely to get a seat first, then the shortest expected trip."""
        return (self.seat_unlikely, self.expected_duration, self.journey.arrives)

    @property
    def missed_arrival(self) -> datetime | None:
        if len(self.journey.legs) < 2:
            return None
        rescue = self.journey.fallback_arrives
        return self.journey.arrives + NO_RESCUE if rescue is None else max(rescue, self.journey.arrives)


def assess(journey: Journey, miss_risk: float, confirm_chance: float | None) -> Assessment:
    if len(journey.legs) < 2:
        return Assessment(journey, 0.0, confirm_chance, journey.arrives)
    rescue = journey.fallback_arrives
    # Missing the change never helps: at best the next train is no slower.
    extra = NO_RESCUE if rescue is None else max(rescue - journey.arrives, timedelta(0))
    return Assessment(journey, miss_risk, confirm_chance, journey.arrives + miss_risk * extra)


def _hours(span: timedelta) -> str:
    minutes = round(abs(span.total_seconds()) / 60)
    return f"{minutes} min" if minutes < 60 else f"{minutes // 60}h {minutes % 60:02d}m"


def percent(p: float) -> str:
    """A chance for people to read: "<1%" rather than a falsely certain "0%"."""
    return "<1%" if p < 0.005 else f"{p:.0%}"


def _when(moment: datetime) -> str:
    return moment.strftime("%a %d %b, %H:%M")


def label(journey: Journey) -> str:
    return " → ".join([journey.legs[0].train_number]
                      + [f"{leg.from_station} → {leg.train_number}" for leg in journey.legs[1:]])


def recommend(options: list[Assessment], station_name: Callable[[str], str] = str
              ) -> tuple[Assessment, list[str]] | None:
    """The option with the best expected outcome, and why, compared with the fastest one."""
    if not options:
        return None
    best = min(options, key=lambda a: a.rank_key)
    fastest = min(options, key=lambda a: (a.journey.duration, a.journey.arrives))
    reasons = []

    if best.miss_risk > 0:
        change = station_name(best.journey.legs[0].to_station)
        late = best.missed_arrival - best.journey.arrives
        rescue = (f"the next train ({best.journey.fallback_train}) still gets you in by "
                  f"{_when(best.missed_arrival)}" + (f", {_hours(late)} later." if late else ", no later.")
                  if best.journey.fallback_train else "there's no later train within a week.")
        reasons.append(f"{percent(best.miss_risk)} chance of missing the change at {change}. If that happens, {rescue}")
    elif len(best.journey.legs) == 1:
        reasons.append("Direct train: no change to miss.")
    if best.confirm_chance is not None:
        reasons.append(f"{best.confirm_chance:.0%} chance your waitlisted ticket(s) confirm "
                       f"({chance_label(best.confirm_chance)}).")
    trip = f"Takes {_hours(best.journey.duration)}"
    if best.expected_duration - best.journey.duration >= timedelta(minutes=1):
        trip += (f" by the timetable, or about {_hours(best.expected_duration)} on average once the chance of "
                 "missing the change is counted")
    reasons.append(trip + (", and it's the fastest option." if fastest is best else "."))

    if fastest is not best:
        paper = best.journey.duration - fastest.journey.duration
        why = []
        if fastest.miss_risk > best.miss_risk:
            why.append(f"a {percent(fastest.miss_risk)} miss risk (next train in by {_when(fastest.missed_arrival)})")
        if fastest.seat_unlikely:
            why.append(f"only a {fastest.confirm_chance:.0%} chance its waitlisted ticket confirms")
        if fastest.expected_duration >= best.expected_duration:
            outcome = f"so on average the trip takes {_hours(fastest.expected_duration - best.expected_duration)} longer."
        elif fastest.seat_unlikely:
            outcome = "so you probably wouldn't get a seat on it."
        else:
            outcome = "and is a little quicker on average, but less certain."
        named = " → ".join([fastest.journey.legs[0].train_number] + [
            f"{station_name(leg.from_station)} → {leg.train_number}" for leg in fastest.journey.legs[1:]])
        reasons.append(f"The fastest option ({named}) is {_hours(paper)} quicker on paper"
                       + (f" but has {' and '.join(why)}" if why else "") + f", {outcome}")
    return best, reasons

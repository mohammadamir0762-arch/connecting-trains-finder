"""Find direct trains and one-change connections between two places on a date.

A train is only offered on dates it actually runs. A second leg is the next
departure of that train, on a day it runs, after you reach the change station
(up to max_layover).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta

from .cities import city_of
from .timetable import MINUTES_PER_DAY, Stop, Timetable


@dataclass(frozen=True)
class SearchOptions:
    min_transfer_same_station: int = 60  # minutes; Indian trains often run late
    min_transfer_other_station: int = 120  # e.g. arrive Lucknow Jn, leave from Lucknow Charbagh
    max_layover: int = 12 * 60


@dataclass(frozen=True)
class Leg:
    train_number: str
    train_name: str
    from_station: str
    to_station: str
    departs: datetime
    arrives: datetime
    start_date: date  # the day this run of the train left its first station
    board_index: int
    alight_index: int


# How far ahead to look for the next train if you miss a connection (weekly trains).
FALLBACK_HORIZON = 7 * MINUTES_PER_DAY


@dataclass(frozen=True)
class Journey:
    legs: tuple[Leg, ...]
    # If train 1 is too late for train 2: when the next train from the change city gets you there.
    fallback_arrives: datetime | None = None
    fallback_train: str | None = None

    @property
    def departs(self) -> datetime:
        return self.legs[0].departs

    @property
    def arrives(self) -> datetime:
        return self.legs[-1].arrives

    @property
    def duration(self) -> timedelta:
        return self.arrives - self.departs

    @property
    def layover(self) -> timedelta | None:
        if len(self.legs) < 2:
            return None
        return self.legs[1].departs - self.legs[0].arrives


def _trains_at(timetable: Timetable, stations: frozenset[str]) -> set[str]:
    return {number for code in stations for number, _ in timetable.station_index.get(code, ())}


def _boarding_index(stops: tuple[Stop, ...], stations: frozenset[str]) -> int | None:
    """The last halt in `stations` before the train leaves them (MAS -> PER -> out: PER)."""
    board = None
    for i, stop in enumerate(stops[:-1]):
        if stop.station in stations:
            board = i
        elif board is not None:
            break
    return board


def _first_index_in(stops: tuple[Stop, ...], stations: frozenset[str], start: int) -> int | None:
    return next((j for j in range(start, len(stops)) if stops[j].station in stations), None)


@dataclass(frozen=True)
class SearchResult:
    direct: list[Journey]
    connections: list[Journey]  # every connection within max_layover, fastest first
    direct_not_running: list[str]  # direct trains on this route that don't run that day


def _legs_into(timetable: Timetable, destination: frozenset[str], numbers) -> dict[str, list]:
    """For trains reaching `destination`: city you could board in -> [(train, board, alight)]."""
    by_city = defaultdict(list)
    for number in numbers:
        stops = timetable.trains[number].stops
        alight = _first_index_in(stops, destination, 1)
        if alight is None:
            continue
        for board in range(alight):
            if stops[board].station not in destination:
                by_city[city_of(stops[board].station)].append((number, board, alight))
    return by_city


def find_journeys(
    timetable: Timetable,
    origin: frozenset[str],
    destination: frozenset[str],
    travel_date: date,
    options: SearchOptions = SearchOptions(),
) -> SearchResult:
    """Direct trains and connections leaving `origin` on `travel_date`."""
    midnight = datetime.combine(travel_date, time())

    def leg(number: str, board: int, alight: int, leaves: int) -> Leg:
        """`leaves` is minutes after travel_date's midnight that the train leaves stop `board`."""
        train = timetable.trains[number]
        start_offset = leaves - train.stops[board].dep  # a whole number of days
        departs = midnight + timedelta(minutes=leaves)
        return Leg(
            train_number=number,
            train_name=train.name,
            from_station=train.stops[board].station,
            to_station=train.stops[alight].station,
            departs=departs,
            arrives=departs + timedelta(minutes=train.stops[alight].arr - train.stops[board].dep),
            start_date=travel_date + timedelta(days=start_offset // MINUTES_PER_DAY),
            board_index=board,
            alight_index=alight,
        )

    def runs(number: str, board: int, leaves: int) -> bool:
        start_offset = leaves - timetable.trains[number].stops[board].dep
        return timetable.trains[number].runs_on(travel_date + timedelta(days=start_offset // MINUTES_PER_DAY))

    # Trains leaving the origin on travel_date either reach the destination
    # (direct) or are candidate first legs.
    direct = {}
    direct_not_running = []
    first_legs = {}
    for number in sorted(_trains_at(timetable, origin)):
        stops = timetable.trains[number].stops
        board = _boarding_index(stops, origin)
        if board is None:
            continue
        alight = _first_index_in(stops, destination, board + 1)
        if not runs(number, board, stops[board].dep % MINUTES_PER_DAY):
            if alight is not None:
                direct_not_running.append(number)
            continue
        if alight is None:
            first_legs[number] = board
        else:
            direct[number] = (board, alight)

    # Trains reaching the destination, keyed by each city you could board them in.
    # A train that runs from the origin itself (on any day) isn't a useful second leg:
    # take it from the origin instead.
    reaching = _trains_at(timetable, destination)
    second_legs_by_city = _legs_into(timetable, destination, reaching - direct.keys() - set(direct_not_running))
    # Any of them, the direct ones included, can rescue you after a missed connection.
    any_leg_by_city = _legs_into(timetable, destination, reaching)
    fallbacks = {}

    def fallback(city: str, station: str, after: int) -> tuple[int, str] | None:
        """Earliest arrival at the destination on a train leaving `city` after minute `after`."""
        if (city, station, after) not in fallbacks:
            found = None
            for number, board, alight in any_leg_by_city.get(city, ()):
                stops = timetable.trains[number].stops
                earliest = after + (0 if stops[board].station == station else options.min_transfer_other_station)
                leaves = earliest + (stops[board].dep - earliest) % MINUTES_PER_DAY
                while leaves - after <= FALLBACK_HORIZON and not runs(number, board, leaves):
                    leaves += MINUTES_PER_DAY
                if leaves - after > FALLBACK_HORIZON:
                    continue
                arrival = leaves + stops[alight].arr - stops[board].dep
                if found is None or arrival < found[0]:
                    found = (arrival, number)
            fallbacks[(city, station, after)] = found
        return fallbacks[(city, station, after)]

    endpoint_cities = {city_of(code) for code in origin | destination}
    # (first train, second train) -> best (arrival, -layover, change, board2, alight2, leaves2)
    best = {}
    for first, board in first_legs.items():
        stops = timetable.trains[first].stops
        leaves = stops[board].dep % MINUTES_PER_DAY
        for change in range(board + 1, len(stops)):
            if stops[change].station in origin:
                break
            city = city_of(stops[change].station)
            if city in endpoint_cities:
                continue
            reaches = leaves + stops[change].arr - stops[board].dep

            for second, board2, alight2 in second_legs_by_city.get(city, ()):
                if second == first:
                    continue
                stops2 = timetable.trains[second].stops
                if stops2[board2].station == stops[change].station:
                    ready = reaches + options.min_transfer_same_station
                else:
                    ready = reaches + options.min_transfer_other_station
                # Next departure after `ready`, skipping days the train doesn't run.
                leaves2 = ready + (stops2[board2].dep - ready) % MINUTES_PER_DAY
                while leaves2 - reaches <= options.max_layover and not runs(second, board2, leaves2):
                    leaves2 += MINUTES_PER_DAY
                layover = leaves2 - reaches
                if layover > options.max_layover:
                    continue
                arrives = leaves2 + stops2[alight2].arr - stops2[board2].dep
                # Same arrival time: prefer the bigger buffer in case the first train is late.
                candidate = (arrives, -layover, change, board2, alight2, leaves2)
                key = (first, second)
                if key not in best or candidate < best[key]:
                    best[key] = candidate

    direct_journeys = [
        Journey((leg(n, b, a, timetable.trains[n].stops[b].dep % MINUTES_PER_DAY),))
        for n, (b, a) in direct.items()
    ]
    direct_journeys.sort(key=lambda j: j.departs)

    connections = []
    for (first, second), (_, _, change, board2, alight2, leaves2) in best.items():
        board = first_legs[first]
        leaves = timetable.trains[first].stops[board].dep % MINUTES_PER_DAY
        station = timetable.trains[first].stops[change].station
        rescue = fallback(city_of(station), station, leaves2 + 1)  # any train after the missed one
        connections.append(Journey(
            (leg(first, board, change, leaves), leg(second, board2, alight2, leaves2)),
            fallback_arrives=midnight + timedelta(minutes=rescue[0]) if rescue else None,
            fallback_train=rescue[1] if rescue else None,
        ))
    connections.sort(key=lambda j: (j.duration, j.arrives))

    return SearchResult(direct_journeys, connections, direct_not_running)

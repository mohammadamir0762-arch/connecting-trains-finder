"""The conditions the delay model can be asked about: rain, festival rush, winter fog.

Each is a feature of every halt in the training data, so the model learns
how much each one changes delays from what actually happened in 2025. A
scenario sets those features for a trip that hasn't happened yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

# IMD's rainfall categories: "heavy rain" starts at 64.5 mm in a day.
HEAVY_RAIN_MM = 64.5

# Festival travel rush: Holi (a week either side) and Durga Puja through
# Diwali to a week after Chhath, when people travel home and back. In 2025
# delays at Bihar and east UP stations rose from ~45 to ~65 min over this
# whole stretch, not just around Diwali day. Dates from bank-holiday lists.
FESTIVAL_SEASONS = [
    (date(2024, 3, 18), date(2024, 4, 1)),    # Holi 25 Mar 2024
    (date(2024, 10, 7), date(2024, 11, 15)),  # Durga Puja 9 Oct, Diwali 31 Oct, Chhath 7 Nov 2024
    (date(2025, 3, 7), date(2025, 3, 21)),    # Holi 14 Mar 2025
    (date(2025, 9, 26), date(2025, 11, 10)),  # Durga Puja 28 Sep, Diwali 20 Oct, Chhath 25-28 Oct 2025
    (date(2026, 2, 24), date(2026, 3, 11)),   # Holi 3-4 Mar 2026
    (date(2026, 10, 15), date(2026, 11, 23)), # Vijayadashami 20 Oct, Diwali 8 Nov, Chhath 16 Nov 2026
]


def is_festival_season(day: date) -> bool:
    return any(start <= day <= end for start, end in FESTIVAL_SEASONS)


def is_winter_fog_season(day: date) -> bool:
    """Mid-December to mid-February, when fog slows trains across north India."""
    return (day.month == 12 and day.day >= 10) or day.month == 1 or (day.month == 2 and day.day <= 15)


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    rain_mm: float = 0.0  # rain at every halt that day
    festival: bool | None = None  # None: decide from the travel date
    winter: bool | None = None

    def festival_on(self, day: date) -> bool:
        return is_festival_season(day) if self.festival is None else self.festival

    def winter_on(self, day: date) -> bool:
        return is_winter_fog_season(day) if self.winter is None else self.winter


SCENARIOS = {
    "Auto (from date)": Scenario(
        "Auto (from date)", "Festival season and winter fog decided by the travel date; no rain assumed."),
    "Normal": Scenario("Normal", "Dry day, no festival rush, no fog.", 0.0, False, False),
    "Heavy rain": Scenario(
        "Heavy rain", f"{HEAVY_RAIN_MM} mm of rain along the route that day (IMD's 'heavy rain').",
        HEAVY_RAIN_MM, False, False),
    "Diwali / Chhath rush": Scenario(
        "Diwali / Chhath rush", "Festival travel season (Durga Puja to Chhath, or Holi).", 0.0, True, False),
    "Winter fog": Scenario("Winter fog", "Mid-December to mid-February fog season.", 0.0, False, True),
}

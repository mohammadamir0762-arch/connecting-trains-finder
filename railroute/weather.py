"""Daily rainfall at railway stations.

IMD's gridded daily rainfall (0.25°, built from rain gauges) for 2024-2025,
read from the NetCDF files IMD Pune publishes. Days IMD hasn't published yet
(2026) come from Open-Meteo's ERA5 archive instead.
"""

from __future__ import annotations

import json
import time
import urllib.request
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np
from scipy.io import netcdf_file

from .timetable import DATA_DIR

IMD_DIR = DATA_DIR / "raw" / "imd"
STATIONS_PATH = DATA_DIR / "raw" / "stations" / "stations.json"
GRID_START = (6.5, 66.5)  # southern-most latitude, western-most longitude of the IMD grid
GRID_STEP = 0.25
NO_DATA = -999.0

# Renamed stations whose old codes have coordinates.
OLD_CODES = {
    "VGLJ": "JHS", "CSMT": "CSTM", "DDU": "MGS", "RKMP": "HBJ", "PRYJ": "ALD", "MMCT": "BCT",
    "AYC": "FD", "KLBG": "GR", "SBIB": "SBI", "SHRN": "SRR",
}


def station_coordinates(routes: list[list[str]], path: Path = STATIONS_PATH) -> dict[str, tuple[float, float]]:
    """(lat, lon) per station; stations missing from the list sit between their neighbours on routes."""
    features = json.loads(path.read_text())["features"]
    known = {
        f["properties"]["code"]: (f["geometry"]["coordinates"][1], f["geometry"]["coordinates"][0])
        for f in features if f.get("geometry") and f["geometry"].get("coordinates")
    }
    coords = dict(known)
    for new, old in OLD_CODES.items():
        if old in known:
            coords[new] = known[old]

    # A few passes: each one can place stations whose neighbours were placed in the last.
    for _ in range(3):
        guesses = defaultdict(list)
        for route in routes:
            for i, code in enumerate(route):
                if code in coords:
                    continue
                neighbours = [route[j] for j in (i - 1, i + 1) if 0 <= j < len(route) and route[j] in coords]
                if neighbours:
                    guesses[code].append(np.mean([coords[n] for n in neighbours], axis=0))
        if not guesses:
            break
        coords.update({code: tuple(np.mean(points, axis=0)) for code, points in guesses.items()})
    return coords


class RainArchive:
    """Rain (mm) at a station on a day, from the IMD grid's nearest cell that has data."""

    def __init__(self, coords: dict[str, tuple[float, float]], imd_dir: Path = IMD_DIR):
        self.grids: dict[int, np.ndarray] = {}
        for path in sorted(imd_dir.glob("rain_*.grd")):
            with netcdf_file(path, mmap=False) as nc:
                self.grids[int(path.stem.split("_")[1])] = np.array(nc.variables["RAINFALL"][:], dtype=np.float32)
        any_grid = next(iter(self.grids.values()))
        land = (any_grid > NO_DATA).any(axis=0)  # coast and sea cells never have data
        land_cells = np.argwhere(land)
        self.cell = {}
        for code, (lat, lon) in coords.items():
            i, j = (lat - GRID_START[0]) / GRID_STEP, (lon - GRID_START[1]) / GRID_STEP
            nearest = land_cells[np.argmin((land_cells[:, 0] - i) ** 2 + (land_cells[:, 1] - j) ** 2)]
            if (nearest[0] - i) ** 2 + (nearest[1] - j) ** 2 <= 9:  # within ~0.75°
                self.cell[code] = tuple(nearest)

    def covers(self, day: date) -> bool:
        return day.year in self.grids

    def rain(self, stations: np.ndarray, days: np.ndarray) -> np.ndarray:
        """Vectorised lookup; NaN where the station or the year isn't covered."""
        out = np.full(len(stations), np.nan, dtype=np.float32)
        days = days.astype("datetime64[D]")
        years = days.astype("datetime64[Y]").astype(int) + 1970
        for year, grid in self.grids.items():
            in_year = years == year
            day_index = (days[in_year] - np.datetime64(f"{year}-01-01")).astype(int)
            cells = [self.cell.get(code) for code in stations[in_year]]
            known = np.array([c is not None for c in cells], dtype=bool)
            if not known.any():
                continue
            ij = np.array([c for c in cells if c is not None])
            values = grid[day_index[known], ij[:, 0], ij[:, 1]]
            target = np.flatnonzero(in_year)[known]
            out[target] = np.where(values > NO_DATA, values, np.nan)
        return out


def open_meteo_rain(points: dict[str, tuple[float, float]], start: date, end: date) -> dict[tuple[str, date], float]:
    """Daily rain from Open-Meteo's ERA5 archive, for days IMD hasn't published (one call per 0.5° cell)."""
    cells = defaultdict(list)
    for code, (lat, lon) in points.items():
        cells[(round(lat * 2) / 2, round(lon * 2) / 2)].append(code)
    rain = {}
    cell_list = list(cells)
    for k in range(0, len(cell_list), 50):
        batch = cell_list[k:k + 50]
        url = ("https://archive-api.open-meteo.com/v1/archive?"
               f"latitude={','.join(str(c[0]) for c in batch)}&longitude={','.join(str(c[1]) for c in batch)}"
               f"&start_date={start}&end_date={end}&daily=precipitation_sum&timezone=Asia%2FKolkata")
        for attempt in range(5):
            try:
                with urllib.request.urlopen(url, timeout=60) as response:
                    results = json.load(response)
                break
            except Exception:
                time.sleep(20 * (attempt + 1))  # rate limit: wait and retry
        else:
            raise RuntimeError("Open-Meteo did not answer")
        results = results if isinstance(results, list) else [results]
        for cell, result in zip(batch, results):
            for day, mm in zip(result["daily"]["time"], result["daily"]["precipitation_sum"]):
                for code in cells[cell]:
                    rain[(code, date.fromisoformat(day))] = mm
        time.sleep(1)
    return rain

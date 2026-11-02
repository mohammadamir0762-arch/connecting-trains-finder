"""Group the stations that serve one city, so "Chennai" covers MAS, MS, PER, ..."""

from __future__ import annotations

from .timetable import Timetable

CITY_STATIONS = {
    "Agra": ["AGC", "AF", "IDH", "RKM"],
    "Ahmedabad": ["ADI", "SBI", "MAN"],
    "Bengaluru": ["SBC", "YPR", "BNC", "KJM", "BAND", "SMVB"],
    "Bhopal": ["BPL", "RKMP"],
    "Chennai": ["MAS", "MS", "MSB", "PER", "TBM"],
    "Delhi": ["NDLS", "DLI", "NZM", "DEE", "ANVT", "DSA", "DEC", "DSJ"],
    "Guwahati": ["GHY", "KYQ"],
    "Hyderabad": ["SC", "HYB", "KCG", "LPI", "CHZ"],
    "Jaipur": ["JP", "GADJ", "DPA"],
    "Kanpur": ["CNB", "CPA", "GOY"],
    "Kochi": ["ERS", "ERN"],
    "Kolkata": ["HWH", "SDAH", "KOAA", "SHM", "SRC"],
    "Lucknow": ["LKO", "LJN", "ASH", "LC", "BNZ", "GTNR"],
    "Mumbai": ["CSMT", "MMCT", "LTT", "DR", "DDR", "BDTS", "TNA", "KYN", "BVI", "PNVL"],
    "Nagpur": ["NGP", "AJNI"],
    "Patna": ["PNBE", "RJPB", "PPTA", "DNR", "PNC"],
    "Prayagraj": ["PRYJ", "PRG", "PCOI", "NYN", "SFG"],
    "Pune": ["PUNE", "SVJR", "HDP"],
    "Surat": ["ST", "UDN"],
    "Thiruvananthapuram": ["TVC", "KCVL"],
    "Varanasi": ["BSB", "BSBS", "BCY"],
}

_CITY_OF_STATION = {code: city for city, codes in CITY_STATIONS.items() for code in codes}


def city_of(station: str) -> str:
    """The city a station belongs to; stations outside CITY_STATIONS are their own city."""
    return _CITY_OF_STATION.get(station, station)


def build_places(timetable: Timetable) -> dict[str, frozenset[str]]:
    """Searchable labels -> station codes: each multi-station city, then every station."""
    places = {}
    for city, codes in CITY_STATIONS.items():
        present = frozenset(c for c in codes if c in timetable.station_index)
        if present:
            places[f"{city} (all stations)"] = present
    for code in sorted(timetable.station_index, key=timetable.station_names.get):
        places[f"{timetable.station_names[code]} ({code})"] = frozenset([code])
    return places

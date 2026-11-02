"""Fetch the data into data/raw/ (about 300 MB; re-running skips what's already there).

- RSTGCN (IIT Kharagpur): September 2024 timetable and per-halt delays.
  https://github.com/KoyenaChowdhury/RSTGCN
- Indian Railways train delays (Hugging Face, CC BY 4.0): a year of runs
  (2025-02-08 .. 2026-02-07) and the dataset's own daily NTES crawl since
  2026-10-10. https://huggingface.co/datasets/pjay-in/indian-railways-delays
- IMD gridded daily rainfall (0.25°) for 2024 and 2025, from IMD Pune.
- Station coordinates (datameet).
- Kaggle "Railway Waitinglist Dataset": real waitlisted tickets. Kaggle needs a
  login, so it is copied from ../irctc/dataset when that project is next to
  this one; otherwise download it by hand.
"""

import json
import shutil
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
RSTGCN_URL = "https://github.com/KoyenaChowdhury/RSTGCN/raw/main/Indian-Railway-Network-and-Delays.zip"
HF_DATASET = "pjay-in/indian-railways-delays"
HF_FOLDERS = ["2025-02-08_2026-02-07", "data"]
IMD_URL = "https://www.imdpune.gov.in/cmpg/Griddata/RF25.php?RF25={year}"
STATIONS_URL = "https://raw.githubusercontent.com/datameet/railways/master/stations.json"
WAITLIST_NAME = "Railway Ticket WaitingList Data.csv"
IRCTC_COPY = ROOT.parent / "irctc" / "dataset" / WAITLIST_NAME


def fetch(url: str, target: Path) -> None:
    """Download to `target` unless it exists, retrying through flaky DNS and rate limits."""
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    for attempt in range(6):
        try:
            urllib.request.urlretrieve(url, partial)
            partial.rename(target)
            return
        except Exception as error:
            last = error
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"could not download {url}: {last}")


def rstgcn() -> None:
    folder = RAW / "rstgcn"
    if (folder / "Indian-Railway-Network-and-Delays" / "train_routes_Sep2024.csv").exists():
        return print("RSTGCN: already downloaded")
    print("RSTGCN: downloading 24 MB…", flush=True)
    fetch(RSTGCN_URL, folder / "rstgcn.zip")
    with zipfile.ZipFile(folder / "rstgcn.zip") as z:
        z.extractall(folder)


def hugging_face() -> None:
    paths = []
    for folder in HF_FOLDERS:
        url = f"https://huggingface.co/api/datasets/{HF_DATASET}/tree/main/{folder}?recursive=true"
        with urllib.request.urlopen(url, timeout=60) as response:
            paths += [item["path"] for item in json.load(response) if item["type"] == "file"]
    missing = [p for p in paths if not (RAW / "ntes" / p).exists()]
    print(f"Train runs: {len(paths) - len(missing)} of {len(paths)} day-files present, fetching {len(missing)}…",
          flush=True)
    base = f"https://huggingface.co/datasets/{HF_DATASET}/resolve/main/"
    with ThreadPoolExecutor(4) as pool:
        list(pool.map(lambda p: fetch(base + p, RAW / "ntes" / p), missing))


def rainfall() -> None:
    for year in (2024, 2025):
        print(f"IMD rainfall {year}…", flush=True)
        fetch(IMD_URL.format(year=year), RAW / "imd" / f"rain_{year}.grd")


def waitlist() -> None:
    target = RAW / "waitlist" / WAITLIST_NAME
    if target.exists():
        return print("Waitlist data: already present")
    if IRCTC_COPY.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(IRCTC_COPY, target)
        return print(f"Waitlist data: copied from {IRCTC_COPY}")
    print(f"Waitlist data: download Kaggle's 'Railway Waitinglist Dataset' and save it as {target}")


if __name__ == "__main__":
    rstgcn()
    hugging_face()
    rainfall()
    fetch(STATIONS_URL, RAW / "stations" / "stations.json")
    waitlist()
    print("Done.")

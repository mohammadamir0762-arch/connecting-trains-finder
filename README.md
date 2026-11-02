# Connecting Trains Finder

**Live app: https://connecting-trains-finder.streamlit.app**

IRCTC only searches direct trains. When there is no good direct train (say
Chennai → Lucknow), travellers find a two-train route by hand: try Chennai →
Jhansi, then Jhansi → Lucknow, then another city, and so on.

This app does that search for you. Pick two cities, a date, a class and the
conditions you expect (normal, heavy rain, Diwali/Chhath rush, winter fog),
and it shows:

1. **A recommended option, with the reasons.** The app picks the option with
   the best *expected* trip time. That counts the chance of missing the change
   and the real next train you would catch if you did.
2. **Direct trains** that run on that date, plus direct trains on the route
   that run on other days (with their days).
3. **Journeys with one change.** For each one: where to change, how long you
   wait, the risk of missing the change, the next train if you miss it, and
   the chance a waitlisted ticket confirms.

Every train is shown only on days it actually runs, including a second leg a
day or two later.

## Machine learning in the project

| Question | Model | Learned from | Held-out result |
|---|---|---|---|
| Will I make the connection? | Delay distribution at each halt, from gradient-boosted trees **and a neural network**, averaged | 3.5M real halts, Feb 2025–Feb 2026, with each day's rainfall and festival and fog seasons | AUC **0.875** on 200,000 real connections from Oct 2026 (waiting time alone: 0.727) |
| Will my waitlisted ticket confirm? | Gradient-boosted trees (beat a neural network), never rising with the WL number | Class and WL number at booking for 53,381 real waitlisted tickets | AUC **0.775** |
| Which option should I take? | Expected trip time from the two models above, then a ranking learned from the options people pick | Search results, model outputs, logged picks | — |

### Rainfall, Diwali and normal conditions

Each halt in the training data carries:

- **Rainfall:** the day's rain at the station and along the route so far, from
  IMD's gridded rain-gauge data.
- **Festival season:** whether the day falls between Durga Puja and a week
  after Chhath, or within a week of Holi.
- **Winter fog season:** whether the day falls between mid-December and
  mid-February.

The model therefore learns what each condition did to real delays, and a
scenario is answered by setting these features. Effects on held-out weeks
(share of halts over 1 h late):

| Condition | Actually happened (in it vs not) | Model what-if (on vs off, same halts) |
|---|---|---|
| Heavy rain along the route | 14.8% vs 13.1% | 15.0% vs 12.7% |
| Winter fog season | 15.8% vs 12.5% | 16.7% vs 12.7% |
| Festival season, all of India | 13.8% vs 13.0% | 13.0% vs 12.7% |
| Festival season, Bihar & east UP | 29.2% vs 24.1% | 26.1% vs 25.1% |

The festival effect is regional. Across the year, in Bihar and east UP, the
share of halts over 1 h late was:

- **18%** in September;
- **27%** in the festival season;
- **33%** in late November to early December;
- **30%** in the fog season.

The rise does not stop when the festivals end. With one year of data the
festival rush cannot be separated from the start of winter, so the model's
Diwali/Chhath scenario understates delays on those routes. The app says so.
Rain's measured effect is modest, and it is an underestimate: the data leaves
out cancelled and diverted trains, which is what floods often cause.

### Deep learning vs gradient-boosted trees

The delay model was trained two ways on the same 1.5M halts and stopped on the
same validation weeks:

- **Gradient-boosted trees** (scikit-learn `HistGradientBoostingClassifier`).
- **A neural network in PyTorch** (`railroute/deep_model.py`). It learns an
  embedding for every train, station, zone, train type and weekday, joins those
  to the numeric features, and passes them through two hidden layers (256 and
  128 units) with dropout.

| Held-out weeks, 891,541 halts | Log loss (lower is better) | AUC, >1 h late | Real connections, Oct 2026: AUC |
|---|---|---|---|
| No features (baseline) | 1.762 | — | 0.727 (waiting time alone) |
| Gradient-boosted trees | 1.325 | 0.898 | 0.863 |
| Neural network | 1.322 | 0.897 | **0.876** |
| **Average of both** (used) | **1.308** | **0.899** | 0.875 |

The validation weeks chose the average, and the test weeks agree. The neural
network generalises better than the trees to a later year (Oct 2026).

For waitlists (2 inputs, 53k tickets), the trees beat a neural network: test
AUC 0.775 vs 0.764. That is too little data for deep learning to help, so the
trees are used.

The waitlist model answers the app's exact question: "IRCTC shows WL n now;
if I book, will it confirm?" It learns from the class and the waitlist number
at booking. It predicts 43% for WL 1 in SL, 28% for WL 5, 10% for WL 45 and 2%
for WL 100. In the data, 37% of WL 1 tickets were confirmed.

An earlier version used the number seen a week before travel. It read a
just-booked WL 1 as a ticket stuck at WL 1, and predicted far too low.

### Recommending the best option

`railroute/recommend.py` works out an expected trip time for every option:

- **On time:** with probability 1 − miss risk, you arrive as scheduled.
- **Missed change:** with the miss risk, you arrive on the next train from the
  change station. The search finds that train in the timetable, including
  train 2 on its next running day.
- **Waitlist:** options whose waitlisted ticket is unlikely to confirm (below
  a 20% chance, the `irctc` project's "good chance" line) come last when a
  better option exists. So WL 1–2 (about 40%) keeps the faster train, while WL
  45 (10%) moves the recommendation to one with a seat. A connection needs both
  tickets confirmed, so the two chances are multiplied.

Among the rest, the lowest expected time wins. The app explains its choice against the fastest
option. For example, Nagpur → Guwahati in winter fog:

> **Recommended: 22511 (direct).** The fastest option (12859 → Kharagpur →
> 22503) is 5 min quicker on paper but has a 43% miss risk, so on average the
> trip takes 8h 05m longer.

Under winter fog, the recommendation differs from the fastest option in 138 of
840 city-pair searches. Picks made in the app are also logged. After 10 picks,
a pairwise logistic model (Bradley–Terry) learns a ranking from them. The app
offers that ranking as an extra sort option, with its cross-validated hit rate.

## Data

| Data | Source | Used for |
|---|---|---|
| Timetable, Sep 2024 | [RSTGCN](https://github.com/KoyenaChowdhury/RSTGCN), IIT Kharagpur | Search, running days, and calibrating miss risk on real Sep 2024 connections |
| 1.09M train runs, Feb 2025–Feb 2026 | [Hugging Face pjay-in/indian-railways-delays](https://huggingface.co/datasets/pjay-in/indian-railways-delays) (CC BY 4.0) | Training the delay model (30% of runs) |
| NTES runs, Oct 2026 | Same dataset's own daily NTES crawl | Final test: a later year with real departure times |
| Daily rainfall 2024–25, 0.25° grid | IMD Pune gridded rainfall; Open-Meteo ERA5 for 2026 days | Rain at every halt |
| Station coordinates | datameet; renamed stations placed between route neighbours | Matching stations to rainfall |
| Waitlisted tickets | Kaggle *Railway Waitinglist Dataset* (as in the `irctc` project) | Waitlist model |

The 2025–26 runs come from a Kaggle set whose original source is not stated,
so they were checked before use:

| Check | Result | What fabricated data would give |
|---|---|---|
| Each train-at-station's mean delay in Sep 2025 vs the independent IIT Kharagpur Sep 2024 data (27,541 pairs) | Spearman correlation **0.48** | 0.00 (checked by shuffling) |
| The same comparison against the dataset's own NTES crawl (Oct 2026 vs Oct 2025) | **0.68** | ~0 |
| Real-world seasons: winter fog, and the festival rush in Bihar | North-India delays more than double in December; the south stays flat. Bihar delays rise from Durga Puja | No regional or seasonal pattern |

## How it works

| File | What it does |
|---|---|
| `railroute/timetable.py` | Loads routes, works out running days, repairs broken timings |
| `railroute/search.py` | Finds direct trains and one-change connections on running days, plus the next train if a change is missed |
| `railroute/history.py` | Turns every delay source into one table of halts with route features |
| `railroute/weather.py` | Gives the IMD rainfall at each station on each day |
| `railroute/seasons.py` | Defines the festival and fog seasons and the scenarios |
| `railroute/delay_model.py` | Delay distributions, miss risk, calibration and every evaluation |
| `railroute/deep_model.py` | The PyTorch neural network |
| `railroute/waitlist_model.py` | Waitlist confirmation, trees vs neural network |
| `railroute/recommend.py` | Expected trip time and the recommendation with reasons |
| `railroute/ranker.py` | Ranking learned from logged picks |
| `scripts/train_models.py` | Trains and scores everything (about 10 min) and writes `models/metrics.json` |
| `app.py` | Streamlit web UI |

To keep the evaluation honest:

- **Splits by whole weeks.** Four weeks in six train the model, one validates
  it and one is held out for testing. All three are spread across the year, so
  each covers every season, and no week appears in more than one.
- **No self-leakage in history features.** A run never contributes to its own
  history features: those are computed out-of-fold by week.
- **A later, separate test.** The miss risk is calibrated on September 2024 and
  tested on October 2026, a different collection from a later year. Only
  times that stations actually reported are used.

## Run it

Needs Python 3.12. The trained models and the timetable cache are in the repo,
so the app runs straight after installing:

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run app.py
```

To rebuild everything from the raw data:

```bash
.venv/bin/python scripts/download_data.py    # ~300 MB, once
.venv/bin/python scripts/train_models.py     # ~10 min; prints every test score
.venv/bin/python scripts/compare_models.py   # ~25 min: baselines vs our models, timings
.venv/bin/pytest
```

## Limits

- **Old timetable.** Timetable and running days are from September 2024.
  Confirm on IRCTC before booking.
- **Festival effect.** One year of data cannot separate the festival rush from
  the start of winter in Bihar and east UP (see above).
- **Rain effect.** Rain's effect is underestimated, because cancelled and
  diverted trains are not in the training data.
- **Waitlist data.** The waitlist data has no dates, trains or routes, so the
  confirm chance cannot account for festival rush or how far ahead you book.
  It depends only on class and waitlist number.
- **Typed-in waitlist numbers.** There is no free live-availability API.

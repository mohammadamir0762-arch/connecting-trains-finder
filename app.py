"""Streamlit UI: find direct trains and one-change journeys, with a recommendation.

Every option shows the risk of missing the change under the chosen conditions
(delay model), what happens if you do miss it (the next train), the chance a
waitlisted ticket confirms (waitlist model), and one option is recommended.

Run with:  streamlit run app.py
"""

from datetime import date, datetime, timedelta

import pandas as pd
import streamlit as st

from railroute import delay_model, ranker, waitlist_model
from railroute.cities import build_places
from railroute.recommend import Assessment, assess, label as option_label, percent, recommend
from railroute.search import Journey, SearchOptions, find_journeys
from railroute.seasons import SCENARIOS, Scenario
from railroute.timetable import WEEKDAYS, load_timetable
from railroute.waitlist_model import chance_label

st.set_page_config(page_title="Connecting Trains Finder", page_icon="🚆", layout="wide")

CLASS_NAMES = {
    "SL": "Sleeper (SL)", "3A": "AC 3-tier (3A)", "2A": "AC 2-tier (2A)",
    "1A": "AC First Class (1A)", "CC": "AC Chair Car (CC)", "2S": "Second Sitting (2S)",
}
CONDITION_NAMES = {
    "Auto (from date)": "📅 Based on the date", "Normal": "☀️ Normal day", "Heavy rain": "🌧️ Heavy rain",
    "Diwali / Chhath rush": "🪔 Diwali / Chhath rush", "Winter fog": "🌫️ Winter fog",
}
DEFAULTS = {
    "origin": "Chennai (all stations)", "destination": "Lucknow (all stations)", "travel_date": date.today(),
    "travel_class": "SL", "scenario": "Auto (from date)", "min_same": SearchOptions.min_transfer_same_station,
    "min_other": SearchOptions.min_transfer_other_station, "max_wait": SearchOptions.max_layover // 60,
}
FOOTER = ("Timetable from September 2024 (RSTGCN, IIT Kharagpur); delays learned from a year of real train runs "
          "and IMD rainfall. Not an official Indian Railways or IRCTC service: always check times on IRCTC "
          "before you book.")
IRCTC_URL = "https://www.irctc.co.in/nget/train-search"
PICK_HELP = ("This doesn't book anything. It tells us which option you'd choose, so the app learns what "
             "travellers prefer.")


# ---------------------------------------------------------------- data and models

@st.cache_resource(show_spinner="Loading timetable and models…")
def load():
    timetable = load_timetable()
    return timetable, build_places(timetable), delay_model.load(), waitlist_model.load()


@st.cache_resource
def load_ranker(choices_mtime: float):
    choices = ranker.load_choices()
    if len(choices) < ranker.MIN_CHOICES:
        return None, len(choices), None
    return ranker.Ranker.fit(choices), len(choices), ranker.evaluate(choices)


@st.cache_data(show_spinner="Searching trains…", max_entries=20)
def run_search(origin: str, destination: str, travel_date: date, options: SearchOptions, scenario: str):
    """All journeys, each connection's miss risk and each journey's usual delay, under `scenario`."""
    result = find_journeys(timetable, places[origin], places[destination], travel_date, options)
    journeys = result.direct + result.connections
    conditions = SCENARIOS[scenario]
    miss = delays.miss_risks(timetable, result.connections, conditions) if delays else None
    typical = delays.typical_arrival_delays(timetable, journeys, conditions) if delays else None
    return result, miss, typical


# ---------------------------------------------------------------- formatting

def clock(moment: datetime) -> str:
    return moment.strftime("%H:%M")


def day(moment: datetime) -> str:
    return moment.strftime("%a %d %b")


def hours(span: timedelta) -> str:
    minutes = round(span.total_seconds() / 60)
    return f"{minutes} min" if minutes < 60 else f"{minutes // 60}h {minutes % 60:02d}m"


def place(label: str) -> str:
    return label.replace(" (all stations)", "")


def station(code: str) -> str:
    return f"{timetable.station_names.get(code, code)} ({code})"


def station_name(code: str) -> str:
    return timetable.station_names.get(code, code)


def train_label(number: str) -> str:
    return f"{number} {timetable.trains[number].name}"


def runs_on(number: str) -> str:
    days = sorted(timetable.trains[number].running_days)
    return "daily" if len(days) == 7 else ", ".join(WEEKDAYS[d] for d in days)


def route_summary(j: Journey) -> str:
    if len(j.legs) == 1:
        return f"🚆 {train_label(j.legs[0].train_number)} — direct"
    first, second = j.legs
    return f"🚆 {first.train_number} → change at {station_name(first.to_station)} → 🚆 {second.train_number}"


def risk_badge(p: float) -> tuple[str, str]:
    if p < 0.10:
        return f"Low risk · {percent(p)} chance to miss the change", "green"
    if p < 0.30:
        return f"Some risk · {percent(p)} chance to miss the change", "orange"
    return f"High risk · {percent(p)} chance to miss the change", "red"


def seat_badge(chance: float | None) -> tuple[str, str] | None:
    if chance is None:
        return None
    color = "green" if chance >= 0.20 else "orange" if chance >= 0.10 else "red"
    return f"Seat: {chance:.0%} chance ({chance_label(chance)})", color


def late_text(minutes: float | None) -> str:
    if minutes is None:
        return "—"
    return "usually on time" if minutes < 5 else (
        f"usually ~{minutes:.0f} min late" if minutes < 90 else f"usually ~{minutes / 60:.1f} h late")


def describe_conditions(scenario: Scenario, on: date) -> str:
    if scenario.name != "Auto (from date)":
        return scenario.description
    found = [name for name, active in (("festival travel season", scenario.festival_on(on)),
                                       ("winter fog season", scenario.winter_on(on))) if active]
    return f"{' and '.join(found).capitalize() if found else 'No festival rush or fog season'} on this date."


# ---------------------------------------------------------------- state and callbacks

for _key, _value in DEFAULTS.items():
    st.session_state.setdefault(_key, _value)
st.session_state.setdefault("waitlists", {})


def current_inputs() -> tuple:
    s = st.session_state
    options = SearchOptions(min_transfer_same_station=s.min_same, min_transfer_other_station=s.min_other,
                            max_layover=s.max_wait * 60)
    return s.origin, s.destination, s.travel_date, s.travel_class, options, s.scenario


def swap_places() -> None:
    s = st.session_state
    s.origin, s.destination = s.destination, s.origin


def search_clicked() -> None:
    s = st.session_state
    if places[s.origin] & places[s.destination]:
        s.search_error = "From and To are the same place — pick two different places."
    else:
        s.pop("search_error", None)
        s.last_search = current_inputs()


def add_waitlist(search_id: int) -> None:
    s = st.session_state
    s.waitlists.setdefault(search_id, {})[s.wl_train] = int(s.wl_number)


def remove_waitlist(search_id: int, train: str) -> None:
    st.session_state.waitlists.get(search_id, {}).pop(train, None)


def log_pick(options: list[dict], chosen: int, context: dict, label: str) -> None:
    ranker.log_choice(options=options, chosen=chosen, context=context)
    st.session_state.picked = label


timetable, places, delays, waitlist = load()
labels = list(places)

# ---------------------------------------------------------------- search panel

st.title("🚆 Connecting Trains Finder")
st.markdown("Find **direct trains and journeys with one change** that IRCTC doesn't show you, see the risk "
            "of missing the change, and get a clear recommendation.")

with st.container(border=True):
    col_from, col_swap, col_to = st.columns([12, 1, 12], vertical_alignment="bottom")
    col_from.selectbox("From", labels, key="origin", help="Type to search a city or station.")
    col_swap.button("⇄", on_click=swap_places, help="Swap From and To", width="stretch")
    col_to.selectbox("To", labels, key="destination", help="Type to search a city or station.")
    col_date, col_class, col_conditions = st.columns(3)
    col_date.date_input("Journey date", key="travel_date", format="DD/MM/YYYY")
    col_class.selectbox("Class", waitlist_model.TRAVEL_CLASSES, key="travel_class", format_func=CLASS_NAMES.get)
    col_conditions.selectbox(
        "Travel conditions", list(SCENARIOS), key="scenario", format_func=CONDITION_NAMES.get,
        help="How delays are predicted. 'Based on the date' checks festival and fog seasons for you; the others "
             "let you see what heavy rain, the festival rush or fog would do to your trip.")
    with st.expander("⚙️ More options: time to change trains and how long you'd wait"):
        col_a, col_b, col_c = st.columns(3)
        col_a.number_input("Minimum time to change trains at the same station (minutes)", 15, 240, step=15,
                           key="min_same")
        col_b.number_input("Minimum time to cross to another station in the same city (minutes)", 30, 360,
                           step=15, key="min_other")
        col_c.slider("Longest wait you'd accept at the change station (hours)", 1, 24, key="max_wait")
    st.button("🔍 Search trains", type="primary", on_click=search_clicked, width="stretch")

if "search_error" in st.session_state:
    st.warning(st.session_state.search_error)

if "last_search" not in st.session_state:
    st.write("")
    for column, (icon, title, text) in zip(st.columns(3), [
        ("📍", "Choose your trip", "Pick where you start, where you're going, the date and your class."),
        ("🔁", "See every way there", "Direct trains first, then journeys with one change, only on days the "
                                    "trains actually run."),
        ("⭐", "Get the best choice", "We check how late the first train usually runs, so you don't pick a "
                                    "change you're likely to miss. Then we tell you which option to take, and why."),
    ]):
        with column.container(border=True):
            st.markdown(f"### {icon} {title}")
            st.write(text)
    st.caption(FOOTER)
    st.stop()

if current_inputs() != st.session_state.last_search:
    st.info("You've changed the search. Press **Search trains** to update the results.", icon="✏️")

# ---------------------------------------------------------------- results

origin, destination, travel_date, travel_class, options, scenario_name = st.session_state.last_search
scenario = SCENARIOS[scenario_name]
search_id = abs(hash(st.session_state.last_search))
result, miss_risks, typical_delays = run_search(origin, destination, travel_date, options, scenario_name)
journeys = result.direct + result.connections

if picked := st.session_state.pop("picked", None):
    st.toast(f"Thanks! Noted your choice: {picked}", icon="✅")

st.markdown(f"## {place(origin)} → {place(destination)}")
st.caption(f"{travel_date:%A %d %B %Y} · {CLASS_NAMES[travel_class]} · {CONDITION_NAMES[scenario_name]}: "
           f"{describe_conditions(scenario, travel_date)}")

if not journeys:
    st.warning("No train runs between these places on this date, directly or with one change. Try another date, "
               "a longer wait under ⚙️ More options, or nearby stations.")
    st.stop()

waitlisted = st.session_state.waitlists.get(search_id, {})
leg_chances: dict[str, float] = {}


def confirm_chance(journey: Journey) -> float | None:
    """Chance every waitlisted leg confirms (legs treated as independent); None if none is waitlisted."""
    if waitlist is None or not any(leg.train_number in waitlisted for leg in journey.legs):
        return None
    chance = 1.0
    for leg in journey.legs:
        number = leg.train_number
        if number not in leg_chances:  # many connections share a train: predict each once
            leg_chances[number] = waitlist.confirm_chance(travel_class, waitlisted.get(number, 0))
        chance *= leg_chances[number]
    return chance


miss = dict(zip(map(id, result.connections), miss_risks)) if delays else {}
typical = dict(zip(map(id, journeys), typical_delays)) if delays else {}
judged: dict[int, Assessment] = {id(j): assess(j, float(miss.get(id(j), 0.0)), confirm_chance(j)) for j in journeys}
features = {id(j): ranker.option_features(j, judged[id(j)].miss_risk, judged[id(j)].confirm_chance or 1.0)
            for j in journeys}
context = {"from": origin, "to": destination, "date": travel_date.isoformat(), "class": travel_class,
           "conditions": scenario_name}

best, reasons = recommend(list(judged.values()), station_name)
fastest = min(judged.values(), key=lambda a: (a.journey.duration, a.journey.arrives))
safest = min(judged.values(), key=lambda a: (a.seat_unlikely, round(a.miss_risk, 2), a.expected_duration))

choices_mtime = ranker.CHOICES_PATH.stat().st_mtime if ranker.CHOICES_PATH.exists() else 0.0
learned, n_choices, ranker_scores = load_ranker(choices_mtime)

sort_keys = {
    "⭐ Best": lambda j: judged[id(j)].rank_key,
    "⚡ Fastest": lambda j: (j.duration, j.arrives),
    "🛡️ Most reliable": lambda j: (round(judged[id(j)].miss_risk, 2), j.duration),
    "🕐 Arrives first": lambda j: (j.arrives, j.duration),
    "⏳ Shortest wait": lambda j: (j.layover or timedelta(0), j.duration),
}
SORT_HELP = ("**⭐ Best**: quickest on average, once the chance of missing the change is counted.  \n"
             "**⚡ Fastest**: shortest by the timetable.  \n"
             "**🛡️ Most reliable**: lowest chance of missing the change.  \n"
             "**🕐 Arrives first** and **⏳ Shortest wait** (at the change station) do what they say.")
if learned is not None:
    score = dict(zip(map(id, journeys), learned.scores([features[id(j)] for j in journeys])))
    sort_keys["👥 Popular picks"] = lambda j: -score[id(j)]
    SORT_HELP += "  \n**👥 Popular picks**: what other travellers chose."


def count(n: int, thing: str) -> str:
    return f"**{n} {thing}{'' if n == 1 else 's'}**" if n else f"no {thing}"


st.markdown(f"On {travel_date:%a %d %b} we found {count(len(result.direct), 'direct train')} and "
            f"{count(len(result.connections), 'journey')} with one change.")

# Origin and destination stations whose name doesn't say the city, e.g. "Aishbagh" is in Lucknow.
city_of_station = {code: place(label) for label in (origin, destination) if label.endswith(" (all stations)")
                   for code in places[label]}


def stop(code: str) -> str:
    name, city = station_name(code), city_of_station.get(code)
    return name if city is None or city.lower() in name.lower() else f"{name} ({city})"


def journey_lines(j: Journey) -> list[str]:
    lines = []
    for i, leg in enumerate(j.legs):
        lines.append(f"🚆 **{train_label(leg.train_number)}**")
        lines.append(f"&emsp;{stop(leg.from_station)} **{clock(leg.departs)}** {leg.departs:%a} → "
                     f"{stop(leg.to_station)} **{clock(leg.arrives)}** {leg.arrives:%a}")
        if i == 0 and len(j.legs) > 1:
            if j.legs[1].from_station != leg.to_station:
                lines.append(f"🚶 Change at **{station_name(leg.to_station)}** and go to **"
                             f"{station_name(j.legs[1].from_station)}** station (**{hours(j.layover)}** between trains)")
            else:
                lines.append(f"↪️ Change at **{station_name(leg.to_station)}**, wait **{hours(j.layover)}**")
    return lines


def missed_text(assessment: Assessment) -> str:
    j, missed = assessment.journey, assessment.missed_arrival
    if not j.fallback_train:
        return "If you miss it: no later train within a week."
    late = missed - j.arrives
    return (f"If you miss it: next train {j.fallback_train} arrives {missed:%a} {clock(missed)} "
            + (f"({hours(late)} later)." if late else "(no later)."))


def badges(assessment: Assessment, explain_miss: bool = True) -> None:
    j = assessment.journey
    if len(j.legs) > 1:
        text, color = risk_badge(assessment.miss_risk)
        st.badge(text, color=color)
        if explain_miss:
            st.caption(missed_text(assessment))
    else:
        st.badge("Direct, no change", color="blue")
        if id(j) in typical:
            st.caption(late_text(typical[id(j)]).capitalize())
    if seat := seat_badge(assessment.confirm_chance):
        st.badge(seat[0], color=seat[1])


def details(assessment: Assessment) -> None:
    j = assessment.journey
    for leg in j.legs:
        st.markdown(f"**{train_label(leg.train_number)}** runs {runs_on(leg.train_number)}  \n"
                    f"{station(leg.from_station)} {clock(leg.departs)} {day(leg.departs)} → "
                    f"{station(leg.to_station)} {clock(leg.arrives)} {day(leg.arrives)}")
    if len(j.legs) > 1:
        st.markdown(f"**{missed_text(assessment)}**")
        st.markdown(f"**Average trip**, counting the chance of a miss: {hours(assessment.expected_duration)} "
                    f"(timetable: {hours(j.duration)}).")
    elif id(j) in typical:
        st.markdown(f"**Arrival:** {late_text(typical[id(j)])} under these conditions.")


def pick_args(j: Journey, on_screen: list[Journey]) -> tuple:
    """What a pick logs: the options the traveller saw, which one they chose, the search and a label."""
    shown = on_screen if j in on_screen else [j] + on_screen
    return ([{**features[id(o)], "label": option_label(o)} for o in shown], shown.index(j), context,
            route_summary(j))


def option_card(j: Journey, key: str, on_screen: list[Journey], tags: list[str] = ()) -> None:
    assessment = judged[id(j)]
    with st.container(border=True):
        col_time, col_route, col_info, col_act = st.columns([2.2, 5, 2.6, 1.4], vertical_alignment="center")
        col_time.markdown(f"#### {clock(j.departs)} → {clock(j.arrives)}")
        col_time.caption(f"{day(j.departs)} → {day(j.arrives)}  \n{hours(j.duration)}"
                         + (" · 1 change" if len(j.legs) > 1 else " · direct"))
        with col_route:
            if tags:
                st.markdown(" ".join(f"**{t}**" for t in tags))
            st.markdown("  \n".join(journey_lines(j)))
        with col_info:
            badges(assessment)
        with col_act:
            with st.popover("Details", width="stretch"):
                details(assessment)
            st.button("👍 I'd choose this", key=f"pick-{key}", width="stretch", on_click=log_pick,
                      args=pick_args(j, on_screen), help=PICK_HELP)


recommendation_box = st.container(border=True)

# ---------------------------------------------------------------- other options worth a look

others = []
if fastest is not best:
    quicker = best.journey.duration - fastest.journey.duration
    why = "Quickest by the timetable" + (f", {hours(quicker)} less than ours" if quicker > timedelta(0) else "")
    if fastest.seat_unlikely:
        why += f", but only a {fastest.confirm_chance:.0%} chance your waitlisted ticket confirms."
    elif len(fastest.journey.legs) > 1:
        why += f", but a {percent(fastest.miss_risk)} chance of missing the change."
    else:
        why += "."
    title = "⚡ Fastest" + (" · 🛡️ Most reliable" if safest is fastest else "")
    others.append((title, fastest, why))
if safest is not best and safest is not fastest:
    longer = safest.journey.duration - best.journey.duration
    why = (f"Lowest chance of missing the change ({percent(safest.miss_risk)})"
           + (f", but takes {hours(longer)} longer than ours." if longer > timedelta(0) else "."))
    others.append(("🛡️ Most reliable", safest, why))
if others:
    st.markdown("#### Other options worth a look")
    for column, (title, a, why) in zip(st.columns(2), others):
        with column.container(border=True):
            st.markdown(f"**{title}** · {clock(a.journey.departs)} {a.journey.departs:%a} → "
                        f"{clock(a.journey.arrives)} {a.journey.arrives:%a} · {hours(a.journey.duration)}")
            st.markdown(route_summary(a.journey))
            st.caption(why)
            badges(a, explain_miss=False)

# ---------------------------------------------------------------- waitlist entry

order = [leg.train_number for a in (best, fastest, safest) for leg in a.journey.legs]
order += sorted({leg.train_number for j in journeys for leg in j.legs})
train_choices = list(dict.fromkeys(order))

with st.expander("🎫 Waitlisted on IRCTC? Add the WL number to see if you'll get a seat"
                 + (f" ({len(waitlisted)} added)" if waitlisted else ""), key="waitlist_box"):
    st.caption(f"If IRCTC shows a waitlist like \"WL 12\" for a train in {CLASS_NAMES[travel_class]}, add it here. "
               "Every option using that train then shows its chance of a confirmed seat, and the recommendation "
               "avoids tickets unlikely to confirm. Trains you don't add are treated as having seats.")
    col_train, col_number, col_add = st.columns([5, 2, 1.2], vertical_alignment="bottom")
    col_train.selectbox("Train", train_choices, format_func=train_label, key="wl_train",
                        help="Trains in our picks come first. Type to search.")
    col_number.number_input("WL number", 1, 999, value=10, key="wl_number")
    col_add.button("Add", on_click=add_waitlist, args=(search_id,), width="stretch", type="primary")
    for train, number in waitlisted.items():
        chance = waitlist.confirm_chance(travel_class, number) if waitlist else None
        col_text, col_remove = st.columns([8, 1.2], vertical_alignment="center")
        col_text.markdown(f"**{train_label(train)}** — WL {number}"
                          + (f" → {chance:.0%} chance of a confirmed seat ({chance_label(chance)})"
                             if chance is not None else ""))
        col_remove.button("Remove", key=f"remove-{train}", on_click=remove_waitlist, args=(search_id, train),
                          width="stretch")
    if scenario.festival_on(travel_date):
        st.caption("⚠️ Seat chances come from waitlist data without dates, so they can't account for the "
                   "festival rush: treat them as optimistic around Diwali, Chhath and Holi.")

# ---------------------------------------------------------------- every option

tab_labels = [f"🚆 Direct trains ({len(result.direct)})", f"🔁 With one change ({len(result.connections)})",
              "🌦️ Conditions", "ℹ️ How it works"]
# Open on the list that has trains in it, not an empty one.
tab_direct, tab_change, tab_conditions, tab_how = st.tabs(
    tab_labels, default=tab_labels[0] if result.direct or not result.connections else tab_labels[1])

with tab_change:
    col_sort, col_show, col_view = st.columns([4, 1.4, 1.6], vertical_alignment="bottom")
    sort_by = col_sort.segmented_control("Sort by", list(sort_keys), default="⭐ Best", key="sort_by",
                                         help=SORT_HELP) or "⭐ Best"
    show = col_show.selectbox("Show", ["10", "30", "100", "All"], key="show")
    as_table = col_view.toggle("Table view", key="as_table")
    connections = sorted(result.connections, key=sort_keys[sort_by])
    connections = connections if show == "All" else connections[: int(show)]

direct = sorted(result.direct, key=lambda j: j.departs)
on_screen = direct + connections

with recommendation_box:
    j = best.journey
    st.markdown("### ⭐ Our recommendation")
    col_trip, col_why = st.columns(2, gap="large")
    with col_trip:
        st.markdown(f"#### {clock(j.departs)} {j.departs:%a} → {clock(j.arrives)} {j.arrives:%a}")
        st.caption(f"{day(j.departs)} → {day(j.arrives)} · {hours(j.duration)} · "
                   + ("1 change" if len(j.legs) > 1 else "direct"))
        badges(best, explain_miss=False)
        st.markdown("  \n".join(journey_lines(j)))
    with col_why:
        st.markdown("**Why this one**")
        st.markdown("\n".join(f"- {reason}" for reason in reasons))
        st.markdown("**How to book**: " + ("two separate tickets, each for the day that train leaves your station."
                                           if len(j.legs) > 1 else "one ticket."))
        st.markdown("\n".join(f"{i}. **{leg.train_number}** {stop(leg.from_station)} → {stop(leg.to_station)}, "
                              f"on **{leg.departs:%a %d %b}**" for i, leg in enumerate(j.legs, 1)))
        st.caption("Times are from the September 2024 timetable, so check them on IRCTC when you book.")
        col_book, col_pick = st.columns(2)
        col_book.link_button("Book on IRCTC ↗", IRCTC_URL, width="stretch")
        col_pick.button("👍 I'd choose this", key="pick-recommended", width="stretch", on_click=log_pick,
                        args=pick_args(j, on_screen), help=PICK_HELP)

with tab_direct:
    if result.direct_not_running:
        others = " · ".join(f"{train_label(n)} (runs {runs_on(n)})" for n in result.direct_not_running)
        st.info(f"Also direct on this route, but not on {travel_date:%a %d %b}: {others}", icon="📅")
    if not direct:
        st.write(f"No direct train runs on {travel_date:%a %d %b}. See **With one change** for other ways there.")
    for j in direct:
        option_card(j, f"direct-{j.legs[0].train_number}", on_screen)

with tab_change:
    if not connections:
        st.write("No journey with one change within your waiting limit. Try a longer wait under "
                 "⚙️ More options.")
    elif as_table:
        st.caption("Read-only table. Switch off Table view to see details and pick an option.")
        train_column = st.column_config.TextColumn(width=150)
        st.dataframe(pd.DataFrame([{
            "Total": hours(j.duration),
            "Average (with misses)": hours(judged[id(j)].expected_duration),
            "Train 1": train_label(j.legs[0].train_number),
            "Departs": f"{day(j.departs)} {clock(j.departs)} · {j.legs[0].from_station}",
            "Change at": station(j.legs[0].to_station),
            "Wait": hours(j.layover),
            "Chance to miss change": percent(judged[id(j)].miss_risk),
            "Train 2": train_label(j.legs[1].train_number),
            "Arrives": f"{day(j.arrives)} {clock(j.arrives)} · {j.legs[1].to_station}",
            "If you miss it": (f"{j.fallback_train} · {day(judged[id(j)].missed_arrival)} "
                               f"{clock(judged[id(j)].missed_arrival)}" if j.fallback_train else "—"),
            "Seat chance": (f"{judged[id(j)].confirm_chance:.0%}"
                            if judged[id(j)].confirm_chance is not None else "—"),
        } for j in connections]), hide_index=True, column_config={"Train 1": train_column, "Train 2": train_column})
    else:
        st.caption(f"Showing {len(connections)} of {len(result.connections)}, with a wait of up to "
                   f"{options.max_layover // 60} h.")
        for j in connections:
            tags = [t for t, a in (("⭐ Recommended", best), ("⚡ Fastest", fastest), ("🛡️ Most reliable", safest))
                    if a.journey is j]
            option_card(j, f"change-{j.legs[0].train_number}-{j.legs[1].train_number}", on_screen, tags)

with tab_conditions:
    if delays:
        j = best.journey
        st.markdown("#### How the conditions change the recommended trip")
        rows = []
        for name, conditions in SCENARIOS.items():
            risk = delays.miss_risks(timetable, [j], conditions)[0] if len(j.legs) > 1 else None
            rows.append({"Conditions": CONDITION_NAMES[name],
                         "Risk of missing the change": percent(risk) if risk is not None else "— (direct)",
                         "Arrival": late_text(delays.typical_arrival_delays(timetable, [j], conditions)[0]),
                         "What it means": conditions.description})
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        st.caption("Learned from a year of real train runs with each day's IMD rainfall and the festival and fog "
                   "seasons. Cancelled and diverted trains aren't in that data, so the worst effects of floods are "
                   "not captured.")
        seasons = delays.metrics.get("observed_by_season")
        if seasons:
            st.markdown("#### Festival travel in the data (share of stops over 1 hour late, 2025)")
            st.dataframe(pd.DataFrame([
                {"Period": name, "Bihar & east UP": f"{v['bihar_east_up']:.0%}",
                 "Rest of India": f"{v['rest_of_india']:.0%}"} for name, v in seasons.items()
            ]), hide_index=True)
            st.caption("Around Diwali and Chhath, Bihar and east UP see far more delays than in September, but the "
                       "rise carries on into the fog season. With one year of data the model can't tell the festival "
                       "rush from the start of winter, so its Diwali / Chhath estimate is low on these routes.")
    else:
        st.write("The delay model isn't trained yet: run `.venv/bin/python scripts/train_models.py`.")

with tab_how:
    st.markdown(
        "**1. Finding trains.** We search the timetable for direct trains and for every way to get there with "
        "one change, using only trains that run on your date — including the second train a day later.\n\n"
        "**2. Will you make the change?** A machine-learning model learned from a year of real train runs how "
        "late each train usually is at each station, and how rain, the festival rush and winter fog change that. "
        "From it we work out the chance your first train arrives too late for the second.\n\n"
        "**3. Will you get a seat?** If you add a waitlist number, a second model — trained on 53,000 real "
        "waitlisted tickets — estimates the chance it gets confirmed.\n\n"
        "**4. The recommendation.** For each option we work out the *expected* trip time: on time if you make "
        "the change, or on the next train if you miss it. We recommend the quickest on average, skipping tickets "
        "unlikely to confirm.\n\n"
        "**5. Learning from you.** When you press *👍 I'd choose this*, we note your choice (nothing is booked). After "
        f"{ranker.MIN_CHOICES} picks, a **👥 Popular picks** sort appears ({n_choices} so far)."
    )
    with st.expander("Technical details and test results"):
        if delays:
            m = delays.metrics
            t = m["trained_on"]
            st.markdown(
                f"**Delay model** — {t['halts']:,} real halts ({t['from']} to {t['to']}, {t['runs_percent']}% of "
                "runs). Gradient-boosted trees and a neural network were trained on the same halts and compared on "
                f"held-out weeks; the app uses **{m['using']}**.")
            halts = m["test_halts"]
            rows = [{"Model": {"trees": "Gradient-boosted trees", "deep": "Neural network (embeddings)",
                               "trees+deep": "Average of both"}.get(k, k), **v}
                    for k, v in halts.items() if isinstance(v, dict)]
            rows.append({"Model": "No features (baseline)", "log_loss": halts["log_loss_no_features"]})
            st.dataframe(pd.DataFrame(rows).rename(columns={
                "log_loss": "Log loss (lower better)", "auc_over_1h_late": "AUC: >1 h late",
                "median_abs_error_min": "Median error (min)"}), hide_index=True)
            c = m["connections"]
            st.markdown(
                f"Tested on **{c['pairs']:,} real connections** from NTES, {c['days']}: AUC **{c['auc']}** (waiting "
                f"time alone: {c['auc_layover_only']}), Brier {c['brier']} (waiting time alone: "
                f"{c['brier_layover_only']}). Trees alone: AUC {c.get('auc_trees')}, neural network alone: "
                f"{c.get('auc_deep')}.")
            st.dataframe(pd.DataFrame(c["calibration"]), hide_index=True)
            st.markdown("**Conditions** — share of held-out stops over 1 h late, observed and predicted, and the "
                        "what-if with only that condition switched on.")
            st.dataframe(pd.DataFrame([{"Condition": k, **v} for k, v in m["conditions"].items()]), hide_index=True)
        if waitlist:
            w = waitlist.metrics
            test = w.get("test", {})
            st.markdown(
                f"**Waitlist model** — class and WL number at booking, {w['tickets']:,} real tickets (Kaggle Railway "
                f"Waitinglist Dataset). Held-out AUC **{w['test_auc']}** with {w.get('using', 'trees')}"
                + (f" (trees {test['trees']['auc']} vs neural network {test['neural_net']['auc']})" if test else "")
                + ".")
            st.dataframe(pd.DataFrame(w["calibration"]), hide_index=True)
        if learned is not None:
            st.markdown(f"**Popular picks** — learned from {n_choices} picks; top option matched the pick "
                        f"{ranker_scores['top1_learned']:.0%} of the time (fastest-first: "
                        f"{ranker_scores['top1_fastest_first']:.0%}).")
            st.json(learned.weight_table())
        st.markdown("**Timetable and running days** — RSTGCN dataset (IIT Kharagpur), September 2024.")

st.divider()
st.caption(FOOTER)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests

st.set_page_config(page_title="Live Drive Probability Engine", page_icon="🏈", layout="wide")

def prob_to_american(p: float) -> str:
    if p <= 0.001: return "+9999"
    if p >= 0.999: return "-9999"
    if p >= 0.5:
        return f"-{round((p / (1.0 - p)) * 100)}"
    return f"+{round(((1.0 - p) / p) * 100)}"

@st.cache_resource
def load_model():
    data = joblib.load("drive_outcome_model.pkl")
    return data["model"], data["features"], data["classes"]

model, feature_cols, classes = load_model()

@st.cache_data(ttl=15)
def fetch_live_nfl_games():
    url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(url, headers=headers, timeout=5)
        if r.status_code == 200:
            return r.json().get("events", [])
    except Exception:
        pass
    return []

st.title("🏈 Live Drive Probability Engine")

# Fetch current scoreboard
events = fetch_live_nfl_games()
game_options = {}

for ev in events:
    name = ev.get("name", "NFL Game")
    state = ev.get("status", {}).get("type", {}).get("state", "pre")
    display_clock = ev.get("status", {}).get("displayClock", "")
    period = ev.get("status", {}).get("period", 0)
    label = f"{name} ({state.upper()} - Q{period} {display_clock})" if state == "in" else f"{name} ({state.upper()})"
    game_options[label] = ev

# Mode Selector in Sidebar
st.sidebar.header("Mode Selection")
mode = st.sidebar.radio("Data Mode", ["📡 Live Game Stream", "🎛️ Manual Controls"])

if mode == "📡 Live Game Stream" and game_options:
    selected_game_label = st.sidebar.selectbox("Select Active Game", list(game_options.keys()))
    selected_ev = game_options[selected_game_label]
    
    status = selected_ev.get("status", {})
    qtr = status.get("period", 1)
    clock_str = status.get("displayClock", "15:00")
    
    try:
        m, s = map(int, clock_str.split(":"))
        qtr_seconds = m * 60 + s
    except Exception:
        qtr_seconds = 900
        
    half_seconds = qtr_seconds + 900 if qtr in [1, 3] else qtr_seconds
    game_seconds = qtr_seconds + (4 - min(qtr, 4)) * 900
    
    competitors = selected_ev["competitions"][0]["competitors"]
    home = next(c for c in competitors if c["homeAway"] == "home")
    away = next(c for c in competitors if c["homeAway"] == "away")
    
    situation = selected_ev["competitions"][0].get("situation", {})
    possession_id = situation.get("possession")
    yardline_100 = situation.get("yardLine", 75)
    
    home_score = int(home.get("score", 0))
    away_score = int(away.get("score", 0))
    
    if possession_id == home["id"]:
        score_diff = home_score - away_score
        poss_team = home["team"]["abbreviation"]
    else:
        score_diff = away_score - home_score
        poss_team = away.get("team", {}).get("abbreviation", "OFF")

    field_start = f"Own {100 - yardline_100}" if yardline_100 > 50 else f"Opp {yardline_100}"
    st.info(f"**Auto-Tracking:** {selected_ev['name']} | Poss: **{poss_team}** | Ball: **{field_start}** | Q{qtr} **{clock_str}** | Margin: **{score_diff:+d}**")

else:
    if mode == "📡 Live Game Stream" and not game_options:
        st.warning("No live NFL games currently active on ESPN's scoreboard. Switched to manual controls.")
    
    st.sidebar.subheader("Manual Game Snapshot")
    qtr = st.sidebar.radio("Quarter", [1, 2, 3, 4], index=2, horizontal=True)
    time_min = st.sidebar.slider("Minutes Remaining", 0, 15, 8)
    time_sec = st.sidebar.slider("Seconds Remaining", 0, 59, 45)
    qtr_seconds = time_min * 60 + time_sec
    half_seconds = qtr_seconds + 900 if qtr in [1, 3] else qtr_seconds
    game_seconds = qtr_seconds + (4 - qtr) * 900

    yardline_100 = st.sidebar.slider("Yards to End Zone", 1, 99, 75)
    score_diff = st.sidebar.slider("Score Margin (Possessing Team)", -35, 35, -3)
    field_start = f"Own {100 - yardline_100}" if yardline_100 > 50 else f"Opp {yardline_100}"

# Run Model
input_df = pd.DataFrame([{
    "yardline_100": yardline_100,
    "half_seconds_remaining": half_seconds,
    "game_seconds_remaining": game_seconds,
    "score_differential": score_diff,
    "qtr": qtr,
    "spread_line": 0.0,
    "total_line": 45.5
}])[feature_cols]

probs = model.predict_proba(input_df)[0]
results = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)

st.subheader("Model Projections & Fair Zero-Vig Odds")
for outcome, p in results:
    clean_name = outcome.replace("_", " ")
    col1, col2, col3 = st.columns([3, 5, 3])
    with col1:
        st.write(f"**{clean_name}**")
    with col2:
        st.progress(float(p))
    with col3:
        st.write(f"**{p * 100:.1f}%** | Fair: `{prob_to_american(p)}`")

st.divider()

# EV Line Checker
st.subheader("💡 Sportsbook Line Checker")
c_bet, c_odds = st.columns(2)
bet_choice = c_bet.selectbox("Offered Market", [r[0].replace("_", " ") for r in results])
offered_line = c_odds.number_input("Book Odds (American, e.g. +350)", value=350, step=10)

selected_prob = dict([(r[0].replace("_", " "), r[1]) for r in results])[bet_choice]
dec_payout = (offered_line / 100.0) + 1.0 if offered_line > 0 else (100.0 / abs(offered_line)) + 1.0
ev = (selected_prob * dec_payout) - 1.0

if ev > 0:
    st.success(f"**+EV Found:** Edge of **+{ev * 100:.1f}%** over book vig.")
else:
    st.error(f"**Negative Value:** {ev * 100:.1f}% house margin (Pass).")

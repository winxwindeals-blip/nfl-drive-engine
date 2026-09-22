import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import math

st.set_page_config(
    page_title="NextDrive | Live Possession & Micro-Prop Analytics",
    page_icon="⚡",
    layout="wide"
)

def prob_to_american(p: float) -> str:
    if p <= 0.001: return "+9999"
    if p >= 0.999: return "-9999"
    if p >= 0.5:
        return f"-{round((p / (1.0 - p)) * 100)}"
    return f"+{round(((1.0 - p) / p) * 100)}"

@st.cache_resource
def load_engine():
    data = joblib.load("drive_outcome_model.pkl")
    return data["model"], data["features"], data["classes"]

model, feature_cols, classes = load_engine()

# Representative situational and personnel reference database
TEAM_DATABASE = {
    "KC": {
        "name": "Kansas City Chiefs",
        "run_pass_split": {"1st_and_10": "62% Pass / 38% Run", "red_zone": "58% Pass / 42% Run", "trailing_4q": "78% Pass / 22% Run"},
        "leaders": {
            "Passing": "Patrick Mahomes (285 yds/g, 2.1 TD/g)",
            "Rushing": "Isiah Pacheco (17.5 carries/g, 4.6 YPC)",
            "Receiving": "Travis Kelce (8.2 tgts/g, 24% target share)",
            "WR1 Target Share": "Rashee Rice (7.8 tgts/g, 22% target share)"
        },
        "target_shares": {"Travis Kelce": 0.24, "Rashee Rice": 0.22, "Xavier Worthy": 0.16, "Isiah Pacheco": 0.10},
        "rush_shares": {"Isiah Pacheco": 0.68, "Carson Steele": 0.18, "Patrick Mahomes": 0.14}
    },
    "SF": {
        "name": "San Francisco 49ers",
        "run_pass_split": {"1st_and_10": "51% Pass / 49% Run", "red_zone": "46% Pass / 54% Run", "trailing_4q": "74% Pass / 26% Run"},
        "leaders": {
            "Passing": "Brock Purdy (268 yds/g, 1.9 TD/g)",
            "Rushing": "Christian McCaffrey (18.2 carries/g, 4.8 YPC)",
            "Receiving": "Deebo Samuel (7.4 tgts/g, 23% target share)",
            "WR1 Target Share": "Brandon Aiyuk (7.1 tgts/g, 21% target share)"
        },
        "target_shares": {"Deebo Samuel": 0.23, "Brandon Aiyuk": 0.21, "George Kittle": 0.20, "Christian McCaffrey": 0.18},
        "rush_shares": {"Christian McCaffrey": 0.72, "Jordan Mason": 0.20, "Brock Purdy": 0.08}
    },
    "BAL": {
        "name": "Baltimore Ravens",
        "run_pass_split": {"1st_and_10": "42% Pass / 58% Run", "red_zone": "38% Pass / 62% Run", "trailing_4q": "69% Pass / 31% Run"},
        "leaders": {
            "Passing": "Lamar Jackson (224 yds/g, 1.7 TD/g)",
            "Rushing": "Derrick Henry (19.4 carries/g, 5.1 YPC)",
            "Receiving": "Zay Flowers (7.9 tgts/g, 26% target share)",
            "TE Target Share": "Mark Andrews (5.4 tgts/g, 18% target share)"
        },
        "target_shares": {"Zay Flowers": 0.26, "Mark Andrews": 0.18, "Rashod Bateman": 0.15, "Derrick Henry": 0.06},
        "rush_shares": {"Derrick Henry": 0.64, "Lamar Jackson": 0.28, "Justice Hill": 0.08}
    },
    "DET": {
        "name": "Detroit Lions",
        "run_pass_split": {"1st_and_10": "48% Pass / 52% Run", "red_zone": "44% Pass / 56% Run", "trailing_4q": "72% Pass / 28% Run"},
        "leaders": {
            "Passing": "Jared Goff (270 yds/g, 1.8 TD/g)",
            "Rushing": "Jahmyr Gibbs (14.2 carries/g, 5.2 YPC) / David Montgomery (13.8 carries/g, 4.4 YPC)",
            "Receiving": "Amon-Ra St. Brown (9.6 tgts/g, 29% target share)",
            "TE Target Share": "Sam LaPorta (6.2 tgts/g, 19% target share)"
        },
        "target_shares": {"Amon-Ra St. Brown": 0.29, "Sam LaPorta": 0.19, "Jameson Williams": 0.17, "Jahmyr Gibbs": 0.13},
        "rush_shares": {"Jahmyr Gibbs": 0.50, "David Montgomery": 0.46, "Jared Goff": 0.04}
    },
    "PHI": {
        "name": "Philadelphia Eagles",
        "run_pass_split": {"1st_and_10": "45% Pass / 55% Run", "red_zone": "40% Pass / 60% Run", "trailing_4q": "70% Pass / 30% Run"},
        "leaders": {
            "Passing": "Jalen Hurts (235 yds/g, 1.5 TD/g)",
            "Rushing": "Saquon Barkley (19.8 carries/g, 4.9 YPC)",
            "Receiving": "A.J. Brown (8.4 tgts/g, 28% target share)",
            "WR2 Target Share": "DeVonta Smith (7.2 tgts/g, 23% target share)"
        },
        "target_shares": {"A.J. Brown": 0.28, "DeVonta Smith": 0.23, "Dallas Goedert": 0.18, "Saquon Barkley": 0.11},
        "rush_shares": {"Saquon Barkley": 0.65, "Jalen Hurts": 0.27, "Kenneth Gainwell": 0.08}
    },
    "BUF": {
        "name": "Buffalo Bills",
        "run_pass_split": {"1st_and_10": "50% Pass / 50% Run", "red_zone": "48% Pass / 52% Run", "trailing_4q": "76% Pass / 24% Run"},
        "leaders": {
            "Passing": "Josh Allen (260 yds/g, 2.0 TD/g)",
            "Rushing": "James Cook (15.5 carries/g, 4.7 YPC)",
            "Receiving": "Khalil Shakir (6.5 tgts/g, 21% target share)",
            "TE Target Share": "Dalton Kincaid (6.1 tgts/g, 19% target share)"
        },
        "target_shares": {"Khalil Shakir": 0.21, "Dalton Kincaid": 0.19, "Keon Coleman": 0.17, "James Cook": 0.12},
        "rush_shares": {"James Cook": 0.62, "Josh Allen": 0.26, "Ray Davis": 0.12}
    }
}

@st.cache_data(ttl=15)
def fetch_live_feed():
    url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(url, headers=headers, timeout=5)
        if r.status_code == 200:
            return r.json().get("events", [])
    except Exception:
        pass
    return []

st.title("⚡ NextDrive")
st.caption("Live NFL Possession Outcome Engine & Situational Micro-Prop Analytics")

tab_drive, tab_tendencies = st.tabs(["🏈 Live Next Drive Engine", "📊 Team & Player Micro-Props"])

# ---------------------------------------------------------
# TAB 1: LIVE DRIVE OUTCOME ENGINE
# ---------------------------------------------------------
with tab_drive:
    events = fetch_live_feed()
    live_games = {}
    for ev in events:
        name = ev.get("name", "NFL Matchup")
        state = ev.get("status", {}).get("type", {}).get("state", "pre")
        clock = ev.get("status", {}).get("displayClock", "")
        period = ev.get("status", {}).get("period", 0)
        label = f"{name} (Q{period} {clock})" if state == "in" else f"{name} ({state.upper()})"
        live_games[label] = ev

    st.sidebar.title("⚡ Drive Controls")
    mode = st.sidebar.radio("Game State Source", ["Live Feed", "Manual Controls"])

    if mode == "Live Feed" and live_games:
        selected_game = st.sidebar.selectbox("Active Matchup", list(live_games.keys()))
        ev = live_games[selected_game]
        status = ev.get("status", {})
        qtr = status.get("period", 1)
        clock_str = status.get("displayClock", "15:00")
        try:
            m, s = map(int, clock_str.split(":"))
            qtr_seconds = m * 60 + s
        except Exception:
            qtr_seconds = 900

        half_seconds = qtr_seconds + 900 if qtr in [1, 3] else qtr_seconds
        game_seconds = qtr_seconds + (4 - min(qtr, 4)) * 900

        competitors = ev["competitions"][0]["competitors"]
        home = next(c for c in competitors if c["homeAway"] == "home")
        away = next(c for c in competitors if c["homeAway"] == "away")
        situation = ev["competitions"][0].get("situation", {})
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
        st.info(f"🏈 **Live Drive Detected:** {selected_game} | Poss: **{poss_team}** | Ball on **{field_start}** | Margin: **{score_diff:+d}**")
    else:
        if mode == "Live Feed" and not live_games:
            st.warning("No active NFL games currently live. Switched to manual entry.")
        st.sidebar.subheader("Possession Inputs")
        qtr = st.sidebar.radio("Quarter", [1, 2, 3, 4], index=2, horizontal=True)
        time_min = st.sidebar.slider("Minutes Remaining", 0, 15, 8)
        time_sec = st.sidebar.slider("Seconds Remaining", 0, 59, 45)
        qtr_seconds = time_min * 60 + time_sec
        half_seconds = qtr_seconds + 900 if qtr in [1, 3] else qtr_seconds
        game_seconds = qtr_seconds + (4 - qtr) * 900

        yardline_100 = st.sidebar.slider("Yards to Opponent End Zone", 1, 99, 75, help="75 = Own 25 (Standard Touchback)")
        score_diff = st.sidebar.slider("Score Margin (Offense)", -35, 35, -3)
        field_start = f"Own {100 - yardline_100}" if yardline_100 > 50 else f"Opp {yardline_100}"

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

    c1, c2, c3 = st.columns(3)
    c1.metric("Quarter & Clock", f"Q{qtr} {qtr_seconds//60:02d}:{qtr_seconds%60:02d}")
    c2.metric("Starting Field Position", field_start)
    c3.metric("Score Margin", f"{score_diff:+d} pts")

    st.subheader("Zero-Vig Drive Outcome Probabilities")
    for outcome, p in results:
        c_name, c_bar, c_odds = st.columns([3, 5, 3])
        c_name.write(f"**{outcome.replace('_', ' ')}**")
        c_bar.progress(float(p))
        c_odds.write(f"**{p * 100:.1f}%** | Fair: `{prob_to_american(p)}`")

    st.divider()
    st.subheader("💡 Commercial Break +EV & Kelly Stake Sizer")
    c_mkt, c_line, c_bank = st.columns(3)
    bet_choice = c_mkt.selectbox("Market Pick", [r[0].replace("_", " ") for r in results])
    offered_line = c_line.number_input("Sportsbook Odds (+350, -120)", value=350, step=10)
    bankroll = c_bank.number_input("Total Bankroll ($)", value=1000, step=100)

    p_sel = dict([(r[0].replace("_", " "), r[1]) for r in results])[bet_choice]
    dec_payout = (offered_line / 100.0) + 1.0 if offered_line > 0 else (100.0 / abs(offered_line)) + 1.0
    ev = (p_sel * dec_payout) - 1.0
    b = dec_payout - 1.0
    full_kelly = (p_sel * b - (1.0 - p_sel)) / b if b > 0 else 0
    quarter_kelly = max(0.0, full_kelly * 0.25)
    stake_dollars = quarter_kelly * bankroll

    if ev > 0:
        st.success(f"✅ **+EV Edge:** **+{ev * 100:.1f}%** edge over book vig. | **1/4 Kelly Stake:** **{quarter_kelly * 100:.1f}%** (${stake_dollars:.2f})")
    else:
        st.error(f"❌ **Negative Edge:** {ev * 100:.1f}% vig disadvantage. Recommended Bet: **$0.00 (PASS)**")

# ---------------------------------------------------------
# TAB 2: TEAM PROFILES & DRIVE MICRO-PROPS
# ---------------------------------------------------------
with tab_tendencies:
    st.subheader("Team Situational Tendencies & Usage Leaders")
    team_abbr = st.selectbox("Select Team Profile", list(TEAM_DATABASE.keys()), format_func=lambda x: f"{TEAM_DATABASE[x]['name']} ({x})")
    team = TEAM_DATABASE[team_abbr]

    col_splits, col_leaders = st.columns(2)
    with col_splits:
        st.markdown("#### Situational Play-Calling Splits")
        st.write(f"• **1st & 10:** {team['run_pass_split']['1st_and_10']}")
        st.write(f"• **Red Zone (Inside 20):** {team['run_pass_split']['red_zone']}")
        st.write(f"• **Trailing 4th Quarter:** {team['run_pass_split']['trailing_4q']}")

    with col_leaders:
        st.markdown("#### Primary Volume Leaders")
        for stat, leader in team["leaders"].items():
            st.write(f"• **{stat}:** {leader}")

    st.divider()
    st.subheader("🎯 Drive-Level Player Micro-Prop Probability Engine")
    st.caption("Calculates the Poisson probability of a player hitting volume benchmarks on the upcoming drive based on expected plays from field position.")

    # Approximate expected plays on drive based on starting yard line
    est_plays = round(max(3.0, 3.2 + (yardline_100 / 100.0) * 3.8), 1)
    st.info(f"Drive starting from **{field_start}** averages approximately **{est_plays} plays** from scrimmage.")

    prop_col1, prop_col2 = st.columns(2)
    
    with prop_col1:
        st.markdown("#### 🏈 1+ Reception on This Drive")
        selected_wr = st.selectbox("Select Target", list(team["target_shares"].keys()))
        t_share = team["target_shares"][selected_wr]
        
        # Expected targets on this drive = est_plays * pass_rate (~56%) * target_share
        exp_targets = est_plays * 0.56 * t_share
        # Poisson probability of at least 1 reception assuming ~68% catch rate
        exp_catches = exp_targets * 0.68
        prob_catch = 1.0 - math.exp(-exp_catches)
        
        st.metric(f"{selected_wr} Probability (1+ Rec)", f"{prob_catch * 100:.1f}%", f"Fair: {prob_to_american(prob_catch)}")
        st.caption(f"Estimated Expected Catches: {exp_catches:.2f} based on {t_share*100:.0f}% target share")

    with prop_col2:
        st.markdown("#### 🏃 10+ Rushing Yards on This Drive")
        selected_rb = st.selectbox("Select Rusher", list(team["rush_shares"].keys()))
        r_share = team["rush_shares"][selected_rb]
        
        # Expected carries on this drive = est_plays * run_rate (~44%) * rush_share
        exp_carries = est_plays * 0.44 * r_share
        # Empirical conversion rate of averaging 10+ yards given expected carries
        prob_10_rush = min(0.95, 1.0 - math.exp(-exp_carries * 0.45))
        
        st.metric(f"{selected_rb} Probability (10+ Rush Yds)", f"{prob_10_rush * 100:.1f}%", f"Fair: {prob_to_american(prob_10_rush)}")
        st.caption(f"Estimated Expected Carries: {exp_carries:.2f} based on {r_share*100:.0f}% carry share")    try:
        m, s = map(int, clock_str.split(":"))
        qtr_seconds = m * 60 + s
    except Exception:
        qtr_seconds = 900
        
    half_seconds = qtr_seconds + 900 if qtr in [1, 3] else qtr_seconds
    game_seconds = qtr_seconds + (4 - min(qtr, 4)) * 900
    
    competitors = ev["competitions"][0]["competitors"]
    home = next(c for c in competitors if c["homeAway"] == "home")
    away = next(c for c in competitors if c["homeAway"] == "away")
    
    situation = ev["competitions"][0].get("situation", {})
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
    
    st.info(f"🏈 **Upcoming Possession:** **{poss_team}** | Ball on **{field_start}** | Clock: **Q{qtr} {clock_str}** | Score Differential: **{score_diff:+d}**")

else:
    if mode == "Live Stream" and not live_games:
        st.warning("No active NFL games detected on the live feed right now. Manual mode enabled.")
        
    st.sidebar.subheader("Possession Setup")
    qtr = st.sidebar.radio("Quarter", [1, 2, 3, 4], index=2, horizontal=True)
    time_min = st.sidebar.slider("Minutes Remaining", 0, 15, 7)
    time_sec = st.sidebar.slider("Seconds Remaining", 0, 59, 30)
    qtr_seconds = time_min * 60 + time_sec
    half_seconds = qtr_seconds + 900 if qtr in [1, 3] else qtr_seconds
    game_seconds = qtr_seconds + (4 - qtr) * 900

    yardline_100 = st.sidebar.slider("Yards to Opponent End Zone", 1, 99, 75, help="75 = Own 25 (Standard Touchback)")
    score_diff = st.sidebar.slider("Score Margin (Offense)", -35, 35, 0)
    field_start = f"Own {100 - yardline_100}" if yardline_100 > 50 else f"Opp {yardline_100}"

# Run NextDrive Engine
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

# Key Metrics Row
col_clock, col_pos, col_diff = st.columns(3)
col_clock.metric("Game Clock", f"Q{qtr} {qtr_seconds//60:02d}:{qtr_seconds%60:02d}")
col_pos.metric("Drive Origin", field_start)
col_diff.metric("Score Margin", f"{score_diff:+d} pts")

st.divider()

# Probability Readout
st.subheader("Next Drive Fair Values")
for outcome, p in results:
    clean_name = outcome.replace("_", " ")
    fair_odds = prob_to_american(p)
    c_label, c_bar, c_odds = st.columns([3, 5, 3])
    with c_label:
        st.write(f"**{clean_name}**")
    with c_bar:
        st.progress(float(p))
    with c_odds:
        st.write(f"**{p * 100:.1f}%** (Fair: `{fair_odds}`)")

st.divider()

# EV Line Checker for the Commercial Break
st.subheader("Commercial Break Line Checker")
c_bet, c_odds = st.columns(2)
bet_choice = c_bet.selectbox("Select Market", [r[0].replace("_", " ") for r in results])
offered_line = c_odds.number_input("Book Odds (American, e.g. +320)", value=320, step=10)

selected_prob = dict([(r[0].replace("_", " "), r[1]) for r in results])[bet_choice]
dec_payout = (offered_line / 100.0) + 1.0 if offered_line > 0 else (100.0 / abs(offered_line)) + 1.0
ev = (selected_prob * dec_payout) - 1.0

if ev > 0:
    st.success(f"**+EV Edge Detected:** **+{ev * 100:.1f}%** over book vig.")
else:
    st.error(f"**Negative Value:** {ev * 100:.1f}% house margin (Pass).")

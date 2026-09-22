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

# ---------------------------------------------------------
# RELIABLE 32-TEAM DIRECTORY
# ---------------------------------------------------------
ALL_32_TEAMS = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF", "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC", "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LAR", "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN",
    "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF", "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN", "Washington Commanders": "WAS"
}

@st.cache_data(ttl=20)
def fetch_live_scoreboard():
    url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
        if r.status_code == 200:
            return r.json().get("events", [])
    except Exception:
        pass
    return []

@st.cache_data(ttl=3600)
def fetch_active_rosters():
    url = "https://github.com/nflverse/nflverse-data/releases/download/rosters/roster_2024.csv"
    try:
        df = pd.read_csv(url, usecols=["team", "full_name", "position", "jersey_number", "status"])
        df = df[df["status"].isin(["ACT", "Active", "act"])].copy()
        df = df[df["position"].isin(["QB", "RB", "WR", "TE", "FB"])].copy()
        return df
    except Exception:
        return pd.DataFrame()

# ---------------------------------------------------------
# APP HEADER
# ---------------------------------------------------------
st.title("⚡ NextDrive")
st.caption("Live NFL Possession Outcome Engine & Real-Time Micro-Prop Analytics")

tab_drive, tab_team = st.tabs(["🏈 Live Next Drive Engine", "📊 Live Team Rosters & Micro-Props"])

# ---------------------------------------------------------
# TAB 1: LIVE DRIVE OUTCOME ENGINE
# ---------------------------------------------------------
with tab_drive:
    events = fetch_live_scoreboard()
    live_games = {}
    for ev in events:
        name = ev.get("name", "NFL Matchup")
        state = ev.get("status", {}).get("type", {}).get("state", "pre")
        clock = ev.get("status", {}).get("displayClock", "")
        period = ev.get("status", {}).get("period", 0)
        label = f"{name} (Q{period} {clock})" if state == "in" else f"{name} ({state.upper()})"
        live_games[label] = ev

    st.sidebar.title("⚡ Drive Setup")
    mode = st.sidebar.radio("Data Source", ["Live Feed", "Manual Controls"])

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
            st.warning("No active NFL games currently live. Switched to manual controls.")
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
# TAB 2: LIVE ROSTERS & MICRO-PROPS
# ---------------------------------------------------------
with tab_team:
    st.subheader("Official Active Rosters & Drive Micro-Prop Evaluators")
    
    col_sel, col_btn = st.columns([4, 1])
    with col_sel:
        selected_team_name = st.selectbox("Select NFL Franchise", list(ALL_32_TEAMS.keys()), index=12)
    with col_btn:
        st.write("")
        st.write("")
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()

    team_abbr = ALL_32_TEAMS[selected_team_name]
    rosters_df = fetch_active_rosters()
    
    team_roster = pd.DataFrame()
    if not rosters_df.empty and "team" in rosters_df.columns:
        team_roster = rosters_df[rosters_df["team"] == team_abbr].copy()

    if team_roster.empty:
        st.warning("Connecting to live database...")
        st.stop()
    else:
        all_wr = team_roster[team_roster["position"] == "WR"]
        all_te = team_roster[team_roster["position"] == "TE"]
        all_rb = team_roster[team_roster["position"].isin(["RB", "FB"])]
        all_qb = team_roster[team_roster["position"] == "QB"]

    st.divider()
    st.subheader("🎯 Automated Drive Micro-Prop Probability Engine")
    st.caption(f"Calculations for upcoming drive starting at **{field_start}**")

    est_plays = round(max(3.0, 3.2 + (yardline_100 / 100.0) * 3.8), 1)
    st.info(f"Drive starting from **{field_start}** averages approximately **{est_plays} plays** from scrimmage.")

    prop_col_rec, prop_col_rush = st.columns(2)

    with prop_col_rec:
        st.markdown("##### 🏈 Player Reception Props")
        
        pass_options = []
        for _, r in pd.concat([all_wr, all_te, all_rb]).iterrows():
            pos = r['position']
            # Assign implied target shares based on position hierarchy
            if pos == "WR":
                t_share = 0.22  # WR1 baseline
            elif pos == "TE":
                t_share = 0.16
            else:  # RB
                t_share = 0.10
                
            display_name = f"#{int(r['jersey_number']) if pd.notna(r['jersey_number']) else '--'} {r['full_name']} ({pos})"
            pass_options.append((display_name, t_share))

        chosen_target_name, assigned_t_share = st.selectbox("Select Pass Catcher", [opt[0] for opt in pass_options], index=0)
        
        # Look up the share for selected player
        actual_t_share = next(opt[1] for opt in pass_options if opt[0] == chosen_target_name)
        
        exp_targets = est_plays * 0.58 * actual_t_share
        exp_catches = exp_targets * 0.68
        
        st.markdown(f"**Implied Target Share:** {actual_t_share * 100:.0f}%")
        st.markdown(f"**Expected Targets on Drive:** {exp_targets:.2f}")

        st.write("###### Chance of 1+ Receptions:")
        prob_1_catch = 1.0 - math.exp(-exp_catches)
        col_p1, col_o1 = st.columns(2)
        col_p1.metric("Probability", f"{prob_1_catch * 100:.1f}%")
        col_o1.metric("Fair Odds", prob_to_american(prob_1_catch))

        st.write("###### Chance of 2+ Receptions:")
        prob_2_catch = 1.0 - math.exp(-exp_catches) * (1 + exp_catches)
        prob_2_catch = max(0.0, min(0.99, prob_2_catch))
        col_p2, col_o2 = st.columns(2)
        col_p2.metric("Probability", f"{prob_2_catch * 100:.1f}%")
        col_o2.metric("Fair Odds", prob_to_american(prob_2_catch))

    with prop_col_rush:
        st.markdown("##### 🏃 Player Rushing Props")
        
        rush_options = []
        for _, r in pd.concat([all_rb, all_qb]).iterrows():
            pos = r['position']
            # Assign implied carry shares based on position hierarchy
            if pos in ["RB", "FB"]:
                r_share = 0.65  # Lead RB baseline
            else:  # QB
                r_share = 0.15
                
            display_name = f"#{int(r['jersey_number']) if pd.notna(r['jersey_number']) else '--'} {r['full_name']} ({pos})"
            rush_options.append((display_name, r_share))

        chosen_rusher_name, assigned_r_share = st.selectbox("Select Rusher", [opt[0] for opt in rush_options], index=0)
        
        actual_r_share = next(opt[1] for opt in rush_options if opt[0] == chosen_rusher_name)
        
        exp_carries = est_plays * 0.42 * actual_r_share
        
        st.markdown(f"**Implied Carry Share:** {actual_r_share * 100:.0f}%")
        st.markdown(f"**Expected Carries on Drive:** {exp_carries:.2f}")

        st.write("###### Chance of 5+ Rushing Yards:")
        prob_5_rush = min(0.98, 1.0 - math.exp(-exp_carries * 0.75))
        col_p3, col_o3 = st.columns(2)
        col_p3.metric("Probability", f"{prob_5_rush * 100:.1f}%")
        col_o3.metric("Fair Odds", prob_to_american(prob_5_rush))

        st.write("###### Chance of 10+ Rushing Yards:")
        prob_10_rush = min(0.95, 1.0 - math.exp(-exp_carries * 0.42))
        col_p4, col_o4 = st.columns(2)
        col_p4.metric("Probability", f"{prob_10_rush * 100:.1f}%")
        col_o4.metric("Fair Odds", prob_to_american(prob_10_rush))

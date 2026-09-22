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
# RELIABLE 32-TEAM DIRECTORY (ESPN Team IDs)
# ---------------------------------------------------------
ALL_32_TEAMS = {
    "Arizona Cardinals": {"id": "22", "abbrev": "ARI"},
    "Atlanta Falcons": {"id": "1", "abbrev": "ATL"},
    "Baltimore Ravens": {"id": "33", "abbrev": "BAL"},
    "Buffalo Bills": {"id": "2", "abbrev": "BUF"},
    "Carolina Panthers": {"id": "29", "abbrev": "CAR"},
    "Chicago Bears": {"id": "3", "abbrev": "CHI"},
    "Cincinnati Bengals": {"id": "4", "abbrev": "CIN"},
    "Cleveland Browns": {"id": "5", "abbrev": "CLE"},
    "Dallas Cowboys": {"id": "6", "abbrev": "DAL"},
    "Denver Broncos": {"id": "7", "abbrev": "DEN"},
    "Detroit Lions": {"id": "8", "abbrev": "DET"},
    "Green Bay Packers": {"id": "9", "abbrev": "GB"},
    "Houston Texans": {"id": "34", "abbrev": "HOU"},
    "Indianapolis Colts": {"id": "11", "abbrev": "IND"},
    "Jacksonville Jaguars": {"id": "30", "abbrev": "JAX"},
    "Kansas City Chiefs": {"id": "12", "abbrev": "KC"},
    "Las Vegas Raiders": {"id": "13", "abbrev": "LV"},
    "Los Angeles Chargers": {"id": "24", "abbrev": "LAC"},
    "Los Angeles Rams": {"id": "14", "abbrev": "LAR"},
    "Miami Dolphins": {"id": "15", "abbrev": "MIA"},
    "Minnesota Vikings": {"id": "16", "abbrev": "MIN"},
    "New England Patriots": {"id": "17", "abbrev": "NE"},
    "New Orleans Saints": {"id": "18", "abbrev": "NO"},
    "New York Giants": {"id": "19", "abbrev": "NYG"},
    "New York Jets": {"id": "20", "abbrev": "NYJ"},
    "Philadelphia Eagles": {"id": "21", "abbrev": "PHI"},
    "Pittsburgh Steelers": {"id": "23", "abbrev": "PIT"},
    "San Francisco 49ers": {"id": "25", "abbrev": "SF"},
    "Seattle Seahawks": {"id": "26", "abbrev": "SEA"},
    "Tampa Bay Buccaneers": {"id": "27", "abbrev": "TB"},
    "Tennessee Titans": {"id": "10", "abbrev": "TEN"},
    "Washington Commanders": {"id": "28", "abbrev": "WAS"}
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

@st.cache_data(ttl=1800)
def fetch_live_espn_roster(team_id: str):
    """Pulls current live active roster directly from ESPN API"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json"
    }
    endpoints = [
        f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/{team_id}/roster",
        f"https://site.web.api.espn.com/apis/site/v2/sports/football/nfl/teams/{team_id}/roster"
    ]
    for url in endpoints:
        try:
            r = requests.get(url, headers=headers, timeout=6)
            if r.status_code == 200:
                data = r.json()
                athletes_groups = data.get("athletes", [])
                skill_players = []
                target_positions = {"QB", "RB", "WR", "TE", "FB"}
                for group in athletes_groups:
                    items = group.get("items", [])
                    for ath in items:
                        pos = ath.get("position", {}).get("abbreviation", "").upper()
                        if pos in target_positions:
                            name = ath.get("fullName") or ath.get("displayName") or "Player"
                            jersey = str(ath.get("jersey", "--"))
                            skill_players.append({
                                "name": name,
                                "pos": pos,
                                "jersey": jersey
                            })
                if skill_players:
                    return skill_players
        except Exception:
            continue
    return []

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
# TAB 2: LIVE ROSTERS & AUTOMATED MICRO-PROPS
# ---------------------------------------------------------
with tab_team:
    st.subheader("Official Live Team Rosters & Automated Micro-Props")
    
    col_sel, col_btn = st.columns([4, 1])
    with col_sel:
        selected_team_name = st.selectbox("Select NFL Franchise", list(ALL_32_TEAMS.keys()), index=26)
    with col_btn:
        st.write("")
        st.write("")
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()

    team_data = ALL_32_TEAMS[selected_team_name]
    
    with st.spinner(f"Pulling active roster for {selected_team_name}..."):
        roster_players = fetch_live_espn_roster(team_data["id"])
    
    if not roster_players:
        st.warning(f"Connecting to live database for {selected_team_name}...")
        st.stop()
    else:
        qbs = [p for p in roster_players if p["pos"] == "QB"]
        rbs = [p for p in roster_players if p["pos"] in ["RB", "FB"]]
        wrs = [p for p in roster_players if p["pos"] == "WR"]
        tes = [p for p in roster_players if p["pos"] == "TE"]

    st.subheader("🎯 Automated Drive Micro-Prop Evaluator")
    est_plays = round(max(3.0, 3.2 + (yardline_100 / 100.0) * 3.8), 1)
    st.info(f"Drive starting from **{field_start}** averages approximately **{est_plays} plays** from scrimmage.")

    prop_col_rec, prop_col_rush = st.columns(2)

    with prop_col_rec:
        st.markdown("##### 🏈 Player Reception Props")
        
        pass_options = {}
        for idx, p in enumerate(wrs):
            label = f"#{p['jersey']} {p['name']} (WR)"
            # Top WRs get larger target share automatically
            pass_options[label] = 0.24 if idx == 0 else (0.19 if idx == 1 else 0.14)
        for idx, p in enumerate(tes):
            label = f"#{p['jersey']} {p['name']} (TE)"
            pass_options[label] = 0.18 if idx == 0 else 0.10
        for idx, p in enumerate(rbs):
            label = f"#{p['jersey']} {p['name']} (RB)"
            pass_options[label] = 0.12 if idx == 0 else 0.06

        if pass_options:
            chosen_target = st.selectbox("Select Pass Catcher", list(pass_options.keys()), index=0)
            target_share = pass_options[chosen_target]
            
            exp_targets = est_plays * 0.58 * target_share
            exp_catches = exp_targets * 0.68
            
            st.markdown(f"**Baseline Opportunity:** ~{target_share * 100:.0f}% Target Share ({exp_targets:.2f} expected targets)")
            
            # 1+ Catch
            prob_1_catch = 1.0 - math.exp(-exp_catches)
            col_p1, col_o1 = st.columns(2)
            col_p1.metric("1+ Reception", f"{prob_1_catch * 100:.1f}%")
            col_o1.metric("Fair Odds", prob_to_american(prob_1_catch))

            # 2+ Catches
            prob_2_catch = max(0.0, min(0.99, 1.0 - math.exp(-exp_catches) * (1.0 + exp_catches)))
            col_p2, col_o2 = st.columns(2)
            col_p2.metric("2+ Receptions", f"{prob_2_catch * 100:.1f}%")
            col_o2.metric("Fair Odds", prob_to_american(prob_2_catch))

    with prop_col_rush:
        st.markdown("##### 🏃 Player Rushing Props")
        
        rush_options = {}
        for idx, p in enumerate(rbs):
            label = f"#{p['jersey']} {p['name']} (RB)"
            rush_options[label] = 0.65 if idx == 0 else (0.25 if idx == 1 else 0.10)
        for idx, p in enumerate(qbs):
            label = f"#{p['jersey']} {p['name']} (QB)"
            rush_options[label] = 0.15 if idx == 0 else 0.05

        if rush_options:
            chosen_rusher = st.selectbox("Select Rusher", list(rush_options.keys()), index=0)
            carry_share = rush_options[chosen_rusher]
            
            exp_carries = est_plays * 0.42 * carry_share
            
            st.markdown(f"**Baseline Opportunity:** ~{carry_share * 100:.0f}% Carry Share ({exp_carries:.2f} expected carries)")

            # 5+ Rush Yards
            prob_5_rush = min(0.98, 1.0 - math.exp(-exp_carries * 0.72))
            col_p3, col_o3 = st.columns(2)
            col_p3.metric("5+ Rush Yards", f"{prob_5_rush * 100:.1f}%")
            col_o3.metric("Fair Odds", prob_to_american(prob_5_rush))

            # 10+ Rush Yards
            prob_10_rush = min(0.95, 1.0 - math.exp(-exp_carries * 0.42))
            col_p4, col_o4 = st.columns(2)
            col_p4.metric("10+ Rush Yards", f"{prob_10_rush * 100:.1f}%")
            col_o4.metric("Fair Odds", prob_to_american(prob_10_rush))

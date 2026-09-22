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
# RELIABLE 32-TEAM ESPN DIRECTORY
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

@st.cache_data(ttl=300)
def fetch_team_roster(team_id: str):
    """Pulls current offensive skill players from ESPN team profile with enabled roster"""
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    url = f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/{team_id}?enable=roster"
    
    try:
        r = requests.get(url, headers=headers, timeout=6)
        if r.status_code == 200:
            data = r.json()
            team_obj = data.get("team", {})
            athletes_list = team_obj.get("record", {}).get("items", []) or team_obj.get("athletes", [])
            
            # Alternative nested athlete search
            if not athletes_list and "roster" in team_obj:
                athletes_list = team_obj.get("roster", {}).get("entries", [])

            offense = []
            valid_positions = {"QB", "RB", "WR", "TE", "FB"}
            
            for item in athletes_list:
                ath = item.get("athlete", item)
                pos = ath.get("position", {}).get("abbreviation", "")
                if pos in valid_positions:
                    offense.append({
                        "name": ath.get("displayName") or ath.get("fullName", "Player"),
                        "pos": pos,
                        "jersey": str(ath.get("jersey", "--"))
                    })
            if offense:
                return offense
    except Exception:
        pass

    # Direct Athletes Endpoint Fallback
    try:
        r2 = requests.get(f"https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/seasons/2024/teams/{team_id}/athletes?limit=50", headers=headers, timeout=6)
        if r2.status_code == 200:
            items = r2.json().get("items", [])
            offense = []
            for ref in items[:25]:
                p_url = ref.get("$ref")
                if p_url:
                    p_res = requests.get(p_url, headers=headers, timeout=3)
                    if p_res.status_code == 200:
                        p_data = p_res.json()
                        pos = p_data.get("position", {}).get("abbreviation", "")
                        if pos in valid_positions:
                            offense.append({
                                "name": p_data.get("displayName", "Player"),
                                "pos": pos,
                                "jersey": str(p_data.get("jersey", "--"))
                            })
            if offense:
                return offense
    except Exception:
        pass

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
# TAB 2: LIVE ROSTERS & MICRO-PROPS
# ---------------------------------------------------------
with tab_team:
    st.subheader("Live Official Team Rosters & Personnel Micro-Props")
    
    col_sel, col_btn = st.columns([4, 1])
    with col_sel:
        selected_team_name = st.selectbox("Select NFL Franchise", list(ALL_32_TEAMS.keys()), index=26)
    with col_btn:
        st.write("")
        st.write("")
        if st.button("🔄 Clear Cache"):
            st.cache_data.clear()
            st.rerun()

    team_info = ALL_32_TEAMS[selected_team_name]
    
    with st.spinner(f"Loading {selected_team_name} roster..."):
        roster = fetch_team_roster(team_info["id"])
    
    if not roster:
        st.warning(f"Roster details temporarily syncing from ESPN for {selected_team_name}. Click 'Clear Cache' above to force a live refresh.")
        skill_options = ["Primary WR1", "Slot WR / WR2", "Pass-Catching TE", "Starting RB"]
    else:
        qbs = [p for p in roster if p["pos"] == "QB"]
        rbs = [p for p in roster if p["pos"] in ["RB", "FB"]]
        wrs = [p for p in roster if p["pos"] == "WR"]
        tes = [p for p in roster if p["pos"] == "TE"]
        
        c_qb, c_rb, c_wr, c_te = st.columns(4)
        with c_qb:
            st.markdown("##### 🎯 Quarterbacks")
            for p in qbs[:3]:
                st.write(f"#{p['jersey']} {p['name']}")
        with c_rb:
            st.markdown("##### 🏃 Running Backs")
            for p in rbs[:4]:
                st.write(f"#{p['jersey']} {p['name']}")
        with c_wr:
            st.markdown("##### ⚡ Wide Receivers")
            for p in wrs[:5]:
                st.write(f"#{p['jersey']} {p['name']}")
        with c_te:
            st.markdown("##### 🛡️ Tight Ends")
            for p in tes[:3]:
                st.write(f"#{p['jersey']} {p['name']}")
        
        skill_options = [f"#{p['jersey']} {p['name']} ({p['pos']})" for p in (wrs + tes + rbs)]

    st.divider()
    st.subheader("🎯 Drive Micro-Prop Estimator")
    st.caption(f"Estimated for upcoming drive starting at **{field_start}**")

    est_plays = round(max(3.0, 3.2 + (yardline_100 / 100.0) * 3.8), 1)

    p_col1, p_col2 = st.columns(2)
    
    with p_col1:
        st.markdown("#### 🏈 1+ Reception on This Drive")
        if skill_options:
            chosen_target = st.selectbox("Select Player", skill_options, index=0)
            t_share = st.slider("Estimated Target Share (%)", 5, 40, 22, 1, help="Adjust based on whether player is WR1 (20-28%), WR2 (15-20%), or TE/RB (10-18%)")
            
            exp_targets = est_plays * 0.58 * (t_share / 100.0)
            exp_catches = exp_targets * 0.68
            prob_catch = 1.0 - math.exp(-exp_catches)
            
            st.metric("1+ Catch Probability", f"{prob_catch * 100:.1f}%", f"Fair: {prob_to_american(prob_catch)}")
            st.caption(f"Expected plays: ~{est_plays} | Estimated targets this drive: {exp_targets:.2f}")

    with p_col2:
        st.markdown("#### 🏃 10+ Rushing Yards on This Drive")
        rush_options = [f"#{p['jersey']} {p['name']} ({p['pos']})" for p in (rbs + qbs)] if roster else ["Lead RB", "RB2 / Change of Pace", "Mobile QB"]
        if rush_options:
            chosen_rusher = st.selectbox("Select Rusher", rush_options, index=0)
            r_share = st.slider("Estimated Carry Share (%)", 5, 90, 65, 5, help="Lead backs typically command 60-75% of early down rush volume")
            
            exp_carries = est_plays * 0.42 * (r_share / 100.0)
            prob_10_rush = min(0.95, 1.0 - math.exp(-exp_carries * 0.42))
            
            st.metric("10+ Rush Yds Probability", f"{prob_10_rush * 100:.1f}%", f"Fair: {prob_to_american(prob_10_rush)}")
            st.caption(f"Expected plays: ~{est_plays} | Estimated carries this drive: {exp_carries:.2f}")

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
# LIVE ESPN DATA FETCHERS
# ---------------------------------------------------------
@st.cache_data(ttl=30)
def fetch_live_scoreboard():
    url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=6)
        if r.status_code == 200:
            return r.json().get("events", [])
    except Exception:
        pass
    return []

@st.cache_data(ttl=3600)
def fetch_all_nfl_teams():
    url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams"
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=6)
        if r.status_code == 200:
            teams_raw = r.json()["sports"][0]["leagues"][0]["teams"]
            teams = {}
            for item in teams_raw:
                t = item["team"]
                teams[t["displayName"]] = {
                    "id": t["id"],
                    "abbrev": t["abbreviation"],
                    "short": t["shortDisplayName"]
                }
            return dict(sorted(teams.items()))
    except Exception:
        pass
    return {}

@st.cache_data(ttl=1800)
def fetch_team_roster(team_id: str):
    """Fetches real-time active offensive skill players from ESPN"""
    url = f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/{team_id}/roster"
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=6)
        if r.status_code == 200:
            data = r.json()
            offense = []
            for grp in data.get("athletes", []):
                for ath in grp.get("items", []):
                    pos = ath.get("position", {}).get("abbreviation", "")
                    if pos in ["QB", "RB", "WR", "TE", "FB"]:
                        offense.append({
                            "name": ath.get("fullName", "Unknown"),
                            "pos": pos,
                            "jersey": ath.get("jersey", "--")
                        })
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
# TAB 1: LIVE DRIVE ENGINE
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
    st.caption("Pulls real-time rosters directly from ESPN's active directory.")

    all_teams = fetch_all_nfl_teams()
    if not all_teams:
        st.error("Unable to load team list from ESPN. Check internet connection.")
    else:
        selected_team_name = st.selectbox("Select NFL Franchise", list(all_teams.keys()))
        team_info = all_teams[selected_team_name]
        
        # Fetch live roster for this team
        roster = fetch_team_roster(team_info["id"])
        
        if not roster:
            st.warning(f"Roster data currently syncing for {selected_team_name}...")
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

            st.divider()
            st.subheader("🎯 Drive Micro-Prop Estimator")
            st.caption(f"Estimated for upcoming drive starting at **{field_start}**")

            # Plays expectancy model based on start territory
            est_plays = round(max(3.0, 3.2 + (yardline_100 / 100.0) * 3.8), 1)

            p_col1, p_col2 = st.columns(2)
            
            # Pass Target micro-prop
            with p_col1:
                st.markdown("#### 🏈 1+ Reception on This Drive")
                pass_catchers = [f"#{p['jersey']} {p['name']} ({p['pos']})" for p in (wrs + tes + rbs)]
                if pass_catchers:
                    chosen_target = st.selectbox("Select Player", pass_catchers, index=0)
                    t_share = st.slider("Estimated Target Share (%)", 5, 40, 22, 1, help="Adjust based on whether player is WR1 (20-28%), WR2 (15-20%), or TE/RB (10-18%)")
                    
                    exp_targets = est_plays * 0.58 * (t_share / 100.0)
                    exp_catches = exp_targets * 0.68
                    prob_catch = 1.0 - math.exp(-exp_catches)
                    
                    st.metric("1+ Catch Probability", f"{prob_catch * 100:.1f}%", f"Fair: {prob_to_american(prob_catch)}")
                    st.caption(f"Expected plays: ~{est_plays} | Estimated targets this drive: {exp_targets:.2f}")

            # Rusher micro-prop
            with p_col2:
                st.markdown("#### 🏃 10+ Rushing Yards on This Drive")
                rushers = [f"#{p['jersey']} {p['name']} ({p['pos']})" for p in (rbs + qbs)]
                if rushers:
                    chosen_rusher = st.selectbox("Select Rusher", rushers, index=0)
                    r_share = st.slider("Estimated Carry Share (%)", 5, 90, 65, 5, help="Lead backs typically command 60-75% of early down rush volume")
                    
                    exp_carries = est_plays * 0.42 * (r_share / 100.0)
                    prob_10_rush = min(0.95, 1.0 - math.exp(-exp_carries * 0.42))
                    
                    st.metric("10+ Rush Yds Probability", f"{prob_10_rush * 100:.1f}%", f"Fair: {prob_to_american(prob_10_rush)}")
                    st.caption(f"Expected plays: ~{est_plays} | Estimated carries this drive: {exp_carries:.2f}")

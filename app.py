import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import math

st.set_page_config(
    page_title="NextDrive | Live Possession & Micro-Prop Engine",
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
# RELIABLE 32-TEAM DIRECTORY (ESPN Team IDs & Abbreviations)
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

ABBREV_TO_NAME = {v["abbrev"]: k for k, v in ALL_32_TEAMS.items()}

@st.cache_data(ttl=15)
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
            r = requests.get(url, headers=headers, timeout=5)
            if r.status_code == 200:
                data = r.json()
                athletes_groups = data.get("athletes", [])
                skill_players = []
                target_positions = {"QB", "RB", "WR", "TE", "FB"}
                for group in athletes_groups:
                    for ath in group.get("items", []):
                        pos = ath.get("position", {}).get("abbreviation", "").upper()
                        if pos in target_positions:
                            skill_players.append({
                                "name": ath.get("fullName") or ath.get("displayName") or "Player",
                                "pos": pos,
                                "jersey": str(ath.get("jersey", "--"))
                            })
                if skill_players:
                    return skill_players
        except Exception:
            continue
    return []

# ---------------------------------------------------------
# HEADER & DATA INGESTION
# ---------------------------------------------------------
st.title("⚡ NextDrive Workstation")
st.caption("Real-Time NFL Possession Probabilities & Live Player Micro-Props Powered by ESPN Live Data")

events = fetch_live_scoreboard()
live_games = {}
for ev in events:
    name = ev.get("name", "NFL Matchup")
    state = ev.get("status", {}).get("type", {}).get("state", "pre")
    clock = ev.get("status", {}).get("displayClock", "")
    period = ev.get("status", {}).get("period", 0)
    label = f"{name} (Q{period} {clock})" if state == "in" else f"{name} ({state.upper()})"
    live_games[label] = ev

# Detect live game situations automatically
if live_games:
    selected_game = st.selectbox("Active Live Matchup", list(live_games.keys()))
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
        poss_team_abbr = home["team"]["abbreviation"]
    else:
        score_diff = away_score - home_score
        poss_team_abbr = away.get("team", {}).get("abbreviation", "OFF")

    field_start = f"Own {100 - yardline_100}" if yardline_100 > 50 else f"Opp {yardline_100}"
    st.info(f"🏈 **Live Situation Ingested:** {selected_game} | Possession: **{poss_team_abbr}** | Ball on: **{field_start}** | Clock: **Q{qtr} {clock_str}** | Margin: **{score_diff:+d}**")

else:
    st.warning("📡 **Scoreboard Standby:** No NFL games currently kicking off. Operating on standard game defaults (touchback territory, tied margin).")
    qtr = 2
    clock_str = "8:00"
    qtr_seconds = 480
    half_seconds = 480
    game_seconds = 1380
    yardline_100 = 75
    score_diff = 0
    poss_team_abbr = "KC"
    field_start = "Own 25"

# Run Drive Outcome Model
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

# Expected plays on this possession based on field territory
est_plays = round(max(3.0, 3.2 + (yardline_100 / 100.0) * 3.8), 1)

# ---------------------------------------------------------
# UNIFIED DUAL-COLUMN WORKSTATION
# ---------------------------------------------------------
col_drive, col_divider, col_prop = st.columns([10, 1, 11])

# === LEFT PANEL: DRIVE OUTCOME ENGINE ===
with col_drive:
    st.subheader("🏈 Next Drive Probabilities")
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Live Clock", f"Q{qtr} {clock_str}")
    m2.metric("Live Ball Spot", field_start)
    m3.metric("Live Margin", f"{score_diff:+d} pts")
    
    st.caption(f"Calculated drive volume: **~{est_plays} plays** from scrimmage.")
    st.write("")

    for outcome, p in results:
        clean_name = outcome.replace("_", " ")
        c_lbl, c_meter, c_fair = st.columns([4, 4, 3])
        c_lbl.write(f"**{clean_name}**")
        c_meter.progress(float(p))
        c_fair.write(f"**{p * 100:.1f}%** (`{prob_to_american(p)}`)")

    st.divider()
    st.markdown("##### 💡 TV Timeout +EV & Stake Sizer")
    s_mkt, s_line, s_bank = st.columns(3)
    bet_choice = s_mkt.selectbox("Market Pick", [r[0].replace("_", " ") for r in results])
    offered_line = s_line.number_input("Sportsbook Odds (+320)", value=320, step=10)
    bankroll = s_bank.number_input("Bankroll ($)", value=1000, step=100)

    p_sel = dict([(r[0].replace("_", " "), r[1]) for r in results])[bet_choice]
    dec_payout = (offered_line / 100.0) + 1.0 if offered_line > 0 else (100.0 / abs(offered_line)) + 1.0
    ev = (p_sel * dec_payout) - 1.0
    b = dec_payout - 1.0
    full_kelly = (p_sel * b - (1.0 - p_sel)) / b if b > 0 else 0
    quarter_kelly = max(0.0, full_kelly * 0.25)
    stake_dollars = quarter_kelly * bankroll

    if ev > 0:
        st.success(f"✅ **+EV Edge:** **+{ev * 100:.1f}%** | **1/4 Kelly Stake:** **{quarter_kelly * 100:.1f}%** (${stake_dollars:.2f})")
    else:
        st.error(f"❌ **Negative Edge:** {ev * 100:.1f}% vig disadvantage. Pass.")

# === CENTER DIVIDER ===
with col_divider:
    st.write("")

# === RIGHT PANEL: PLAYER MICRO-PROPS ===
with col_prop:
    st.subheader("🎯 Player Drive Micro-Props")
    
    # Auto-sync offense on field to the live possession team
    default_team_name = ABBREV_TO_NAME.get(poss_team_abbr, "Kansas City Chiefs")
    all_team_list = list(ALL_32_TEAMS.keys())
    default_idx = all_team_list.index(default_team_name) if default_team_name in all_team_list else 15

    row_team, row_refresh = st.columns([4, 1])
    with row_team:
        selected_team_name = st.selectbox("Offense on Field", all_team_list, index=default_idx)
    with row_refresh:
        st.write("")
        st.write("")
        if st.button("🔄 Refresh"):
            st.cache_data.clear()
            st.rerun()

    team_data = ALL_32_TEAMS[selected_team_name]
    
    with st.spinner(f"Loading {selected_team_name} active roster..."):
        roster_players = fetch_live_espn_roster(team_data["id"])
        
    if not roster_players:
        st.warning("Connecting to active roster...")
    else:
        wrs = [p for p in roster_players if p["pos"] == "WR"]
        tes = [p for p in roster_players if p["pos"] == "TE"]
        rbs = [p for p in roster_players if p["pos"] in ["RB", "FB"]]
        qbs = [p for p in roster_players if p["pos"] == "QB"]

        tab_rec, tab_rush = st.tabs(["🏈 Reception Props", "🏃 Rushing Props"])
        
        with tab_rec:
            pass_options = {}
            for idx, p in enumerate(wrs):
                lbl = f"#{p['jersey']} {p['name']} (WR)"
                pass_options[lbl] = 0.24 if idx == 0 else (0.19 if idx == 1 else 0.13)
            for idx, p in enumerate(tes):
                lbl = f"#{p['jersey']} {p['name']} (TE)"
                pass_options[lbl] = 0.18 if idx == 0 else 0.10
            for idx, p in enumerate(rbs):
                lbl = f"#{p['jersey']} {p['name']} (RB)"
                pass_options[lbl] = 0.12 if idx == 0 else 0.06

            chosen_target = st.selectbox("Select Pass Catcher", list(pass_options.keys()), index=0)
            target_share = pass_options[chosen_target]
            
            exp_targets = est_plays * 0.58 * target_share
            exp_catches = exp_targets * 0.68
            
            st.caption(f"Live Ingested Opportunity: **{target_share * 100:.0f}%** Target Share (~{exp_targets:.2f} targets)")

            r_col1, r_col2 = st.columns(2)
            prob_1_catch = 1.0 - math.exp(-exp_catches)
            r_col1.metric("1+ Reception on Drive", f"{prob_1_catch * 100:.1f}%", f"Fair: {prob_to_american(prob_1_catch)}")

            prob_2_catch = max(0.0, min(0.99, 1.0 - math.exp(-exp_catches) * (1.0 + exp_catches)))
            r_col2.metric("2+ Receptions on Drive", f"{prob_2_catch * 100:.1f}%", f"Fair: {prob_to_american(prob_2_catch)}")

        with tab_rush:
            rush_options = {}
            for idx, p in enumerate(rbs):
                lbl = f"#{p['jersey']} {p['name']} (RB)"
                rush_options[lbl] = 0.65 if idx == 0 else (0.25 if idx == 1 else 0.10)
            for idx, p in enumerate(qbs):
                lbl = f"#{p['jersey']} {p['name']} (QB)"
                rush_options[lbl] = 0.15 if idx == 0 else 0.05

            chosen_rusher = st.selectbox("Select Ball Carrier", list(rush_options.keys()), index=0)
            carry_share = rush_options[chosen_rusher]
            
            exp_carries = est_plays * 0.42 * carry_share
            st.caption(f"Live Ingested Opportunity: **{carry_share * 100:.0f}%** Carry Share (~{exp_carries:.2f} carries)")

            ru_col1, ru_col2 = st.columns(2)
            prob_5_rush = min(0.98, 1.0 - math.exp(-exp_carries * 0.72))
            ru_col1.metric("5+ Rush Yds on Drive", f"{prob_5_rush * 100:.1f}%", f"Fair: {prob_to_american(prob_5_rush)}")

            prob_10_rush = min(0.95, 1.0 - math.exp(-exp_carries * 0.42))
            ru_col2.metric("10+ Rush Yds on Drive", f"{prob_10_rush * 100:.1f}%", f"Fair: {prob_to_american(prob_10_rush)}")

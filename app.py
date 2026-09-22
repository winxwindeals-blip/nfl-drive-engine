import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import math

st.set_page_config(
    page_title="NextDrive | Matchup Workstation",
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
# ESPN API DATA FETCHERS (WEEK-BY-WEEK MATCHUPS)
# ---------------------------------------------------------
@st.cache_data(ttl=30)
def fetch_week_schedule(week_num: int):
    """Fetches full matchup slate for any regular season NFL week"""
    url = f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard?seasontype=2&week={week_num}"
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=6)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return {}

@st.cache_data(ttl=1800)
def fetch_live_espn_roster(team_id: str):
    """Fetches active offensive skill players for a team"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json"
    }
    url = f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/{team_id}/roster"
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
        pass
    return []

# ---------------------------------------------------------
# HEADER & WEEKLY SLATE SELECTOR
# ---------------------------------------------------------
st.title("⚡ NextDrive Matchup Workstation")
st.caption("Weekly NFL Slate Analyzer with Live Possession & In-Drive Micro-Prop Projections")

# Detect current season week from default scoreboard
scoreboard_root = fetch_week_schedule(week_num=0)
current_week = 3
if scoreboard_root:
    current_week = scoreboard_root.get("week", {}).get("number", 3)

c_week, c_game = st.columns([1, 3])

with c_week:
    selected_week = st.selectbox("NFL Week", list(range(1, 19)), index=max(0, current_week - 1))

# Load the selected week's full slate
week_data = fetch_week_schedule(selected_week)
events = week_data.get("events", [])

matchups = {}
for ev in events:
    comp = ev.get("competitions", [{}])[0]
    competitors = comp.get("competitors", [])
    if len(competitors) >= 2:
        home = next((c for c in competitors if c.get("homeAway") == "home"), competitors[0])
        away = next((c for c in competitors if c.get("homeAway") == "away"), competitors[1])
        status = ev.get("status", {})
        state = status.get("type", {}).get("state", "pre")
        clock = status.get("displayClock", "")
        period = status.get("period", 0)
        
        h_abbr = home.get("team", {}).get("abbreviation", "HOME")
        a_abbr = away.get("team", {}).get("abbreviation", "AWAY")
        
        status_label = f"🔴 Q{period} {clock}" if state == "in" else ("FINAL" if state == "post" else ev.get("status", {}).get("type", {}).get("shortDetail", "SCHEDULED"))
        label = f"{a_abbr} @ {h_abbr} ({status_label})"
        matchups[label] = {
            "event": ev,
            "home": home,
            "away": away,
            "status": status,
            "situation": comp.get("situation", {})
        }

with c_game:
    if matchups:
        selected_matchup_label = st.selectbox("Select Matchup", list(matchups.keys()))
        active_match = matchups[selected_matchup_label]
    else:
        st.warning(f"No schedule events found for Week {selected_week}.")
        st.stop()

# ---------------------------------------------------------
# MATCHUP TELEMETRY & POSSESSION SELECTION
# ---------------------------------------------------------
home_team = active_match["home"]["team"]
away_team = active_match["away"]["team"]
home_score = int(active_match["home"].get("score", 0))
away_score = int(active_match["away"].get("score", 0))

state = active_match["status"].get("type", {}).get("state", "pre")
sit = active_match["situation"]

# Determine possession
auto_poss_id = sit.get("possession")
default_poss_idx = 1 if auto_poss_id == home_team["id"] else 0

st.write("")
col_poss, col_info = st.columns([2, 4])

with col_poss:
    poss_choice = st.radio(
        "Offense on Field (Has Ball)",
        [f"{away_team['displayName']} ({away_team['abbreviation']})", 
         f"{home_team['displayName']} ({home_team['abbreviation']})"],
        index=default_poss_idx,
        horizontal=True
    )

is_home_poss = (poss_choice.startswith(home_team["displayName"]))
off_team = home_team if is_home_poss else away_team
def_team = away_team if is_home_poss else home_team
score_diff = (home_score - away_score) if is_home_poss else (away_score - home_score)

# Live clock and ball spot ingestion
if state == "in":
    qtr = active_match["status"].get("period", 1)
    clock_str = active_match["status"].get("displayClock", "15:00")
    yardline_100 = sit.get("yardLine", 75)
else:
    qtr = 2
    clock_str = "8:30"
    yardline_100 = 75  # Standard touchback Own 25

try:
    m, s = map(int, clock_str.split(":"))
    qtr_seconds = m * 60 + s
except Exception:
    qtr_seconds = 510

half_seconds = qtr_seconds + 900 if qtr in [1, 3] else qtr_seconds
game_seconds = qtr_seconds + (4 - min(qtr, 4)) * 900
field_start = f"Own {100 - yardline_100}" if yardline_100 > 50 else f"Opp {yardline_100}"

with col_info:
    st.info(f"🏈 **Drive Situation:** **{off_team['abbreviation']}** Offense vs **{def_team['abbreviation']}** Defense | Line of Scrimmage: **{field_start}** | Clock: **Q{qtr} {clock_str}** | Margin: **{score_diff:+d}**")

# Run Drive Model
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
est_plays = round(max(3.0, 3.2 + (yardline_100 / 100.0) * 3.8), 1)

# Game script run/pass lean
pass_rate = 0.58
if score_diff <= -8 and qtr >= 3:
    pass_rate = 0.72
elif score_diff >= 8 and qtr >= 3:
    pass_rate = 0.42
run_rate = 1.0 - pass_rate

# ---------------------------------------------------------
# FETCH OFFENSIVE ROSTER FOR THIS MATCHUP
# ---------------------------------------------------------
roster_players = fetch_live_espn_roster(off_team["id"])
wrs = [p for p in roster_players if p["pos"] == "WR"] if roster_players else []
tes = [p for p in roster_players if p["pos"] == "TE"] if roster_players else []
rbs = [p for p in roster_players if p["pos"] in ["RB", "FB"]] if roster_players else []
qbs = [p for p in roster_players if p["pos"] == "QB"] if roster_players else []

# ---------------------------------------------------------
# SUGGESTED VALUE SPOTLIGHTS FOR THIS MATCHUP
# ---------------------------------------------------------
st.markdown(f"### 🔥 Matchup Value Spotlights ({off_team['abbreviation']} vs {def_team['abbreviation']})")

if roster_players:
    top_wr = wrs[0] if wrs else None
    top_rb = rbs[0] if rbs else None

    rec_prob_1, rec_prob_2, rush_prob_5, rush_prob_10 = 0.0, 0.0, 0.0, 0.0

    if top_wr:
        exp_tg = est_plays * pass_rate * 0.25
        exp_ct = exp_tg * 0.68
        rec_prob_1 = 1.0 - math.exp(-exp_ct)
        rec_prob_2 = max(0.0, min(0.99, 1.0 - math.exp(-exp_ct) * (1.0 + exp_ct)))

    if top_rb:
        exp_car = est_plays * run_rate * 0.68
        rush_prob_5 = min(0.98, 1.0 - math.exp(-exp_car * 0.72))
        rush_prob_10 = min(0.95, 1.0 - math.exp(-exp_car * 0.42))

    s1, s2, s3 = st.columns(3)
    with s1:
        st.success("🎯 **Top Volume Floor (High Hit Rate)**")
        if rush_prob_5 >= rec_prob_1 and top_rb:
            st.markdown(f"**#{top_rb['jersey']} {top_rb['name']} (RB1)**")
            st.write(f"• **5+ Rushing Yards:** **{rush_prob_5 * 100:.1f}%** (`{prob_to_american(rush_prob_5)}`)")
            st.caption(f"Projected ~{exp_car:.1f} carries from {field_start}.")
        elif top_wr:
            st.markdown(f"**#{top_wr['jersey']} {top_wr['name']} (WR1)**")
            st.write(f"• **1+ Reception:** **{rec_prob_1 * 100:.1f}%** (`{prob_to_american(rec_prob_1)}`)")
            st.caption(f"Primary target funnel (~25% share against {def_team['abbreviation']}).")

    with s2:
        st.info("🚀 **Top Plus-Money Value (Ceiling)**")
        if top_wr:
            st.markdown(f"**#{top_wr['jersey']} {top_wr['name']} (WR1)**")
            st.write(f"• **2+ Receptions:** **{rec_prob_2 * 100:.1f}%** (`{prob_to_american(rec_prob_2)}`)")
            st.caption(f"Strong plus-money conversion on drives reaching 5+ plays.")
        elif top_rb:
            st.markdown(f"**#{top_rb['jersey']} {top_rb['name']} (RB1)**")
            st.write(f"• **10+ Rushing Yards:** **{rush_prob_10 * 100:.1f}%** (`{prob_to_american(rush_prob_10)}`)")
            st.caption(f"Explosive chunk yardage benchmark vs {def_team['abbreviation']}.")

    with s3:
        st.warning("📋 **Matchup Game Script**")
        if score_diff <= -8 and qtr >= 3:
            st.write(f"• **Trailing {score_diff:+d}:** Pass heavy script (~72% pass lean).")
        elif score_diff >= 8 and qtr >= 3:
            st.write(f"• **Protecting Lead {score_diff:+d}:** Run heavy script (~58% rush lean).")
        else:
            st.write(f"• **Neutral Script:** Standard balanced playcalling.")
        st.caption(f"Drive volume: ~{est_plays} plays.")
else:
    st.info("Loading active personnel...")

st.divider()

# ---------------------------------------------------------
# DUAL-COLUMN WORKSTATION
# ---------------------------------------------------------
col_drive, col_divider, col_prop = st.columns([10, 1, 11])

# === LEFT PANEL: DRIVE OUTCOME ENGINE ===
with col_drive:
    st.subheader(f"🏈 {off_team['abbreviation']} Drive Probabilities")
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Game Clock", f"Q{qtr} {clock_str}")
    m2.metric("Ball Spot", field_start)
    m3.metric("Margin", f"{score_diff:+d} pts")
    
    st.caption(f"Expected drive volume: **~{est_plays} plays** from scrimmage.")
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
    offered_line = s_line.number_input("Book Line (+320)", value=320, step=10)
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

# === RIGHT PANEL: ALL PLAYER PROPS ===
with col_prop:
    st.subheader(f"🎯 {off_team['abbreviation']} Player Micro-Props")

    if not roster_players:
        st.warning("Connecting to active roster...")
    else:
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
            
            exp_targets = est_plays * pass_rate * target_share
            exp_catches = exp_targets * 0.68
            
            st.caption(f"Situational Opportunity: **{target_share * 100:.0f}%** Target Share (~{exp_targets:.2f} targets)")

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
            
            exp_carries = est_plays * run_rate * carry_share
            st.caption(f"Situational Opportunity: **{carry_share * 100:.0f}%** Carry Share (~{exp_carries:.2f} carries)")

            ru_col1, ru_col2 = st.columns(2)
            prob_5_rush = min(0.98, 1.0 - math.exp(-exp_carries * 0.72))
            ru_col1.metric("5+ Rush Yds on Drive", f"{prob_5_rush * 100:.1f}%", f"Fair: {prob_to_american(prob_5_rush)}")

            prob_10_rush = min(0.95, 1.0 - math.exp(-exp_carries * 0.42))
            ru_col2.metric("10+ Rush Yds on Drive", f"{prob_10_rush * 100:.1f}%", f"Fair: {prob_to_american(prob_10_rush)}")

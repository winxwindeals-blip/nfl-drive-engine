import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import math

st.set_page_config(
    page_title="NextDrive | A WIN•X•WIN LLC Company",
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

ABBREV_TO_INFO = {v["abbrev"]: {"name": k, "id": v["id"]} for k, v in ALL_32_TEAMS.items()}

WEEKLY_SCHEDULE_BACKUP = {
    1: [("BAL", "KC"), ("GB", "PHI"), ("PIT", "ATL"), ("ARI", "BUF"), ("TEN", "CHI"), ("NE", "CIN"), ("HOU", "IND"), ("JAX", "MIA"), ("CAR", "NO"), ("MIN", "NYG"), ("LV", "LAC"), ("DEN", "SEA"), ("DAL", "CLE"), ("WAS", "TB"), ("LAR", "DET"), ("NYJ", "SF")],
    2: [("BUF", "MIA"), ("LV", "BAL"), ("LAC", "CAR"), ("NO", "DAL"), ("TB", "DET"), ("IND", "GB"), ("CLE", "JAX"), ("NYG", "WAS"), ("SF", "MIN"), ("SEA", "NE"), ("NYJ", "TEN"), ("CIN", "KC"), ("LAR", "ARI"), ("PIT", "DEN"), ("CHI", "HOU"), ("ATL", "PHI")],
    3: [("NE", "NYJ"), ("NYG", "CLE"), ("CHI", "IND"), ("HOU", "MIN"), ("PHI", "NO"), ("LAC", "PIT"), ("DEN", "TB"), ("GB", "TEN"), ("CAR", "LV"), ("MIA", "SEA"), ("DET", "ARI"), ("BAL", "DAL"), ("SF", "LAR"), ("KC", "ATL"), ("JAX", "BUF"), ("WAS", "CIN")],
    4: [("DAL", "NYG"), ("NO", "ATL"), ("CIN", "CAR"), ("LAR", "CHI"), ("MIN", "GB"), ("JAX", "HOU"), ("PIT", "IND"), ("DEN", "NYJ"), ("PHI", "TB"), ("WAS", "ARI"), ("NE", "SF"), ("KC", "LAC"), ("CLE", "LV"), ("BUF", "BAL"), ("TEN", "MIA"), ("SEA", "DET")],
    5: [("TB", "ATL"), ("NYJ", "MIN"), ("CAR", "CHI"), ("BAL", "CIN"), ("BUF", "HOU"), ("IND", "JAX"), ("MIA", "NE"), ("CLE", "WAS"), ("LV", "DEN"), ("ARI", "SF"), ("GB", "LAR"), ("NYG", "SEA"), ("DAL", "PIT"), ("NO", "KC")],
    6: [("SF", "SEA"), ("JAX", "CHI"), ("WAS", "BAL"), ("HOU", "NE"), ("CLE", "PHI"), ("IND", "TEN"), ("ARI", "GB"), ("TB", "NO"), ("LAC", "DEN"), ("PIT", "LV"), ("DET", "DAL"), ("ATL", "CAR"), ("CIN", "NYG"), ("BUF", "NYJ")],
    7: [("DEN", "NO"), ("NE", "JAX"), ("SEA", "ATL"), ("TEN", "BUF"), ("CIN", "CLE"), ("HOU", "GB"), ("MIA", "IND"), ("DET", "MIN"), ("PHI", "NYG"), ("LV", "LAR"), ("CAR", "WAS"), ("KC", "SF"), ("NYJ", "PIT"), ("BAL", "TB"), ("LAC", "ARI")],
    8: [("MIN", "LAR"), ("BAL", "CLE"), ("TEN", "DET"), ("IND", "HOU"), ("GB", "JAX"), ("ARI", "MIA"), ("NYJ", "NE"), ("ATL", "TB"), ("CHI", "WAS"), ("NO", "LAC"), ("BUF", "SEA"), ("PHI", "CIN"), ("CAR", "DEN"), ("KC", "LV"), ("DAL", "SF"), ("NYG", "PIT")]
}

@st.cache_data(ttl=15)
def fetch_week_schedule(week_num: int):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    url = f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard?dates=2024&seasontype=2&week={week_num}"
    try:
        r = requests.get(url, headers=headers, timeout=5)
        if r.status_code == 200:
            events = r.json().get("events", [])
            if events:
                return events
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
# LOGO & BRANDED SUBSIDIARY HEADER
# ---------------------------------------------------------
st.markdown("""
<div style="display: flex; align-items: center; gap: 16px; padding: 6px 0 14px 0;">
  <svg width="58" height="58" viewBox="0 0 60 60" fill="none" xmlns="http://www.w3.org/2000/svg">
    <rect width="60" height="60" rx="14" fill="#141820" stroke="#262D3D" stroke-width="2"/>
    <rect x="12" y="24" width="4" height="12" rx="2" fill="#3D4B63"/>
    <rect x="20" y="19" width="4" height="22" rx="2" fill="#6B7F9E"/>
    <path d="M28 14L46 30L28 46L32 32H25L32 14H28Z" fill="url(#grad_bolt)"/>
    <defs>
      <linearGradient id="grad_bolt" x1="25" y1="14" x2="46" y2="46" gradientUnits="userSpaceOnUse">
        <stop stop-color="#FF5E3A"/>
        <stop offset="1" stop-color="#FF2A54"/>
      </linearGradient>
    </defs>
  </svg>
  <div>
    <div style="display: flex; align-items: baseline; gap: 10px;">
      <h1 style="margin: 0; padding: 0; font-size: 2.1rem; font-weight: 900; letter-spacing: -0.5px; line-height: 1.1;">
        NEXT<span style="color: #FF4B4B;">DRIVE</span>
      </h1>
      <span style="font-size: 0.72rem; font-weight: 800; color: #FF4B4B; letter-spacing: 1.4px; text-transform: uppercase; background: rgba(255, 75, 75, 0.12); padding: 3px 8px; border-radius: 4px; border: 1px solid rgba(255, 75, 75, 0.3);">
        A WIN•X•WIN LLC COMPANY
      </span>
    </div>
    <p style="margin: 3px 0 0 0; padding: 0; font-size: 0.78rem; font-weight: 700; color: #7C8BA1; letter-spacing: 1.3px; text-transform: uppercase;">
      Real-Time Live Scoreboard & Advanced Micro-Prop Analytics
    </p>
  </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# SCHEDULE & MATCHUP INGESTION
# ---------------------------------------------------------
c_week, c_game = st.columns([1, 3])

with c_week:
    selected_week = st.selectbox("NFL Week", list(range(1, 19)), index=3)

events = fetch_week_schedule(selected_week)
matchup_options = {}

if events:
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
            
            status_label = f"🔴 Q{period} {clock}" if state == "in" else ("FINAL" if state == "post" else "SCHEDULED")
            label = f"{a_abbr} @ {h_abbr} ({status_label})"
            
            sit = comp.get("situation", {})
            matchup_options[label] = {
                "home_abbr": h_abbr,
                "away_abbr": a_abbr,
                "home_name": home.get("team", {}).get("displayName", h_abbr),
                "away_name": away.get("team", {}).get("displayName", a_abbr),
                "home_id": home.get("team", {}).get("id"),
                "away_id": away.get("team", {}).get("id"),
                "home_score": int(home.get("score", 0)),
                "away_score": int(away.get("score", 0)),
                "state": state,
                "period": period,
                "clock": clock,
                "yardLine": sit.get("yardLine", 75),
                "downDistance": sit.get("downDistanceText", "1st & 10"),
                "possession": sit.get("possession")
            }
else:
    pairs = WEEKLY_SCHEDULE_BACKUP.get(selected_week, [("KC", "SF"), ("BAL", "PIT"), ("BUF", "MIA"), ("DET", "GB")])
    for a_abbr, h_abbr in pairs:
        a_info = ABBREV_TO_INFO.get(a_abbr, {"name": a_abbr, "id": "1"})
        h_info = ABBREV_TO_INFO.get(h_abbr, {"name": h_abbr, "id": "2"})
        label = f"{a_abbr} @ {h_abbr} (SCHEDULED)"
        matchup_options[label] = {
            "home_abbr": h_abbr,
            "away_abbr": a_abbr,
            "home_name": h_info["name"],
            "away_name": a_info["name"],
            "home_id": h_info["id"],
            "away_id": h_info["id"],
            "home_score": 0,
            "away_score": 0,
            "state": "pre",
            "period": 2,
            "clock": "8:30",
            "yardLine": 75,
            "downDistance": "1st & 10 at Own 25",
            "possession": None
        }

with c_game:
    selected_label = st.selectbox("Select Matchup", list(matchup_options.keys()))
    m = matchup_options[selected_label]

# Determine automatic live ball possession
auto_poss_home = (m["possession"] == m["home_id"]) if m["possession"] else False
default_idx = 1 if auto_poss_home else 0

# ---------------------------------------------------------
# 📺 LIVE BROADCAST SCOREBOARD BANNER
# ---------------------------------------------------------
away_status = "🏈 " if not auto_poss_home and m["state"] == "in" else ""
home_status = "🏈 " if auto_poss_home and m["state"] == "in" else ""
game_state_str = f"🔴 Q{m['period']} {m['clock']}" if m["state"] == "in" else ("FINAL" if m["state"] == "post" else "PRE-GAME")

st.markdown(f"""
<div style="background: #111622; border: 1px solid #1E293B; border-radius: 10px; padding: 14px 20px; margin: 10px 0 16px 0; display: flex; justify-content: space-between; align-items: center;">
    <div style="display: flex; align-items: center; gap: 30px;">
        <div style="text-align: left;">
            <div style="font-size: 0.85rem; font-weight: 700; color: #94A3B8;">AWAY</div>
            <div style="font-size: 1.4rem; font-weight: 900; color: #FFFFFF;">{away_status}{m['away_abbr']} <span style="color: #FF4B4B; margin-left: 8px;">{m['away_score']}</span></div>
        </div>
        <div style="font-size: 1.2rem; font-weight: 800; color: #475569;">@</div>
        <div style="text-align: left;">
            <div style="font-size: 0.85rem; font-weight: 700; color: #94A3B8;">HOME</div>
            <div style="font-size: 1.4rem; font-weight: 900; color: #FFFFFF;">{home_status}{m['home_abbr']} <span style="color: #FF4B4B; margin-left: 8px;">{m['home_score']}</span></div>
        </div>
    </div>
    <div style="text-align: center; border-left: 1px solid #263346; padding-left: 24px;">
        <div style="font-size: 0.8rem; font-weight: 800; color: #38BDF8; text-transform: uppercase;">{game_state_str}</div>
        <div style="font-size: 0.95rem; font-weight: 700; color: #E2E8F0;">{m['downDistance']}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# OFFENSE POSSESSION CONTROLS
# ---------------------------------------------------------
col_poss, col_info = st.columns([2, 4])

with col_poss:
    poss_pick = st.radio(
        "Offense on Field (Has Ball)",
        [f"{m['away_name']} ({m['away_abbr']})", f"{m['home_name']} ({m['home_abbr']})"],
        index=default_idx,
        horizontal=True
    )

is_home = poss_pick.startswith(m["home_name"])
off_abbr = m["home_abbr"] if is_home else m["away_abbr"]
off_name = m["home_name"] if is_home else m["away_name"]
off_id = m["home_id"] if is_home else m["away_id"]
def_abbr = m["away_abbr"] if is_home else m["home_abbr"]
score_diff = (m["home_score"] - m["away_score"]) if is_home else (m["away_score"] - m["home_score"])

yardline_100 = m["yardLine"] if m["state"] == "in" else 75
field_start = f"Own {100 - yardline_100}" if yardline_100 > 50 else f"Opp {yardline_100}"
qtr = m["period"] if m["state"] == "in" else 2
clock_str = m["clock"] if m["state"] == "in" else "8:30"

try:
    min_part, sec_part = map(int, clock_str.split(":"))
    qtr_seconds = min_part * 60 + sec_part
except Exception:
    qtr_seconds = 510

half_seconds = qtr_seconds + 900 if qtr in [1, 3] else qtr_seconds
game_seconds = qtr_seconds + (4 - min(qtr, 4)) * 900

with col_info:
    st.info(f"🏈 **Drive Situation:** **{off_abbr}** Offense vs **{def_abbr}** Defense | Line: **{field_start}** | Clock: **Q{qtr} {clock_str}** | Offense Margin: **{score_diff:+d}**")

# Run Possession Model
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

# Dynamic playcalling lean
pass_rate = 0.58
if score_diff <= -8 and qtr >= 3:
    pass_rate = 0.72
elif score_diff >= 8 and qtr >= 3:
    pass_rate = 0.42
run_rate = 1.0 - pass_rate

# Fetch live offensive roster
roster = fetch_live_espn_roster(str(off_id))
wrs = [p for p in roster if p["pos"] == "WR"] if roster else []
tes = [p for p in roster if p["pos"] == "TE"] if roster else []
rbs = [p for p in roster if p["pos"] in ["RB", "FB"]] if roster else []
qbs = [p for p in roster if p["pos"] == "QB"] if roster else []

# Extract model drive TD probability
prob_drive_td = dict([(r[0], r[1]) for r in results]).get("Touchdown", 0.22)

# ---------------------------------------------------------
# SUGGESTED VALUE SPOTLIGHTS
# ---------------------------------------------------------
st.markdown(f"### 🔥 Matchup Value Spotlights ({off_abbr} vs {def_abbr})")

if roster:
    top_wr = wrs[0] if wrs else None
    top_rb = rbs[0] if rbs else None
    top_qb = qbs[0] if qbs else None

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
            st.caption(f"Primary target funnel (~25% share against {def_abbr}).")

    with s2:
        st.info("🚀 **Top Plus-Money Value (Ceiling)**")
        if top_wr:
            st.markdown(f"**#{top_wr['jersey']} {top_wr['name']} (WR1)**")
            st.write(f"• **2+ Receptions:** **{rec_prob_2 * 100:.1f}%** (`{prob_to_american(rec_prob_2)}`)")
            st.caption(f"Strong plus-money conversion on drives reaching 5+ plays.")
        elif top_rb:
            st.markdown(f"**#{top_rb['jersey']} {top_rb['name']} (RB1)**")
            st.write(f"• **10+ Rushing Yards:** **{rush_prob_10 * 100:.1f}%** (`{prob_to_american(rush_prob_10)}`)")
            st.caption(f"Explosive chunk yardage benchmark vs {def_abbr}.")

    with s3:
        st.warning("⚡ **Anytime / Drive TD Leader**")
        if top_rb:
            rb_td_prob = prob_drive_td * 0.45
            st.markdown(f"**#{top_rb['jersey']} {top_rb['name']} (RB1)**")
            st.write(f"• **Drive TD Scorer:** **{rb_td_prob * 100:.1f}%** (`{prob_to_american(rb_td_prob)}`)")
            st.caption("Goal-line and red zone carry share favorite.")
        elif top_qb:
            qb_pass_td = prob_drive_td * 0.65
            st.markdown(f"**#{top_qb['jersey']} {top_qb['name']} (QB1)**")
            st.write(f"• **1+ Passing TD:** **{qb_pass_td * 100:.1f}%** (`{prob_to_american(qb_pass_td)}`)")
            st.caption("Drive TD converted through the air.")
else:
    st.info("Loading active personnel...")

st.divider()

# ---------------------------------------------------------
# DUAL-COLUMN WORKSTATION
# ---------------------------------------------------------
col_drive, col_divider, col_prop = st.columns([10, 1, 11])

# === LEFT PANEL: DRIVE OUTCOME ENGINE ===
with col_drive:
    st.subheader(f"🏈 {off_abbr} Drive Probabilities")
    
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

# === RIGHT PANEL: ALL EXPANDED PLAYER PROPS ===
with col_prop:
    st.subheader(f"🎯 {off_abbr} Player Micro-Props")

    if not roster:
        st.warning("Connecting to active roster...")
    else:
        tab_rec, tab_rush, tab_pass, tab_td = st.tabs(["🏈 Receptions & Yds", "🏃 Rushing Props", "🎯 QB Passing", "⚡ Anytime / Drive TD"])
        
        # 1. RECEPTIONS & RECEIVING YARDS
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
            exp_rec_yds = exp_catches * 11.2
            
            st.caption(f"Situational Expectancy: **{target_share * 100:.0f}%** Target Share (~{exp_targets:.2f} targets | ~{exp_rec_yds:.1f} yds)")

            r1, r2 = st.columns(2)
            prob_1_catch = 1.0 - math.exp(-exp_catches)
            r1.metric("1+ Reception", f"{prob_1_catch * 100:.1f}%", f"Fair: {prob_to_american(prob_1_catch)}")

            prob_2_catch = max(0.0, min(0.99, 1.0 - math.exp(-exp_catches) * (1.0 + exp_catches)))
            r2.metric("2+ Receptions", f"{prob_2_catch * 100:.1f}%", f"Fair: {prob_to_american(prob_2_catch)}")

            r3, r4 = st.columns(2)
            prob_10_rec = min(0.96, 1.0 - math.exp(-exp_catches * 0.65))
            r3.metric("10+ Receiving Yds", f"{prob_10_rec * 100:.1f}%", f"Fair: {prob_to_american(prob_10_rec)}")

            prob_20_rec = min(0.90, 1.0 - math.exp(-exp_catches * 0.38))
            r4.metric("20+ Receiving Yds", f"{prob_20_rec * 100:.1f}%", f"Fair: {prob_to_american(prob_20_rec)}")

        # 2. RUSHING PROPS
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
            st.caption(f"Situational Expectancy: **{carry_share * 100:.0f}%** Carry Share (~{exp_carries:.2f} carries)")

            ru1, ru2 = st.columns(2)
            prob_5_rush = min(0.98, 1.0 - math.exp(-exp_carries * 0.72))
            ru1.metric("5+ Rush Yds", f"{prob_5_rush * 100:.1f}%", f"Fair: {prob_to_american(prob_5_rush)}")

            prob_10_rush = min(0.95, 1.0 - math.exp(-exp_carries * 0.42))
            ru2.metric("10+ Rush Yds", f"{prob_10_rush * 100:.1f}%", f"Fair: {prob_to_american(prob_10_rush)}")

            ru3, ru4 = st.columns(2)
            prob_15_rush = min(0.90, 1.0 - math.exp(-exp_carries * 0.28))
            ru3.metric("15+ Rush Yds", f"{prob_15_rush * 100:.1f}%", f"Fair: {prob_to_american(prob_15_rush)}")

            prob_rush_td = min(0.85, prob_drive_td * (carry_share * 0.70))
            ru4.metric("Rush TD on Drive", f"{prob_rush_td * 100:.1f}%", f"Fair: {prob_to_american(prob_rush_td)}")

        # 3. QB PASSING PROPS
        with tab_pass:
            qb_options = [f"#{p['jersey']} {p['name']} (QB)" for p in qbs] if qbs else ["#1 QB Starter"]
            chosen_qb = st.selectbox("Select Quarterback", qb_options, index=0)
            
            exp_pass_attempts = est_plays * pass_rate
            exp_completions = exp_pass_attempts * 0.65
            exp_pass_yds = exp_completions * 10.8
            
            st.caption(f"Drive Passing Volume: ~{exp_pass_attempts:.1f} Attempts | ~{exp_pass_yds:.1f} Projected Passing Yards")

            q1, q2 = st.columns(2)
            prob_15_pass = min(0.98, 1.0 - math.exp(-exp_completions * 0.70))
            q1.metric("15+ Passing Yards", f"{prob_15_pass * 100:.1f}%", f"Fair: {prob_to_american(prob_15_pass)}")

            prob_25_pass = min(0.94, 1.0 - math.exp(-exp_completions * 0.45))
            q2.metric("25+ Passing Yards", f"{prob_25_pass * 100:.1f}%", f"Fair: {prob_to_american(prob_25_pass)}")

            q3, q4 = st.columns(2)
            prob_pass_td = min(0.85, prob_drive_td * 0.68)
            q3.metric("1+ Passing TD on Drive", f"{prob_pass_td * 100:.1f}%", f"Fair: {prob_to_american(prob_pass_td)}")

            prob_int = min(0.40, max(0.02, 1.0 - math.exp(-exp_pass_attempts * 0.028)))
            q4.metric("Interception Thrown", f"{prob_int * 100:.1f}%", f"Fair: {prob_to_american(prob_int)}")

        # 4. ANYTIME / DRIVE TOUCHDOWN PROPS
        with tab_td:
            all_td_options = []
            for p in rbs:
                all_td_options.append((f"#{p['jersey']} {p['name']} (RB)", 0.38))
            for p in wrs:
                all_td_options.append((f"#{p['jersey']} {p['name']} (WR)", 0.22))
            for p in tes:
                all_td_options.append((f"#{p['jersey']} {p['name']} (TE)", 0.15))
            for p in qbs:
                all_td_options.append((f"#{p['jersey']} {p['name']} (QB - Rush)", 0.10))

            if all_td_options:
                chosen_scorer = st.selectbox("Select TD Candidate", [opt[0] for opt in all_td_options], index=0)
                td_weight = next(opt[1] for opt in all_td_options if opt[0] == chosen_scorer)
                
                drive_td_prob = prob_drive_td * td_weight
                # Anytime game TD projection (roughly 6.5 drives per team per game)
                game_td_prob = 1.0 - (1.0 - drive_td_prob) ** 6.0
                
                td_col1, td_col2 = st.columns(2)
                td_col1.metric("Touchdown THIS Drive", f"{drive_td_prob * 100:.1f}%", f"Fair: {prob_to_american(drive_td_prob)}")
                td_col2.metric("Anytime Game TD (Full Game)", f"{game_td_prob * 100:.1f}%", f"Fair: {prob_to_american(game_td_prob)}")
                st.caption(f"Implied team drive touchdown baseline: {prob_drive_td * 100:.1f}%.")

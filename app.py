import streamlit as st
import pandas as pd
import numpy as np
import joblib

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

st.title("🏈 Live Drive Probability Engine")
st.caption("Objective zero-vig next-drive possession probabilities | Commercial break decision support")

# Sidebar Controls
st.sidebar.header("Commercial Break Snapshot")
qtr = st.sidebar.radio("Quarter", [1, 2, 3, 4], index=2, horizontal=True)
time_min = st.sidebar.slider("Minutes Remaining in Qtr", 0, 15, 8)
time_sec = st.sidebar.slider("Seconds Remaining in Qtr", 0, 59, 45)
qtr_seconds = time_min * 60 + time_sec

half_seconds = qtr_seconds + 900 if qtr in [1, 3] else qtr_seconds
game_seconds = qtr_seconds + (4 - qtr) * 900

yardline_100 = st.sidebar.slider("Yards to Opponent End Zone", 1, 99, 75, help="75 = Own 25-yard line (Touchback)")
score_diff = st.sidebar.slider("Score Differential (Offense - Defense)", -35, 35, -3)

# Prediction
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

# Main Output
field_start = f"Own {100 - yardline_100}" if yardline_100 > 50 else f"Opp {yardline_100}"
st.markdown(f"### Drive Snapshot: **Q{qtr} {time_min:02d}:{time_sec:02d}** | Origin: **{field_start}** | Margin: **{score_diff:+d} pts**")
st.divider()

st.subheader("Model Projections & Fair Zero-Vig Odds")
for outcome, p in results:
    clean_name = outcome.replace("_", " ")
    fair_odds = prob_to_american(p)
    col1, col2, col3 = st.columns([3, 5, 3])
    with col1:
        st.write(f"**{clean_name}**")
    with col2:
        st.progress(float(p))
    with col3:
        st.write(f"**{p * 100:.1f}%** | Fair: `{fair_odds}`")

st.divider()

# Market EV Checker
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

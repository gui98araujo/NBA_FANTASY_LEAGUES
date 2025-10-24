# app.py
# -*- coding: utf-8 -*-
import time
from datetime import datetime
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from nba_api.stats.endpoints import leaguegamelog

# =========================
# ----- Page Config + CSS
# =========================
st.set_page_config(page_title="Fantasy NBA Records", page_icon="🏀", layout="wide")

# Dark theme (pure black)
st.markdown(
    """
    <style>
    :root { --primary-color: #00E5FF; }
    .stApp { background-color: #000000; color: #F1F1F1; }
    .st-emotion-cache-1r4qj8v, .st-emotion-cache-uhkwxv, .st-emotion-cache-1dp5vir {
        background-color: #111111 !important;
    }
    .stButton>button, .stDownloadButton>button {
        background-color: #00E5FF !important; color: #000 !important; border: none;
    }
    .stSelectbox label, .stNumberInput label, .stCheckbox label {
        color: #F1F1F1 !important;
    }
    .stDataFrame table { color: #F1F1F1; }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# ----- Helpers
# =========================
def season_label_from_start_year(start_year: int) -> str:
    return f"{start_year}-{str((start_year + 1) % 100).zfill(2)}"

def infer_current_season_label() -> str:
    dt = datetime.today()
    start_year = dt.year if dt.month >= 10 else dt.year - 1
    return season_label_from_start_year(start_year)

def seasons_from(start_year: int) -> List[str]:
    last_start_year = int(infer_current_season_label()[:4])
    return [season_label_from_start_year(y) for y in range(start_year, last_start_year + 1)]

def parse_opponent_from_matchup(matchup: str) -> Tuple[str, str, bool]:
    parts = str(matchup).split()
    if len(parts) != 3:
        return ("", "", False)
    team, sep, opp = parts
    return team, opp, sep.lower() == "vs"

# =========================
# ----- Data Fetch
# =========================
VALID_SEASON_TYPES = ["Regular Season", "Playoffs"]

def _fetch_logs(season_label: str, season_type: str) -> pd.DataFrame:
    try:
        r = leaguegamelog.LeagueGameLog(
            season=season_label,
            season_type_all_star=season_type,
            player_or_team_abbreviation="P",
        )
        df = r.get_data_frames()[0].copy()
        df["SEASON"] = season_label
        df["SEASON_TYPE"] = season_type
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
        parsed = df["MATCHUP"].apply(parse_opponent_from_matchup)
        df["TEAM_ABBREV_FROM_MATCHUP"] = parsed.apply(lambda x: x[0])
        df["OPPONENT_ABBREVIATION"] = parsed.apply(lambda x: x[1])
        df["IS_HOME"] = parsed.apply(lambda x: x[2])
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(show_spinner=True)
def load_logs(start_season_label: str, include_playoffs: bool) -> pd.DataFrame:
    start_year = int(start_season_label[:4])
    seasons = seasons_from(start_year)
    season_types = ["Regular Season"] + (["Playoffs"] if include_playoffs else [])
    parts = []
    for s in seasons:
        for t in season_types:
            df = _fetch_logs(s, t)
            if not df.empty:
                parts.append(df)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

# =========================
# ----- Fantasy Points
# =========================
DEFAULT_SCORING = {
    "points": 0.7, "assist": 1.1, "steal": 2.2, "block": 2.0, "turnover": -1.0,
    "ft_missed": -0.1, "three_made": 0.7, "three_missed": -0.2,
    "oreb": 1.2, "dreb": 0.95, "tech_foul": -1.0, "flagrant_foul": -2.0,
    "double_double": 2.0, "triple_double": 3.0,
    "bonus_40": 2.0, "bonus_50": 2.0, "bonus_15_ast": 2.0, "bonus_20_reb": 2.0,
    "stack_dd_td": True, "stack_40_50": True
}

def compute_fantasy_points(df: pd.DataFrame, s: Dict[str, float]) -> pd.DataFrame:
    out = df.copy()
    for col in ["PTS","AST","STL","BLK","TOV","FG3M","FG3A","FTM","FTA","OREB","DREB"]:
        if col not in out: out[col] = 0
    out["REB"] = out["OREB"] + out["DREB"]
    out["_ft_missed"] = (out["FTA"] - out["FTM"]).clip(lower=0)
    out["_fg3_missed"] = (out["FG3A"] - out["FG3M"]).clip(lower=0)
    out["_b40"] = np.where(out["PTS"]>=40,s["bonus_40"],0)
    out["_b50"] = np.where(out["PTS"]>=50,s["bonus_50"],0)
    out["_b15_ast"] = np.where(out["AST"]>=15,s["bonus_15_ast"],0)
    out["_b20_reb"] = np.where(out["REB"]>=20,s["bonus_20_reb"],0)
    dd_count = sum(cat>=10 for cat in [out["PTS"],out["REB"],out["AST"],out["STL"],out["BLK"]])
    out["_is_dd"] = (dd_count>=2).astype(int)
    out["_is_td"] = (dd_count>=3).astype(int)
    out["_dd_points"] = out["_is_dd"]*s["double_double"]+out["_is_td"]*s["triple_double"]
    fp = (
        s["points"]*out["PTS"]+s["assist"]*out["AST"]+s["steal"]*out["STL"]+s["block"]*out["BLK"]+
        s["turnover"]*out["TOV"]+s["ft_missed"]*out["_ft_missed"]+s["three_made"]*out["FG3M"]+
        s["three_missed"]*out["_fg3_missed"]+s["oreb"]*out["OREB"]+s["dreb"]*out["DREB"]+
        out["_b40"]+out["_b50"]+out["_b15_ast"]+out["_b20_reb"]+out["_dd_points"]
    )
    out["fantasy_points"] = fp
    return out

# =========================
# ----- UI
# =========================
page = st.sidebar.radio("Navigate", ["Setup", "Records"], index=0)

if "scoring" not in st.session_state: st.session_state["scoring"]=DEFAULT_SCORING.copy()
if "raw_df" not in st.session_state: st.session_state["raw_df"]=pd.DataFrame()
if "started" not in st.session_state: st.session_state["started"]=False

if page=="Setup":
    st.title("⚙️ Setup")
    st.subheader("Set your League Scoring Settings")
    s=st.session_state["scoring"]
    with st.form("setup_form"):
        c1,c2=st.columns(2)
        with c1:
            s["points"]=st.number_input("Points",value=s["points"])
            s["assist"]=st.number_input("Assist",value=s["assist"])
            s["steal"]=st.number_input("Steal",value=s["steal"])
            s["block"]=st.number_input("Block",value=s["block"])
        with c2:
            s["turnover"]=st.number_input("Turnover",value=s["turnover"])
            s["three_made"]=st.number_input("3PT Made",value=s["three_made"])
            s["three_missed"]=st.number_input("3PT Missed",value=s["three_missed"])
            s["oreb"]=st.number_input("Offensive Rebound",value=s["oreb"])
        include_playoffs=st.checkbox("Include Playoffs",value=True)
        seasons_all=seasons_from(1996)
        start_season=st.selectbox("Starting Season",seasons_all,index=0)
        submitted=st.form_submit_button("Save Settings")
        if submitted:
            st.session_state["scoring"]=s
            st.session_state["include_playoffs"]=include_playoffs
            st.session_state["start_season"]=start_season
            st.success("Settings saved.")
    if st.button("Let's Start"):
        with st.spinner(f"Downloading data from {start_season}..."):
            raw=load_logs(start_season,include_playoffs)
        st.session_state["raw_df"]=raw
        st.session_state["started"]=True
        st.success(f"Loaded {len(raw)} rows.")

elif page=="Records":
    st.title("🏀 Records")
    if not st.session_state["started"]:
        st.error("Go to Setup and click Let's Start.")
    else:
        df=compute_fantasy_points(st.session_state["raw_df"],st.session_state["scoring"])
        top_n=st.selectbox("Top N",[10,20,50,100],index=0)
        st.subheader("Top Players by Fantasy Points")
        agg=df.groupby("PLAYER_NAME",as_index=False)["fantasy_points"].sum().sort_values("fantasy_points",ascending=False).head(top_n)
        st.dataframe(agg)
``

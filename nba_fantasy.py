# app.py
# -*- coding: utf-8 -*-
import time
from datetime import datetime
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from nba_api.stats.endpoints import leaguegamelog, commonplayerinfo

# =========================
# ----- Page Config + CSS
# =========================
st.set_page_config(page_title="Fantasy NBA Records", page_icon="🏀", layout="wide")

# Light theme override: white background, black text
st.markdown(
    """
    <style>
    :root { --primary-color: #0057FF; }
    .stApp { background-color: #FFFFFF; color: #000000; }
    .block-container, .stDataFrame, .stMarkdown, .stText, .stSelectbox, .stNumberInput, .stCheckbox, .stRadio {
        color: #000000 !important; background-color: #FFFFFF !important;
    }
    label, .stSelectbox label, .stNumberInput label, .stCheckbox label, .stTextInput label, .stRadio label {
        color: #000000 !important;
    }
    h1, h2, h3, h4, h5, h6, p, span, div, code, .markdown-text-container {
        color: #000000 !important;
    }
    .stButton>button, .stDownloadButton>button {
        background-color: #0057FF !important; color: #FFFFFF !important;
        border: 0; border-radius: 6px;
    }
    .stButton>button:hover, .stDownloadButton>button:hover { background-color: #0043C6 !important; }
    .stDataFrame table, .stDataFrame thead tr th, .stDataFrame tbody tr td {
        color: #000000 !important; background-color: #FFFFFF !important;
    }
    section[data-testid="stSidebar"] { background-color: #F7F9FC !important; color: #000000 !important; }
    .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] > div {
        color: #000000 !important; background-color: #FFFFFF !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# ----- Helpers: Seasons
# =========================
def season_label_from_start_year(start_year: int) -> str:
    return f"{start_year}-{str((start_year + 1) % 100).zfill(2)}"

def infer_current_season_label(today: Optional[datetime] = None) -> str:
    dt = today or datetime.today()
    start_year = dt.year if dt.month >= 10 else dt.year - 1
    return season_label_from_start_year(start_year)

def all_seasons_available(start_year: int = 1996, end_season: Optional[str] = None) -> List[str]:
    last = end_season or infer_current_season_label()
    last_start_year = int(last[:4])
    return [season_label_from_start_year(y) for y in range(start_year, last_start_year + 1)]

def parse_opponent_from_matchup(matchup: str) -> Tuple[str, str, bool]:
    parts = str(matchup).split()
    if len(parts) != 3:
        return ("", "", False)
    team, sep, opp = parts
    is_home = (sep.lower() == "vs")
    return team, opp, is_home

# =========================
# ----- Data Fetch
# =========================
VALID_SEASON_TYPES = ["Regular Season", "Playoffs"]

def _fetch_player_game_logs_for_season(
    season_label: str,
    season_type: str,
    retries: int = 3,
    sleep_sec: float = 0.6,
) -> pd.DataFrame:
    assert season_type in VALID_SEASON_TYPES
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            r = leaguegamelog.LeagueGameLog(
                season=season_label,
                season_type_all_star=season_type,
                player_or_team_abbreviation="P",
            )
            df = r.get_data_frames()[0].copy()
            df["SEASON"] = season_label
            df["SEASON_TYPE"] = season_type
            if "GAME_DATE" in df.columns:
                df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
            if "MATCHUP" in df.columns:
                parsed = df["MATCHUP"].apply(parse_opponent_from_matchup)
                df["TEAM_ABBREV_FROM_MATCHUP"] = parsed.apply(lambda x: x[0])
                df["OPPONENT_ABBREVIATION"] = parsed.apply(lambda x: x[1])
                df["IS_HOME"] = parsed.apply(lambda x: x[2])
            return df
        except Exception as e:
            last_err = e
            time.sleep(sleep_sec * attempt)
    st.warning(f"Failed to fetch {season_label} / {season_type}: {last_err}")
    return pd.DataFrame()

@st.cache_data(show_spinner=True, persist=True)
def load_all_player_game_logs(include_playoffs: bool = True) -> pd.DataFrame:
    seasons = all_seasons_available(start_year=1996)
    season_types = ["Regular Season"] + (["Playoffs"] if include_playoffs else [])
    parts = []
    for s in seasons:
        for t in season_types:
            df = _fetch_player_game_logs_for_season(s, t)
            if not df.empty:
                parts.append(df)
    if not parts:
        return pd.DataFrame()
    big = pd.concat(parts, ignore_index=True)
    sort_cols = [c for c in ["GAME_DATE", "GAME_ID", "PLAYER_ID"] if c in big.columns]
    if sort_cols:
        big = big.sort_values(sort_cols).reset_index(drop=True)
    return big
def clear_all_caches():
    load_all_player_game_logs.clear()

# =========================
# ----- Fantasy Scoring
# =========================
DEFAULT_SCORING: Dict[str, float | bool] = {
    "points": 0.7, "assist": 1.1, "steal": 2.2, "block": 2.0, "turnover": -1.0,
    "ft_missed": -0.1, "three_made": 0.7, "three_missed": -0.2,
    "oreb": 1.2, "dreb": 0.95, "tech_foul": -1.0, "flagrant_foul": -2.0,
    "double_double": 2.0, "triple_double": 3.0, "bonus_40": 2.0, "bonus_50": 2.0,
    "bonus_15_ast": 2.0, "bonus_20_reb": 2.0,
    "stack_dd_td": True, "stack_40_50": True,
}

TECH_CANDIDATES = ["TECH", "TECH_FOULS", "TECHNICALS", "TECHNICAL_FOUL", "TF"]
FLAGRANT_CANDIDATES = ["FLAGRANT", "FLAGRANT_FOULS", "FLAGRANT1", "FLAGRANT2", "FF"]

def compute_fantasy_points(df: pd.DataFrame, scoring: Dict[str, float | bool]) -> pd.DataFrame:
    s = {**DEFAULT_SCORING, **(scoring or {})}
    out = df.copy()
    def ensure(col, default=0):
        if col not in out.columns: out[col] = default
    for c in ["PTS","AST","STL","BLK","TOV","FG3M","FG3A","FTM","FTA","OREB","DREB"]:
        ensure(c, 0)
    if "REB" not in out.columns:
        out["REB"] = out["OREB"].fillna(0) + out["DREB"].fillna(0)
    tech_col = next((c for c in TECH_CANDIDATES if c in out.columns), None)
    flag_col = next((c for c in FLAGRANT_CANDIDATES if c in out.columns), None)
    if tech_col is None: out["__TECH__"] = 0; tech_col = "__TECH__"
    if flag_col is None: out["__FLAG__"] = 0; flag_col = "__FLAG__"
    out["_ft_missed"] = (out["FTA"].fillna(0) - out["FTM"].fillna(0)).clip(lower=0)
    out["_fg3_missed"] = (out["FG3A"].fillna(0) - out["FG3M"].fillna(0)).clip(lower=0)
    if s.get("stack_40_50", True):
        out["_b40"] = np.where(out["PTS"] >= 40, s["bonus_40"], 0.0)
        out["_b50"] = np.where(out["PTS"] >= 50, s["bonus_50"], 0.0)
    else:
        out["_b40"] = np.where((out["PTS"] >= 40)&(out["PTS"]<50), s["bonus_40"], 0.0)
        out["_b50"] = np.where(out["PTS"] >= 50, s["bonus_50"], 0.0)
    out["_b15_ast"] = np.where(out["AST"] >= 15, s["bonus_15_ast"], 0.0)
    out["_b20_reb"] = np.where(out["REB"] >= 20, s["bonus_20_reb"], 0.0)
    cats = [out["PTS"].fillna(0), out["REB"].fillna(0), out["AST"].fillna(0),
            out["STL"].fillna(0), out["BLK"].fillna(0)]
    dd_count = sum(cat >= 10 for cat in cats)
    out["_is_dd"] = (dd_count >= 2).astype(int)
    out["_is_td"] = (dd_count >= 3).astype(int)
    if bool(s.get("stack_dd_td", True)):
        out["_dd_points"] = out["_is_dd"]*s["double_double"] + out["_is_td"]*s["triple_double"]
    else:
        out["_dd_points"] = np.where(out["_is_td"]==1, s["triple_double"],
                                     np.where(out["_is_dd"]==1, s["double_double"], 0.0))
    fp = (
        s["points"]*out["PTS"].fillna(0) + s["assist"]*out["AST"].fillna(0) +
        s["steal"]*out["STL"].fillna(0) + s["block"]*out["BLK"].fillna(0) +
        s["turnover"]*out["TOV"].fillna(0) + s["ft_missed"]*out["_ft_missed"] +
        s["three_made"]*out["FG3M"].fillna(0) + s["three_missed"]*out["_fg3_missed"] +
        s["oreb"]*out["OREB"].fillna(0) + s["dreb"]*out["DREB"].fillna(0) +
        s["tech_foul"]*out[tech_col].fillna(0) + s["flagrant_foul"]*out[flag_col].fillna(0) +
        out["_b40"] + out["_b50"] + out["_b15_ast"] + out["_b20_reb"] + out["_dd_points"]
    )
    out["fantasy_points"] = fp.astype(float)
    for c in ["_ft_missed","_fg3_missed","_b40","_b50","_b15_ast","_b20_reb","_is_dd","_is_td","_dd_points","__TECH__","__FLAG__"]:
        if c in out.columns: out.drop(columns=[c], inplace=True)
    return out

# =========================
# ----- Player Headshots
# =========================
def player_headshot_url(player_id: int) -> str:
    return f"https://cdn.nba.com/headshots/nba/latest/1040x760/{int(player_id)}.png"

# =========================
# ----- Player Positions (cached on demand)
# =========================
@st.cache_data(show_spinner=False)
def get_player_position(player_id: int) -> str:
    """
    Fetch POSITION via CommonPlayerInfo; cached per PLAYER_ID.
    Returns 'UNK' if not available.
    """
    try:
        info = commonplayerinfo.CommonPlayerInfo(player_id=player_id).get_data_frames()[0]
        pos = str(info.get("POSITION", ["UNK"])[0]) if "POSITION" in info.columns else "UNK"
        return pos if pos else "UNK"
    except Exception:
        return "UNK"

def ensure_position_column(df: pd.DataFrame, player_id_col: str = "PLAYER_ID") -> pd.Series:
    """Return a Series with positions aligned with df rows, querying only unknown IDs once."""
    ids = df[player_id_col].dropna().astype(int).unique().tolist()
    positions_map = {}
    for pid in ids:
        positions_map[pid] = get_player_position(int(pid))
        time.sleep(0.05)  # be nice to the API; cached after first call
    return df[player_id_col].map(lambda x: positions_map.get(int(x), "UNK"))

# =========================
# ----- Records helpers
# =========================
def teams_from_df(df: pd.DataFrame) -> List[str]:
    cols = [c for c in ["TEAM_ABBREVIATION", "TEAM_ABBREV_FROM_MATCHUP"] if c in df.columns]
    if not cols: return []
    vals = set()
    for c in cols:
        vals.update(df[c].dropna().astype(str).unique().tolist())
    return sorted([v for v in vals if v and v != "nan"])

# =========================
# ----- Session defaults
# =========================
if "scoring" not in st.session_state:
    st.session_state["scoring"] = DEFAULT_SCORING.copy()
if "include_playoffs" not in st.session_state:
    st.session_state["include_playoffs"] = True
if "analysis_start_season" not in st.session_state:
    st.session_state["analysis_start_season"] = "1996-97"
if "raw_df" not in st.session_state:
    st.session_state["raw_df"] = pd.DataFrame()
if "started" not in st.session_state:
    st.session_state["started"] = False
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []  # list of {'role': 'user'/'assistant', 'content': str}

# =========================
# ----- Sidebar Nav
# =========================
page = st.sidebar.radio("Navigate", ["Setup", "Records", "Chat"], index=0)
st.sidebar.write("---")
if st.sidebar.button("Refresh data (clear cache)"):
    clear_all_caches()
    st.session_state["raw_df"] = pd.DataFrame()
    st.session_state["started"] = False
    st.sidebar.success("Cache cleared. Go to Setup and click 'Let's Start' again.")
st.sidebar.info("Tip: First click **Let's Start** on Setup to download and cache all data.")

# =========================
# ----- Setup Page
# =========================
if page == "Setup":
    st.title("⚙️ Setup")
    st.caption("Set your League Scoring Settings and analysis start. Click **Let's Start** to download data and begin.")

    s = st.session_state["scoring"]

    st.subheader("Set your League Scoring Settings")
    with st.form("scoring_form"):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            s["points"] = st.number_input("Points Scored", value=float(s["points"]), step=0.05, format="%.2f")
            s["assist"] = st.number_input("Assist", value=float(s["assist"]), step=0.05, format="%.2f")
            s["steal"]  = st.number_input("Steal",  value=float(s["steal"]),  step=0.1,  format="%.2f")
            s["block"]  = st.number_input("Block",  value=float(s["block"]),  step=0.1,  format="%.2f")
            s["turnover"] = st.number_input("Turnover", value=float(s["turnover"]), step=0.1, format="%.2f")
        with c2:
            s["ft_missed"] = st.number_input("FT Missed", value=float(s["ft_missed"]), step=0.05, format="%.2f")
            s["three_made"] = st.number_input("3PT Made", value=float(s["three_made"]), step=0.05, format="%.2f")
            s["three_missed"] = st.number_input("3PT Missed", value=float(s["three_missed"]), step=0.05, format="%.2f")
            s["oreb"] = st.number_input("Offensive Rebound", value=float(s["oreb"]), step=0.05, format="%.2f")
            s["dreb"] = st.number_input("Defensive Rebound", value=float(s["dreb"]), step=0.05, format="%.2f")
        with c3:
            s["double_double"] = st.number_input("Double Double Bonus", value=float(s["double_double"]), step=0.5, format="%.2f")
            s["triple_double"] = st.number_input("Triple Double Bonus", value=float(s["triple_double"]), step=0.5, format="%.2f")
            s["bonus_40"] = st.number_input("40+ Points Bonus", value=float(s["bonus_40"]), step=0.5, format="%.2f")
            s["bonus_50"] = st.number_input("50+ Points Bonus", value=float(s["bonus_50"]), step=0.5, format="%.2f")
        with c4:
            s["bonus_15_ast"] = st.number_input("15+ Assists Bonus", value=float(s["bonus_15_ast"]), step=0.5, format="%.2f")
            s["bonus_20_reb"] = st.number_input("20+ Rebounds Bonus", value=float(s["bonus_20_reb"]), step=0.5, format="%.2f")
            s["tech_foul"] = st.number_input("Technical Foul", value=float(s["tech_foul"]), step=0.5, format="%.2f")
            s["flagrant_foul"] = st.number_input("Flagrant Foul", value=float(s["flagrant_foul"]), step=0.5, format="%.2f")

        c5, c6 = st.columns(2)
        with c5:
            s["stack_dd_td"] = st.checkbox("Stack Double-Double and Triple-Double", value=bool(s["stack_dd_td"]))
        with c6:
            s["stack_40_50"] = st.checkbox("Stack 40+ and 50+ Bonuses", value=bool(s["stack_40_50"]))

        st.markdown("---")
        st.subheader("Data Options")
        include_playoffs = st.checkbox("Include Playoffs", value=st.session_state["include_playoffs"])
        seasons_all = all_seasons_available(start_year=1996)
        start_idx = seasons_all.index(st.session_state["analysis_start_season"]) if st.session_state["analysis_start_season"] in seasons_all else 0
        analysis_start = st.selectbox("Starting season for analysis (filters the leaderboards)", seasons_all, index=start_idx,
                                      help="Example: 2003-04, 2004-05. App downloads all seasons, filters visualizations from this season onward.")
        saved = st.form_submit_button("Save Settings")
        if saved:
            st.session_state["scoring"] = s
            st.session_state["include_playoffs"] = include_playoffs
            st.session_state["analysis_start_season"] = analysis_start
            st.success("Settings saved.")

    st.markdown("### Start")
    cstart1, cstart2 = st.columns([1,3])
    with cstart1:
        start_clicked = st.button("Let's Start")
    with cstart2:
        st.caption("Downloads all seasons (1996-97 to current) and caches the dataset for this deployment.")
    if start_clicked:
        st.session_state["include_playoffs"] = include_playoffs
        st.session_state["analysis_start_season"] = analysis_start
        with st.spinner("Downloading ALL player game logs (1996-97 to current). First run may take several minutes..."):
            raw = load_all_player_game_logs(include_playoffs=include_playoffs)
        if raw.empty:
            st.error("No data could be loaded. Try again or reduce options.")
        else:
            st.session_state["raw_df"] = raw
            st.session_state["started"] = True
            st.success(f"Loaded {len(raw):,} player-game rows from {raw['SEASON'].min()} to {raw['SEASON'].max()}. Go to **Records**.")
    if isinstance(st.session_state.get("raw_df"), pd.DataFrame) and not st.session_state["raw_df"].empty:
        st.markdown("#### Sample of Loaded Data")
        st.dataframe(st.session_state["raw_df"].head(20), use_container_width=True)

# =========================
# ----- Records Page
# =========================
elif page == "Records":
    st.title("🏀 Records")
    st.caption("Leaderboards by Season, Career, and Game. Use the filters to customize your view.")

    if not st.session_state.get("started", False) or st.session_state.get("raw_df", pd.DataFrame()).empty:
        st.error("No data available. Please go to the **Setup** page and click **Let's Start**.")
        st.stop()

    raw = st.session_state["raw_df"].copy()
    scoring = st.session_state["scoring"]
    analysis_start = st.session_state["analysis_start_season"]

    try:
        start_year_filter = int(analysis_start[:4])
    except Exception:
        start_year_filter = 1996
    raw["_SEASON_START_YEAR"] = raw["SEASON"].str[:4].astype(int)
    df = raw.loc[raw["_SEASON_START_YEAR"] >= start_year_filter].copy()
    df.drop(columns=["_SEASON_START_YEAR"], inplace=True)

    @st.cache_data(show_spinner=False)
    def compute_fp_cached(df_in: pd.DataFrame, scoring_key: str, window_key: str) -> pd.DataFrame:
        return compute_fantasy_points(df_in, scoring)

    scoring_key = str(sorted([(k, scoring[k]) for k in scoring.keys()]))
    window_key = f"{analysis_start}"
    df = compute_fp_cached(df, scoring_key, window_key)

    st.subheader("Filters")
    left, right = st.columns([2, 1])
    with left:
        teams = teams_from_df(df)
        sel_teams = st.multiselect("Select Team(s)", options=teams, default=teams)
    with right:
        top_n = st.selectbox("Top N", options=[10, 15, 20, 50, 100], index=0)
    df_f = df[df["TEAM_ABBREVIATION"].isin(sel_teams)].copy() if sel_teams and len(sel_teams) < len(teams) else df.copy()

    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["Season Records", "Career Records", "Game Records"])

    def add_photo_and_name(dfi: pd.DataFrame) -> pd.DataFrame:
        out = dfi.copy()
        out["Photo"] = out["PLAYER_ID"].apply(lambda x: player_headshot_url(int(x)))
        out.rename(columns={"PLAYER_NAME": "Player"}, inplace=True)
        return out

    def render_table(table: pd.DataFrame):
        st.dataframe(
            table, use_container_width=True, hide_index=True,
            column_config={
                "Rank": st.column_config.NumberColumn("Rank", format="%d", width="small"),
                "Player": st.column_config.TextColumn("Player", width="medium"),
                "Photo": st.column_config.ImageColumn("Photo", width="small"),
                "Team": st.column_config.TextColumn("Team", width="small"),
                "Season": st.column_config.TextColumn("Season", width="small"),
                "Fantasy Points": st.column_config.NumberColumn("Fantasy Points", format="%.2f"),
                "Game Date": st.column_config.DateColumn("Game Date"),
                "Opponent": st.column_config.TextColumn("Opponent", width="small"),
            },
        )

    with tab1:
        st.subheader("Season Records")
        c1, c2 = st.columns(2)
        with c1:
            season_types = sorted(df_f["SEASON_TYPE"].dropna().unique().tolist())
            sel_season_types = st.multiselect("Season Type", options=season_types, default=season_types)
        with c2:
            seasons = sorted(df_f["SEASON"].dropna().unique().tolist())
            sel_seasons = st.multiselect("Season", options=seasons, default=seasons)

        tmp = df_f.copy()
        if sel_season_types: tmp = tmp[tmp["SEASON_TYPE"].isin(sel_season_types)]
        if sel_seasons: tmp = tmp[tmp["SEASON"].isin(sel_seasons)]

        season_totals = tmp.groupby(["SEASON","PLAYER_ID","PLAYER_NAME"], as_index=False).agg(FP=("fantasy_points","sum"))
        team_per_season = tmp.groupby(["SEASON","PLAYER_ID","TEAM_ABBREVIATION"], as_index=False).agg(FP=("fantasy_points","sum"))
        main_team = (team_per_season.sort_values(["SEASON","PLAYER_ID","FP"], ascending=[True,True,False])
                                   .drop_duplicates(["SEASON","PLAYER_ID"])
                                   .rename(columns={"TEAM_ABBREVIATION":"Team"}))[["SEASON","PLAYER_ID","Team"]]
        season_totals = season_totals.merge(main_team, on=["SEASON","PLAYER_ID"], how="left")
        season_totals.rename(columns={"FP":"Fantasy Points"}, inplace=True)
        topN = (season_totals.sort_values(["SEASON","Fantasy Points"], ascending=[True, False])
                            .groupby("SEASON", group_keys=False)
                            .apply(lambda g: g.nlargest(top_n, "Fantasy Points")))
        topN = add_photo_and_name(topN)
        topN["Rank"] = topN.groupby("SEASON")["Fantasy Points"].rank(ascending=False, method="first").astype(int)
        topN = topN[["Rank","Photo","Player","Team","SEASON","Fantasy Points","PLAYER_ID"]].rename(columns={"SEASON":"Season"})
        render_table(topN.drop(columns=["PLAYER_ID"]))
        st.download_button("Download Season Records (CSV)",
                           data=topN.drop(columns=["Photo","PLAYER_ID"]).to_csv(index=False).encode("utf-8"),
                           file_name="season_records.csv", mime="text/csv")

    with tab2:
        st.subheader("Career Records")
        career_totals = (df_f.groupby(["PLAYER_ID","PLAYER_NAME"], as_index=False)
                           .agg(Fantasy_Points=("fantasy_points","sum"))
                           .sort_values("Fantasy_Points", ascending=False).head(top_n))
        team_totals = df_f.groupby(["PLAYER_ID","TEAM_ABBREVIATION"], as_index=False).agg(FP=("fantasy_points","sum"))
        primary_team = (team_totals.sort_values(["PLAYER_ID","FP"], ascending=[True, False])
                                   .drop_duplicates(["PLAYER_ID"])
                                   .rename(columns={"TEAM_ABBREVIATION":"Team"}))[["PLAYER_ID","Team"]]
        career_totals = career_totals.merge(primary_team, on="PLAYER_ID", how="left").rename(columns={"Fantasy_Points":"Fantasy Points"})
        out = add_photo_and_name(career_totals)
        out["Rank"] = range(1, len(out) + 1)
        out = out[["Rank","Photo","Player","Team","Fantasy Points","PLAYER_ID"]]
        render_table(out.drop(columns=["PLAYER_ID"]))
        st.download_button("Download Career Records (CSV)",
                           data=out.drop(columns=["Photo","PLAYER_ID"]).to_csv(index=False).encode("utf-8"),
                           file_name="career_records.csv", mime="text/csv")

    with tab3:
        st.subheader("Game Records")
        game_top = df_f.sort_values("fantasy_points", ascending=False).head(top_n).copy()
        game_top["Team"] = game_top["TEAM_ABBREVIATION"]
        game_top["Opponent"] = game_top.get("OPPONENT_ABBREVIATION", "")
        game_top["Season"] = game_top["SEASON"]
        game_top["Fantasy Points"] = game_top["fantasy_points"]
        game_top["Game Date"] = pd.to_datetime(game_top.get("GAME_DATE", pd.NaT)).dt.date
        out = add_photo_and_name(game_top)
        out["Rank"] = range(1, len(out) + 1)
        cols_order = ["Rank","Photo","Player","Team","Season","Game Date","Opponent","Fantasy Points","PLAYER_ID"]
        for col in cols_order:
            if col not in out.columns: out[col] = pd.NA
        out = out[cols_order]
        render_table(out.drop(columns=["PLAYER_ID"]))
        st.download_button("Download Game Records (CSV)",
                           data=out.drop(columns=["Photo","PLAYER_ID"]).to_csv(index=False).encode("utf-8"),
                           file_name="game_records.csv", mime="text/csv")

# =========================
# ----- Chat Page (NEW)
# =========================
elif page == "Chat":
    st.title("💬 Chat")
    st.caption("Ask questions about the loaded dataset. Answers are computed with pandas (no hallucinations).")

    if not st.session_state.get("started", False) or st.session_state.get("raw_df", pd.DataFrame()).empty:
        st.error("No data available. Please go to the **Setup** page and click **Let's Start**.")
        st.stop()

    raw = st.session_state["raw_df"].copy()
    scoring = st.session_state["scoring"]
    analysis_start = st.session_state["analysis_start_season"]

    try:
        start_year_filter = int(analysis_start[:4])
    except Exception:
        start_year_filter = 1996
    raw["_SEASON_START_YEAR"] = raw["SEASON"].str[:4].astype(int)
    df = raw.loc[raw["_SEASON_START_YEAR"] >= start_year_filter].copy()
    df.drop(columns=["_SEASON_START_YEAR"], inplace=True)

    @st.cache_data(show_spinner=False)
    def compute_fp_cached_for_chat(df_in: pd.DataFrame, scoring_key: str, window_key: str) -> pd.DataFrame:
        return compute_fantasy_points(df_in, scoring)

    scoring_key = str(sorted([(k, scoring[k]) for k in scoring.keys()]))
    window_key = f"{analysis_start}"
    df = compute_fp_cached_for_chat(df, scoring_key, window_key)

    # ---------- Utility functions
    def normalize_season_str(s: str) -> Optional[str]:
        s = s.strip().replace(" ", "")
        # Accept forms like '24/25', '2024/25', '2024-25', '24-25'
        if "/" in s: a,b = s.split("/")
        elif "-" in s: a,b = s.split("-")
        else: return None
        a = int(a) if len(a) == 4 else (2000 + int(a) if int(a) < 50 else 1900 + int(a))
        b = int(b) if len(b) == 4 else int(b)
        b_full = a + 1
        return f"{a}-{str(b_full % 100).zfill(2)}"

    def find_player_id_by_name_like(name: str) -> Optional[int]:
        name_l = name.lower().strip()
        # Exact match first
        exact = df[df["PLAYER_NAME"].str.lower() == name_l]
        if not exact.empty:
            return int(exact["PLAYER_ID"].iloc[0])
        # Contains match
        cand = df[df["PLAYER_NAME"].str.lower().str.contains(name_l, na=False)]
        if not cand.empty:
            # Prefer most frequent
            top = cand["PLAYER_ID"].value_counts().idxmax()
            return int(top)
        return None

    def with_without_teammate(player_id: int, teammate_id: int, season: Optional[str]) -> pd.DataFrame:
        d = df.copy()
        if season:
            d = d[d["SEASON"] == season]
        # filter to player's games
        p_games = d[d["PLAYER_ID"] == player_id][["GAME_ID","PLAYER_ID","fantasy_points","PTS","AST","REB","STL","BLK","TOV","TEAM_ID","TEAM_ABBREVIATION","GAME_DATE"]].copy()
        # teammate presence in same TEAM and GAME_ID with minutes > 0 (approx: any row means dressed; nba_api doesn't give MIN in this endpoint as time, but presence suffices)
        tm_games = d[d["PLAYER_ID"] == teammate_id][["GAME_ID","TEAM_ID"]].drop_duplicates()
        with_tm = p_games.merge(tm_games, on=["GAME_ID","TEAM_ID"], how="inner")
        without_tm = p_games.merge(tm_games, on=["GAME_ID","TEAM_ID"], how="left", indicator=True)
        without_tm = without_tm[without_tm["_merge"] == "left_only"].drop(columns=["_merge","TEAM_ID_y"]).rename(columns={"TEAM_ID_x":"TEAM_ID"})
        # Averages
        def metrics(x: pd.DataFrame) -> pd.Series:
            return pd.Series({
                "Games": len(x),
                "Fantasy Avg": x["fantasy_points"].mean() if not x.empty else np.nan,
                "PTS Avg": x["PTS"].mean() if not x.empty else np.nan,
                "AST Avg": x["AST"].mean() if not x.empty else np.nan,
                "REB Avg": x["REB"].mean() if not x.empty else np.nan,
            })
        res = pd.DataFrame({
            "With Teammate": metrics(with_tm),
            "Without Teammate": metrics(without_tm),
        }).T.reset_index().rename(columns={"index":"Split"})
        return res

    def top_scorers_position_distribution(season: str, top_n: int = 100, metric: str = "PTS") -> pd.DataFrame:
        d = df[df["SEASON"] == season].copy()
        # Season totals per player
        totals = d.groupby(["PLAYER_ID","PLAYER_NAME"], as_index=False).agg(Value=(metric, "sum"))
        top_df = totals.sort_values("Value", ascending=False).head(top_n).copy()
        # Attach position
        top_df["POSITION"] = ensure_position_column(top_df, player_id_col="PLAYER_ID")
        dist = top_df["POSITION"].fillna("UNK").replace("", "UNK").value_counts().sort_index()
        dist_df = dist.reset_index().rename(columns={"index":"Position","POSITION":"Count"})
        dist_df["Percent"] = (dist_df["Count"] / dist_df["Count"].sum() * 100).round(2)
        return dist_df

    def centers_mean_std(season: Optional[str], metric: str = "fantasy_points") -> pd.DataFrame:
        d = df.copy()
        if season: d = d[d["SEASON"] == season]
        # Attach position for the filtered rows' players (unique mapping)
        small = d[["PLAYER_ID"]].drop_duplicates().copy()
        small["POSITION"] = ensure_position_column(small, player_id_col="PLAYER_ID")
        d = d.merge(small, on="PLAYER_ID", how="left")
        centers = d[d["POSITION"].fillna("UNK").str.contains("C")].copy()  # include C, C-F, F-C, etc
        if centers.empty:
            return pd.DataFrame({"Metric":[metric], "Games":[0], "Mean":[np.nan], "Std":[np.nan]})
        return pd.DataFrame({
            "Metric":[metric],
            "Games":[len(centers)],
            "Mean":[centers[metric].mean()],
            "Std":[centers[metric].std(ddof=1) if len(centers)>1 else 0.0]
        })

    def weekly_max_average(season: Optional[str], metric: str = "fantasy_points", top_n: int = 20) -> pd.DataFrame:
        d = df.copy()
        if season: d = d[d["SEASON"] == season]
        if "GAME_DATE" not in d.columns:
            st.warning("GAME_DATE not available for weekly computation.")
            return pd.DataFrame()
        tmp = d.copy()
        tmp["ISO_Week"] = pd.to_datetime(tmp["GAME_DATE"]).dt.isocalendar().week.astype(int)
        tmp["ISO_Year"] = pd.to_datetime(tmp["GAME_DATE"]).dt.isocalendar().year.astype(int)
        weekly_max = (tmp.groupby(["PLAYER_ID","PLAYER_NAME","ISO_Year","ISO_Week"], as_index=False)
                        .agg(WeeklyMax=(metric,"max")))
        avg_by_player = (weekly_max.groupby(["PLAYER_ID","PLAYER_NAME"], as_index=False)
                           .agg(AvgWeeklyMax=("WeeklyMax","mean"), Weeks=("WeeklyMax","count")))
        topk = avg_by_player.sort_values("AvgWeeklyMax", ascending=False).head(top_n)
        return topk

    # ---------- Intent parser
    def answer_question(q: str):
        ql = q.lower()

        # Season extraction (optional)
        season = None
        import re
        m = re.search(r'(\d{2,4})\/\-', ql)
        if m:
            season = normalize_season_str(m.group(0))

        # Metric detection
        metric = "fantasy_points"
        if " fantasy" in ql or "fantasy" in ql:
            metric = "fantasy_points"
        elif "assist" in ql:
            metric = "AST"
        elif "rebound" in ql or "rebounds" in ql or "rebote" in ql:
            metric = "REB"
        elif "steal" in ql:
            metric = "STL"
        elif "block" in ql or "toco" in ql:
            metric = "BLK"
        elif "turnover" in ql:
            metric = "TOV"
        elif "point" in ql or "ponto" in ql or "pontuador" in ql:
            metric = "PTS"

        # With/Without teammate
        if ("with" in ql and "without" in ql) or ("com" in ql and "sem" in ql):
            # naive parse: "... of <player> ... with <teammate> ... without ..."
            # Extract two names by capitalization heuristic from original q
            tokens = [t.strip(",. ") for t in q.split() if t[:1].isupper()]
            if len(tokens) >= 2:
                p_name = " ".join(tokens[0:2]) if len(tokens)>=2 else tokens[0]
                # a bit more robust: get two longest capitalized spans
                # fallback: try first proper name then the next
                pid = find_player_id_by_name_like(p_name)
                if pid is None:
                    # try first token only
                    pid = find_player_id_by_name_like(tokens[0])
                # teammate name: search after 'with'/'com'
                after_with = q.split("with")[-1] if "with" in ql else q.split("com")[-1]
                t_tokens = [t for t in after_with.split() if t[:1].isupper()]
                t_name = " ".join(t_tokens[:2]) if t_tokens else (tokens[2] if len(tokens)>2 else "")
                tid = find_player_id_by_name_like(t_name) if t_name else None

                if pid is None or tid is None:
                    st.warning("Could not confidently identify the player and/or teammate. Please specify full names.")
                    return

                res = with_without_teammate(pid, tid, season)
                title = f"Averages for {df[df.PLAYER_ID==pid]['PLAYER_NAME'].iloc[0]} with/without {df[df.PLAYER_ID==tid]['PLAYER_NAME'].iloc[0]}" + (f" in {season}" if season else "")
                st.markdown(f"**{title}**")
                st.dataframe(res, use_container_width=True)
                return

        # Position distribution among Top-N scorers
        if ("distribution" in ql and "position" in ql) or ("distribuição" in ql and "posição" in ql):
            # detect Top N
            import re
            m2 = re.search(r'top\s*(\d+)', ql)
            top_n = int(m2.group(1)) if m2 else 100
            # metric: if explicitly says 'points', use PTS; else fantasy_points
            met = "PTS" if ("point" in ql or "ponto" in ql or "pontuador" in ql) else "fantasy_points"
            if not season:
                st.warning("Please specify a season (e.g., 2025-26 or 25/26) for position distribution.")
                return
            dist_df = top_scorers_position_distribution(season, top_n=top_n, metric=met)
            st.markdown(f"**Position distribution among Top {top_n} by {met} – {season}**")
            st.dataframe(dist_df, use_container_width=True)
            st.bar_chart(dist_df.set_index("Position")["Count"])
            return

        # Mean and std of centers
        if ("center" in ql or "centers" in ql or "pivot" in ql or "pivô" in ql) and ("std" in ql or "desvio" in ql or "standard" in ql):
            res = centers_mean_std(season, metric=metric)
            st.markdown(f"**Mean and Std of Centers – Metric: {metric}**" + (f" – {season}" if season else ""))
            st.dataframe(res, use_container_width=True)
            return

        # Top players by weekly max average
        if ("weekly" in ql and "max" in ql) or ("semana" in ql and "maior" in ql):
            import re
            m3 = re.search(r'top\s*(\d+)', ql)
            top_n = int(m3.group(1)) if m3 else 20
            res = weekly_max_average(season, metric=metric, top_n=top_n)
            if res.empty:
                st.warning("No data for weekly computation.")
                return
            st.markdown(f"**Top {top_n} by Average Weekly Max – Metric: {metric}**" + (f" – {season}" if season else ""))
            st.dataframe(res, use_container_width=True)
            return

        # Generic: per-player season average on chosen metric
        if "average" in ql or "média" in ql or "mean" in ql:
            # find a player name
            tokens_cap = [t.strip(",. ") for t in q.split() if t[:1].isupper()]
            if tokens_cap:
                candidate = " ".join(tokens_cap[:2])
                pid = find_player_id_by_name_like(candidate) or find_player_id_by_name_like(tokens_cap[0])
                if pid:
                    d = df[df["PLAYER_ID"] == pid].copy()
                    if season: d = d[d["SEASON"] == season]
                    if d.empty:
                        st.warning("No games found for the specified player/season.")
                        return
                    avg = d[metric].mean()
                    st.markdown(f"**Average {metric} – {d['PLAYER_NAME'].iloc[0]}**" + (f" – {season}" if season else ""))
                    st.write(f"{avg:.2f} over {len(d)} games.")
                    return

        st.info("Sorry, I couldn't parse that. Try one of the examples below or specify season/player clearly.")

    # ---------- Chat UI
    st.markdown("**Examples you can try:**")
    st.markdown("- Average of Franz Wagner in 24/25 **with** Banchero and **without** Banchero")
    st.markdown("- Among the **Top 100** scorers in 25/26, **position distribution**")
    st.markdown("- **Mean and std** of **centers** in 24/25 (fantasy points)")
    st.markdown("- **Top 20** players by **weekly max average** in 24/25")

    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    query = st.chat_input("Ask about the dataset...")
    if query:
        st.session_state["chat_history"].append({"role":"user","content":query})
        with st.chat_message("user"):
            st.write(query)
        with st.chat_message("assistant"):
            try:
                answer_question(query)
                st.session_state["chat_history"].append({"role":"assistant","content":"(see result above)"})
            except Exception as e:
                st.error(f"Error while answering: {e}")

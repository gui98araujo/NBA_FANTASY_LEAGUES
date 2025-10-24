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

# Light theme override: white background, black text
st.markdown(
    """
    <style>
    :root {
        --primary-color: #0057FF;
    }
    .stApp {
        background-color: #FFFFFF;
        color: #000000;
    }
    /* containers, cards, widgets */
    .block-container,
    .stDataFrame,
    .stMarkdown,
    .stText,
    .stSelectbox,
    .stNumberInput,
    .stCheckbox,
    .stRadio {
        color: #000000 !important;
        background-color: #FFFFFF !important;
    }
    /* labels */
    label, .stSelectbox label, .stNumberInput label, .stCheckbox label,
    .stTextInput label, .stRadio label, .stCaption, .st-emotion-cache {
        color: #000000 !important;
    }
    /* titles, headers */
    h1, h2, h3, h4, h5, h6, p, span, div, code, .markdown-text-container {
        color: #000000 !important;
    }
    /* buttons */
    .stButton>button, .stDownloadButton>button {
        background-color: #0057FF !important;
        color: #FFFFFF !important;
        border: 0;
        border-radius: 6px;
    }
    .stButton>button:hover, .stDownloadButton>button:hover {
        background-color: #0043C6 !important;
    }
    /* tables */
    .stDataFrame table,
    .stDataFrame thead tr th,
    .stDataFrame tbody tr td {
        color: #000000 !important;
        background-color: #FFFFFF !important;
    }
    /* sidebar */
    section[data-testid="stSidebar"] {
        background-color: #F7F9FC !important;
        color: #000000 !important;
    }
    /* inputs */
    .stTextInput input,
    .stNumberInput input,
    .stSelectbox div[data-baseweb="select"] > div {
        color: #000000 !important;
        background-color: #FFFFFF !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# ----- Helpers: Seasons
# =========================
def season_label_from_start_year(start_year: int) -> str:
    """e.g., 2005 -> '2005-06'"""
    return f"{start_year}-{str((start_year + 1) % 100).zfill(2)}"

def infer_current_season_label(today: Optional[datetime] = None) -> str:
    dt = today or datetime.today()
    start_year = dt.year if dt.month >= 10 else dt.year - 1
    return season_label_from_start_year(start_year)

def all_seasons_available(start_year: int = 1996, end_season: Optional[str] = None) -> List[str]:
    """
    Build list of seasons from start_year to current.
    LeagueGameLog is stable from ~1996-97 onwards.
    """
    last = end_season or infer_current_season_label()
    last_start_year = int(last[:4])
    return [season_label_from_start_year(y) for y in range(start_year, last_start_year + 1)]

def parse_opponent_from_matchup(matchup: str) -> Tuple[str, str, bool]:
    """
    'DEN vs LAL' (home) or 'DEN @ LAL' (away) -> (team, opp, is_home)
    """
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

    # On persistent failure, warn and return empty to keep flow
    st.warning(f"Failed to fetch {season_label} / {season_type}: {last_err}")
    return pd.DataFrame()

@st.cache_data(show_spinner=True, persist=True)
def load_all_player_game_logs(include_playoffs: bool = True) -> pd.DataFrame:
    """
    Download all available player game logs (1996-97 .. current).
    Cached to avoid re-downloading on subsequent runs.
    """
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
    # event weights
    "points": 0.7,
    "assist": 1.1,
    "steal": 2.2,
    "block": 2.0,
    "turnover": -1.0,
    "ft_missed": -0.1,
    "three_made": 0.7,
    "three_missed": -0.2,
    "oreb": 1.2,
    "dreb": 0.95,
    "tech_foul": -1.0,
    "flagrant_foul": -2.0,
    # bonuses
    "double_double": 2.0,
    "triple_double": 3.0,
    "bonus_40": 2.0,
    "bonus_50": 2.0,
    "bonus_15_ast": 2.0,
    "bonus_20_reb": 2.0,
    # stacking behavior
    "stack_dd_td": True,   # True => DD (+2) + TD (+3) can stack; False => TD overrides DD
    "stack_40_50": True,   # True => 50+ stacks with 40+; False => 50+ replaces 40+
}

TECH_CANDIDATES = ["TECH", "TECH_FOULS", "TECHNICALS", "TECHNICAL_FOUL", "TF"]
FLAGRANT_CANDIDATES = ["FLAGRANT", "FLAGRANT_FOULS", "FLAGRANT1", "FLAGRANT2", "FF"]

def compute_fantasy_points(df: pd.DataFrame, scoring: Dict[str, float | bool]) -> pd.DataFrame:
    s = {**DEFAULT_SCORING, **(scoring or {})}
    out = df.copy()

    # Ensure columns exist
    def ensure(col, default=0):
        if col not in out.columns:
            out[col] = default

    for c in ["PTS", "AST", "STL", "BLK", "TOV", "FG3M", "FG3A", "FTM", "FTA", "OREB", "DREB"]:
        ensure(c, 0)
    if "REB" not in out.columns:
        out["REB"] = out["OREB"].fillna(0) + out["DREB"].fillna(0)

    tech_col = next((c for c in TECH_CANDIDATES if c in out.columns), None)
    flag_col = next((c for c in FLAGRANT_CANDIDATES if c in out.columns), None)
    if tech_col is None:
        out["__TECH__"] = 0
        tech_col = "__TECH__"
    if flag_col is None:
        out["__FLAG__"] = 0
        flag_col = "__FLAG__"

    # Misses
    out["_ft_missed"] = (out["FTA"].fillna(0) - out["FTM"].fillna(0)).clip(lower=0)
    out["_fg3_missed"] = (out["FG3A"].fillna(0) - out["FG3M"].fillna(0)).clip(lower=0)

    # Bonuses
    if s.get("stack_40_50", True):
        out["_b40"] = np.where(out["PTS"] >= 40, s["bonus_40"], 0.0)
        out["_b50"] = np.where(out["PTS"] >= 50, s["bonus_50"], 0.0)
    else:
        out["_b40"] = np.where((out["PTS"] >= 40) & (out["PTS"] < 50), s["bonus_40"], 0.0)
        out["_b50"] = np.where(out["PTS"] >= 50, s["bonus_50"], 0.0)

    out["_b15_ast"] = np.where(out["AST"] >= 15, s["bonus_15_ast"], 0.0)
    out["_b20_reb"] = np.where(out["REB"] >= 20, s["bonus_20_reb"], 0.0)

    # Double / Triple double
    cats = [
        out["PTS"].fillna(0),
        out["REB"].fillna(0),
        out["AST"].fillna(0),
        out["STL"].fillna(0),
        out["BLK"].fillna(0),
    ]
    dd_count = sum(cat >= 10 for cat in cats)
    out["_is_dd"] = (dd_count >= 2).astype(int)
    out["_is_td"] = (dd_count >= 3).astype(int)
    if bool(s.get("stack_dd_td", True)):
        out["_dd_points"] = out["_is_dd"] * s["double_double"] + out["_is_td"] * s["triple_double"]
    else:
        out["_dd_points"] = np.where(out["_is_td"] == 1, s["triple_double"],
                                     np.where(out["_is_dd"] == 1, s["double_double"], 0.0))

    fp = (
        s["points"] * out["PTS"].fillna(0)
        + s["assist"] * out["AST"].fillna(0)
        + s["steal"] * out["STL"].fillna(0)
        + s["block"] * out["BLK"].fillna(0)
        + s["turnover"] * out["TOV"].fillna(0)
        + s["ft_missed"] * out["_ft_missed"]
        + s["three_made"] * out["FG3M"].fillna(0)
        + s["three_missed"] * out["_fg3_missed"]
        + s["oreb"] * out["OREB"].fillna(0)
        + s["dreb"] * out["DREB"].fillna(0)
        + s["tech_foul"] * out[tech_col].fillna(0)
        + s["flagrant_foul"] * out[flag_col].fillna(0)
        + out["_b40"] + out["_b50"] + out["_b15_ast"] + out["_b20_reb"] + out["_dd_points"]
    )
    out["fantasy_points"] = fp.astype(float)

    # Cleanup aux columns
    for c in ["_ft_missed","_fg3_missed","_b40","_b50","_b15_ast","_b20_reb","_is_dd","_is_td","_dd_points","__TECH__","__FLAG__"]:
        if c in out.columns:
            out.drop(columns=[c], inplace=True)
    return out

# =========================
# ----- Player Headshots
# =========================
def player_headshot_url(player_id: int) -> str:
    # NBA CDN
    pid = int(player_id)
    return f"https://cdn.nba.com/headshots/nba/latest/1040x760/{pid}.png"

# =========================
# ----- Records helpers
# =========================
def teams_from_df(df: pd.DataFrame) -> List[str]:
    cols = [c for c in ["TEAM_ABBREVIATION", "TEAM_ABBREV_FROM_MATCHUP"] if c in df.columns]
    if not cols:
        return []
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

# =========================
# ----- Sidebar Nav
# =========================
page = st.sidebar.radio("Navigate", ["Setup", "Records"], index=0, help="Switch between Setup and Records")

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
        include_playoffs = st.checkbox("Include Playoffs", value=st.session_state["include_playoffs"], help="Include Playoffs besides Regular Season.")

        seasons_all = all_seasons_available(start_year=1996)
        # Choose starting season for analysis (filter applied after download)
        start_idx = 0  # earliest by default
        if st.session_state["analysis_start_season"] in seasons_all:
            start_idx = seasons_all.index(st.session_state["analysis_start_season"])

        analysis_start = st.selectbox(
            "Starting season for analysis (filters the leaderboards)", 
            seasons_all, index=start_idx, 
            help="Example: 2003-04, 2004-05. The app downloads all seasons, but records will filter from this season onward."
        )

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

    # Preview
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

    # Filter dataset to analysis start season
    try:
        start_year_filter = int(analysis_start[:4])
    except Exception:
        start_year_filter = 1996
    raw["_SEASON_START_YEAR"] = raw["SEASON"].str[:4].astype(int)
    df = raw.loc[raw["_SEASON_START_YEAR"] >= start_year_filter].copy()
    df.drop(columns=["_SEASON_START_YEAR"], inplace=True)

    # Compute fantasy points (cached per scoring settings + window)
    @st.cache_data(show_spinner=False)
    def compute_fp_cached(df_in: pd.DataFrame, scoring_key: str, window_key: str) -> pd.DataFrame:
        return compute_fantasy_points(df_in, scoring)

    scoring_key = str(sorted([(k, scoring[k]) for k in scoring.keys()]))
    window_key = f"{analysis_start}"
    df = compute_fp_cached(df, scoring_key, window_key)

    # ---- Filters
    st.subheader("Filters")
    left, right = st.columns([2, 1])

    with left:
        teams = teams_from_df(df)
        sel_teams = st.multiselect(
            "Select Team(s)", options=teams, default=teams,
            help="Filter by the player's team in that game."
        )
    with right:
        top_n = st.selectbox("Top N", options=[10, 15, 20, 50, 100], index=0)

    if sel_teams and len(sel_teams) < len(teams):
        df_f = df[df["TEAM_ABBREVIATION"].isin(sel_teams)].copy()
    else:
        df_f = df.copy()

    st.markdown("---")

    # ---- Tabs
    tab1, tab2, tab3 = st.tabs(["Season Records", "Career Records", "Game Records"])

    def add_photo_and_name(dfi: pd.DataFrame) -> pd.DataFrame:
        out = dfi.copy()
        out["Photo"] = out["PLAYER_ID"].apply(lambda x: player_headshot_url(int(x)))
        out.rename(columns={"PLAYER_NAME": "Player"}, inplace=True)
        return out

    def render_table(table: pd.DataFrame):
        st.dataframe(
            table,
            use_container_width=True,
            hide_index=True,
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

    # ---- Season Records
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
        if sel_season_types:
            tmp = tmp[tmp["SEASON_TYPE"].isin(sel_season_types)]
        if sel_seasons:
            tmp = tmp[tmp["SEASON"].isin(sel_seasons)]

        season_totals = (
            tmp.groupby(["SEASON", "PLAYER_ID", "PLAYER_NAME"], as_index=False)
               .agg(FP=("fantasy_points", "sum"))
        )

        # Primary team in that season by FP
        team_per_season = (
            tmp.groupby(["SEASON","PLAYER_ID","TEAM_ABBREVIATION"], as_index=False)
               .agg(FP=("fantasy_points","sum"))
        )
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

        st.download_button(
            "Download Season Records (CSV)",
            data=topN.drop(columns=["Photo","PLAYER_ID"]).to_csv(index=False).encode("utf-8"),
            file_name="season_records.csv", mime="text/csv"
        )

    # ---- Career Records
    with tab2:
        st.subheader("Career Records")

        career_totals = (
            df_f.groupby(["PLAYER_ID","PLAYER_NAME"], as_index=False)
                .agg(Fantasy_Points=("fantasy_points","sum"))
                .sort_values("Fantasy_Points", ascending=False)
                .head(top_n)
        )

        team_totals = (
            df_f.groupby(["PLAYER_ID","TEAM_ABBREVIATION"], as_index=False)
                .agg(FP=("fantasy_points","sum"))
        )
        primary_team = (team_totals.sort_values(["PLAYER_ID","FP"], ascending=[True, False])
                                   .drop_duplicates(["PLAYER_ID"])
                                   .rename(columns={"TEAM_ABBREVIATION":"Team"}))[["PLAYER_ID","Team"]]
        career_totals = career_totals.merge(primary_team, on="PLAYER_ID", how="left")
        career_totals.rename(columns={"Fantasy_Points":"Fantasy Points"}, inplace=True)

        out = add_photo_and_name(career_totals)
        out["Rank"] = range(1, len(out) + 1)
        out = out[["Rank","Photo","Player","Team","Fantasy Points","PLAYER_ID"]]
        render_table(out.drop(columns=["PLAYER_ID"]))

        st.download_button(
            "Download Career Records (CSV)",
            data=out.drop(columns=["Photo","PLAYER_ID"]).to_csv(index=False).encode("utf-8"),
            file_name="career_records.csv", mime="text/csv"
        )

    # ---- Game Records
    with tab3:
        st.subheader("Game Records")

        # Top N best single games
        game_top = (
            df_f.sort_values("fantasy_points", ascending=False)
                .head(top_n)
                .copy()
        )

        # Idempotent helper columns (avoid .insert)
        game_top["Team"] = game_top["TEAM_ABBREVIATION"]
        game_top["Opponent"] = game_top.get("OPPONENT_ABBREVIATION", "")
        game_top["Season"] = game_top["SEASON"]
        game_top["Fantasy Points"] = game_top["fantasy_points"]
        game_top["Game Date"] = pd.to_datetime(game_top.get("GAME_DATE", pd.NaT)).dt.date

        # Output with photo and names
        out = add_photo_and_name(game_top)
        out["Rank"] = range(1, len(out) + 1)

        cols_order = ["Rank", "Photo", "Player", "Team", "Season", "Game Date", "Opponent", "Fantasy Points", "PLAYER_ID"]
        for col in cols_order:
            if col not in out.columns:
                out[col] = pd.NA
        out = out[cols_order]

        render_table(out.drop(columns=["PLAYER_ID"]))

        st.download_button(
            "Download Game Records (CSV)",
            data=out.drop(columns=["Photo","PLAYER_ID"]).to_csv(index=False).encode("utf-8"),
            file_name="game_records.csv", mime="text/csv"
        )
``

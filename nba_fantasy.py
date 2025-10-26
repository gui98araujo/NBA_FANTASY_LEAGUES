# app.py
# -*- coding: utf-8 -*-
import time
import random
from datetime import datetime
from typing import List, Dict, Optional, Iterable, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import streamlit as st
import openai

from nba_api.stats.endpoints import leaguegamelog, commonplayerinfo, commonteamroster

# =========================
# Page Config + Light Theme (white bg, black text)
# =========================
st.set_page_config(page_title="Fantasy NBA App", page_icon="🏀", layout="wide")
st.markdown(
    """
    <style>
    .stApp { background-color: #FFFFFF; color: #000000; }
    h1,h2,h3,h4,h5,h6,label,p,span,div,code { color: #000000 !important; }
    .stButton>button, .stDownloadButton>button {
        background-color: #0057FF !important; color: #FFFFFF !important; border: 0; border-radius: 6px;
        padding: 0.5rem 1rem; font-weight: 600;
    }
    .stButton>button:hover, .stDownloadButton>button:hover { background-color: #0043C6 !important; }
    section[data-testid="stSidebar"] { background-color: #F7F9FC !important; color: #000000 !important; }
    .stDataFrame table, .stDataFrame thead tr th, .stDataFrame tbody tr td {
        color: #000000 !important; background-color: #FFFFFF !important;
    }
    /* KPI cards */
    .kpi-card {
        border: 1px solid #e6e8eb; border-radius: 10px; padding: 14px 16px; background: #fff;
        box-shadow: 0 1px 3px rgba(16,24,40,0.06);
    }
    .kpi-title { font-size: 0.9rem; color: #475467; margin-bottom: 6px; }
    .kpi-value { font-size: 1.6rem; font-weight: 700; color: #101828; }
    .kpi-sub { font-size: 0.85rem; color: #667085; }
    /* Colored legend bullets */
    .legend-bullet { display:inline-block; width:12px; height:12px; border-radius:3px; margin-right:6px; vertical-align:middle;}
    .legend { font-size: 0.95rem; font-weight: 600; color:#101828; }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# OpenAI (Chat) – opcional
# =========================
openai.api_key = st.secrets.get("OPENAI_API_KEY", "")

# =========================
# Helpers: Seasons
# =========================
EARLIEST_SEASON_START = 2020  # 2020-21 and forward (Regular Season only)

def season_label_from_start_year(start_year: int) -> str:
    """2005 -> '2005-06'"""
    return f"{start_year}-{str((start_year + 1) % 100).zfill(2)}"

def infer_current_season_label(today: Optional[datetime] = None) -> str:
    dt = today or datetime.today()
    start_year = dt.year if dt.month >= 10 else dt.year - 1
    return season_label_from_start_year(start_year)

def seasons_from_2020() -> List[str]:
    last = infer_current_season_label()
    last_start_year = int(last[:4])
    return [season_label_from_start_year(y) for y in range(EARLIEST_SEASON_START, last_start_year + 1)]

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
# Data Fetch (cached per season) — Regular Season only (2020+)
# =========================
@st.cache_data(show_spinner=False, persist=True)
def fetch_one_season_regular(season_label: str, timeout_s: int = 10) -> pd.DataFrame:
    """
    Download player game logs for one season (Regular Season only).
    Cached to avoid re-download.
    """
    try:
        r = leaguegamelog.LeagueGameLog(
            season=season_label,
            season_type_all_star="Regular Season",
            player_or_team_abbreviation="P",
            timeout=timeout_s,
        )
        df = r.get_data_frames()[0].copy()
        df["SEASON"] = season_label
        df["SEASON_TYPE"] = "Regular Season"
        if "GAME_DATE" in df.columns:
            df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
        if "MATCHUP" in df.columns:
            parsed = df["MATCHUP"].apply(parse_opponent_from_matchup)
            df["TEAM_ABBREV_FROM_MATCHUP"] = parsed.apply(lambda x: x[0])
            df["OPPONENT_ABBREVIATION"] = parsed.apply(lambda x: x[1])
            df["IS_HOME"] = parsed.apply(lambda x: x[2])
        return df
    except Exception:
        return pd.DataFrame()

def load_regular_2020_to_current(
    max_workers: int = 6,
    timeout_s: int = 10,
) -> Tuple[pd.DataFrame, List[Tuple[str, str]]]:
    """
    Downloads all seasons from 2020-21 to current (Regular Season only) with parallelism.
    Returns (big_df, failures_list[ (season, status) ]).
    """
    seasons = seasons_from_2020()
    total = len(seasons)
    progress = st.progress(0, text=f"Loading data... (0/{total})")
    parts: List[pd.DataFrame] = []
    failures: List[Tuple[str, str]] = []
    completed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(fetch_one_season_regular, s, timeout_s): s for s in seasons}
        for fut in as_completed(futures):
            s = futures[fut]
            try:
                df = fut.result()
                if df is not None and not df.empty:
                    parts.append(df)
                else:
                    failures.append((s, "empty or failed"))
            except Exception as e:
                failures.append((s, str(e)))
            completed += 1
            progress.progress(min(1.0, completed / total), text=f"Loading data... ({completed}/{total})")

    progress.empty()
    if parts:
        big = pd.concat(parts, ignore_index=True)
        sort_cols = [c for c in ["GAME_DATE", "GAME_ID", "PLAYER_ID"] if c in big.columns]
        if sort_cols:
            big = big.sort_values(sort_cols).reset_index(drop=True)
    else:
        big = pd.DataFrame()

    return big, failures

def clear_all_caches():
    fetch_one_season_regular.clear()
    if hasattr(season_positions_map, "clear"):
        season_positions_map.clear()
    if hasattr(get_player_position, "clear"):
        get_player_position.clear()

# =========================
# Fantasy Scoring (full rules)
# =========================
DEFAULT_SCORING: Dict[str, float | bool] = {
    # base events
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
    # stacking
    "stack_dd_td": True,
    "stack_40_50": True,
}

TECH_CANDIDATES = ["TECH", "TECH_FOULS", "TECHNICALS", "TECHNICAL_FOUL", "TF"]
FLAGRANT_CANDIDATES = ["FLAGRANT", "FLAGRANT_FOULS", "FLAGRANT1", "FLAGRANT2", "FF"]

def compute_fantasy_points(df: pd.DataFrame, scoring: Dict[str, float | bool]) -> pd.DataFrame:
    s = {**DEFAULT_SCORING, **(scoring or {})}
    out = df.copy()

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

    out["_ft_missed"] = (out["FTA"].fillna(0) - out["FTM"].fillna(0)).clip(lower=0)
    out["_fg3_missed"] = (out["FG3A"].fillna(0) - out["FG3M"].fillna(0)).clip(lower=0)

    if s.get("stack_40_50", True):
        out["_b40"] = np.where(out["PTS"] >= 40, s["bonus_40"], 0.0)
        out["_b50"] = np.where(out["PTS"] >= 50, s["bonus_50"], 0.0)
    else:
        out["_b40"] = np.where((out["PTS"] >= 40) & (out["PTS"] < 50), s["bonus_40"], 0.0)
        out["_b50"] = np.where(out["PTS"] >= 50, s["bonus_50"], 0.0)

    out["_b15_ast"] = np.where(out["AST"] >= 15, s["bonus_15_ast"], 0.0)
    out["_b20_reb"] = np.where(out["REB"] >= 20, s["bonus_20_reb"], 0.0)

    cats = [out["PTS"].fillna(0), out["REB"].fillna(0), out["AST"].fillna(0), out["STL"].fillna(0), out["BLK"].fillna(0)]
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

    for c in ["_ft_missed","_fg3_missed","_b40","_b50","_b15_ast","_b20_reb","_is_dd","_is_td","_dd_points","__TECH__","__FLAG__"]:
        if c in out.columns:
            out.drop(columns=[c], inplace=True)
    return out

# =========================
# Images & misc helpers
# =========================
def player_headshot_url(player_id: int) -> str:
    return f"https://cdn.nba.com/headshots/nba/latest/1040x760/{int(player_id)}.png"

def teams_from_df(df: pd.DataFrame) -> List[str]:
    cols = [c for c in ["TEAM_ABBREVIATION", "TEAM_ABBREV_FROM_MATCHUP"] if c in df.columns]
    if not cols:
        return []
    vals = set()
    for c in cols:
        vals.update(df[c].dropna().astype(str).unique().tolist())
    return sorted([v for v in vals if v and v != "nan"])

@st.cache_data(show_spinner=False, persist=True)
def get_player_position(player_id: int) -> str:
    """Fallback: fetch primary position via CommonPlayerInfo (one player)."""
    try:
        info = commonplayerinfo.CommonPlayerInfo(player_id=int(player_id))
        df = info.get_data_frames()[0]
        pos = df.loc[0, "POSITION"] if "POSITION" in df.columns else ""
        return str(pos) if pd.notna(pos) else ""
    except Exception:
        return ""

def primary_position_letter(position: str) -> str:
    """Map 'G', 'F', 'C', 'G-F', 'F-C' -> primary letter."""
    if not position:
        return ""
    return position.strip().upper()[0]  # first letter (G/F/C)

@st.cache_data(show_spinner=False, persist=True)
def season_positions_map(season_label: str, team_ids: Tuple[int, ...]) -> pd.DataFrame:
    """
    Build PLAYER_ID -> POSITION map for a season using CommonTeamRoster
    (far fewer calls than per-player CommonPlayerInfo).
    """
    rows = []
    for tid in team_ids:
        try:
            roster = commonteamroster.CommonTeamRoster(season=season_label, team_id=int(tid))
            rdf = roster.get_data_frames()[0]
            rows.append(rdf[["PLAYER_ID", "POSITION"]])
        except Exception:
            continue
        time.sleep(0.2)  # light throttle
    if rows:
        pos_df = pd.concat(rows, ignore_index=True).drop_duplicates("PLAYER_ID")
    else:
        pos_df = pd.DataFrame(columns=["PLAYER_ID", "POSITION"])
    pos_df["PLAYER_ID"] = pos_df["PLAYER_ID"].astype("Int64")
    return pos_df

# =========================
# Session Defaults
# =========================
if "scoring" not in st.session_state:
    st.session_state["scoring"] = DEFAULT_SCORING.copy()
if "raw_df" not in st.session_state:
    st.session_state["raw_df"] = pd.DataFrame()
if "started" not in st.session_state:
    st.session_state["started"] = False

# =========================
# Sidebar & Navigation
# =========================
page = st.radio("Navigate", ["Setup", "Records", "Player Insights", "League Insights", "Chat"], index=0)
clear_all_caches()
st.session_state["raw_df"] = pd.DataFrame()
st.session_state["started"] = False

# =========================
# Setup Page (2020-21+ Regular only)
# =========================
if page == "Setup":
    st.title("⚙️ Setup")
    st.caption("This app uses Regular Season data only (no Playoffs).")
    st.info("**Data coverage:** Regular Seasons from **2020‑21** to the current season. No Playoffs.")

    # Scoring settings
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

        saved = st.form_submit_button("Save Settings")
        if saved:
            st.session_state["scoring"] = s
            st.success("Settings saved.")

    st.markdown("---")
    st.subheader("Load Data (2020‑21 → current, Regular Season only)")
    if st.button("Let's Start"):
        with st.spinner("Downloading player game logs (2020‑21 to current)..."):
            raw, failures = load_regular_2020_to_current(max_workers=6, timeout_s=10)

        if raw.empty:
            st.error("No data could be loaded. Try again in a moment.")
        else:
            st.session_state["raw_df"] = raw
            st.session_state["started"] = True
            st.success(f"Loaded {len(raw):,} player-game rows. Seasons: {raw['SEASON'].min()} → {raw['SEASON'].max()}.")

        if failures:
            st.warning("Some seasons failed or returned empty and were skipped:")
            st.dataframe(pd.DataFrame(failures, columns=["Season", "Status"]), use_container_width=True)

    if not st.session_state["raw_df"].empty:
        st.markdown("#### Sample of Loaded Data")
        st.dataframe(st.session_state["raw_df"].head(20), use_container_width=True)

        st.markdown("#### Export")
        raw = st.session_state["raw_df"]
        st.download_button(
            "Download full dataset (CSV)",
            data=raw.to_csv(index=False).encode("utf-8"),
            file_name="player_game_logs_2020_to_current.csv",
            mime="text/csv"
        )
        try:
            import pyarrow as pa  # noqa: F401
            parquet_bytes = raw.to_parquet(index=False)
            st.download_button(
                "Download full dataset (Parquet)",
                data=parquet_bytes,
                file_name="player_game_logs_2020_to_current.parquet",
                mime="application/octet-stream"
            )
        except Exception:
            st.caption("Install `pyarrow` to enable Parquet export.")

# =========================
# Records Page
# =========================
elif page == "Records":
    st.title("🏀 Records")
    st.caption("Leaderboards by Season, Career, and Game (Regular Season only, 2020‑present).")

    if not st.session_state.get("started", False) or st.session_state.get("raw_df", pd.DataFrame()).empty:
        st.error("No data available. Please go to the **Setup** page and click **Let's Start**.")
        st.stop()

    raw = st.session_state["raw_df"].copy()
    scoring = st.session_state["scoring"]

    # Compute fantasy points (cached by scoring)
    @st.cache_data(show_spinner=False)
    def compute_fp_cached(df_in: pd.DataFrame, scoring_key: str) -> pd.DataFrame:
        return compute_fantasy_points(df_in, scoring)

    scoring_key = str(sorted([(k, scoring[k]) for k in scoring.keys()]))
    df = compute_fp_cached(raw, scoring_key)

    # Filters
    st.subheader("Filters")
    left, right = st.columns([2, 1])
    with left:
        teams = teams_from_df(df)
        sel_teams = st.multiselect("Select Team(s)", options=teams, default=teams)
    with right:
        top_n = st.selectbox("Top N", options=[10, 15, 20, 50, 100], index=0)

    if sel_teams and len(sel_teams) < len(teams):
        df_f = df[df["TEAM_ABBREVIATION"].isin(sel_teams)].copy()
    else:
        df_f = df.copy()

    st.markdown("---")
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

    # Season Records
    with tab1:
        st.subheader("Season Records")
        seasons = sorted(df_f["SEASON"].dropna().unique().tolist())
        sel_seasons = st.multiselect("Season", options=seasons, default=seasons)

        tmp = df_f[df_f["SEASON"].isin(sel_seasons)].copy()

        season_totals = (
            tmp.groupby(["SEASON", "PLAYER_ID", "PLAYER_NAME"], as_index=False)
               .agg(FP=("fantasy_points", "sum"))
        )
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

    # Career Records
    with tab2:
        st.subheader("Career Records (in 2020‑present window)")
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

    # Game Records
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
        cols_order = ["Rank", "Photo", "Player", "Team", "Season", "Game Date", "Opponent", "Fantasy Points", "PLAYER_ID"]
        for col in cols_order:
            if col not in out.columns:
                out[col] = pd.NA
        out = out[cols_order]
        render_table(out.drop(columns=["PLAYER_ID"]))

# =========================
# Player Insights Page (Generate Insights button) – league-wide position rank
# =========================
elif page == "Player Insights":
    st.title("📊 Player Insights")
    st.caption("Filter by Season, Team and Player. Regular Season only (2020‑present).")

    if not st.session_state.get("started", False) or st.session_state.get("raw_df", pd.DataFrame()).empty:
        st.error("No data available. Please go to the **Setup** page and click **Let's Start**.")
        st.stop()

    raw = st.session_state["raw_df"].copy()
    scoring = st.session_state["scoring"]

    @st.cache_data(show_spinner=False)
    def compute_fp_cached(df_in: pd.DataFrame, scoring_key: str) -> pd.DataFrame:
        return compute_fantasy_points(df_in, scoring)

    scoring_key = str(sorted([(k, scoring[k]) for k in scoring.keys()]))
    df = compute_fp_cached(raw, scoring_key)

    # ---- Filters: Season -> Team -> Player
    seasons = sorted(df["SEASON"].dropna().unique().tolist())
    sel_season = st.selectbox("Season", options=seasons, index=len(seasons)-1)

    df_season = df[df["SEASON"] == sel_season].copy()
    teams = sorted(df_season["TEAM_ABBREVIATION"].dropna().unique().tolist())
    sel_team = st.selectbox("Team", options=teams, index=0)

    df_season_team = df_season[df_season["TEAM_ABBREVIATION"] == sel_team].copy()
    players = df_season_team[["PLAYER_ID","PLAYER_NAME"]].drop_duplicates().sort_values("PLAYER_NAME")
    if players.empty:
        st.warning("No players found for this team/season.")
        st.stop()

    player_display = players["PLAYER_NAME"].tolist()
    player_ids = players["PLAYER_ID"].tolist()
    name_to_id = dict(zip(player_display, player_ids))
    sel_player_name = st.selectbox("Player", options=player_display, index=0)
    sel_player_id = name_to_id[sel_player_name]

    # ---- Button to generate insights
    generate = st.button("Generate Insights")
    if not generate:
        st.info("Set the filters above and click **Generate Insights** to view the visuals.")
        st.stop()

    # Build season-wide positions via team rosters (fast)
    season_team_ids = tuple(sorted(df_season["TEAM_ID"].dropna().astype(int).unique().tolist()))
    pos_map_df = season_positions_map(sel_season, season_team_ids)
    # Fallback para o jogador selecionado, se não vier no roster
    if pos_map_df.empty or sel_player_id not in pos_map_df["PLAYER_ID"].astype(int).tolist():
        fallback_pos = get_player_position(sel_player_id)
        pos_map_df = pd.concat([pos_map_df, pd.DataFrame([{"PLAYER_ID": sel_player_id, "POSITION": fallback_pos}])], ignore_index=True)

    # Headshot + position + KPIs
    colA, colB = st.columns([1, 3])
    with colA:
        st.image(player_headshot_url(sel_player_id), width=220)
        pos_row = pos_map_df[pos_map_df["PLAYER_ID"].astype(int) == int(sel_player_id)]
        pos_str = (pos_row["POSITION"].iloc[0] if not pos_row.empty else "") or "N/A"
        st.write(f"**Position:** {pos_str}")

    # Player games for selection
    p_games = df_season_team[df_season_team["PLAYER_ID"] == sel_player_id].copy()
    if p_games.empty:
        st.warning("No games found for this player/season/team.")
        st.stop()

    # Weekly metrics
    season_start_date = df_season["GAME_DATE"].min()
    p_games["WEEK"] = ((p_games["GAME_DATE"] - season_start_date).dt.days // 7 + 1).astype(int)
    weekly_avg = p_games.groupby("WEEK", as_index=False).agg(avg_fp=("fantasy_points","mean"))
    weekly_max = p_games.groupby("WEEK", as_index=False).agg(max_fp=("fantasy_points","max"))
    avg_fp_season = p_games["fantasy_points"].mean()
    avg_weekly_max = weekly_max["max_fp"].mean() if not weekly_max.empty else float("nan")

    # League-wide overall & position ranks (by avg_fp in season)
    per_player = df_season.groupby(["PLAYER_ID","PLAYER_NAME"], as_index=False).agg(
        GP=("GAME_ID","nunique"),
        avg_fp=("fantasy_points","mean")
    )
    # Attach positions from map
    per_player = per_player.merge(pos_map_df, on="PLAYER_ID", how="left")
    per_player["POS_PRIMARY"] = per_player["POSITION"].fillna("").apply(primary_position_letter)

    # Overall rank
    league_sorted = per_player.sort_values("avg_fp", ascending=False).reset_index(drop=True)
    overall_rank = (league_sorted.index[league_sorted["PLAYER_ID"] == sel_player_id][0] + 1) if sel_player_id in league_sorted["PLAYER_ID"].values else None
    overall_count = len(league_sorted)

    # League-wide position rank (G/F/C bucket)
    p_primary = primary_position_letter(pos_str if pos_str != "N/A" else "")
    in_pos = per_player[per_player["POS_PRIMARY"] == p_primary].sort_values("avg_fp", ascending=False).reset_index(drop=True)
    if not in_pos.empty and sel_player_id in in_pos["PLAYER_ID"].values:
        pos_rank = in_pos.index[in_pos["PLAYER_ID"] == sel_player_id][0] + 1
        pos_count = len(in_pos)
    else:
        pos_rank, pos_count = None, None

    with colB:
        # KPI cards (4 lado a lado)
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(f"""
                <div class="kpi-card">
                  <div class="kpi-title">Overall Rank</div>
                  <div class="kpi-value">{('#'+str(overall_rank)) if overall_rank else 'N/A'}</div>
                  <div class="kpi-sub">out of {overall_count}</div>
                </div>
            """, unsafe_allow_html=True)
        with k2:
            st.markdown(f"""
                <div class="kpi-card">
                  <div class="kpi-title">Position Rank ({p_primary or 'N/A'})</div>
                  <div class="kpi-value">{('#'+str(pos_rank)) if pos_rank else 'N/A'}</div>
                  <div class="kpi-sub">league-wide, out of {pos_count if pos_count else '—'}</div>
                </div>
            """, unsafe_allow_html=True)
        with k3:
            st.markdown(f"""
                <div class="kpi-card">
                  <div class="kpi-title">Avg FPTS (season)</div>
                  <div class="kpi-value">{avg_fp_season:.2f}</div>
                  <div class="kpi-sub">per game</div>
                </div>
            """, unsafe_allow_html=True)
        with k4:
            st.markdown(f"""
                <div class="kpi-card">
                  <div class="kpi-title">Avg Weekly Max</div>
                  <div class="kpi-value">{(0.0 if np.isnan(avg_weekly_max) else float(avg_weekly_max)):.2f}</div>
                  <div class="kpi-sub">mean of weekly best game</div>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Weekly Trend")

    # Two-line chart: weekly average (blue) and weekly max (red) + visual legend
    try:
        import altair as alt
        chart1 = alt.Chart(weekly_avg).mark_line(point=True, color="#1A73E8").encode(
            x=alt.X("WEEK:O", title="Week"),
            y=alt.Y("avg_fp:Q", title="Fantasy Points"),
            tooltip=[alt.Tooltip("WEEK:O"), alt.Tooltip("avg_fp:Q", format=".2f", title="Weekly average")]
        )
        chart2 = alt.Chart(weekly_max).mark_line(point=True, color="#D93025").encode(
            x=alt.X("WEEK:O"),
            y=alt.Y("max_fp:Q"),
            tooltip=[alt.Tooltip("WEEK:O"), alt.Tooltip("max_fp:Q", format=".2f", title="Weekly max")]
        )
        st.altair_chart(chart1 + chart2, use_container_width=True)

        st.markdown(
            """
            <div class="legend">
              <span class="legend-bullet" style="background:#1A73E8"></span>Weekly average
              &nbsp;&nbsp;&nbsp;&nbsp;
              <span class="legend-bullet" style="background:#D93025"></span>Weekly max
            </div>
            """, unsafe_allow_html=True
        )
    except Exception:
        merged = weekly_avg.merge(weekly_max, on="WEEK", how="outer").sort_values("WEEK")
        merged = merged.set_index("WEEK")
        st.line_chart(merged.rename(columns={"avg_fp":"Weekly average", "max_fp":"Weekly max"}))

    st.markdown("---")
    st.subheader("Teammate Impact (with vs without)")
    # Consider only games the player played for this team/season
    team_games = df_season_team[df_season_team["GAME_ID"].isin(p_games["GAME_ID"].unique())].copy()

    # Teammates (exclude the selected player)
    teammate_rows = team_games[team_games["PLAYER_ID"] != sel_player_id][["PLAYER_ID","PLAYER_NAME"]].drop_duplicates()
    results = []
    # Player FP by game
    player_fp_by_game = p_games.groupby("GAME_ID")["fantasy_points"].mean()
    games_all = set(player_fp_by_game.index)

    for _, row in teammate_rows.iterrows():
        tm_id = int(row["PLAYER_ID"])
        tm_name = row["PLAYER_NAME"]

        tm_games_present = set(team_games.loc[team_games["PLAYER_ID"] == tm_id, "GAME_ID"].unique().tolist())
        with_games = games_all.intersection(tm_games_present)
        without_games = games_all.difference(tm_games_present)

        avg_with = player_fp_by_game.loc[list(with_games)].mean() if with_games else np.nan
        avg_without = player_fp_by_game.loc[list(without_games)].mean() if without_games else np.nan

        results.append({"Teammate": tm_name, "With": avg_with, "Without": avg_without})

    impact_df = pd.DataFrame(results).sort_values("With", ascending=False)
    show_df = impact_df.copy()
    for c in ["With","Without"]:
        show_df[c] = show_df[c].map(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
    st.dataframe(show_df, use_container_width=True)

# =========================
# League Insights Page (season filter only here)
# =========================
elif page == "League Insights":
    st.title("📈 League Insights")
    st.caption("Season-wide position analysis and distributions (Regular Season, 2020‑present).")

    if not st.session_state.get("started", False) or st.session_state.get("raw_df", pd.DataFrame()).empty:
        st.error("No data available. Please go to the **Setup** page and click **Let's Start**.")
        st.stop()

    raw = st.session_state["raw_df"].copy()
    scoring = st.session_state["scoring"]

    @st.cache_data(show_spinner=False)
    def compute_fp_cached(df_in: pd.DataFrame, scoring_key: str) -> pd.DataFrame:
        return compute_fantasy_points(df_in, scoring)

    scoring_key = str(sorted([(k, scoring[k]) for k in scoring.keys()]))
    df = compute_fp_cached(raw, scoring_key)

    # Season filter (only here)
    seasons = sorted(df["SEASON"].dropna().unique().tolist())
    sel_season = st.selectbox("Season", options=seasons, index=len(seasons)-1)
    df_s = df[df["SEASON"] == sel_season].copy()

    # Build season-wide positions via team rosters (fast)
    season_team_ids = tuple(sorted(df_s["TEAM_ID"].dropna().astype(int).unique().tolist()))
    pos_map_df = season_positions_map(sel_season, season_team_ids)
    df_s = df_s.merge(pos_map_df, on="PLAYER_ID", how="left")
df_s = df_s[df_s["POS_PRIMARY"] != "U"]
    # Fallback letter (if any missing)
df_s["POS_PRIMARY"] = df_s["POSITION"].fillna("").apply(primary_position_letter)
df_s["POS_PRIMARY"] = df_s["POS_PRIMARY"].replace({"": "U"})  # U = Unknown

st.markdown("### Position boxplot (FPTS by position)")
try:
    import altair as alt
    bp = alt.Chart(df_s).mark_violin().encode(
        x=alt.X("POS_PRIMARY:N", title="Position (primary: G/F/C/U)"),
        y=alt.Y("fantasy_points:Q", title="Fantasy points (per game)"),
        color=alt.Color("POS_PRIMARY:N", legend=None)
        
    )
    st.altair_chart(bp, use_container_width=True)
except Exception:
    st.write("Altair not available, showing simple table sample:")
    st.dataframe(df_s[["PLAYER_NAME","POS_PRIMARY","fantasy_points"]].head(30))

st.markdown("---")
st.markdown("### What stats contribute most to FPTS? (season average per game)")

    # Build per-row contributions (base categories only, bonuses excluded)
s = st.session_state["scoring"]
d = df_s.fillna(0).copy()
contrib = pd.DataFrame({
        "PTS": s["points"] * d["PTS"],
        "AST": s["assist"] * d["AST"],
        "STL": s["steal"] * d["STL"],
        "BLK": s["block"] * d["BLK"],
        "TOV": s["turnover"] * d["TOV"],        # negative weight expected
        "FT Missed": s["ft_missed"] * (d["FTA"] - d["FTM"]).clip(lower=0),
        "3PM": s["three_made"] * d["FG3M"],
        "3PMissed": s["three_missed"] * (d["FG3A"] - d["FG3M"]).clip(lower=0),
        "OREB": s["oreb"] * d["OREB"],
        "DREB": s["dreb"] * d["DREB"],
    })
avg_contrib = contrib.mean().sort_values(ascending=False).reset_index()
avg_contrib.columns = ["Stat", "Avg FPTS contribution"]

try:
    import altair as alt
    bars = alt.Chart(avg_contrib).mark_bar().encode(
        x=alt.X("Avg FPTS contribution:Q", title="Avg FPTS contribution per game"),
        y=alt.Y("Stat:N", sort="-x"),
        color=alt.condition(alt.datum["Avg FPTS contribution"] > 0, alt.value("#1A73E8"), alt.value("#D93025")),
        tooltip=[alt.Tooltip("Stat:N"), alt.Tooltip("Avg FPTS contribution:Q", format=".2f")]
        )
    st.altair_chart(bars, use_container_width=True)
except Exception:
    st.dataframe(avg_contrib)

st.markdown("---")
st.markdown("### Position distribution in Top brackets")

    # Rank players by avg fpts (season), then count POS_PRIMARY in ranges
per_player = df_s.groupby(["PLAYER_ID","PLAYER_NAME","POS_PRIMARY"], as_index=False).agg(
        avg_fp=("fantasy_points","mean"), GP=("GAME_ID","nunique")
    ).sort_values("avg_fp", ascending=False).reset_index(drop=True)

def count_pos_in_range(dfpp, start, end):
    sl = dfpp.iloc[start:end]
    return sl["POS_PRIMARY"].value_counts().rename(f"{start+1}-{end}")

top50 = count_pos_in_range(per_player, 0, 50)
top50_100 = count_pos_in_range(per_player, 50, 100)
top100_150 = count_pos_in_range(per_player, 100, 150)
dist = pd.concat([top50, top50_100, top100_150], axis=1).fillna(0).astype(int)
dist = dist.reindex(["G","F","C"]).fillna(0).astype(int)

st.dataframe(dist.reset_index().rename(columns={"index":"Pos"}), use_container_width=True)


# === Analytical Insights ===
st.markdown("### Insights")

st.markdown("#### Violin Plot Insight")
st.info("The violin plot reveals the distribution of fantasy points across player positions. Guards (G) tend to have a wider spread and higher median FPTS, indicating variability and potential for high performance. Forwards (F) and Centers (C) show more consistent distributions, with Centers generally having slightly higher median FPTS.")

st.markdown("#### FPTS Contribution Insight")
st.info("The bar chart of average FPTS contributions highlights that points scored (PTS), assists (AST), and rebounds (OREB/DREB) are the most significant contributors to fantasy points. Turnovers (TOV) and missed 3-pointers negatively impact FPTS, emphasizing the importance of efficiency.")
st.markdown("---")
st.markdown("### Top 50 – game count by FPTS buckets (heatmap)")

    # ====== PATCHED SECTION (fix KeyError with MultiIndex) ======
    # Faixas (bins)
bins = [-1, 20, 25, 30, 35, 40, 45, 50, 1e9]
labels = ["<20","20-25","25-30","30-35","35-40","40-45","45-50","50+"]

    # Top 50 por média de FPTS na temporada selecionada
per_player_top = df_s.groupby(["PLAYER_ID","PLAYER_NAME","POS_PRIMARY"], as_index=False) \
                         .agg(avg_fp=("fantasy_points","mean"), GP=("GAME_ID","nunique")) \
                         .sort_values("avg_fp", ascending=False).reset_index(drop=True)

top50_ids = per_player_top.head(50)["PLAYER_ID"].astype(int).tolist()

    # Filtrar jogos desses jogadores
df_top = df_s[df_s["PLAYER_ID"].astype(int).isin(top50_ids)].copy()
df_top["bucket"] = pd.cut(df_top["fantasy_points"], bins=bins, labels=labels)

    # Contagem por faixa (somente PLAYER_ID no índice)
tmp_counts = (
        df_top.groupby(["PLAYER_ID","bucket"])
              .size()
              .unstack(fill_value=0)
              .reindex(columns=labels, fill_value=0)
              .reset_index()
    )

    # Colar nomes (merge) e reordenar na ordem do Top 50 (por avg_fp)
names_map = per_player_top[["PLAYER_ID","PLAYER_NAME"]].drop_duplicates()
counts = tmp_counts.merge(names_map, on="PLAYER_ID", how="left")

    # Reordenar pelo ranking do per_player_top
order_idx = pd.Index(top50_ids, dtype=int)
counts["PLAYER_ID"] = counts["PLAYER_ID"].astype(int)
counts = counts.set_index("PLAYER_ID").reindex(order_idx).reset_index()

    # Colocar PLAYER_NAME como primeira coluna
counts.insert(0, "PLAYER_NAME", counts.pop("PLAYER_NAME"))

    # Preencher faltas de nome (raro) com '—'
counts["PLAYER_NAME"] = counts["PLAYER_NAME"].fillna("—")

    # Heatmap por coluna (gradiente)
heat_cols = labels
styled = counts[["PLAYER_NAME"] + heat_cols] \
        .style \
        .background_gradient(axis=0, cmap="YlOrRd", subset=heat_cols) \
        .format(na_rep="0")

st.dataframe(styled, use_container_width=True)
    # ====== END PATCHED SECTION ======

# =========================
# Chat Page (GPT‑3.5) – opcional
# =========================
elif page == "Chat":
    st.title("💬 Fantasy NBA Chat")
    st.caption("Ask anything about NBA Fantasy. This uses OpenAI GPT‑3.5. Free usage depends on your account limits.")

    if "OPENAI_API_KEY" not in st.session_state:
        st.session_state["OPENAI_API_KEY"] = st.secrets.get("OPENAI_API_KEY", "")

    with st.expander("API Key (optional override)"):
        key_in = st.text_input("OpenAI API Key (stored only for this session)", type="password", value=st.session_state["OPENAI_API_KEY"])
        if st.button("Use this key for this session"):
            st.session_state["OPENAI_API_KEY"] = key_in
            st.success("API key set for this session.")

    if st.session_state["OPENAI_API_KEY"]:
        openai.api_key = st.session_state["OPENAI_API_KEY"]

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask me anything about NBA Fantasy..."):
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                resp = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role":"system","content":"You are an NBA Fantasy expert assistant."}] + st.session_state["messages"],
                    temperature=0.7,
                    max_tokens=500
                )
                reply = resp.choices[0].message["content"]
            except Exception as e:
                reply = f"⚠️ Sorry, I couldn't reach the chat service. Error: {e}\n\nTip: Check your API key and quota."
            st.markdown(reply)
            st.session_state["messages"].append({"role": "assistant", "content": reply})

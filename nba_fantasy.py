import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import json
import os
from typing import Dict, List, Optional

# Configuração da página
st.set_page_config(
    page_title="NFL Fantasy Analytics",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado para design profissional
st.markdown("""
<style>
    /* Tema principal */
    .main {
        background: linear-gradient(135deg, #1e3a5f 0%, #2c5282 100%);
        color: white;
    }
    
    /* Header customizado */
    .header-container {
        background: linear-gradient(90deg, #1e3a5f 0%, #2c5282 100%);
        padding: 1rem 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .header-title {
        font-size: 2.5rem;
        font-weight: bold;
        color: white;
        text-align: center;
        margin: 0;
    }
    
    .header-subtitle {
        font-size: 1.2rem;
        color: #e2e8f0;
        text-align: center;
        margin: 0.5rem 0 0 0;
    }
    
    /* Cards de estatísticas */
    .stat-card {
        background: rgba(255, 255, 255, 0.1);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #e74c3c;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
    }
    
    .stat-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #ffffff;
        margin: 0;
    }
    
    .stat-label {
        font-size: 1rem;
        color: #cbd5e0;
        margin: 0.5rem 0 0 0;
    }
    
    /* Sidebar customizada */
    .css-1d391kg {
        background: linear-gradient(180deg, #2d3748 0%, #1a202c 100%);
    }
    
    /* Filtros */
    .filter-container {
        background: rgba(255, 255, 255, 0.05);
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Métricas */
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        color: white;
    }
    
    /* Botões */
    .stButton > button {
        background: linear-gradient(90deg, #e74c3c 0%, #c0392b 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    
    .stButton > button:hover {
        background: linear-gradient(90deg, #c0392b 0%, #a93226 100%);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    /* Selectbox */
    .stSelectbox > div > div {
        background-color: rgba(255, 255, 255, 0.1);
        color: white;
    }
    
    /* Multiselect */
    .stMultiSelect > div > div {
        background-color: rgba(255, 255, 255, 0.1);
        color: white;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255, 255, 255, 0.1);
        color: white;
        border-radius: 5px 5px 0 0;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #e74c3c;
        color: white;
    }
    
    /* Esconder elementos do Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Carrega os dados da NFL"""
    
    data_dir = "/home/ubuntu/nfl_data"
    
    try:
        # Carregar dados consolidados
        df = pd.read_csv(f"{data_dir}/consolidated_fantasy_data.csv")
        
        # Carregar dados de times
        teams_df = pd.read_csv(f"{data_dir}/team_data.csv")
        
        # Carregar resumo
        with open(f"{data_dir}/data_summary.json", 'r') as f:
            summary = json.load(f)
        
        return df, teams_df, summary
        
    except Exception as e:
        st.error(f"Erro ao carregar dados: {str(e)}")
        return None, None, None

def create_header():
    """Cria o header da aplicação"""
    
    st.markdown("""
    <div class="header-container">
        <h1 class="header-title">🏈 NFL Fantasy Analytics</h1>
        <p class="header-subtitle">Dashboard Avançado de Análise de Fantasy Football (2010-2024)</p>
    </div>
    """, unsafe_allow_html=True)

def create_sidebar_filters(df: pd.DataFrame, teams_df: pd.DataFrame):
    """Cria filtros na sidebar"""
    
    st.sidebar.markdown("## 🎯 Filtros")
    
    # Filtro de temporada
    seasons = sorted(df['season'].unique(), reverse=True)
    selected_seasons = st.sidebar.multiselect(
        "📅 Temporadas",
        options=seasons,
        default=seasons[:3],  # Últimas 3 temporadas por padrão
        help="Selecione as temporadas para análise"
    )
    
    # Filtro de posição
    positions = ['QB', 'RB', 'WR', 'TE']
    selected_positions = st.sidebar.multiselect(
        "🎯 Posições",
        options=positions,
        default=positions,
        help="Selecione as posições para análise"
    )
    
    # Filtro de times
    teams = sorted(df['recent_team'].unique())
    selected_teams = st.sidebar.multiselect(
        "🏈 Times",
        options=teams,
        default=teams,
        help="Selecione os times para análise"
    )
    
    # Filtro de tipo de temporada
    season_type = st.sidebar.selectbox(
        "📊 Tipo de Temporada",
        options=['REG', 'POST', 'ALL'],
        index=0,
        help="Tipo de jogos para análise"
    )
    
    # Filtro de semanas
    if selected_seasons:
        filtered_df = df[df['season'].isin(selected_seasons)]
        weeks = sorted(filtered_df['week'].unique())
        selected_weeks = st.sidebar.multiselect(
            "📈 Semanas",
            options=weeks,
            default=weeks,
            help="Selecione as semanas para análise"
        )
    else:
        selected_weeks = []
    
    return {
        'seasons': selected_seasons,
        'positions': selected_positions,
        'teams': selected_teams,
        'season_type': season_type,
        'weeks': selected_weeks
    }

def filter_data(df: pd.DataFrame, filters: Dict):
    """Aplica filtros aos dados"""
    
    filtered_df = df.copy()
    
    if filters['seasons']:
        filtered_df = filtered_df[filtered_df['season'].isin(filters['seasons'])]
    
    if filters['positions']:
        filtered_df = filtered_df[filtered_df['position'].isin(filters['positions'])]
    
    if filters['teams']:
        filtered_df = filtered_df[filtered_df['recent_team'].isin(filters['teams'])]
    
    if filters['weeks']:
        filtered_df = filtered_df[filtered_df['week'].isin(filters['weeks'])]
    
    return filtered_df

def create_overview_metrics(df: pd.DataFrame):
    """Cria métricas de visão geral"""
    
    st.markdown("## 📊 Visão Geral")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_players = df['player_id'].nunique()
        st.markdown(f"""
        <div class="stat-card">
            <p class="stat-value">{total_players:,}</p>
            <p class="stat-label">Jogadores</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        total_games = len(df)
        st.markdown(f"""
        <div class="stat-card">
            <p class="stat-value">{total_games:,}</p>
            <p class="stat-label">Jogos</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        total_seasons = df['season'].nunique()
        st.markdown(f"""
        <div class="stat-card">
            <p class="stat-value">{total_seasons}</p>
            <p class="stat-label">Temporadas</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        total_teams = df['recent_team'].nunique()
        st.markdown(f"""
        <div class="stat-card">
            <p class="stat-value">{total_teams}</p>
            <p class="stat-label">Times</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        avg_fantasy_points = df['fantasy_points_ppr'].mean()
        st.markdown(f"""
        <div class="stat-card">
            <p class="stat-value">{avg_fantasy_points:.1f}</p>
            <p class="stat-label">Média PPR</p>
        </div>
        """, unsafe_allow_html=True)

def create_player_profile(df: pd.DataFrame, teams_df: pd.DataFrame):
    """Cria seção de perfil do jogador"""
    
    st.markdown("## 👤 Perfil do Jogador")
    
    # Seletor de jogador
    players = df.groupby(['player_display_name', 'position', 'recent_team']).agg({
        'fantasy_points_ppr': 'sum',
        'season': 'max'
    }).reset_index()
    
    players = players.sort_values('fantasy_points_ppr', ascending=False)
    
    player_options = [f"{row['player_display_name']} ({row['position']}) - {row['recent_team']}" 
                     for _, row in players.head(100).iterrows()]
    
    selected_player_str = st.selectbox(
        "🔍 Selecionar Jogador",
        options=player_options,
        help="Selecione um jogador para análise detalhada"
    )
    
    if selected_player_str:
        # Extrair informações do jogador
        player_name = selected_player_str.split(' (')[0]
        player_data = df[df['player_display_name'] == player_name].copy()
        
        if not player_data.empty:
            # Informações básicas do jogador
            latest_data = player_data.sort_values('season', ascending=False).iloc[0]
            
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col1:
                # Foto do jogador (placeholder)
                st.markdown(f"""
                <div style="text-align: center; padding: 1rem; background: rgba(255,255,255,0.1); border-radius: 10px;">
                    <div style="width: 100px; height: 100px; background: #e74c3c; border-radius: 50%; margin: 0 auto; display: flex; align-items: center; justify-content: center; font-size: 2rem;">
                        {latest_data['position']}
                    </div>
                    <h3 style="color: white; margin: 1rem 0 0 0;">{latest_data['player_display_name']}</h3>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                # Logo do time e estatísticas principais
                team_info = teams_df[teams_df['team_abbr'] == latest_data['recent_team']]
                
                if not team_info.empty:
                    team_logo = team_info.iloc[0]['team_logo_espn']
                    team_name = team_info.iloc[0]['team_name']
                    
                    st.markdown(f"""
                    <div style="text-align: center; padding: 1rem;">
                        <img src="{team_logo}" width="80" style="margin-bottom: 1rem;">
                        <h4 style="color: white; margin: 0;">{team_name}</h4>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Estatísticas principais
                total_fantasy_points = player_data['fantasy_points_ppr'].sum()
                total_games = len(player_data)
                avg_points_per_game = total_fantasy_points / total_games if total_games > 0 else 0
                
                col2_1, col2_2, col2_3 = st.columns(3)
                
                with col2_1:
                    st.metric("Total PPR", f"{total_fantasy_points:.1f}")
                
                with col2_2:
                    st.metric("Jogos", total_games)
                
                with col2_3:
                    st.metric("PPR/Jogo", f"{avg_points_per_game:.1f}")
            
            with col3:
                # Estatísticas específicas por posição
                position = latest_data['position']
                
                if position == 'QB':
                    passing_yards = player_data['passing_yards'].sum()
                    passing_tds = player_data['passing_tds'].sum()
                    st.metric("Jardas Passe", f"{passing_yards:,.0f}")
                    st.metric("TDs Passe", f"{passing_tds:.0f}")
                
                elif position in ['RB']:
                    rushing_yards = player_data['rushing_yards'].sum()
                    rushing_tds = player_data['rushing_tds'].sum()
                    st.metric("Jardas Corrida", f"{rushing_yards:,.0f}")
                    st.metric("TDs Corrida", f"{rushing_tds:.0f}")
                
                elif position in ['WR', 'TE']:
                    receiving_yards = player_data['receiving_yards'].sum()
                    receiving_tds = player_data['receiving_tds'].sum()
                    receptions = player_data['receptions'].sum()
                    st.metric("Recepções", f"{receptions:.0f}")
                    st.metric("Jardas Recepção", f"{receiving_yards:,.0f}")
                    st.metric("TDs Recepção", f"{receiving_tds:.0f}")
            
            return player_data
    
    return None

def create_teammate_impact_analysis(player_data: pd.DataFrame, all_data: pd.DataFrame):
    """
    Cria a seção de análise de impacto do companheiro de equipe,
    incluindo a contagem de jogos com e sem o companheiro.
    """
    st.markdown("### 🤝 Teammate Impact (with vs without)")
    
    # 1. Identificar possíveis companheiros de equipe (jogadores no mesmo time/semana/temporada)
    # Excluir o próprio jogador
    player_id = player_data['player_id'].iloc[0]
    
    # Criar um DataFrame de jogos do jogador principal
    player_games = player_data[['game_id', 'season', 'week', 'recent_team', 'fantasy_points_ppr']].copy()
    player_games.rename(columns={'fantasy_points_ppr': 'player_fpts_with'}, inplace=True)
    
    # Filtrar todos os dados para jogos do mesmo time e temporada
    team_games = all_data[
        (all_data['recent_team'].isin(player_games['recent_team'].unique())) &
        (all_data['season'].isin(player_games['season'].unique())) &
        (all_data['player_id'] != player_id)
    ].copy()
    
    # Agrupar por companheiro de equipe
    teammate_stats = team_games.groupby('player_id').agg(
        teammate_name=('player_display_name', 'first'),
        teammate_position=('position', 'first'),
        teammate_team=('recent_team', 'first'),
        total_games_teammate=('game_id', 'nunique')
    ).reset_index()
    
    # 2. Calcular impacto
    impact_data = []
    
    # Limitar aos companheiros com um número mínimo de jogos juntos (ex: 5)
    min_games_together = 5 
    
    for index, row in teammate_stats.iterrows():
        teammate_id = row['player_id']
        
        # Jogos com o companheiro (teammate_id estava em campo)
        games_with = team_games[team_games['player_id'] == teammate_id]['game_id'].unique()
        df_with = player_games[player_games['game_id'].isin(games_with)]
        
        # Jogos sem o companheiro (teammate_id não estava em campo, mas o jogador principal estava)
        all_player_games = player_games['game_id'].unique()
        games_without = [g for g in all_player_games if g not in games_with]
        df_without = player_games[player_games['game_id'].isin(games_without)]
        
        count_with = len(df_with)
        count_without = len(df_without)
        
        if count_with >= min_games_together:
            avg_fpts_with = df_with['player_fpts_with'].mean()
            avg_fpts_without = df_without['player_fpts_with'].mean()
            
            impact = avg_fpts_with - avg_fpts_without
            
            impact_data.append({
                'Teammate': f"{row['teammate_name']} ({row['teammate_position']})",
                'Team': row['teammate_team'],
                'Avg FPTS With': f"{avg_fpts_with:.1f}",
                'Avg FPTS Without': f"{avg_fpts_without:.1f}",
                'Impact (With - Without)': f"{impact:.1f}",
                'Games With': count_with, # NOVO: Contagem de jogos com
                'Games Without': count_without # NOVO: Contagem de jogos sem
            })
            
    if impact_data:
        impact_df = pd.DataFrame(impact_data)
        impact_df['Impact (With - Without)'] = impact_df['Impact (With - Without)'].astype(float)
        impact_df = impact_df.sort_values('Impact (With - Without)', ascending=False).head(10)
        
        st.dataframe(
            impact_df,
            column_order=['Teammate', 'Team', 'Games With', 'Games Without', 'Avg FPTS With', 'Avg FPTS Without', 'Impact (With - Without)'],
            column_config={
                'Impact (With - Without)': st.column_config.Progress(
                    "Impact (With - Without)",
                    format="%.1f",
                    min_value=impact_df['Impact (With - Without)'].min(),
                    max_value=impact_df['Impact (With - Without)'].max(),
                ),
                'Games With': "Jogos Com",
                'Games Without': "Jogos Sem",
                'Avg FPTS With': "Média FPTS Com",
                'Avg FPTS Without': "Média FPTS Sem",
            },
            hide_index=True,
            use_container_width=True
        )
        st.info(f"💡 **Nota**: A análise considera companheiros que jogaram pelo menos {min_games_together} jogos com o jogador principal.")
    else:
        st.info("Não há dados de impacto de companheiros de equipe suficientes para exibir.")

def create_player_kpis(player_data: pd.DataFrame, all_data: pd.DataFrame):
    """
    Cria a seção de KPIs de Desempenho e Rankings do jogador,
    incluindo a posição em FPTS gerados por categoria estatística.
    """
    st.markdown("### 🏆 KPIs de Desempenho e Rankings")
    
    # 1. Definir as colunas de estatísticas e seus pesos em FPTS (PPR)
    # Baseado em regras PPR (1 ponto por recepção, 6 por TD, 1 ponto a cada 10 jardas, -2 por INT/Fumble)
    # Para simplificar, focaremos nas colunas de estatísticas que mais contribuem para FPTS
    
    stat_cols = {
        'passing_yards': {'label': 'Jardas Passe', 'fpts_per_unit': 0.04}, # 1/25
        'passing_tds': {'label': 'TDs Passe', 'fpts_per_unit': 4},
        'rushing_yards': {'label': 'Jardas Corrida', 'fpts_per_unit': 0.1}, # 1/10
        'rushing_tds': {'label': 'TDs Corrida', 'fpts_per_unit': 6},
        'receiving_yards': {'label': 'Jardas Recepção', 'fpts_per_unit': 0.1}, # 1/10
        'receiving_tds': {'label': 'TDs Recepção', 'fpts_per_unit': 6},
        'receptions': {'label': 'Recepções', 'fpts_per_unit': 1},
    }
    
    # Estatísticas negativas (perda de pontos)
    neg_stat_cols = {
        'interceptions': {'label': 'INTs (-2)', 'fpts_per_unit': -2},
        'fumbles_lost': {'label': 'Fumbles Perdidos (-2)', 'fpts_per_unit': -2},
    }
    
    # 2. Calcular FPTS médios por jogo gerados por cada categoria para TODOS os jogadores
    
    # Agrupar dados de todos os jogadores por jogo
    all_players_agg = all_data.groupby('player_id').agg(
        player_name=('player_display_name', 'first'),
        position=('position', 'first'),
        games_played=('game_id', 'nunique'),
        **{col: ('mean') for col in stat_cols.keys()},
        **{col: ('mean') for col in neg_stat_cols.keys()}
    ).reset_index()
    
    # Filtrar jogadores com pelo menos 3 jogos
    all_players_agg = all_players_agg[all_players_agg['games_played'] >= 3].copy()
    
    # Calcular FPTS médios gerados por categoria
    for col, info in stat_cols.items():
        all_players_agg[f'fpts_avg_by_{col}'] = all_players_agg[col] * info['fpts_per_unit']
    
    for col, info in neg_stat_cols.items():
        all_players_agg[f'fpts_avg_by_{col}'] = all_players_agg[col] * info['fpts_per_unit']
        
    # 3. Determinar o ranking (posição) do jogador principal em cada categoria
    
    player_id = player_data['player_id'].iloc[0]
    player_position = player_data['position'].iloc[0]
    
    # Filtrar apenas jogadores da mesma posição
    pos_agg = all_players_agg[all_players_agg['position'] == player_position].copy()
    
    kpi_data = []
    
    # KPIs positivos (maior é melhor)
    for col, info in stat_cols.items():
        fpts_col = f'fpts_avg_by_{col}'
        
        # Calcular ranking (posição)
        pos_agg['rank'] = pos_agg[fpts_col].rank(ascending=False, method='min')
        
        player_row = pos_agg[pos_agg['player_id'] == player_id].iloc[0]
        
        kpi_data.append({
            'KPI': f"FPTS/Jogo por {info['label']}",
            'Valor do Jogador': f"{player_row[fpts_col]:.2f}",
            'Posição (Rank)': f"{int(player_row['rank'])}/{len(pos_agg)}",
            'Tipo': 'Positivo'
        })

    # KPIs negativos (menor é melhor)
    for col, info in neg_stat_cols.items():
        fpts_col = f'fpts_avg_by_{col}'
        
        # Calcular ranking (posição) - menor valor absoluto (mais próximo de 0) é melhor
        # Como os valores são negativos, o rank deve ser crescente (ascending=True)
        pos_agg['rank'] = pos_agg[fpts_col].rank(ascending=True, method='min')
        
        player_row = pos_agg[pos_agg['player_id'] == player_id].iloc[0]
        
        kpi_data.append({
            'KPI': f"Perda FPTS/Jogo por {info['label'].split('(')[0].strip()}",
            'Valor do Jogador': f"{player_row[fpts_col]:.2f}", # Valor negativo
            'Posição (Rank)': f"{int(player_row['rank'])}/{len(pos_agg)}",
            'Tipo': 'Negativo'
        })
        
    # 4. Exibir KPIs
    kpi_df = pd.DataFrame(kpi_data)
    
    st.dataframe(
        kpi_df,
        column_config={
            'KPI': 'Métrica',
            'Valor do Jogador': 'Média FPTS/Jogo',
            'Posição (Rank)': f'Posição entre {player_position}s',
        },
        hide_index=True,
        use_container_width=True
    )
    st.info(f"💡 **Interpretação**: O ranking indica a posição do jogador em relação a todos os outros jogadores da posição {player_position} com pelo menos 3 jogos, classificado pela média de FPTS/Jogo gerada pela métrica.")

def create_league_insights(df: pd.DataFrame, teams_df: pd.DataFrame):
    """
    Cria a seção League Insights com tabelas de FPTS cedidos por time por categoria.
    """
    st.markdown("### 🌐 League Insights: Defesas vs. Posições")
    
    # 1. Definir as categorias de estatísticas defensivas
    # Usaremos as mesmas estatísticas de contribuição de FPTS para calcular o que a defesa cede
    
    stat_cols = {
        'fantasy_points_ppr': 'FPTS Total',
        'passing_yards': 'Jardas Passe',
        'rushing_yards': 'Jardas Corrida',
        'receiving_yards': 'Jardas Recepção',
        'receptions': 'Recepções',
        'passing_tds': 'TDs Passe',
        'rushing_tds': 'TDs Corrida',
        'receiving_tds': 'TDs Recepção',
        'interceptions': 'INTs (Negativo)',
        'fumbles_lost': 'Fumbles Perdidos (Negativo)',
    }
    
    # Mapeamento para as tabelas solicitadas pelo usuário (adaptado para NFL)
    user_stat_map = {
        'fantasy_points_ppr': 'Total FPTS',
        'passing_yards': 'Passe',
        'rushing_yards': 'Corrida',
        'receiving_yards': 'Recepção',
        'receptions': 'Recepções',
        'passing_tds': 'TDs Passe',
        'rushing_tds': 'TDs Corrida',
        'receiving_tds': 'TDs Recepção',
    }
    
    # 2. Calcular FPTS médios cedidos por jogo pela defesa (opponent_team)
    
    # Agrupar por time adversário (defesa)
    defense_agg = df.groupby(['opponent_team']).agg(
        games_played=('game_id', 'nunique'),
        **{col: ('sum') for col in stat_cols.keys()}
    ).reset_index()
    
    # Calcular média por jogo
    for col in stat_cols.keys():
        defense_agg[f'avg_{col}_cedido'] = defense_agg[col] / defense_agg['games_played']
        
    # 3. Criar e exibir as tabelas
    
    # Dicionário para mapear abreviação do time para nome completo
    team_name_map = teams_df.set_index('team_abbr')['team_name'].to_dict()
    defense_agg['Team Name'] = defense_agg['opponent_team'].map(team_name_map)
    
    # Colunas a serem exibidas nas tabelas
    display_cols = ['Team Name', 'avg_fantasy_points_ppr_cedido']
    
    # Tabela 1: Média de FPTS cedidos por partida (Total)
    st.markdown("#### 🏈 FPTS Cedidos por Partida (Total)")
    fpts_table = defense_agg.sort_values('avg_fantasy_points_ppr_cedido', ascending=False)[display_cols].copy()
    fpts_table.columns = ['Equipe', 'Média FPTS Cedidos']
    st.dataframe(fpts_table.head(15).style.format({'Média FPTS Cedidos': "{:.1f}"}), hide_index=True, use_container_width=True)
    
    # Tabelas por categoria estatística
    
    st.markdown("#### 📊 FPTS Cedidos por Categoria Estatística")
    
    # Usar colunas para organizar as tabelas (3 colunas por linha)
    cols = st.columns(3)
    col_idx = 0
    
    for stat_col, stat_label in user_stat_map.items():
        
        # Coluna de média cedida
        avg_col = f'avg_{stat_col}_cedido'
        
        # Título da tabela
        title = f"Média {stat_label} Cedidos"
        
        # Tabela
        table_data = defense_agg.sort_values(avg_col, ascending=False)[['Team Name', avg_col]].copy()
        table_data.columns = ['Equipe', 'Média Cedida']
        
        with cols[col_idx % 3]:
            st.markdown(f"**{title}**")
            st.dataframe(table_data.head(10).style.format({'Média Cedida': "{:.1f}"}), hide_index=True, use_container_width=True)
        
        col_idx += 1
        
    # Tabela de Perda de Pontos (Turnovers)
    
    st.markdown("#### 🚨 Perda de Pontos por Turnovers")
    
    # Calcular FPTS de Turnovers cedidos (INTs e Fumbles)
    defense_agg['avg_turnover_fpts_cedido'] = (
        defense_agg['avg_interceptions_cedido'] * -2 + 
        defense_agg['avg_fumbles_lost_cedido'] * -2
    )
    
    # O time que cede MAIS perda de pontos (valor absoluto negativo maior) é o PIOR
    # Queremos do maior para o menor (do mais negativo para o menos negativo)
    turnover_table = defense_agg.sort_values('avg_turnover_fpts_cedido', ascending=True)[['Team Name', 'avg_turnover_fpts_cedido']].copy()
    turnover_table.columns = ['Equipe', 'Média FPTS Perdidos Cedidos']
    
    st.dataframe(
        turnover_table.head(10).style.format({'Média FPTS Perdidos Cedidos': "{:.1f}"}), 
        hide_index=True, 
        use_container_width=True
    )
    st.info("💡 **Interpretação**: Quanto mais negativo o valor, mais FPTS a defesa adversária gera por turnovers (INTs e Fumbles Perdidos).")

def main():
    """Função principal da aplicação"""
    
    # Criar header
    create_header()
    
    # Carregar dados
    with st.spinner("🔄 Carregando dados da NFL..."):
        df, teams_df, summary = load_data()
    
    if df is None:
        st.error("❌ Não foi possível carregar os dados. Verifique se os arquivos estão disponíveis.")
        return
    
    # Criar filtros na sidebar
    filters = create_sidebar_filters(df, teams_df)
    
    # Aplicar filtros
    filtered_df = filter_data(df, filters)
    
    if filtered_df.empty:
        st.warning("⚠️ Nenhum dado encontrado com os filtros selecionados.")
        return
    
    # Criar métricas de visão geral
    create_overview_metrics(filtered_df)
    
    # Criar tabs principais
    tab1, tab2, tab3 = st.tabs([
        "👤 Player Insights", 
        "📊 Rankings & Tendências", 
        "🌐 League Insights"
    ])
    
    with tab1:
        # 1. Perfil do Jogador
        player_data = create_player_profile(filtered_df, teams_df)
        
        if player_data is not None:
            # 2. Gráficos do jogador (usando placeholders para as funções originais)
            st.markdown("### 📈 Performance ao Longo do Tempo (Placeholder)")
            st.info("Visualizações de timeline e dual-bar chart seriam exibidas aqui.")
            
            # 3. Teammate Impact (Com as novas colunas)
            create_teammate_impact_analysis(player_data, filtered_df)
            
            # 4. Player KPIs (Novos KPIs)
            create_player_kpis(player_data, filtered_df)
            
    with tab2:
        st.markdown("### 🏆 Rankings por Posição (Placeholder)")
        st.info("Rankings e análises de tendência por posição seriam exibidos aqui.")
        
    with tab3:
        # League Insights (Novas tabelas)
        create_league_insights(filtered_df, teams_df)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #cbd5e0; padding: 1rem;">
        <p>🏈 NFL Fantasy Analytics Dashboard | Dados: nfl-data-py | Desenvolvido com Streamlit</p>
        <p>📅 Dados atualizados até: {}</p>
    </div>
    """.format(datetime.now().strftime("%d/%m/%Y")), unsafe_allow_html=True)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
NFL Fantasy Football Analytics Dashboard
Aplicação Streamlit para análise avançada de fantasy football da NFL
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
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
            
            
# KPIs adicionais de fantasy points por tipo de ação
metrics = {
    'FPTS por Pontos': 'points',
    'FPTS por Rebotes Ofensivos': 'offensive_rebounds',
    'FPTS por Rebotes Defensivos': 'defensive_rebounds',
    'FPTS por Bloqueios': 'blocks',
    'FPTS por Roubos': 'steals',
    'FPTS por Assistências': 'assists',
    'FPTS perdidos por Turnovers': 'turnovers',
    'FPTS perdidos por Faltas': 'fumbles'
}

for label, col in metrics.items():
    if col in player_data.columns:
        avg = player_data[col].mean()
        st.metric(label, f"{avg:.2f}")

    return player_data
    
    return None

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
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "👤 Perfil do Jogador", 
        "📊 Rankings", 
        "📈 Tendências", 
        "🆚 Comparações",
        "🔍 Insights Avançados"
    ])
    
    with tab1:
        player_data = create_player_profile(filtered_df, teams_df)
        
        if player_data is not None:
            # Gráficos do jogador
            st.markdown("### 📈 Performance ao Longo do Tempo")
            
            from visualizations import (
                create_player_timeline_chart, create_dual_bar_chart, 
                create_stacked_bar_with_line
            )
            
            # Criar visualizações do jogador
            col1, col2 = st.columns(2)
            
            with col1:
                # Gráfico de timeline similar ao design de referência
                timeline_fig = create_player_timeline_chart(player_data, 'fantasy_points_ppr')
                st.plotly_chart(timeline_fig, use_container_width=True)
            
            with col2:
                # Gráfico de barras duplas para métricas específicas da posição
                position = player_data['position'].iloc[0]
                
                if position == 'QB':
                    dual_fig = create_dual_bar_chart(player_data, 'passing_yards', 'rushing_yards')
                elif position == 'RB':
                    dual_fig = create_dual_bar_chart(player_data, 'rushing_yards', 'receiving_yards')
                elif position in ['WR', 'TE']:
                    dual_fig = create_dual_bar_chart(player_data, 'receiving_yards', 'targets')
                else:
                    dual_fig = create_dual_bar_chart(player_data, 'fantasy_points_ppr', 'fantasy_points')
                
                st.plotly_chart(dual_fig, use_container_width=True)
            
            # Gráfico adicional de análise avançada
            if position == 'QB':
                st.markdown("#### Análise de Eficiência (TD/TO Ratio)")
                efficiency_fig = create_stacked_bar_with_line(player_data)
                st.plotly_chart(efficiency_fig, use_container_width=True)
    
    with tab2:
        st.markdown("### 🏆 Rankings por Posição")
        
        # Rankings por posição
        for position in filters['positions']:
            pos_data = filtered_df[filtered_df['position'] == position]
            
            if not pos_data.empty:
                st.markdown(f"#### {position}")
                
                # Top 10 jogadores da posição
                top_players = pos_data.groupby(['player_display_name', 'recent_team']).agg({
                    'fantasy_points_ppr': 'sum',
                    'fantasy_points': 'sum'
                }).reset_index().sort_values('fantasy_points_ppr', ascending=False).head(10)
                
                # Criar gráfico de barras
                fig = px.bar(
                    top_players,
                    x='fantasy_points_ppr',
                    y='player_display_name',
                    orientation='h',
                    title=f"Top 10 {position} - Fantasy Points PPR",
                    labels={'fantasy_points_ppr': 'Fantasy Points PPR', 'player_display_name': 'Jogador'}
                )
                
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white',
                    yaxis={'categoryorder': 'total ascending'}
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("### 📈 Análise de Tendências")
        
        # Tendências por temporada
        season_trends = filtered_df.groupby(['season', 'position']).agg({
            'fantasy_points_ppr': 'mean'
        }).reset_index()
        
        fig = px.line(
            season_trends,
            x='season',
            y='fantasy_points_ppr',
            color='position',
            title="Média de Fantasy Points PPR por Temporada",
            labels={'season': 'Temporada', 'fantasy_points_ppr': 'Média Fantasy Points PPR'}
        )
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        from player_comparison import create_player_comparison_interface
        
        create_player_comparison_interface(filtered_df)
    
    with tab5:

st.markdown("### 🏈 League Insights - FPTS cedidos por tipo de ação")

actions = {
    'Pontos': 'points_allowed',
    'Rebotes': 'rebounds_allowed',
    'Assistências': 'assists_allowed',
    'Roubos': 'steals_allowed',
    'Bloqueios': 'blocks_allowed',
    'Turnovers': 'turnovers_forced',
    'Roubos que geram perda de pontos': 'steals_points_lost'
}

for label, col in actions.items():
    if col in teams_df.columns:
        table_df = teams_df[['team_name', col]].copy()
        table_df = table_df.rename(columns={col: f'FPTS cedidos por {label}'})
        table_df = table_df.sort_values(by=f'FPTS cedidos por {label}', ascending=False)
        st.markdown(f"#### {label}")
        st.dataframe(table_df)

        from insights import display_insights_summary, create_advanced_filters
        from visualizations import (
            create_consistency_chart, create_rookie_analysis_chart, 
            create_weekly_trends_chart, create_position_scarcity_chart,
            create_breakout_analysis_chart
        )
        
        st.markdown("### 🔍 Insights Avançados")
        
        # Filtros avançados
        advanced_filters = create_advanced_filters()
        
        # Exibir insights principais
        display_insights_summary(filtered_df)
        
        st.markdown("---")
        st.markdown("### 📊 Visualizações Avançadas")
        
        # Criar sub-tabs para diferentes análises
        viz_tab1, viz_tab2, viz_tab3, viz_tab4, viz_tab5 = st.tabs([
            "📊 Consistência", "🆕 Rookies", "📈 Tendências", "💎 Escassez", "🚀 Breakouts"
        ])
        
        with viz_tab1:
            st.markdown("#### Análise de Consistência por Posição")
            selected_pos_consistency = st.selectbox(
                "Selecionar Posição para Análise de Consistência:",
                options=['QB', 'RB', 'WR', 'TE'],
                key="consistency_pos"
            )
            
            consistency_fig = create_consistency_chart(filtered_df, selected_pos_consistency)
            st.plotly_chart(consistency_fig, use_container_width=True)
            
            st.info("💡 **Interpretação**: Jogadores no canto inferior direito têm alta performance e alta consistência (ideais para fantasy).")
        
        with viz_tab2:
            st.markdown("#### Performance de Rookies vs Veteranos")
            rookie_fig = create_rookie_analysis_chart(filtered_df)
            st.plotly_chart(rookie_fig, use_container_width=True)
            
            st.info("💡 **Insight**: Posições com menor gap rookie-veterano podem oferecer melhor valor em rookies.")
        
        with viz_tab3:
            st.markdown("#### Tendências Semanais por Posição")
            selected_pos_trends = st.selectbox(
                "Selecionar Posição para Análise de Tendências:",
                options=['QB', 'RB', 'WR', 'TE'],
                key="trends_pos"
            )
            
            trends_fig = create_weekly_trends_chart(filtered_df, selected_pos_trends)
            st.plotly_chart(trends_fig, use_container_width=True)
            
            st.info("💡 **Estratégia**: Use essas tendências para timing de trades e decisões de lineup.")
        
        with viz_tab4:
            st.markdown("#### Análise de Escassez Posicional")
            scarcity_fig = create_position_scarcity_chart(filtered_df)
            st.plotly_chart(scarcity_fig, use_container_width=True)
            
            st.info("💡 **Draft Strategy**: Posições com maior variabilidade (QB, TE) são consideradas 'premium' - priorize early.")
        
        with viz_tab5:
            st.markdown("#### Análise de Breakout (Primeira vs Segunda Temporada)")
            breakout_fig = create_breakout_analysis_chart(filtered_df)
            st.plotly_chart(breakout_fig, use_container_width=True)
            
            st.info("💡 **Sleeper Picks**: Jogadores acima da linha diagonal tiveram breakout na segunda temporada.")
    
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

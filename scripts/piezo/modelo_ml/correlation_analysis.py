import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import warnings
from pathlib import Path
import os
import sys
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

# Configurações
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Caminhos
BASE_DIR = Path(__file__).parent.parent
CSV_POZOS = Path(__file__).parent / 'csv_modelo/aquifer_depth_piezo.csv'
CSV_PRECIP = Path(__file__).parent / 'csv_modelo/bd_precipitacao.csv'
RESULTS_DIR = BASE_DIR / 'resources/correlation_analysis'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def log(msg):
    """Função de logging simples."""
    print(f'[LOG] {msg}')

def calculate_distance(lat1, lon1, lat2, lon2):
    """Calcula distância euclidiana entre duas coordenadas em metros."""
    return np.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2)

def find_nearby_precipitation_stations(poço_coords, precip_df, max_distance_km=50, n_stations=3):
    """Encontra estações de precipitação próximas ao poço."""
    station_coords = precip_df[['codigo', 'coord_x_m', 'coord_y_m']].drop_duplicates()
    
    distances = []
    for _, station in station_coords.iterrows():
        try:
            x = float(station['coord_x_m'])
            y = float(station['coord_y_m'])
            distance = calculate_distance(poço_coords[0], poço_coords[1], x, y)
            distances.append({
                'codigo': station['codigo'],
                'distance_m': distance,
                'distance_km': distance / 1000
            })
        except (ValueError, TypeError):
            continue
    
    nearby_stations = [s for s in distances if s['distance_km'] <= max_distance_km]
    nearby_stations.sort(key=lambda x: x['distance_km'])
    selected_stations = nearby_stations[:n_stations]
    
    if selected_stations:
        log(f"Estações de precipitação encontradas:")
        for station in selected_stations:
            log(f"  {station['codigo']}: {station['distance_km']:.1f} km")
    else:
        log(f"Nenhuma estação de precipitação encontrada dentro de {max_distance_km} km")
    
    return [s['codigo'] for s in selected_stations]

def load_and_prepare_data(codigo_poco):
    """Carrega e prepara dados do poço e precipitação."""
    # Permitir código com '_' ou '/'
    codigo_poco = str(codigo_poco).replace('_', '/')
    log(f"Carregando dados para poço {codigo_poco}")
    
    # Carregar dados do poço
    if not CSV_POZOS.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {CSV_POZOS}")
    
    df_poco = pd.read_csv(CSV_POZOS)
    # Remover aspas dos códigos se existirem
    df_poco['codigo'] = df_poco['codigo'].astype(str).str.strip('"')
    df_poco = df_poco[df_poco['codigo'] == codigo_poco]
    
    if df_poco.empty:
        raise ValueError(f'Poço {codigo_poco} não encontrado')
    
    # Extrair coordenadas
    poço_coords = None
    if 'coord_x_m' in df_poco.columns and 'coord_y_m' in df_poco.columns:
        try:
            x = float(df_poco['coord_x_m'].iloc[0])
            y = float(df_poco['coord_y_m'].iloc[0])
            poço_coords = (x, y)
            log(f"Coordenadas do poço: ({x:.0f}, {y:.0f})")
        except (ValueError, TypeError):
            log("Coordenadas do poço não disponíveis")
    
    # Preparar dados do poço
    df_poco = df_poco[['data', 'profundidade_nivel_agua']].copy()
    df_poco['data'] = pd.to_datetime(df_poco['data'])
    df_poco['gwl'] = -pd.to_numeric(df_poco['profundidade_nivel_agua'], errors='coerce')
    df_poco = df_poco.drop(columns=['profundidade_nivel_agua']).dropna()
    
    # Agregar GWL por mês
    df_poco_monthly = df_poco.groupby(pd.Grouper(key='data', freq='M'))['gwl'].mean().reset_index()
    df_poco_monthly = df_poco_monthly.dropna()
    
    log(f"Dados do poço: {len(df_poco_monthly)} meses")
    
    # Carregar dados de precipitação
    if not CSV_PRECIP.exists():
        log("Arquivo de precipitação não encontrado")
        return df_poco_monthly, None, None
    
    df_precip = pd.read_csv(CSV_PRECIP)
    df_precip['data'] = pd.to_datetime(df_precip['data'])
    
    # Encontrar estações próximas
    if poço_coords:
        nearby_stations = find_nearby_precipitation_stations(poço_coords, df_precip)
        if nearby_stations:
            # Filtrar estações selecionadas
            df_precip_filtered = df_precip[df_precip['codigo'].isin(nearby_stations)].copy()
            
            # Agregar precipitação mensal
            monthly_precip = df_precip_filtered.groupby([
                pd.Grouper(key='data', freq='M'),
                'codigo'
            ])['precipitacao_dia_mm'].sum().reset_index()
            
            # Pivotar para ter uma coluna por estação
            monthly_precip_pivot = monthly_precip.pivot(
                index='data', 
                columns='codigo', 
                values='precipitacao_dia_mm'
            ).fillna(0)
            
            # Adicionar média das estações
            monthly_precip_pivot['precip_media'] = monthly_precip_pivot.mean(axis=1)
            monthly_precip_pivot = monthly_precip_pivot.reset_index()
            
            log(f"Dados de precipitação: {len(monthly_precip_pivot)} meses")
            return df_poco_monthly, monthly_precip_pivot, nearby_stations
    
    log("Não foi possível preparar dados de precipitação")
    return df_poco_monthly, None, None

def merge_gwl_precipitation_data(df_gwl, df_precip):
    """Combina dados de GWL e precipitação."""
    if df_precip is None:
        return df_gwl
    
    # Merge por data
    merged_df = pd.merge(df_gwl, df_precip, on='data', how='inner')
    
    log(f"Dados combinados: {len(merged_df)} meses")
    return merged_df

def calculate_correlations(df, gwl_col='gwl', precip_col='precip_media'):
    """Calcula correlações entre GWL e precipitação."""
    if precip_col not in df.columns:
        log("Coluna de precipitação não encontrada")
        return None
    
    # Remover valores nulos
    df_clean = df[[gwl_col, precip_col]].dropna()
    
    if len(df_clean) < 10:
        log("Dados insuficientes para análise de correlação")
        return None
    
    # Correlações
    pearson_corr, pearson_p = pearsonr(df_clean[gwl_col], df_clean[precip_col])
    spearman_corr, spearman_p = spearmanr(df_clean[gwl_col], df_clean[precip_col])
    
    correlations = {
        'pearson': {'correlation': pearson_corr, 'p_value': pearson_p},
        'spearman': {'correlation': spearman_corr, 'p_value': spearman_p},
        'n_samples': len(df_clean)
    }
    
    log(f"Correlação Pearson: {pearson_corr:.3f} (p={pearson_p:.3f})")
    log(f"Correlação Spearman: {spearman_corr:.3f} (p={spearman_p:.3f})")
    
    return correlations

def analyze_lag_correlations(df, gwl_col='gwl', precip_col='precip_media', max_lag=12):
    """Analisa correlações com diferentes lags temporais."""
    if precip_col not in df.columns:
        return None
    
    df_clean = df[[gwl_col, precip_col, 'data']].dropna().sort_values('data')
    
    lag_correlations = []
    
    for lag in range(max_lag + 1):
        # Criar lag na precipitação
        df_lag = df_clean.copy()
        df_lag[f'{precip_col}_lag_{lag}'] = df_lag[precip_col].shift(lag)
        
        # Calcular correlação
        corr_data = df_lag[[gwl_col, f'{precip_col}_lag_{lag}']].dropna()
        
        if len(corr_data) > 10:
            pearson_corr, pearson_p = pearsonr(corr_data[gwl_col], corr_data[f'{precip_col}_lag_{lag}'])
            spearman_corr, spearman_p = spearmanr(corr_data[gwl_col], corr_data[f'{precip_col}_lag_{lag}'])
            
            lag_correlations.append({
                'lag': lag,
                'pearson_corr': pearson_corr,
                'pearson_p': pearson_p,
                'spearman_corr': spearman_corr,
                'spearman_p': spearman_p,
                'n_samples': len(corr_data)
            })
    
    return pd.DataFrame(lag_correlations)

def create_correlation_plots(df, correlations, lag_correlations, codigo_poco):
    """Cria gráficos de correlação."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Análise de Correlação: GWL vs Precipitação - Poço {codigo_poco}', fontsize=16)
    
    # 1. Scatter plot GWL vs Precipitação
    if 'precip_media' in df.columns:
        ax1 = axes[0, 0]
        df_clean = df[['gwl', 'precip_media']].dropna()
        ax1.scatter(df_clean['precip_media'], df_clean['gwl'], alpha=0.6)
        ax1.set_xlabel('Precipitação Mensal (mm)')
        ax1.set_ylabel('Nível Freático (m)')
        ax1.set_title('GWL vs Precipitação')
        
        # Adicionar linha de tendência
        if len(df_clean) > 2:
            z = np.polyfit(df_clean['precip_media'], df_clean['gwl'], 1)
            p = np.poly1d(z)
            ax1.plot(df_clean['precip_media'], p(df_clean['precip_media']), "r--", alpha=0.8)
    
    # 2. Série temporal
    ax2 = axes[0, 1]
    ax2.plot(df['data'], df['gwl'], label='GWL', color='blue')
    ax2.set_xlabel('Data')
    ax2.set_ylabel('Nível Freático (m)', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    
    if 'precip_media' in df.columns:
        ax2_twin = ax2.twinx()
        ax2_twin.plot(df['data'], df['precip_media'], label='Precipitação', color='red', alpha=0.7)
        ax2_twin.set_ylabel('Precipitação (mm)', color='red')
        ax2_twin.tick_params(axis='y', labelcolor='red')
    
    ax2.set_title('Série Temporal: GWL e Precipitação')
    
    # 3. Correlação por lag
    if lag_correlations is not None and not lag_correlations.empty:
        ax3 = axes[1, 0]
        ax3.plot(lag_correlations['lag'], lag_correlations['pearson_corr'], 
                marker='o', label='Pearson', color='blue')
        ax3.plot(lag_correlations['lag'], lag_correlations['spearman_corr'], 
                marker='s', label='Spearman', color='red')
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax3.set_xlabel('Lag (meses)')
        ax3.set_ylabel('Correlação')
        ax3.set_title('Correlação por Lag')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # 4. Heatmap de correlação
    ax4 = axes[1, 1]
    if 'precip_media' in df.columns:
        corr_cols = ['gwl', 'precip_media']
        corr_matrix = df[corr_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, ax=ax4)
        ax4.set_title('Matriz de Correlação')
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f'correlation_analysis_{codigo_poco}.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_seasonal_analysis(df, codigo_poco):
    """Análise sazonal da correlação."""
    if 'precip_media' not in df.columns:
        return
    
    df_clean = df[['gwl', 'precip_media', 'data']].dropna()
    df_clean['month'] = df_clean['data'].dt.month
    df_clean['season'] = pd.cut(df_clean['month'], 
                               bins=[0, 3, 6, 9, 12], 
                               labels=['Winter', 'Spring', 'Summer', 'Fall'])
    
    # Correlação por estação
    seasonal_corr = df_clean.groupby('season').apply(
        lambda x: pearsonr(x['gwl'], x['precip_media'])[0] if len(x) > 5 else np.nan
    )
    
    # Gráfico sazonal
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f'Análise Sazonal - Poço {codigo_poco}', fontsize=14)
    
    # Box plot por estação
    df_clean.boxplot(column='gwl', by='season', ax=ax1)
    ax1.set_title('Distribuição GWL por Estação')
    ax1.set_xlabel('Estação')
    ax1.set_ylabel('Nível Freático (m)')
    
    # Correlação por estação
    seasonal_corr.plot(kind='bar', ax=ax2, color='skyblue')
    ax2.set_title('Correlação GWL-Precipitação por Estação')
    ax2.set_xlabel('Estação')
    ax2.set_ylabel('Correlação de Pearson')
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f'seasonal_analysis_{codigo_poco}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return seasonal_corr

def export_correlation_results(correlations, lag_correlations, seasonal_corr, codigo_poco):
    """Exporta resultados da análise de correlação."""
    results = {
        'poço': codigo_poco,
        'data_analise': datetime.now().isoformat(),
        'correlacoes_gerais': correlations,
        'lag_correlations': lag_correlations.to_dict('records') if lag_correlations is not None else None,
        'seasonal_correlation': seasonal_corr.to_dict() if seasonal_corr is not None else None
    }
    
    # Salvar como JSON
    output_file = RESULTS_DIR / f'correlation_results_{codigo_poco}.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    log(f"Resultados salvos em: {output_file}")

def analyze_correlation_for_poco(codigo_poco):
    """Função principal para analisar correlação de um poço."""
    log(f"=== ANÁLISE DE CORRELAÇÃO PARA POÇO {codigo_poco} ===")
    
    try:
        # Carregar dados
        df_gwl, df_precip, nearby_stations = load_and_prepare_data(codigo_poco)
        
        # Combinar dados
        df_combined = merge_gwl_precipitation_data(df_gwl, df_precip)
        
        if df_precip is None:
            log("Não foi possível analisar correlação - dados de precipitação não disponíveis")
            return
        
        # Calcular correlações
        correlations = calculate_correlations(df_combined)
        lag_correlations = analyze_lag_correlations(df_combined)
        seasonal_corr = create_seasonal_analysis(df_combined, codigo_poco)
        
        # Criar gráficos
        create_correlation_plots(df_combined, correlations, lag_correlations, codigo_poco)
        
        # Exportar resultados
        export_correlation_results(correlations, lag_correlations, seasonal_corr, codigo_poco)
        
        # Resumo
        log("\n=== RESUMO DA ANÁLISE ===")
        if correlations:
            log(f"Correlação geral (Pearson): {correlations['pearson']['correlation']:.3f}")
            log(f"Significância (p-value): {correlations['pearson']['p_value']:.3f}")
        
        if lag_correlations is not None and not lag_correlations.empty:
            best_lag = lag_correlations.loc[lag_correlations['pearson_corr'].abs().idxmax()]
            log(f"Melhor lag: {best_lag['lag']} meses (corr={best_lag['pearson_corr']:.3f})")
        
        if seasonal_corr is not None:
            best_season = seasonal_corr.idxmax()
            log(f"Melhor correlação sazonal: {best_season} ({seasonal_corr[best_season]:.3f})")
        
    except Exception as e:
        log(f"Erro na análise: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import json
    
    # Lista de poços para analisar (pode ser modificada)
    poços_para_analisar = ['330_183']
    
    for codigo_poco in poços_para_analisar:
        analyze_correlation_for_poco(codigo_poco)
        print("\n" + "="*50 + "\n") 
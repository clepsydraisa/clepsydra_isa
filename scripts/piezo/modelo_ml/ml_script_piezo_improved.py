import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.interpolate import interp1d
import pymannkendall as mk
from statsmodels.tsa.seasonal import STL
import warnings
from pathlib import Path
import os
import sys
import json
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import euclidean_distances
warnings.filterwarnings('ignore')

# ===================== CONFIGURAÇÕES =====================
# Caminhos principais
BASE_DIR = Path(__file__).parent.parent
CSV_POZOS = Path(__file__).parent / 'csv_modelo/aquifer_depth_piezo.csv'
CSV_PRECIP = Path(__file__).parent / 'csv_modelo/bd_precipitacao.csv'
RESULTS_DIR = BASE_DIR / 'resources/trends_piezo'
MODELS_DIR = BASE_DIR / 'resources/models_piezo'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Parâmetros principais
LAG = 3  # Número de lags para ML
TEST_SIZE = 0.2  # Proporção para teste
RANDOM_STATE = 42
MIN_GAP_MONTHS = 6  # Lacuna mínima para segmentação
ALPHA = 0.05  # Significância estatística
FORECAST_MONTHS = 3  # Número de meses para prever
MAX_DISTANCE_KM = 50  # Distância máxima para estações de precipitação (km)

# ===================== FUNÇÕES AUXILIARES =====================
def log(msg):
    """Função de logging simples."""
    print(f'[LOG] {msg}')

def calculate_distance(lat1, lon1, lat2, lon2):
    """Calcula distância euclidiana entre duas coordenadas em metros."""
    return np.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2)

def find_nearby_precipitation_stations(poço_coords, precip_df, max_distance_km=MAX_DISTANCE_KM, n_stations=3):
    """
    Encontra estações de precipitação próximas ao poço.
    
    Args:
        poço_coords: (x, y) coordenadas do poço em metros
        precip_df: DataFrame com dados de precipitação
        max_distance_km: distância máxima em km
        n_stations: número máximo de estações a retornar
    
    Returns:
        Lista de códigos das estações mais próximas
    """
    # Obter coordenadas únicas das estações
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
    
    # Filtrar por distância máxima e ordenar
    nearby_stations = [s for s in distances if s['distance_km'] <= max_distance_km]
    nearby_stations.sort(key=lambda x: x['distance_km'])
    
    # Retornar as n estações mais próximas
    selected_stations = nearby_stations[:n_stations]
    
    if selected_stations:
        log(f"Estações de precipitação encontradas:")
        for station in selected_stations:
            log(f"  {station['codigo']}: {station['distance_km']:.1f} km")
    else:
        log(f"Nenhuma estação de precipitação encontrada dentro de {max_distance_km} km")
    
    return [s['codigo'] for s in selected_stations]

def aggregate_precipitation_monthly(precip_df, station_codes):
    """
    Agrega dados de precipitação diária para mensal.
    
    Args:
        precip_df: DataFrame com dados de precipitação diária
        station_codes: Lista de códigos das estações
    
    Returns:
        DataFrame com precipitação mensal agregada
    """
    # Filtrar estações selecionadas
    filtered_df = precip_df[precip_df['codigo'].isin(station_codes)].copy()
    
    if filtered_df.empty:
        log("Nenhum dado de precipitação encontrado para as estações selecionadas")
        return None
    
    # Converter data
    filtered_df['data'] = pd.to_datetime(filtered_df['data'])
    
    # Agregar por mês e estação
    monthly_precip = filtered_df.groupby([
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
    
    log(f"Precipitação mensal preparada: {len(monthly_precip_pivot)} meses")
    return monthly_precip_pivot

def load_poco_data(codigo_poco):
    """Carrega dados do poço a partir do CSV principal."""
    if not CSV_POZOS.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {CSV_POZOS}")
    
    df = pd.read_csv(CSV_POZOS)
    if 'codigo' not in df.columns:
        raise ValueError('Coluna "codigo" não encontrada no CSV.')
    
    df = df[df['codigo'] == codigo_poco]
    if df.empty:
        log(f'Poço {codigo_poco} não encontrado ou sem dados.')
        return None
    
    if 'data' not in df.columns or 'profundidade_nivel_agua' not in df.columns:
        raise ValueError('Colunas necessárias não encontradas.')
    
    # Extrair coordenadas do poço
    poço_coords = None
    if 'coord_x_m' in df.columns and 'coord_y_m' in df.columns:
        try:
            x = float(df['coord_x_m'].iloc[0])
            y = float(df['coord_y_m'].iloc[0])
            poço_coords = (x, y)
            log(f"Coordenadas do poço: ({x:.0f}, {y:.0f})")
        except (ValueError, TypeError):
            log("Coordenadas do poço não disponíveis")
    
    df = df[['data', 'profundidade_nivel_agua']].copy()
    df['nivel'] = -pd.to_numeric(df['profundidade_nivel_agua'], errors='coerce')
    df = df.drop(columns=['profundidade_nivel_agua'])
    
    log(f'Carregados {len(df)} registos para poço {codigo_poco}')
    return df.dropna(), poço_coords

# ===================== CLASSE PRINCIPAL DE ANÁLISE =====================
class GroundwaterTrendAnalysis:
    """
    Análise completa de tendências para dados piezométricos (GWL).
    Mantém todas as funcionalidades do script original.
    """
    
    def __init__(self, data, date_col='data', value_col='nivel'):
        """Initialize with groundwater data."""
        self.data = data.copy()
        self.data[date_col] = pd.to_datetime(self.data[date_col])
        self.data = self.data.set_index(date_col).sort_index()
        self.value_col = value_col
        self.monthly_data = None
        self.segments = []
        self.filled_segments = []
        self.trend_results = []
        
    def step1_monthly_aggregation(self, method='mean'):
        """Step 1: Keep one value per month."""
        log("=== STEP 1: MONTHLY AGGREGATION ===")
        
        if method == 'mean':
            self.monthly_data = self.data.groupby(pd.Grouper(freq='M'))[self.value_col].mean()
        elif method == 'median':
            self.monthly_data = self.data.groupby(pd.Grouper(freq='M'))[self.value_col].median()
        elif method == 'first':
            self.monthly_data = self.data.groupby(pd.Grouper(freq='M'))[self.value_col].first()
        elif method == 'last':
            self.monthly_data = self.data.groupby(pd.Grouper(freq='M'))[self.value_col].last()
        else:
            raise ValueError("Method must be 'mean', 'median', 'first', or 'last'")
            
        full_range = pd.date_range(start=self.monthly_data.index.min(), end=self.monthly_data.index.max(), freq='M')
        self.monthly_data = self.monthly_data.reindex(full_range)
        
        log(f"Original data points: {len(self.data)}")
        log(f"Monthly data points: {len(self.monthly_data)}")
        log(f"Missing months: {self.monthly_data.isna().sum()}")
        log(f"Data completeness: {(1 - self.monthly_data.isna().mean())*100:.1f}%")
        
        # Visualização
        plt.figure(figsize=(15, 6))
        plt.plot(self.monthly_data.index, self.monthly_data.values, 'b-o', markersize=3, alpha=0.7)
        plt.title(f'Monthly Aggregated Data ({method})')
        plt.xlabel('Date')
        plt.ylabel('Water Level')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def exploratory_analysis(self):
        """Comprehensive EDA for groundwater data."""
        log("=== EXPLORATORY DATA ANALYSIS ===")
        
        log(f"Data period: {self.data.index.min()} to {self.data.index.max()}")
        log(f"Total observations: {len(self.monthly_data)}")
        log(f"Missing values: {self.monthly_data.isna().sum()}")

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        axes[0,0].plot(self.monthly_data.index, self.monthly_data, 'b-', alpha=0.7)
        axes[0,0].set_title('Groundwater Level Time Series')
        axes[0,0].set_ylabel('Water Level')
        axes[0,0].grid(True, alpha=0.3)
        
        try:
            stl = STL(self.monthly_data.dropna(), seasonal=13, period=12)
            decomp = stl.fit()
            axes[0,1].plot(decomp.trend, 'r-', linewidth=2)
            axes[0,1].set_title('Extracted Trend Component')
            axes[0,1].set_ylabel('Trend')
            axes[0,1].grid(True, alpha=0.3)
        except Exception as e:
            axes[0,1].text(0.5, 0.5, f'STL decomposition failed:\n{str(e)}', ha='center', va='center', transform=axes[0,1].transAxes)
        
        monthly_data_2 = self.monthly_data.groupby(self.monthly_data.index.month).agg(['mean', 'std'])
        axes[1,0].errorbar(monthly_data_2.index, monthly_data_2['mean'], yerr=monthly_data_2['std'], marker='o', capsize=5)
        axes[1,0].set_title('Seasonal Pattern (Monthly Averages)')
        axes[1,0].set_xlabel('Month')
        axes[1,0].set_ylabel('Water Level')
        axes[1,0].set_xticks(range(1, 13))
        axes[1,0].grid(True, alpha=0.3)
        
        annual_data = self.monthly_data.groupby(self.monthly_data.index.year).mean()
        axes[1,1].plot(annual_data.index, annual_data.values, 'go-', markersize=4)
        axes[1,1].set_title('Annual Mean Water Levels')
        axes[1,1].set_xlabel('Year')
        axes[1,1].set_ylabel('Water Level')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        try:
            decomp.plot()
            plt.show()
        except:
            pass

    def step2_segment_by_gaps(self, min_gap_months=MIN_GAP_MONTHS):
        """Step 2: Split time series based on gaps."""
        log(f"=== STEP 2: SEGMENT BY GAPS (≥{min_gap_months} months) ===")
        
        if self.monthly_data is None:
            raise ValueError("Run step1_monthly_aggregation first")
            
        na_mask = self.monthly_data.isna()
        gap_starts = []
        gap_ends = []
        
        in_gap = False
        gap_start = None
        
        for i, (date, is_na) in enumerate(na_mask.items()):
            if is_na and not in_gap:
                gap_start = i
                in_gap = True
            elif not is_na and in_gap:
                gap_length = i - gap_start
                if gap_length >= min_gap_months:
                    gap_starts.append(gap_start)
                    gap_ends.append(i)
                in_gap = False
                
        if in_gap and (len(na_mask) - gap_start) >= min_gap_months:
            gap_starts.append(gap_start)
            gap_ends.append(len(na_mask))
            
        log(f"Found {len(gap_starts)} significant gaps:")
        
        segment_boundaries = [0]
        
        for gap_start, gap_end in zip(gap_starts, gap_ends):
            segment_boundaries.append(gap_start)
            segment_boundaries.append(gap_end)
            
        segment_boundaries = sorted(list(set(segment_boundaries)))
        segment_boundaries.append(len(self.monthly_data))
        
        self.segments = []
        for i in range(0, len(segment_boundaries)-1, 2):
            start_idx = segment_boundaries[i]
            end_idx = segment_boundaries[i+1] if i+1 < len(segment_boundaries) else len(self.monthly_data)
            
            start_date = self.monthly_data.index[start_idx]
            end_date = self.monthly_data.index[min(end_idx-1, len(self.monthly_data)-1)]
            
            segment_data = self.monthly_data.iloc[start_idx:end_idx]
            
            if segment_data.notna().sum() > 0:
                self.segments.append({
                    'id': len(self.segments) + 1,
                    'start_date': start_date,
                    'end_date': end_date,
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'data': segment_data,
                    'length_months': len(segment_data),
                    'valid_months': segment_data.notna().sum(),
                    'completeness': segment_data.notna().sum() / len(segment_data)
                })
                
        log(f"Created {len(self.segments)} segments:")
        for seg in self.segments:
            log(f"  Segment {seg['id']}: {seg['start_date'].strftime('%Y-%m')} to {seg['end_date'].strftime('%Y-%m')}")
            log(f"    Length: {seg['length_months']} months, Valid: {seg['valid_months']} ({seg['completeness']*100:.1f}%)")
            
        self._plot_segments()
        
    def _plot_segments(self):
        """Visualize the segmented time series."""
        plt.figure(figsize=(15, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(self.monthly_data.index, self.monthly_data.values, 'k-', alpha=0.3, label='All data')
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(self.segments)))
        for seg, color in zip(self.segments, colors):
            seg_data = seg['data'].dropna()
            if len(seg_data) > 0:
                plt.plot(seg_data.index, seg_data.values, 'o-', color=color, linewidth=2, markersize=4, label=f"Segment {seg['id']}")
                
        plt.title('Time Series Segments')
        plt.ylabel('Water Level')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 1, 2)
        seg_ids = [seg['id'] for seg in self.segments]
        lengths = [seg['length_months'] for seg in self.segments]
        completeness = [seg['completeness']*100 for seg in self.segments]
        
        x = np.arange(len(seg_ids))
        width = 0.35
        
        plt.bar(x - width/2, lengths, width, label='Total months', alpha=0.7)
        plt.bar(x + width/2, completeness, width, label='Completeness %', alpha=0.7)
        
        plt.xlabel('Segment ID')
        plt.ylabel('Count / Percentage')
        plt.title('Segment Statistics')
        plt.xticks(x, seg_ids)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    def step3_fill_missing_data(self, method='seasonal_decompose', **kwargs):
        """Step 3: Fill missing data within each segment."""
        log(f"=== STEP 3: FILL MISSING DATA ({method}) ===")
        
        if not self.segments:
            raise ValueError("Run step2_segment_by_gaps first")
            
        self.filled_segments = []
        
        for seg in self.segments:
            log(f"Processing Segment {seg['id']}:")
            log(f"  Missing values: {seg['data'].isna().sum()}/{len(seg['data'])}")
            
            if seg['data'].isna().sum() == 0:
                log("  No missing values - using original data")
                filled_data = seg['data'].copy()
            else:
                filled_data = self._fill_segment_gaps(seg['data'], method, **kwargs)
                
            filled_segment = seg.copy()
            filled_segment['filled_data'] = filled_data
            filled_segment['fill_method'] = method
            self.filled_segments.append(filled_segment)
            
            log(f"  After filling: {filled_data.isna().sum()} missing values")
            
        self._plot_filled_segments()
        
    def _fill_segment_gaps(self, data, method, **kwargs):
        """Fill gaps in a single segment."""
        filled_data = data.copy()
        
        if method == 'interpolate':
            interp_method = kwargs.get('interp_method', 'linear')
            if interp_method in ['linear', 'quadratic', 'cubic']:
                filled_data = data.interpolate(method=interp_method)
            else:
                valid_idx = ~data.isna()
                if valid_idx.sum() >= 2:
                    valid_dates = np.arange(len(data))[valid_idx]
                    valid_values = data.values[valid_idx]
                    
                    if len(valid_values) > 1:
                        f = interp1d(valid_dates, valid_values, kind=interp_method, bounds_error=False, fill_value='extrapolate')
                        filled_data = pd.Series(f(np.arange(len(data))), index=data.index)
                        
        elif method == 'seasonal_decompose':
            min_periods = kwargs.get('min_periods', 24)
            if len(data.dropna()) >= min_periods:
                try:
                    temp_filled = data.interpolate(method='linear', limit=3)
                    if temp_filled.notna().sum() >= min_periods:
                        stl = STL(temp_filled.dropna(), seasonal=13, robust=True)
                        decomp = stl.fit()
                        trend_filled = decomp.trend.reindex(data.index).interpolate()
                        seasonal_filled = decomp.seasonal.reindex(data.index).fillna(method='pad')
                        filled_data = trend_filled + seasonal_filled
                    else:
                        filled_data = data.interpolate(method='linear')
                except:
                    filled_data = data.interpolate(method='linear')
            else:
                filled_data = data.interpolate(method='linear')
                
        elif method == 'forward_fill':
            filled_data = data.fillna(method='ffill')
            
        elif method == 'backward_fill':
            filled_data = data.fillna(method='bfill')
            
        elif method == 'mean':
            mean_value = data.mean()
            filled_data = data.fillna(mean_value)
            
        else:
            raise ValueError(f"Unknown fill method: {method}")
            
        return filled_data
    
    def _plot_filled_segments(self):
        """Visualize original vs filled data for each segment."""
        n_segments = len(self.filled_segments)
        fig, axes = plt.subplots(n_segments, 1, figsize=(15, 4*n_segments))
        
        if n_segments == 1:
            axes = [axes]
            
        for i, seg in enumerate(self.filled_segments):
            axes[i].plot(seg['data'].index, seg['data'].values, 'ko-', markersize=3, alpha=0.7, label='Original')
            axes[i].plot(seg['filled_data'].index, seg['filled_data'].values, 'r-', linewidth=2, alpha=0.8, label='Filled')
            
            filled_mask = seg['data'].isna()
            if filled_mask.sum() > 0:
                filled_points = seg['filled_data'][filled_mask]
                axes[i].plot(filled_points.index, filled_points.values, 'ro', markersize=5, label='Filled points')
            
            axes[i].set_title(f"Segment {seg['id']}: {seg['start_date'].strftime('%Y-%m')} to {seg['end_date'].strftime('%Y-%m')}")
            axes[i].set_ylabel('Water Level')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.show()
        
    def step4_test_trends(self, alpha=ALPHA):
        """Step 4: Test for significant trends in each filled segment."""
        log(f"=== STEP 4: TREND TESTING (α={alpha}) ===")
        
        if not self.filled_segments:
            raise ValueError("Run step3_fill_missing_data first")
            
        self.trend_results = []
        
        for seg in self.filled_segments:
            log(f"--- Segment {seg['id']} Trend Analysis ---")
            
            data = seg['filled_data'].dropna()
            if len(data) < 12:
                log("  Insufficient data for trend analysis")
                continue
                
            result = self._test_segment_trend(data, seg['id'], alpha)
            self.trend_results.append(result)
            
        self._create_trend_summary()
        
    def _test_segment_trend(self, data, segment_id, alpha):
        """Test trends for a single segment."""
        result = {
            'segment_id': segment_id,
            'n_observations': len(data),
            'data_years': len(data) / 12,
            'start_date': data.index[0],
            'end_date': data.index[-1]
        }
        
        try:
            mk_result = mk.original_test(data.values, alpha=alpha)
            result['mann_kendall'] = {
                'trend': mk_result.trend,
                'p_value': mk_result.p,
                'tau': mk_result.Tau,
                'slope': mk_result.slope,
                'slope_per_year': mk_result.slope * 12,
                'significant': mk_result.p < alpha
            }
            log(f"  Mann-Kendall: {mk_result.trend} (p={mk_result.p:.4f}, τ={mk_result.Tau:.3f})")
            log(f"  Sen's slope: {mk_result.slope*12:.4f} units/year")
        except Exception as e:
            log(f"  Mann-Kendall test failed: {e}")
            
        if len(data) >= 24:
            try:
                smk_result = mk.seasonal_test(data.values, period=12, alpha=alpha)
                result['seasonal_mann_kendall'] = {
                    'trend': smk_result.trend,
                    'p_value': smk_result.p,
                    'tau': smk_result.Tau,
                    'slope': smk_result.slope,
                    'slope_per_year': smk_result.slope * 12,
                    'significant': smk_result.p < alpha
                }
                log(f"  Seasonal Mann-Kendall: {smk_result.trend} (p={smk_result.p:.4f})")
                log(f"  Seasonal Sen's slope: {smk_result.slope*12:.4f} units/year")
            except Exception as e:
                log(f"  Seasonal Mann-Kendall test failed: {e}")
                
        try:
            HRmk_result = mk.hamed_rao_modification_test(data.values, alpha=alpha)
            result['hamed_rao_modification'] = {
                'trend': HRmk_result.trend,
                'p_value': HRmk_result.p,
                'tau': HRmk_result.Tau,
                'slope': HRmk_result.slope,
                'slope_per_year': HRmk_result.slope * 12,
                'significant': HRmk_result.p < alpha
            }
            log(f"  Hamed Rao Mann-Kendall: {HRmk_result.trend} (p={HRmk_result.p:.4f}, τ={HRmk_result.Tau:.3f})")
            log(f"  Sen's slope: {HRmk_result.slope*12:.4f} units/year")
        except Exception as e:
            log(f" Hamed Rao Mann-Kendall test failed: {e}")
            
        try:
            x = np.arange(len(data))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, data.values)
            slope_per_year = slope * 12
            result['linear_regression'] = {
                'slope_per_year': slope_per_year,
                'intercept': intercept,
                'r_value': r_value,
                'r_squared': r_value**2,
                'p_value': p_value,
                'std_err_per_year': std_err * 12,
                'significant': p_value < alpha
            }
            trend_direction = 'increasing' if slope_per_year > 0 else 'decreasing'
            log(f"  Linear regression: {trend_direction} (p={p_value:.4f}, R²={r_value**2:.3f})")
            log(f"  Linear slope: {slope_per_year:.4f} ± {std_err*12:.4f} units/year")
        except Exception as e:
            log(f"  Linear regression failed: {e}")
            
        return result
    
    def _create_trend_summary(self):
        """Create summary table of all trend results."""
        log(f"{'='*80}")
        log("TREND ANALYSIS SUMMARY")
        log(f"{'='*80}")
        
        if not self.trend_results:
            log("No trend results available")
            return
            
        summary_data = []
        for result in self.trend_results:
            row = {
                'Segment': result['segment_id'],
                'Period': f"{result['start_date'].strftime('%Y-%m')} to {result['end_date'].strftime('%Y-%m')}",
                'Years': f"{result['data_years']:.1f}",
                'N_obs': result['n_observations']
            }
            
            if 'mann_kendall' in result:
                mk = result['mann_kendall']
                row['MK_Trend'] = mk['trend']
                row['MK_p_value'] = f"{mk['p_value']:.4f}"
                row['MK_Slope_yr'] = f"{mk['slope_per_year']:.4f}"
                
            if 'seasonal_mann_kendall' in result:
                smk = result['seasonal_mann_kendall']
                row['SMK_Trend'] = smk['trend']
                row['SMK_p_value'] = f"{smk['p_value']:.4f}"
                row['SMK_Slope_yr'] = f"{smk['slope_per_year']:.4f}"
                
            if 'hamed_rao_modification' in result:
                HRsmk = result['hamed_rao_modification']
                row['HRMK_Trend'] = HRsmk['trend']
                row['HRMK_p_value'] = f"{HRsmk['p_value']:.4f}"
                row['HRSMK_Slope_yr'] = f"{HRsmk['slope_per_year']:.4f}"
           
            if 'linear_regression' in result:
                lr = result['linear_regression']
                row['Lin_Slope_yr'] = f"{lr['slope_per_year']:.4f}"
                row['Lin_p_value'] = f"{lr['p_value']:.4f}"
                row['R²'] = f"{lr['r_squared']:.3f}"
                
            summary_data.append(row)
            
        df_summary = pd.DataFrame(summary_data).transpose()
        log("\nDetailed Results:")
        print(df_summary.to_string(index=True))
        
        total_segments = len(self.trend_results)
        mk_significant = sum(1 for r in self.trend_results if r.get('mann_kendall', {}).get('significant', False))
        smk_significant = sum(1 for r in self.trend_results if r.get('seasonal_mann_kendall', {}).get('significant', False))
        HRsmk_significant = sum(1 for r in self.trend_results if r.get('hamed_rao_modification', {}).get('significant', False))
        lr_significant = sum(1 for r in self.trend_results if r.get('linear_regression', {}).get('significant', False))
        
        log(f"Total segments analyzed: {total_segments}")
        log(f"Significant trends (Mann-Kendall): {mk_significant}/{total_segments}")
        log(f"Significant trends (Seasonal MK): {smk_significant}/{total_segments}")
        log(f"Significant trends (Hamed Rao modified MK): {HRsmk_significant}/{total_segments}")
        log(f"Significant trends (Linear regression): {lr_significant}/{total_segments}")
        
        mk_trends = [r['mann_kendall']['trend'] for r in self.trend_results if 'mann_kendall' in r and r['mann_kendall']['significant']]
        if mk_trends:
            trend_counts = pd.Series(mk_trends).value_counts()
            log(f"Significant trend directions (Mann-Kendall): {dict(trend_counts)}")
            
        self._plot_trend_results()
        
    def _plot_trend_results(self):
        """Visualize trend test results."""
        if not self.trend_results:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        ax1 = axes[0, 0]
        for i, seg in enumerate(self.filled_segments):
            color = 'red' if any(r['segment_id'] == seg['id'] and r.get('hamed_rao_modification', {}).get('significant', False) for r in self.trend_results) else 'blue'
            ax1.plot(seg['filled_data'].index, seg['filled_data'].values, color=color, alpha=0.7, linewidth=2, label=f"Segment {seg['id']}" + (" (sig.)" if color=='red' else ""))
        ax1.set_title('Segments with Significant Trends (Red)')
        ax1.set_ylabel('Water Level')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[0, 1]
        segments = [r['segment_id'] for r in self.trend_results]
        mk_pvals = [r.get('mann_kendall', {}).get('p_value', 1) for r in self.trend_results]
        smk_pvals = [r.get('seasonal_mann_kendall', {}).get('p_value', 1) for r in self.trend_results]
        hrmk_pvals = [r.get('hamed_rao_modification', {}).get('p_value', 1) for r in self.trend_results]
        x = np.arange(len(segments))
        width = 0.25
        
        ax2.bar(x - width, mk_pvals, width, label='Mann-Kendall', alpha=0.7)
        ax2.bar(x, smk_pvals, width, label='Seasonal MK', alpha=0.7)
        ax2.bar(x + width, hrmk_pvals, width, label='Hamed Rao MK', alpha=0.7)
        ax2.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='α=0.05')
        ax2.set_xlabel('Segment')
        ax2.set_ylabel('p-value')
        ax2.set_title('Trend Test p-values')
        ax2.set_xticks(x)
        ax2.set_xticklabels(segments)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
        ax3 = axes[1, 0]
        mk_slopes = [r.get('mann_kendall', {}).get('slope_per_year', 0) for r in self.trend_results]
        smk_slopes = [r.get('seasonal_mann_kendall', {}).get('slope_per_year', 0) for r in self.trend_results]
        hrmk_slopes = [r.get('hamed_rao_modification', {}).get('slope_per_year', 0) for r in self.trend_results]
        
        ax3.bar(x - width, mk_slopes, width, label='Mann-Kendall', alpha=0.7)
        ax3.bar(x, smk_slopes, width, label='Seasonal MK', alpha=0.7)
        ax3.bar(x + width, hrmk_slopes, width, label='Hamed Rao MK', alpha=0.7)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax3.set_xlabel('Segment')
        ax3.set_ylabel('Trend Slope (units/year)')
        ax3.set_title('Trend Magnitudes')
        ax3.set_xticks(x)
        ax3.set_xticklabels(segments)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        ax4 = axes[1, 1]
        r_squared = [r.get('linear_regression', {}).get('r_squared', 0) for r in self.trend_results]
        
        ax4.bar(segments, r_squared, alpha=0.7, color='green')
        ax4.set_xlabel('Segment')
        ax4.set_ylabel('R² (Linear Regression)')
        ax4.set_title('Trend Fit Quality')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# ===================== FUNÇÕES DE MACHINE LEARNING =====================
def prepare_data_for_ml(analyzer, precip_data=None, lag=LAG):
    """
    Prepara dados para previsão de GWL usando outputs da análise de tendências.
    Versão melhorada com integração de precipitação e variáveis categóricas.
    """
    if not analyzer.filled_segments:
        log("Sem segmentos válidos para ML.")
        return None
    
    last_seg = analyzer.filled_segments[-1]
    trend = next((t for t in analyzer.trend_results if t['segment_id'] == last_seg['id']), {})
    
    df = last_seg['filled_data'].to_frame(name='gwl')
    
    # Adicionar lags
    for i in range(1, lag + 1):
        df[f'gwl_lag_{i}'] = df['gwl'].shift(i)
    
    # Adicionar tendências de MK
    df['trend_slope'] = trend.get('hamed_rao_modification', {}).get('slope_per_year', 0)
    
    # Adicionar variáveis categóricas melhoradas
    trend_direction = trend.get('hamed_rao_modification', {}).get('trend', 'no trend')
    df['trend_direction'] = trend_direction
    
    # Adicionar sazonalidade (mês do ano)
    df['month'] = df.index.month
    df['season'] = pd.cut(df.index.month, bins=[0, 3, 6, 9, 12], labels=['Winter', 'Spring', 'Summer', 'Fall'])
    
    # Adicionar indicadores de tendência
    df['trend_increasing'] = int(trend_direction == 'increasing')
    df['trend_decreasing'] = int(trend_direction == 'decreasing')
    df['trend_no_trend'] = int(trend_direction == 'no trend')
    
    # Merge com dados de precipitação se disponível
    if precip_data is not None:
        try:
            # Garantir que todas as colunas de precipitação são float
            for col in precip_data.columns:
                precip_data[col] = pd.to_numeric(precip_data[col], errors='coerce')
            df = df.merge(precip_data, left_index=True, right_index=True, how='left').fillna(0)
            log(f"Precipitação integrada: {len(precip_data.columns)} estações")
        except Exception as e:
            log(f"Erro ao integrar precipitação: {e}")
    
    df = df.dropna()
    if not df.empty:
        df.to_csv(BASE_DIR / "resources" / "ml_data_prepared.csv", index=True)
        log(f"Dados preparados para ML: {len(df)} registos")
    return df

def train_predict_gwl(df, poço_id):
    """
    Treina XGBoost para prever GWL.
    Versão melhorada com variáveis categóricas e avaliação detalhada.
    """
    if df is None or df.empty:
        log(f"Poço {poço_id}: Sem dados para ML.")
        return None, None
    
    X = df.drop('gwl', axis=1)
    y = df['gwl']
    
    # Identificar colunas categóricas
    categorical_cols = ['trend_direction', 'season']
    numerical_cols = [col for col in X.columns if col not in categorical_cols]
    
    # Codificar categóricas
    X_encoded = X.copy()
    for col in categorical_cols:
        if col in X_encoded.columns:
            X_encoded = pd.get_dummies(X_encoded, columns=[col], drop_first=True)
    
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=TEST_SIZE, shuffle=False, random_state=RANDOM_STATE)
    
    # Modelo XGBoost com hiperparâmetros otimizados
    model = xgb.XGBRegressor(
        objective='reg:squarederror', 
        n_estimators=200, 
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE
    )
    
    # Treinar modelo
    model.fit(X_train, y_train)
    
    # Avaliação
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # Métricas adicionais
    mae = np.mean(np.abs(y_test - y_pred))
    r2 = model.score(X_test, y_test)
    
    log(f"Poço {poço_id} - RMSE: {rmse:.3f}, MAE: {mae:.3f}, R²: {r2:.3f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': model.feature_names_in_,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    log("Top 10 features mais importantes:")
    for i, row in feature_importance.head(10).iterrows():
        log(f"  {row['feature']}: {row['importance']:.3f}")
    
    # Salvar modelo
    model_path = MODELS_DIR / f"model_{poço_id.replace('/', '_')}.json"
    model.save_model(model_path)
    log(f"Modelo salvo em: {model_path}")
    
    # Retornar dados para visualização
    results = {
        'y_test': y_test,
        'y_pred': y_pred,
        'X_test': X_test,
        'feature_importance': feature_importance
    }
    
    return model, rmse, results

def predict_future_gwl(model, last_data, trend_results, precip_forecast=None, months_ahead=FORECAST_MONTHS):
    """
    Faz previsão de GWL para os próximos meses.
    
    Args:
        model: Modelo XGBoost treinado
        last_data: Últimos dados conhecidos
        trend_results: Resultados da análise de tendência
        precip_forecast: Previsão de precipitação (opcional)
        months_ahead: Número de meses para prever
    
    Returns:
        DataFrame com previsões
    """
    log(f"=== PREVISÃO PARA OS PRÓXIMOS {months_ahead} MESES ===")
    
    # Preparar dados para previsão
    predictions = []
    current_data = last_data.copy()
    
    # Obter parâmetros de tendência
    trend_slope = trend_results.get('hamed_rao_modification', {}).get('slope_per_year', 0)
    trend_direction = trend_results.get('hamed_rao_modification', {}).get('trend', 'no trend')
    
    # Definir last_date como o último índice do current_data
    last_date = current_data.index[-1]
    if isinstance(last_date, str):
        last_date = pd.to_datetime(last_date)
    
    for month in range(1, months_ahead + 1):
        # Criar features para o próximo mês
        features = {}
        
        # Lags (usar valores conhecidos ou previsões anteriores)
        for i in range(1, LAG + 1):
            if i <= len(current_data):
                features[f'gwl_lag_{i}'] = current_data.iloc[-i]['gwl']
            else:
                features[f'gwl_lag_{i}'] = current_data.iloc[-1]['gwl']  # Usar último valor conhecido
        
        # Tendência
        features['trend_slope'] = trend_slope
        features['trend_direction'] = trend_direction
        
        # Sazonalidade
        forecast_date = last_date + pd.DateOffset(months=month)
        features['month'] = forecast_date.month
        features['season'] = pd.cut([forecast_date.month], bins=[0, 3, 6, 9, 12], labels=['Winter', 'Spring', 'Summer', 'Fall'])[0]
        
        # Indicadores de tendência
        features['trend_increasing'] = int(trend_direction == 'increasing')
        features['trend_decreasing'] = int(trend_direction == 'decreasing')
        features['trend_no_trend'] = int(trend_direction == 'no trend')
        
        # Precipitação (se disponível)
        if precip_forecast is not None and month <= len(precip_forecast):
            for col in precip_forecast.columns:
                if col in precip_forecast.columns:
                    features[col] = precip_forecast.iloc[month-1][col]
        
        # Converter para DataFrame
        features_df = pd.DataFrame([features])
        
        # Codificar variáveis categóricas
        categorical_cols = ['trend_direction', 'season']
        for col in categorical_cols:
            if col in features_df.columns:
                features_df = pd.get_dummies(features_df, columns=[col], drop_first=True)
        
        # Garantir que as colunas correspondem ao modelo treinado
        model_features = model.feature_names_in_
        for col in model_features:
            if col not in features_df.columns:
                features_df[col] = 0
        features_df = features_df[model_features]
        
        # Fazer previsão
        prediction = model.predict(features_df)[0]
        
        # Calcular data da previsão
        forecast_date = last_date + pd.DateOffset(months=month)
        
        predictions.append({
            'data': forecast_date,
            'gwl_previsto': prediction,
            'mes': month
        })
        
        # Adicionar à série atual para próximas previsões
        new_row = pd.DataFrame([{
            'gwl': prediction,
            'gwl_lag_1': current_data.iloc[-1]['gwl'],
            'gwl_lag_2': current_data.iloc[-2]['gwl'] if len(current_data) > 1 else current_data.iloc[-1]['gwl'],
            'gwl_lag_3': current_data.iloc[-3]['gwl'] if len(current_data) > 2 else current_data.iloc[-1]['gwl'],
            'trend_slope': trend_slope,
            'trend_direction': trend_direction,
            'month': forecast_date.month,
            'season': pd.cut([forecast_date.month], bins=[0, 3, 6, 9, 12], labels=['Winter', 'Spring', 'Summer', 'Fall'])[0],
            'trend_increasing': int(trend_direction == 'increasing'),
            'trend_decreasing': int(trend_direction == 'decreasing'),
            'trend_no_trend': int(trend_direction == 'no trend')
        }], index=[forecast_date])
        current_data = pd.concat([current_data, new_row])
        last_date = forecast_date # Update last_date for the next iteration
    
    predictions_df = pd.DataFrame(predictions)
    predictions_df = predictions_df.set_index('data')
    
    log(f"Previsões geradas para {len(predictions_df)} meses")
    for _, row in predictions_df.iterrows():
        log(f"  {row.name.strftime('%Y-%m')}: {row['gwl_previsto']:.3f}")
    
    return predictions_df

def plot_results(analyzer, model_results, predictions, codigo_poco):
    """
    Cria gráfico final com valores reais vs previstos e previsões futuras.
    """
    log("=== CRIANDO GRÁFICO FINAL ===")
    
    # Obter dados do último segmento
    last_seg = analyzer.filled_segments[-1]
    historical_data = last_seg['filled_data']
    
    # Criar figura
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Série temporal completa com previsões
    ax1 = axes[0, 0]
    ax1.plot(historical_data.index, historical_data.values, 'b-', linewidth=2, label='Dados históricos', alpha=0.8)
    
    if model_results and 'y_test' in model_results:
        # Plotar dados de teste vs preditos
        test_dates = model_results['X_test'].index
        ax1.plot(test_dates, model_results['y_test'], 'go', markersize=4, label='Dados de teste', alpha=0.7)
        ax1.plot(test_dates, model_results['y_pred'], 'ro', markersize=4, label='Previsões (teste)', alpha=0.7)
    
    if predictions is not None:
        # Plotar previsões futuras
        ax1.plot(predictions.index, predictions['gwl_previsto'], 'r--', linewidth=3, label='Previsões futuras', alpha=0.8)
        ax1.scatter(predictions.index, predictions['gwl_previsto'], c='red', s=100, alpha=0.8, zorder=5)
    
    ax1.set_title(f'Groundwater Level - Poço {codigo_poco}\nHistórico vs Previsões')
    ax1.set_xlabel('Data')
    ax1.set_ylabel('Nível de Água (m)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Scatter plot: Real vs Predito
    ax2 = axes[0, 1]
    if model_results and 'y_test' in model_results:
        ax2.scatter(model_results['y_test'], model_results['y_pred'], alpha=0.6, c='blue')
        ax2.plot([model_results['y_test'].min(), model_results['y_test'].max()], 
                [model_results['y_test'].min(), model_results['y_test'].max()], 'r--', linewidth=2)
        ax2.set_xlabel('Valores Reais')
        ax2.set_ylabel('Valores Previstos')
        ax2.set_title('Real vs Predito (Teste)')
        ax2.grid(True, alpha=0.3)
    
    # 3. Feature Importance
    ax3 = axes[1, 0]
    if model_results and 'feature_importance' in model_results:
        top_features = model_results['feature_importance'].head(10)
        ax3.barh(range(len(top_features)), top_features['importance'])
        ax3.set_yticks(range(len(top_features)))
        ax3.set_yticklabels(top_features['feature'])
        ax3.set_xlabel('Importância')
        ax3.set_title('Top 10 Features Mais Importantes')
        ax3.grid(True, alpha=0.3)
    
    # 4. Resíduos
    ax4 = axes[1, 1]
    if model_results and 'y_test' in model_results:
        residuals = model_results['y_test'] - model_results['y_pred']
        ax4.scatter(model_results['y_pred'], residuals, alpha=0.6, c='green')
        ax4.axhline(y=0, color='red', linestyle='--', linewidth=2)
        ax4.set_xlabel('Valores Previstos')
        ax4.set_ylabel('Resíduos')
        ax4.set_title('Análise de Resíduos')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Salvar gráfico
    plot_path = RESULTS_DIR / f"plot_{codigo_poco.replace('/', '_')}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    log(f"Gráfico salvo em: {plot_path}")
    
    plt.show()

def export_trend_json(analyzer, output_path):
    """Exporta resultados de tendência em JSON. Mantém formato original."""
    segments_json = []
    for seg, trend in zip(analyzer.filled_segments, analyzer.trend_results):
        x = np.arange(len(seg['filled_data']))
        if 'linear_regression' in trend:
            slope = trend['linear_regression']['slope_per_year'] / 12
            intercept = trend['linear_regression']['intercept']
            trend_line = [float(intercept + slope * i) for i in range(len(x))]
            trend_points = [{"x": str(date.date()), "y": float(val)} for date, val in zip(seg['filled_data'].index, trend_line)]
        else:
            trend_points = []

        segments_json.append({
            "start_date": str(seg['start_date'].date()),
            "end_date": str(seg['end_date'].date()),
            "trend_type": trend.get('mann_kendall', {}).get('trend', 'no trend'),
            "p_value": trend.get('mann_kendall', {}).get('p_value', None),
            "slope_per_year": trend.get('mann_kendall', {}).get('slope_per_year', None),
            "r_squared": trend.get('linear_regression', {}).get('r_squared', None),
            "trend_points": trend_points
        })

    result = {
        "segments": segments_json,
        "overall": {
            "significant": any(seg.get("p_value", 1) is not None and seg.get("p_value", 1) < 0.05 for seg in segments_json),
            "method": "Mann-Kendall"
        }
    }
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

# ===================== FUNÇÃO PRINCIPAL MELHORADA =====================
def analyze_groundwater_trends(df, date_col='data', value_col='nivel', 
                               monthly_method='mean', fill_method='seasonal_decompose', 
                               min_gap_months=MIN_GAP_MONTHS, alpha=ALPHA):
    """
    Complete 4-step groundwater trend analysis workflow for piezometric data.
    Mantém funcionalidade original.
    """
    analyzer = GroundwaterTrendAnalysis(df, date_col, value_col)
    log("Starting 4-step groundwater trend analysis...")
    analyzer.step1_monthly_aggregation(method=monthly_method)
    analyzer.exploratory_analysis()
    analyzer.step2_segment_by_gaps(min_gap_months=min_gap_months)
    analyzer.step3_fill_missing_data(method=fill_method)
    analyzer.step4_test_trends(alpha=alpha)
    return analyzer

def analyze_single_poco(codigo_poco, use_precipitation=True):
    """
    Função principal para análise individual de um poço.
    Versão melhorada com integração de precipitação e previsão futura.
    """
    log(f"=== ANÁLISE COMPLETA PARA POÇO {codigo_poco} ===")
    
    # 1. Carregar dados do poço
    df, poço_coords = load_poco_data(codigo_poco)
    if df is None:
        return None
    
    # 2. Carregar e integrar dados de precipitação
    precip_data = None
    if use_precipitation and poço_coords and CSV_PRECIP.exists():
        try:
            log("Carregando dados de precipitação...")
            precip_df = pd.read_csv(CSV_PRECIP)
            
            # Encontrar estações próximas
            nearby_stations = find_nearby_precipitation_stations(poço_coords, precip_df)
            
            if nearby_stations:
                # Agregar precipitação mensal
                precip_data = aggregate_precipitation_monthly(precip_df, nearby_stations)
            else:
                log("Nenhuma estação de precipitação próxima encontrada")
        except Exception as e:
            log(f"Erro ao carregar precipitação: {e}")
    
    # 3. Análise de tendências
    analyzer = analyze_groundwater_trends(df)
    
    # 4. Exportar JSON de tendências
    output_path = RESULTS_DIR / f"trend_{codigo_poco.replace('/', '_')}.json"
    export_trend_json(analyzer, output_path)
    log(f"Tendências exportadas para: {output_path}")
    
    # 5. Preparar dados para ML
    ml_df = prepare_data_for_ml(analyzer, precip_data)
    
    # 6. Treinar e avaliar modelo
    model, rmse, model_results = train_predict_gwl(ml_df, codigo_poco)
    
    # 7. Fazer previsão para os próximos meses
    predictions = None
    if model is not None and ml_df is not None:
        # Obter resultados de tendência do último segmento
        last_segment_trend = None
        for trend in analyzer.trend_results:
            if trend['segment_id'] == analyzer.filled_segments[-1]['id']:
                last_segment_trend = trend
                break
        
        if last_segment_trend:
            # Fazer previsão
            predictions = predict_future_gwl(model, ml_df, last_segment_trend, precip_data)
            
            # Salvar previsões
            forecast_path = RESULTS_DIR / f"forecast_{codigo_poco.replace('/', '_')}.json"
            predictions.to_json(forecast_path, orient='index')
            log(f"Previsões salvas em: {forecast_path}")
    
    # 8. Criar gráfico final
    if model_results and predictions is not None:
        plot_results(analyzer, model_results, predictions, codigo_poco)
    
    log(f"Análise completa para poço {codigo_poco} finalizada.")
    return analyzer, model, rmse, predictions

# ===================== EXECUÇÃO DIRETA =====================
if __name__ == '__main__':
    # Exemplo: Analise um poço individual com precipitação
    # Para desativar precipitação: analyze_single_poco('330/183', use_precipitation=False)
    analyze_single_poco('330/183')
    
    # Se quiser analisar outro poço, chame a função novamente:
    # analyze_single_poco('331/15')

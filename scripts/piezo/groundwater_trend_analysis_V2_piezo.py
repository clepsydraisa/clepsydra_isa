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
warnings.filterwarnings('ignore')

# identificar pasta de trabalho
try:
    # .../clepsydra_isa/scripts/piezo/groundwater_trend_analysis_2905.py
    working_dir=Path(__file__).parent.parent.parent # working directory from script location: scripts are in 'scripts' folder
except:
    # pasta MJ
    working_dir=Path(r"C:\Users\mjmartins\OneDrive - Universidade de Lisboa\Documentos\Clepsydra_ISA\clepsydra_isa-main")


# Example usage:
"""
# Load your evenly spaced groundwater data
df = pd.read_csv('your_data.csv')

# Run complete 4-step analysis
analyzer = analyze_groundwater_trends(
    df, 
    date_col='date', 
    value_col='water_level',
    monthly_method='mean',
    fill_method='interpolate',
    min_gap_months=6,
    alpha=0.05
)

# Access results
monthly_data = analyzer.monthly_data
segments = analyzer.segments
filled_segments = analyzer.filled_segments  
trend_results = analyzer.trend_results
"""

def main():
    codigos = [
        '330/183', '331/15', '331/2', '341/17', '342/78', '342/97', '377/54', '377/59', '377/84', '377/86',
        '377/94', '390/208', '390/99', '391/243', '391/33', '391/437', '404/69', '405/17', '405/34', '418/15', '418/4'
    ]
    fn= Path(working_dir) /"resources" / "aquifer_depth_piezo.csv"
    output_dir = Path(working_dir) / "resources" / "trends"
    output_dir.mkdir(parents=True, exist_ok=True)
    for codigo_poco in codigos:
        df=pd.read_csv(fn)
        df=df[df['codigo']==codigo_poco]
        df=df[['data', 'profundidade_nivel_agua']]
        df['nivel'] = -pd.to_numeric(df['profundidade_nivel_agua'], errors='coerce')
        df=df.drop(columns=['profundidade_nivel_agua'] ) 
        analyzer = analyze_groundwater_trends(
            df, 
            date_col='data', 
            value_col='nivel',
            monthly_method='mean',
            fill_method='seasonal_decompose',
            min_gap_months=24,
            alpha=0.05
        )
        output_path = output_dir / f"trend_{codigo_poco.replace('/', '_')}.json"
        export_trend_json(analyzer, output_path)

    # --- Nitrato ---
    codigos_nitrato = [
        "329/341", "329/6", "330/186", "330/187", "330/230", "330/231", "330/233", "330/234", "330/235",
        "331/126", "331/127", "331/129", "331/130", "331/131", "331/133", "341/252", "341/253", "341/254",
        "341/255", "342/113", "342/114", "342/115", "353/22", "353/373", "353/87", "364/259", "364/297",
        "365/15", "365/470", "377/262", "377/264", "377/287", "377/94", "391/244", "391/33", "391/404",
        "391/AG14", "404/69", "405/17", "405/AG6", "418/4", "418/AG49", "419/AG3"
    ]
    fn_nitrato = Path(working_dir) / "resources" / "model_data" / "nitrato" / "nitrato_model_al.csv"
    output_dir_nitrato = Path(working_dir) / "resources" / "trends_nitrato"
    output_dir_nitrato.mkdir(parents=True, exist_ok=True)
    if fn_nitrato.exists():
        df_nitrato = pd.read_csv(fn_nitrato)
        for codigo_poco in codigos_nitrato:
            df = df_nitrato[df_nitrato['codigo'] == codigo_poco][['data', 'nitrato']].copy()
            df['nitrato'] = pd.to_numeric(df['nitrato'], errors='coerce')
            analyzer = analyze_groundwater_trends(
                df,
                date_col='data',
                value_col='nitrato',
                monthly_method='mean',
                fill_method='seasonal_decompose',
                min_gap_months=24,
                alpha=0.05
            )
            output_path = output_dir_nitrato / f"trend_{str(codigo_poco).replace('/', '_')}.json"
            export_trend_json(analyzer, output_path)
    else:
        print(f"Arquivo não encontrado: {fn_nitrato}")

class GroundwaterTrendAnalysis:
    """
    Streamlined groundwater trend analysis for evenly spaced time series
    """
    
    def __init__(self, data, date_col='date', value_col='water_level'):
        """
        Initialize with groundwater data
        
        Parameters:
        data: DataFrame with date and water level columns
        date_col: name of date column
        value_col: name of water level column
        """
        self.data = data.copy()
        self.data[date_col] = pd.to_datetime(self.data[date_col])
        self.data = self.data.set_index(date_col).sort_index()
        self.value_col = value_col
        self.monthly_data = None
        self.segments = []
        self.filled_segments = []
        self.trend_results = []
        
    def step1_monthly_aggregation(self, method='mean'):
        """
        Step 1: Keep one value per month
        
        Parameters:
        method: aggregation method ('mean', 'median', 'first', 'last')
        """
        print("=== STEP 1: MONTHLY AGGREGATION ===")
        
        # Group by year-month and aggregate
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
            
        # Create complete monthly index
        full_range = pd.date_range(start=self.monthly_data.index.min(), 
                                 end=self.monthly_data.index.max(), 
                                 freq='M')
        self.monthly_data = self.monthly_data.reindex(full_range)
        
        print(f"Original data points: {len(self.data)}")
        print(f"Monthly data points: {len(self.monthly_data)}")
        print(f"Missing months: {self.monthly_data.isna().sum()}")
        print(f"Data completeness: {(1 - self.monthly_data.isna().mean())*100:.1f}%")
        
        # Visualize monthly data
        plt.figure(figsize=(15, 6))
        plt.plot(self.monthly_data.index, self.monthly_data.values, 'b-o', markersize=3, alpha=0.7)
        plt.title(f'Monthly Aggregated Data ({method})')
        plt.xlabel('Date')
        plt.ylabel('Water Level')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        

# ADICIONADO 
    def exploratory_analysis(self):
        """Step new: Comprehensive EDA for groundwater data"""
        print(f"\n=== EXPLORATORY DATA ANALYSIS ===")
        
        # Basic statistics
        print(f"Data period: {self.data.index.min()} to {self.data.index.max()}")
        print(f"Total observations: {len(self.monthly_data)}")
        print(f"Missing values: {self.monthly_data.isna().sum()}")

        
        # Identify gaps
        """
        gaps = self.monthly_data.isna()
        if gaps.sum() > 0:
            gap_periods = self.monthly_data._identify_gap_periods()
            print(f"\nLarge gaps (>6 months): {len(gap_periods)}")
            for start, end, length in gap_periods:
                print(f"  {start} to {end}: {length} months")
        """
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Time series plot
        axes[0,0].plot(self.monthly_data.index, self.monthly_data, 'b-', alpha=0.7)
        axes[0,0].set_title('Groundwater Level Time Series')
        axes[0,0].set_ylabel('Water Level')
        axes[0,0].grid(True, alpha=0.3)
        
        # Seasonal decomposition (if enough data)
        try:
            stl = STL(self.monthly_data.dropna(), seasonal=13,period=12)
            decomp = stl.fit()
            axes[0,1].plot(decomp.trend, 'r-', linewidth=2)
            axes[0,1].set_title('Extracted Trend Component')
            axes[0,1].set_ylabel('Trend')
            axes[0,1].grid(True, alpha=0.3)
            
            """
            # Store trend for later use
            self.monthly_data.trend_component = decomp.trend
            """
            
        except Exception as e:
            axes[0,1].text(0.5, 0.5, f'STL decomposition failed:\n{str(e)}', 
                          ha='center', va='center', transform=axes[0,1].transAxes)
        
        # Seasonal patterns
        monthly_data_2 = self.monthly_data.groupby(self.monthly_data.index.month).agg(['mean', 'std'])
        axes[1,0].errorbar(monthly_data_2.index, monthly_data_2['mean'], 
                          yerr=monthly_data_2['std'], marker='o', capsize=5)
        axes[1,0].set_title('Seasonal Pattern (Monthly Averages)')
        axes[1,0].set_xlabel('Month')
        axes[1,0].set_ylabel('Water Level')
        axes[1,0].set_xticks(range(1, 13))
        axes[1,0].grid(True, alpha=0.3)
        
        # Annual means
        annual_data = self.monthly_data.groupby(self.monthly_data.index.year).mean()
        axes[1,1].plot(annual_data.index, annual_data.values, 'go-', markersize=4)
        axes[1,1].set_title('Annual Mean Water Levels')
        axes[1,1].set_xlabel('Year')
        axes[1,1].set_ylabel('Water Level')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # ver a decomposicao da ts
        decomp.plot()
        plt.show()

# FIM DE ADICIONADO        
               
    def step2_segment_by_gaps(self, min_gap_months=6):
        """
        Step 2: Split time series based on gaps of 6+ months
        
        Parameters:
        min_gap_months: minimum gap length to create segment boundary
        """
        print(f"\n=== STEP 2: SEGMENT BY GAPS (≥{min_gap_months} months) ===")
        
        if self.monthly_data is None:
            raise ValueError("Run step1_monthly_aggregation first")
            
        # Find gap periods
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
                
        # Handle case where series ends with a gap
        if in_gap and (len(na_mask) - gap_start) >= min_gap_months:
            gap_starts.append(gap_start)
            gap_ends.append(len(na_mask))
            
        print(f"Found {len(gap_starts)} significant gaps:")
        
        # Create segments based on gaps
        segment_boundaries = [0]
        
        for gap_start, gap_end in zip(gap_starts, gap_ends):
            segment_boundaries.append(gap_start)
            segment_boundaries.append(gap_end)
            
        segment_boundaries.append(len(self.monthly_data))
        
        # Remove duplicates and sort
        segment_boundaries = sorted(list(set(segment_boundaries)))
        
        # Create segments (skip gap periods)
        self.segments = []
        for i in range(0, len(segment_boundaries)-1, 2):
            start_idx = segment_boundaries[i]
            end_idx = segment_boundaries[i+1] if i+1 < len(segment_boundaries) else len(self.monthly_data)
            
            start_date = self.monthly_data.index[start_idx]
            end_date = self.monthly_data.index[min(end_idx-1, len(self.monthly_data)-1)]
            
            segment_data = self.monthly_data.iloc[start_idx:end_idx]
            
            # Only keep segments with some data
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
                
        print(f"Created {len(self.segments)} segments:")
        for seg in self.segments:
            print(f"  Segment {seg['id']}: {seg['start_date'].strftime('%Y-%m')} to {seg['end_date'].strftime('%Y-%m')}")
            print(f"    Length: {seg['length_months']} months, Valid: {seg['valid_months']} ({seg['completeness']*100:.1f}%)")
            
        # Visualize segments
        self._plot_segments()
        
    def _plot_segments(self):
        """Visualize the segmented time series"""
        plt.figure(figsize=(15, 8))
        
        # Plot original monthly data
        plt.subplot(2, 1, 1)
        plt.plot(self.monthly_data.index, self.monthly_data.values, 'k-', alpha=0.3, label='All data')
        
        # Color each segment differently
        colors = plt.cm.Set1(np.linspace(0, 1, len(self.segments)))
        for seg, color in zip(self.segments, colors):
            seg_data = seg['data'].dropna()
            if len(seg_data) > 0:
                plt.plot(seg_data.index, seg_data.values, 'o-', 
                        color=color, linewidth=2, markersize=4, 
                        label=f"Segment {seg['id']}")
                
        plt.title('Time Series Segments')
        plt.ylabel('Water Level')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot segment statistics
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
        
        
    def step3_fill_missing_data(self, method='interpolate', **kwargs):
        """
        Step 3: Fill missing data within each segment
        
        Parameters:
        method: filling method ('interpolate', 'seasonal_decompose', 'forward_fill', 'backward_fill', 'mean')
        **kwargs: additional parameters for the chosen method
        """
        print(f"\n=== STEP 3: FILL MISSING DATA ({method}) ===")
        
        if not self.segments:
            raise ValueError("Run step2_segment_by_gaps first")
            
        self.filled_segments = []
        
        for seg in self.segments:
            print(f"\nProcessing Segment {seg['id']}:")
            print(f"  Missing values: {seg['data'].isna().sum()}/{len(seg['data'])}")
            
            if seg['data'].isna().sum() == 0:
                print("  No missing values - using original data")
                filled_data = seg['data'].copy()
            else:
                filled_data = self._fill_segment_gaps(seg['data'], method, **kwargs)
                
            filled_segment = seg.copy()
            filled_segment['filled_data'] = filled_data
            filled_segment['fill_method'] = method
            self.filled_segments.append(filled_segment)
            
            print(f"  After filling: {filled_data.isna().sum()} missing values")
            
        # Visualize filled data
        self._plot_filled_segments()
        
    def _fill_segment_gaps(self, data, method, **kwargs):
        """Fill gaps in a single segment"""
        filled_data = data.copy()
        
        if method == 'interpolate':
            # Linear interpolation (default) or other scipy methods
            interp_method = kwargs.get('interp_method', 'linear')
            if interp_method in ['linear', 'quadratic', 'cubic']:
                filled_data = data.interpolate(method=interp_method)
            else:
                # Use scipy interpolation for more options
                valid_idx = ~data.isna()
                if valid_idx.sum() >= 2:  # Need at least 2 points
                    valid_dates = np.arange(len(data))[valid_idx]
                    valid_values = data.values[valid_idx]
                    
                    if len(valid_values) > 1:
                        f = interp1d(valid_dates, valid_values, 
                                   kind=interp_method, 
                                   bounds_error=False, 
                                   fill_value='extrapolate')
                        filled_data = pd.Series(f(np.arange(len(data))), index=data.index)
                        
        elif method == 'seasonal_decompose':
            # Fill using seasonal decomposition
            min_periods = kwargs.get('min_periods', 24)  # Need 2 years minimum
            if len(data.dropna()) >= min_periods:
                try:
                    # First do basic interpolation for STL
                    temp_filled = data.interpolate(method='linear', limit=3)
                    
                    if temp_filled.notna().sum() >= min_periods:
                        stl = STL(temp_filled.dropna(), seasonal=13, robust=True)
                        decomp = stl.fit()
                        
                        # Predict missing values using trend + seasonal
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
        """Visualize original vs filled data for each segment"""
        n_segments = len(self.filled_segments)
        fig, axes = plt.subplots(n_segments, 1, figsize=(15, 4*n_segments))
        
        if n_segments == 1:
            axes = [axes]
            
        for i, seg in enumerate(self.filled_segments):
            # Plot original data
            axes[i].plot(seg['data'].index, seg['data'].values, 'ko-', 
                        markersize=3, alpha=0.7, label='Original')
            
            # Plot filled data
            axes[i].plot(seg['filled_data'].index, seg['filled_data'].values, 'r-', 
                        linewidth=2, alpha=0.8, label='Filled')
            
            # Highlight filled points
            filled_mask = seg['data'].isna()
            if filled_mask.sum() > 0:
                filled_points = seg['filled_data'][filled_mask]
                axes[i].plot(filled_points.index, filled_points.values, 'ro', 
                           markersize=5, label='Filled points')
            
            axes[i].set_title(f"Segment {seg['id']}: {seg['start_date'].strftime('%Y-%m')} to {seg['end_date'].strftime('%Y-%m')}")
            axes[i].set_ylabel('Water Level')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.show()
        
    def step4_test_trends(self, alpha=0.05):
        """
        Step 4: Test for significant trends in each filled segment
        
        Parameters:
        alpha: significance level for trend tests
        """
        print(f"\n=== STEP 4: TREND TESTING (α={alpha}) ===")
        
        if not self.filled_segments:
            raise ValueError("Run step3_fill_missing_data first")
            
        self.trend_results = []
        
        for seg in self.filled_segments:
            print(f"\n--- Segment {seg['id']} Trend Analysis ---")
            
            data = seg['filled_data'].dropna()
            if len(data) < 12:  # Need at least 1 year
                print("  Insufficient data for trend analysis")
                continue
                
            result = self._test_segment_trend(data, seg['id'], alpha)
            self.trend_results.append(result)
            
        # Summary table
        self._create_trend_summary()
        
    def _test_segment_trend(self, data, segment_id, alpha):
        """Test trends for a single segment"""
        result = {
            'segment_id': segment_id,
            'n_observations': len(data),
            'data_years': len(data) / 12,
            'start_date': data.index[0],
            'end_date': data.index[-1]
        }
        
        # 1. Mann-Kendall Test
        try:
            mk_result = mk.original_test(data.values, alpha=alpha)
            result['mann_kendall'] = {
                'trend': mk_result.trend,
                'p_value': mk_result.p,
                'tau': mk_result.Tau,
                'slope': mk_result.slope,
                'slope_per_year': mk_result.slope * 12,  # Convert to per-year
                'significant': mk_result.p < alpha
            }
            print(f"  Mann-Kendall: {mk_result.trend} (p={mk_result.p:.4f}, τ={mk_result.Tau:.3f})")
            print(f"  Sen's slope: {mk_result.slope*12:.4f} units/year")
            
        except Exception as e:
            print(f"  Mann-Kendall test failed: {e}")
            
        # 2. Seasonal Mann-Kendall Test (if enough data)
        if len(data) >= 24:  # Need at least 2 years
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
                print(f"  Seasonal Mann-Kendall: {smk_result.trend} (p={smk_result.p:.4f})")
                print(f"  Seasonal Sen's slope: {smk_result.slope*12:.4f} units/year")
                
            except Exception as e:
                print(f"  Seasonal Mann-Kendall test failed: {e}")
  
       
       # 3.  Hamed Rao modified Mann-Kendall test for autocorrelated data
       
        try:
           HRmk_result = mk.hamed_rao_modification_test(data.values, alpha=alpha)
           result['hamed_rao_modification'] = {
               'trend': HRmk_result.trend,
               'p_value': HRmk_result.p,
               'tau': HRmk_result.Tau,
               'slope': HRmk_result.slope,
               'slope_per_year': HRmk_result.slope * 12,  # Convert to per-year
               'significant': HRmk_result.p < alpha
           }
           print(f"  Hamed Rao Mann-Kendall: {HRmk_result.trend} (p={HRmk_result.p:.4f}, τ={HRmk_result.Tau:.3f})")
           print(f"  Sen's slope: {HRmk_result.slope*12:.4f} units/year")
           
        except Exception as e:
           print(f" Hamed Rao Mann-Kendall test failed: {e}")
      
    
        # 4. Linear Regression
        try:
            x = np.arange(len(data))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, data.values)
            
            # Convert to per-year
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
            print(f"  Linear regression: {trend_direction} (p={p_value:.4f}, R²={r_value**2:.3f})")
            print(f"  Linear slope: {slope_per_year:.4f} ± {std_err*12:.4f} units/year")
            
        except Exception as e:
            print(f"  Linear regression failed: {e}")
            
        return result
    
    def _create_trend_summary(self):
        """Create summary table of all trend results"""
        print(f"\n{'='*80}")
        print("TREND ANALYSIS SUMMARY")
        print(f"{'='*80}")
        
        if not self.trend_results:
            print("No trend results available")
            return
            
        # Create summary table
        summary_data = []
        for result in self.trend_results:
            row = {
                'Segment': result['segment_id'],
                'Period': f"{result['start_date'].strftime('%Y-%m')} to {result['end_date'].strftime('%Y-%m')}",
                'Years': f"{result['data_years']:.1f}",
                'N_obs': result['n_observations']
            }
            
            # Add Mann-Kendall results
            if 'mann_kendall' in result:
                mk = result['mann_kendall']
                row['MK_Trend'] = mk['trend']
                row['MK_p_value'] = f"{mk['p_value']:.4f}"
                row['MK_Slope_yr'] = f"{mk['slope_per_year']:.4f}"
                
            # Add Seasonal Mann-Kendall if available
            if 'seasonal_mann_kendall' in result:
                smk = result['seasonal_mann_kendall']
                row['SMK_Trend'] = smk['trend']
                row['SMK_p_value'] = f"{smk['p_value']:.4f}"
                row['SMK_Slope_yr'] = f"{smk['slope_per_year']:.4f}"
                
            # Add Hamed-Rao modified Mann-Kendall if available
            if 'hamed_rao_modification' in result:
                HRsmk = result['hamed_rao_modification']
                row['HRMK_Trend'] = HRsmk['trend']
                row['HRMK_p_value'] = f"{HRsmk['p_value']:.4f}"
                row['HRSMK_Slope_yr'] = f"{HRsmk['slope_per_year']:.4f}"
           
            # Add linear regression
            if 'linear_regression' in result:
                lr = result['linear_regression']
                row['Lin_Slope_yr'] = f"{lr['slope_per_year']:.4f}"
                row['Lin_p_value'] = f"{lr['p_value']:.4f}"
                row['R²'] = f"{lr['r_squared']:.3f}"
                
            summary_data.append(row)
            
        # Display table
        df_summary = pd.DataFrame(summary_data).transpose()
        print("\nDetailed Results:")
        print(df_summary.to_string(index=True))
        
        # Overall summary
        print(f"\n{'='*50}")
        print("OVERALL SUMMARY")
        print(f"{'='*50}")
        
        total_segments = len(self.trend_results)
        
        # Count significant trends
        mk_significant = sum(1 for r in self.trend_results 
                           if r.get('mann_kendall', {}).get('significant', False))
        smk_significant = sum(1 for r in self.trend_results 
                            if r.get('seasonal_mann_kendall', {}).get('significant', False))
        HRsmk_significant = sum(1 for r in self.trend_results 
                    if r.get('hamed_rao_modification', {}).get('significant', False))
        lr_significant = sum(1 for r in self.trend_results 
                           if r.get('linear_regression', {}).get('significant', False))
        
        print(f"Total segments analyzed: {total_segments}")
        print(f"Significant trends (Mann-Kendall): {mk_significant}/{total_segments}")
        print(f"Significant trends (Seasonal MK): {smk_significant}/{total_segments}")
        print(f"Significant trends (Hamed Rao modified MK): {HRsmk_significant}/{total_segments}")
        print(f"Significant trends (Linear regression): {lr_significant}/{total_segments}")
        
        # Trend directions
        mk_trends = [r['mann_kendall']['trend'] for r in self.trend_results 
                    if 'mann_kendall' in r and r['mann_kendall']['significant']]
        if mk_trends:
            trend_counts = pd.Series(mk_trends).value_counts()
            print(f"\nSignificant trend directions (Mann-Kendall): {dict(trend_counts)}")
            
        # Plot trend results
        self._plot_trend_results()
        
    def _plot_trend_results(self):
        """Visualize trend test results"""
        if not self.trend_results:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Segments with data and trends
        ax1 = axes[0, 0]
        for i, seg in enumerate(self.filled_segments):
            color = 'red' if any(r['segment_id'] == seg['id'] and 
                               r.get('hamed_rao_modification', {}).get('significant', False) 
                               for r in self.trend_results) else 'blue'
            ax1.plot(seg['filled_data'].index, seg['filled_data'].values, 
                    color=color, alpha=0.7, linewidth=2, 
                    label=f"Segment {seg['id']}" + (" (sig.)" if color=='red' else ""))
        ax1.set_title('Segments with Significant Trends (Red)')
        ax1.set_ylabel('Water Level')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: P-values
        ax2 = axes[0, 1]
        segments = [r['segment_id'] for r in self.trend_results]
        mk_pvals = [r.get('mann_kendall', {}).get('p_value', 1) for r in self.trend_results]
        smk_pvals = [r.get('seasonal_mann_kendall', {}).get('p_value', 1) for r in self.trend_results]
        hrmk_pvals = [r.get('hamed_rao_modification', {}).get('p_value', 1) for r in self.trend_results]
        x = np.arange(len(segments))
        width = 0.25
        
        ax2.bar(x - width, mk_pvals, width, label='Mann-Kendall', alpha=0.7)
        ax2.bar(x , smk_pvals, width, label='Seasonal MK', alpha=0.7)
        ax2.bar(x + width, hrmk_pvals, width, label='Hamed Rao MK', alpha=0.7)
        ax2.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='α=0.05')
        ax2.set_xlabel('Segment')
        ax2.set_ylabel('p-value')
        ax2.set_title('Trend Test p-values')
        ax2.set_xticks(x)
        ax2.set_xticklabels(segments)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
        
        # Plot 3: Trend slopes
        ax3 = axes[1, 0]
        mk_slopes = [r.get('mann_kendall', {}).get('slope_per_year', 0) for r in self.trend_results]
        smk_slopes = [r.get('seasonal_mann_kendall', {}).get('slope_per_year', 0) for r in self.trend_results]
        hrmk_slopes = [r.get('hamed_rao_modification', {}).get('slope_per_year', 0) for r in self.trend_results]
        
        ax3.bar(x - width, mk_slopes, width, label='Mann-Kendall', alpha=0.7)
        ax3.bar(x , smk_slopes, width, label='Seasonal MK', alpha=0.7)
        ax3.bar(x + width, hrmk_slopes, width, label='Hamed Rao MK', alpha=0.7)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax3.set_xlabel('Segment')
        ax3.set_ylabel('Trend Slope (units/year)')
        ax3.set_title('Trend Magnitudes')
        ax3.set_xticks(x)
        ax3.set_xticklabels(segments)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: R-squared values
        ax4 = axes[1, 1]
        r_squared = [r.get('linear_regression', {}).get('r_squared', 0) for r in self.trend_results]
        
        ax4.bar(segments, r_squared, alpha=0.7, color='green')
        ax4.set_xlabel('Segment')
        ax4.set_ylabel('R² (Linear Regression)')
        ax4.set_title('Trend Fit Quality')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# Main function for your specific workflow
def analyze_groundwater_trends(df, date_col='date', value_col='water_level', 
                             monthly_method='mean', fill_method='interpolate', 
                             min_gap_months=6, alpha=0.05):
    """
    Complete 4-step groundwater trend analysis workflow
    
    Parameters:
    df: DataFrame with evenly spaced time series data
    date_col: name of date column
    value_col: name of water level column
    monthly_method: aggregation method for monthly data ('mean', 'median', 'first', 'last')
    fill_method: method for filling missing data ('interpolate', 'seasonal_decompose', etc.)
    min_gap_months: minimum gap size to split segments
    alpha: significance level for trend tests
    
    Returns:
    GroundwaterTrendAnalysis object with all results
    """
    
    # Initialize analyzer
    analyzer = GroundwaterTrendAnalysis(df, date_col, value_col)
    
    # Execute 4-step workflow
    print("Starting 4-step groundwater trend analysis...\n")
    
    # Step 1: Monthly aggregation
    analyzer.step1_monthly_aggregation(method=monthly_method)
    
    # Step new: Exploratory analysis
    analyzer.exploratory_analysis()
    
    # Step 2: Segment by gaps
    analyzer.step2_segment_by_gaps(min_gap_months=min_gap_months)
    
    # Step 3: Fill missing data
    analyzer.step3_fill_missing_data(method=fill_method)
    
    # Step 4: Test trends
    analyzer.step4_test_trends(alpha=alpha)
    
    return analyzer

def export_trend_json(analyzer, output_path):
    segments_json = []
    for seg, trend in zip(analyzer.filled_segments, analyzer.trend_results):
        # Linear regression points for trend line
        x = np.arange(len(seg['filled_data']))
        y = seg['filled_data'].values
        if 'linear_regression' in trend:
            slope = trend['linear_regression']['slope_per_year'] / 12  # per month
            intercept = trend['linear_regression']['intercept']
            trend_line = [float(intercept + slope * i) for i in range(len(x))]
            trend_points = [
                {"x": str(date.date()), "y": float(val)}
                for date, val in zip(seg['filled_data'].index, trend_line)
            ]
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

if __name__ == '__main__':
    main()
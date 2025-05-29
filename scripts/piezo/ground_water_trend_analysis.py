import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import kendalltau
import ruptures as rpt
from statsmodels.tsa.seasonal import STL
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller
import pymannkendall as mk
import warnings
warnings.filterwarnings('ignore')

class GroundwaterTrendAnalysis:
    """
    Comprehensive trend analysis for groundwater level time series
    with gaps, seasonality, and potential non-linear trends
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
        self.changepoints = []
        self.segments = []
        self.results = {}
        
    def exploratory_analysis(self):
        """Step 1: Comprehensive EDA for groundwater data"""
        print("=== EXPLORATORY DATA ANALYSIS ===")
        
        # Basic statistics
        print(f"Data period: {self.data.index.min()} to {self.data.index.max()}")
        print(f"Total observations: {len(self.data)}")
        print(f"Missing values: {self.data[self.value_col].isna().sum()}")
        print(f"Data completeness: {(1 - self.data[self.value_col].isna().mean())*100:.1f}%")
        
        # Identify gaps
        gaps = self.data[self.value_col].isna()
        if gaps.sum() > 0:
            gap_periods = self._identify_gap_periods()
            print(f"\nLarge gaps (>6 months): {len(gap_periods)}")
            for start, end, length in gap_periods:
                print(f"  {start} to {end}: {length} months")
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Time series plot
        axes[0,0].plot(self.data.index, self.data[self.value_col], 'b-', alpha=0.7)
        axes[0,0].set_title('Groundwater Level Time Series')
        axes[0,0].set_ylabel('Water Level')
        axes[0,0].grid(True, alpha=0.3)
        
        # Seasonal decomposition (if enough data)
        try:
            stl = STL(self.data[self.value_col].dropna(), seasonal=13)
            decomp = stl.fit()
            
            axes[0,1].plot(decomp.trend, 'r-', linewidth=2)
            axes[0,1].set_title('Extracted Trend Component')
            axes[0,1].set_ylabel('Trend')
            axes[0,1].grid(True, alpha=0.3)
            
            # Store trend for later use
            self.trend_component = decomp.trend
            
        except Exception as e:
            axes[0,1].text(0.5, 0.5, f'STL decomposition failed:\n{str(e)}', 
                          ha='center', va='center', transform=axes[0,1].transAxes)
        
        # Seasonal patterns
        monthly_data = self.data.groupby(self.data.index.month)[self.value_col].agg(['mean', 'std'])
        axes[1,0].errorbar(monthly_data.index, monthly_data['mean'], 
                          yerr=monthly_data['std'], marker='o', capsize=5)
        axes[1,0].set_title('Seasonal Pattern (Monthly Averages)')
        axes[1,0].set_xlabel('Month')
        axes[1,0].set_ylabel('Water Level')
        axes[1,0].set_xticks(range(1, 13))
        axes[1,0].grid(True, alpha=0.3)
        
        # Annual means
        annual_data = self.data.groupby(self.data.index.year)[self.value_col].mean()
        axes[1,1].plot(annual_data.index, annual_data.values, 'go-', markersize=4)
        axes[1,1].set_title('Annual Mean Water Levels')
        axes[1,1].set_xlabel('Year')
        axes[1,1].set_ylabel('Water Level')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Autocorrelation analysis
        valid_data = self.data[self.value_col].dropna()
        if len(valid_data) > 24:
            ljung_box = acorr_ljungbox(valid_data, lags=12, return_df=True)
            print(f"\nAutocorrelation test (Ljung-Box):")
            print(f"Strong autocorrelation detected: {(ljung_box['lb_pvalue'] < 0.05).any()}")
            
    def _identify_gap_periods(self, min_gap_months=6):
        """Identify significant gaps in the data"""
        gaps = []
        na_series = self.data[self.value_col].isna()
        
        in_gap = False
        gap_start = None
        
        for date, is_na in na_series.items():
            if is_na and not in_gap:
                gap_start = date
                in_gap = True
            elif not is_na and in_gap:
                gap_length = (date.year - gap_start.year) * 12 + (date.month - gap_start.month)
                if gap_length >= min_gap_months:
                    gaps.append((gap_start, date, gap_length))
                in_gap = False
                
        return gaps
    
    def detect_changepoints(self, method='pelt', min_segment_length=24):
        """Step 2: Detect changepoints in the time series"""
        print("\n=== CHANGEPOINT DETECTION ===")
        
        # Prepare data (interpolate short gaps for changepoint detection)
        data_for_cp = self.data[self.value_col].interpolate(method='time', limit=6)
        valid_data = data_for_cp.dropna().values
        
        if len(valid_data) < 100:
            print("Insufficient data for reliable changepoint detection")
            return
        
        # Apply changepoint detection
        if method == 'pelt':
            # PELT algorithm - good for multiple changepoints
            algo = rpt.Pelt(model="rbf", min_size=min_segment_length).fit(valid_data)
            changepoints = algo.predict(pen=3.0)  # Adjust penalty as needed
            
        elif method == 'binary_seg':
            # Binary segmentation
            algo = rpt.Binseg(model="l2", min_size=min_segment_length).fit(valid_data)
            changepoints = algo.predict(n_bkps=3)  # Max 3 changepoints
            
        elif method == 'window':
            # Window-based detection
            algo = rpt.Window(width=min_segment_length, model="l2").fit(valid_data)
            changepoints = algo.predict(pen=1.0)
        
        # Convert indices back to dates
        valid_dates = data_for_cp.dropna().index
        self.changepoints = [valid_dates[cp-1] for cp in changepoints[:-1]]  # Exclude last point
        
        print(f"Detected {len(self.changepoints)} changepoints using {method}:")
        for i, cp in enumerate(self.changepoints):
            print(f"  Changepoint {i+1}: {cp.strftime('%Y-%m')}")
            
        # Create segments
        self._create_segments()
        
        # Visualize changepoints
        self._plot_changepoints()
        
    def _create_segments(self):
        """Create time segments based on detected changepoints"""
        segment_bounds = [self.data.index.min()] + self.changepoints + [self.data.index.max()]
        
        self.segments = []
        for i in range(len(segment_bounds)-1):
            start_date = segment_bounds[i]
            end_date = segment_bounds[i+1]
            segment_data = self.data.loc[start_date:end_date, self.value_col]
            
            self.segments.append({
                'start': start_date,
                'end': end_date,
                'data': segment_data,
                'length_years': (end_date - start_date).days / 365.25,
                'n_obs': segment_data.notna().sum()
            })
            
    def _plot_changepoints(self):
        """Visualize detected changepoints"""
        plt.figure(figsize=(15, 6))
        plt.plot(self.data.index, self.data[self.value_col], 'b-', alpha=0.7, label='Water Level')
        
        # Mark changepoints
        for cp in self.changepoints:
            plt.axvline(x=cp, color='red', linestyle='--', alpha=0.8, linewidth=2)
            
        # Color code segments
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.segments)))
        for i, (segment, color) in enumerate(zip(self.segments, colors)):
            segment_data = segment['data'].dropna()
            if len(segment_data) > 0:
                plt.fill_between(segment_data.index, 
                               segment_data.values.min(), 
                               segment_data.values.max(), 
                               alpha=0.2, color=color, 
                               label=f'Segment {i+1}')
        
        plt.title('Changepoint Detection Results')
        plt.xlabel('Date')
        plt.ylabel('Water Level')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
    def analyze_segments(self):
        """Step 3: Analyze trends within each segment"""
        print("\n=== SEGMENT-WISE TREND ANALYSIS ===")
        
        if not self.segments:
            print("No segments defined. Run changepoint detection first.")
            return
            
        self.results = {'segments': []}
        
        for i, segment in enumerate(self.segments):
            print(f"\n--- Segment {i+1}: {segment['start'].strftime('%Y-%m')} to {segment['end'].strftime('%Y-%m')} ---")
            print(f"Duration: {segment['length_years']:.1f} years")
            print(f"Observations: {segment['n_obs']}")
            
            if segment['n_obs'] < 24:  # Need at least 2 years for seasonal analysis
                print("Insufficient data for trend analysis")
                continue
                
            segment_result = self._analyze_single_segment(segment['data'], i+1)
            self.results['segments'].append(segment_result)
            
    def _analyze_single_segment(self, data, segment_num):
        """Analyze trend in a single segment"""
        valid_data = data.dropna()
        
        if len(valid_data) < 12:
            return {'segment': segment_num, 'insufficient_data': True}
            
        results = {'segment': segment_num, 'insufficient_data': False}
        
        # 1. Seasonal Mann-Kendall Test
        try:
            # Create monthly series for seasonal MK
            monthly_series = []
            for month in range(1, 13):
                month_data = valid_data[valid_data.index.month == month]
                if len(month_data) >= 3:  # Need at least 3 years of data for a month
                    monthly_series.extend(month_data.values)
                    
            if len(monthly_series) > 0:
                # Seasonal Mann-Kendall
                smk_result = mk.seasonal_test(valid_data.values, period=12)
                results['seasonal_mk'] = {
                    'trend': smk_result.trend,
                    'p_value': smk_result.p,
                    'tau': smk_result.Tau,
                    'slope': smk_result.slope,
                    'significant': smk_result.p < 0.05
                }
                
                print(f"  Seasonal Mann-Kendall: {smk_result.trend} (p={smk_result.p:.4f})")
                print(f"  Sen's slope: {smk_result.slope:.4f} units/year")
                
        except Exception as e:
            print(f"  Seasonal Mann-Kendall failed: {e}")
            
        # 2. Regular Mann-Kendall (for comparison)
        try:
            mk_result = mk.original_test(valid_data.values)
            results['mann_kendall'] = {
                'trend': mk_result.trend,
                'p_value': mk_result.p,
                'tau': mk_result.Tau,
                'slope': mk_result.slope,
                'significant': mk_result.p < 0.05
            }
            print(f"  Mann-Kendall: {mk_result.trend} (p={mk_result.p:.4f})")
            
        except Exception as e:
            print(f"  Mann-Kendall failed: {e}")
            
        # 3. Linear regression trend
        try:
            # Convert dates to numeric for regression
            x = np.arange(len(valid_data))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, valid_data.values)
            
            # Convert slope to per-year basis
            obs_per_year = len(valid_data) / ((valid_data.index[-1] - valid_data.index[0]).days / 365.25)
            annual_slope = slope * obs_per_year
            
            results['linear_trend'] = {
                'slope_per_year': annual_slope,
                'r_squared': r_value**2,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
            
            print(f"  Linear trend: {annual_slope:.4f} units/year (p={p_value:.4f}, R²={r_value**2:.3f})")
            
        except Exception as e:
            print(f"  Linear regression failed: {e}")
            
        return results
    
    def compare_segments(self):
        """Step 4: Compare trends across segments"""
        print("\n=== SEGMENT COMPARISON ===")
        
        if not self.results.get('segments'):
            print("No segment results available")
            return
            
        # Create comparison table
        comparison_data = []
        for result in self.results['segments']:
            if result.get('insufficient_data'):
                continue
                
            row = {'Segment': result['segment']}
            
            if 'seasonal_mk' in result:
                smk = result['seasonal_mk']
                row['SMK_Trend'] = smk['trend']
                row['SMK_p_value'] = f"{smk['p_value']:.4f}"
                row['Sens_slope'] = f"{smk['slope']:.4f}"
                
            if 'linear_trend' in result:
                lt = result['linear_trend']
                row['Linear_slope'] = f"{lt['slope_per_year']:.4f}"
                row['Linear_p'] = f"{lt['p_value']:.4f}"
                row['R_squared'] = f"{lt['r_squared']:.3f}"
                
            comparison_data.append(row)
            
        if comparison_data:
            df_comparison = pd.DataFrame(comparison_data)
            print("\nTrend Comparison Table:")
            print(df_comparison.to_string(index=False))
            
            # Summary statistics
            significant_trends = []
            for result in self.results['segments']:
                if not result.get('insufficient_data') and 'seasonal_mk' in result:
                    if result['seasonal_mk']['significant']:
                        significant_trends.append(result['seasonal_mk']['trend'])
                        
            print(f"\nSummary:")
            print(f"Segments with significant trends: {len(significant_trends)}")
            if significant_trends:
                trend_counts = pd.Series(significant_trends).value_counts()
                print(f"Trend directions: {dict(trend_counts)}")
    
    def generate_report(self):
        """Generate comprehensive analysis report"""
        print("\n" + "="*60)
        print("GROUNDWATER TREND ANALYSIS REPORT")
        print("="*60)
        
        # Data summary
        print(f"\nDATA SUMMARY:")
        print(f"Period: {self.data.index.min().strftime('%Y-%m')} to {self.data.index.max().strftime('%Y-%m')}")
        print(f"Total duration: {(self.data.index.max() - self.data.index.min()).days/365.25:.1f} years")
        print(f"Data completeness: {(1 - self.data[self.value_col].isna().mean())*100:.1f}%")
        
        # Changepoint summary
        print(f"\nCHANGEPOINT SUMMARY:")
        print(f"Number of changepoints detected: {len(self.changepoints)}")
        print(f"Number of segments: {len(self.segments)}")
        
        # Overall assessment
        if self.results.get('segments'):
            significant_segments = [r for r in self.results['segments'] 
                                  if not r.get('insufficient_data') and 
                                  r.get('seasonal_mk', {}).get('significant', False)]
            
            print(f"\nOVERALL ASSESSMENT:")
            print(f"Segments with significant trends: {len(significant_segments)}/{len(self.results['segments'])}")
            
            if significant_segments:
                trends = [r['seasonal_mk']['trend'] for r in significant_segments]
                slopes = [r['seasonal_mk']['slope'] for r in significant_segments]
                
                print(f"Predominant trend direction: {max(set(trends), key=trends.count)}")
                print(f"Average trend magnitude: {np.mean(slopes):.4f} units/year")
                print(f"Range of trend magnitudes: {min(slopes):.4f} to {max(slopes):.4f} units/year")
        
        print("\nRECOMMendations:")
        print("- Consider hydrogeological factors that might explain changepoints")
        print("- Examine correlation with precipitation/climate data")
        print("- Assess anthropogenic influences (pumping, land use changes)")
        print("- Consider additional analysis for non-linear trends if needed")

# Example usage function
def analyze_groundwater_data(df, date_col='date', value_col='water_level'):
    """
    Main function to run complete analysis
    
    Parameters:
    df: DataFrame with date and water level columns
    date_col: name of date column  
    value_col: name of water level column
    
    Returns:
    GroundwaterTrendAnalysis object with results
    """
    
    # Initialize analyzer
    analyzer = GroundwaterTrendAnalysis(df, date_col, value_col)
    
    # Run complete analysis
    analyzer.exploratory_analysis()
    analyzer.detect_changepoints(method='pelt')
    analyzer.analyze_segments()
    analyzer.compare_segments()
    analyzer.generate_report()
    
    return analyzer

# Example of how to use with your data:
"""
# Load your data
df = pd.read_csv('groundwater_data.csv')  # Replace with your file

# Run analysis
analyzer = analyze_groundwater_data(df, date_col='date', value_col='water_level')

# Access results
results = analyzer.results
changepoints = analyzer.changepoints
segments = analyzer.segments
"""

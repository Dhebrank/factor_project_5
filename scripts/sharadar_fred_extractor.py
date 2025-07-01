#!/usr/bin/env python3
"""
Sharadar FRED Economic Data Extractor for Economic Regime Analysis
Extracts comprehensive economic indicators from Sharadar's economic_data table
for 4-regime framework analysis.

Discovered Database Structure:
- economic_data: Contains time series data with series_id, date, value
- economic_indicators: Contains metadata about indicators
- economic_regimes: Contains pre-computed regime analysis

Author: Claude Code
Date: July 1, 2025
"""

import pandas as pd
import numpy as np
import psycopg2
import logging
from pathlib import Path
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SharadarFREDExtractor:
    """Extract FRED economic data from Sharadar database"""
    
    def __init__(self, output_dir="data/processed"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Database connection parameters
        self.connection_string = "postgresql://futures_user:databento_futures_2025@localhost:5432/sharadar"
        
        # Key FRED indicators for economic regime analysis based on commonly available series
        self.key_indicators = [
            'GDPC1',      # Real GDP
            'CPIAUCSL',   # Consumer Price Index
            'CPILFESL',   # Core CPI
            'UNRATE',     # Unemployment Rate
            'FEDFUNDS',   # Federal Funds Rate
            'DGS10',      # 10-Year Treasury Rate
            'DGS2',       # 2-Year Treasury Rate
            'T10Y2Y',     # 10Y-2Y Treasury Spread
            'T10Y3M',     # 10Y-3M Treasury Spread
            'PAYEMS',     # Nonfarm Payrolls
            'INDPRO',     # Industrial Production
            'HOUST',      # Housing Starts
            'UMCSENT',    # Consumer Sentiment
            'NAPM',       # ISM Manufacturing PMI
            'DCOILWTICO', # Oil Price
            'M2SL',       # Money Supply M2
            'AHETPI',     # Average Hourly Earnings
            'RSAFS',      # Retail Sales
            'PPIACO',     # Producer Price Index
            'VIXCLS'      # VIX (if available)
        ]
        
    def connect_to_database(self):
        """Connect to Sharadar database"""
        try:
            conn = psycopg2.connect(self.connection_string)
            logger.info("Successfully connected to Sharadar database")
            return conn
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            return None
    
    def get_available_indicators(self, conn):
        """Get all available economic indicators from the database"""
        query = """
        SELECT indicator_id, series_id, title, units, frequency, 
               seasonal_adjustment, category, subcategory, source
        FROM economic_indicators 
        WHERE is_active = true
        ORDER BY series_id;
        """
        
        try:
            indicators_df = pd.read_sql(query, conn)
            logger.info(f"Found {len(indicators_df)} available economic indicators")
            return indicators_df
        except Exception as e:
            logger.error(f"Error getting indicators: {e}")
            return pd.DataFrame()
    
    def get_economic_data_for_series(self, conn, series_id, start_date='1990-01-01'):
        """Extract data for a specific economic series"""
        query = """
        SELECT ed.date, ed.value, ed.change_1d, ed.change_1w, ed.change_1m, 
               ed.percentile_1y, ed.z_score_1y
        FROM economic_data ed
        JOIN economic_indicators ei ON ed.indicator_id = ei.indicator_id
        WHERE ei.series_id = %s 
        AND ed.date >= %s
        AND ed.value IS NOT NULL
        ORDER BY ed.date;
        """
        
        try:
            df = pd.read_sql(query, conn, params=[series_id, start_date])
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df['series_id'] = series_id
                logger.info(f"✅ {series_id}: {len(df)} observations from {df['date'].min()} to {df['date'].max()}")
            else:
                logger.warning(f"❌ {series_id}: No data found")
            return df
        except Exception as e:
            logger.error(f"Error extracting {series_id}: {e}")
            return pd.DataFrame()
    
    def extract_all_economic_data(self, start_date='1990-01-01'):
        """Extract all key economic indicators"""
        logger.info("Starting comprehensive economic data extraction from Sharadar...")
        
        conn = self.connect_to_database()
        if not conn:
            return None, {}
        
        try:
            # Get available indicators
            available_indicators = self.get_available_indicators(conn)
            logger.info(f"Available series: {available_indicators['series_id'].tolist()}")
            
            # Find intersection with our key indicators
            available_series = set(available_indicators['series_id'].tolist())
            key_series = set(self.key_indicators)
            extractable_series = key_series.intersection(available_series)
            missing_series = key_series - available_series
            
            logger.info(f"🎯 Can extract {len(extractable_series)} of {len(key_series)} key indicators")
            logger.info(f"📊 Extractable: {sorted(extractable_series)}")
            if missing_series:
                logger.info(f"❌ Missing: {sorted(missing_series)}")
            
            # Extract data for each available series
            all_series_data = []
            extraction_log = {}
            
            for series_id in sorted(extractable_series):
                logger.info(f"Extracting {series_id}...")
                series_df = self.get_economic_data_for_series(conn, series_id, start_date)
                
                if not series_df.empty:
                    all_series_data.append(series_df)
                    extraction_log[series_id] = {
                        'status': 'success',
                        'observations': len(series_df),
                        'start_date': series_df['date'].min().strftime('%Y-%m-%d'),
                        'end_date': series_df['date'].max().strftime('%Y-%m-%d')
                    }
                else:
                    extraction_log[series_id] = {'status': 'no_data'}
            
            # Combine all series into wide format
            if all_series_data:
                logger.info("Combining all series into master dataset...")
                
                # Create master dataset
                master_df = pd.DataFrame()
                
                for series_df in all_series_data:
                    series_id = series_df['series_id'].iloc[0]
                    
                    # Prepare series for merge (keep date and value)
                    merge_df = series_df[['date', 'value']].copy()
                    merge_df.columns = ['date', series_id]
                    
                    # Merge with master
                    if master_df.empty:
                        master_df = merge_df.copy()
                    else:
                        master_df = pd.merge(master_df, merge_df, on='date', how='outer')
                
                # Sort by date
                master_df = master_df.sort_values('date').reset_index(drop=True)
                
                logger.info(f"Master dataset created: {len(master_df)} observations, {len(master_df.columns)-1} indicators")
                logger.info(f"Date range: {master_df['date'].min()} to {master_df['date'].max()}")
                
                return master_df, extraction_log, available_indicators
            else:
                logger.error("No data extracted successfully")
                return None, extraction_log, available_indicators
                
        except Exception as e:
            logger.error(f"Error in extraction process: {e}")
            return None, {}, pd.DataFrame()
        finally:
            conn.close()
    
    def create_derived_indicators(self, df):
        """Create derived indicators for regime analysis"""
        df = df.copy()
        
        # Calculate year-over-year changes for key indicators
        indicators_for_yoy = ['GDPC1', 'CPIAUCSL', 'CPILFESL', 'PAYEMS', 'INDPRO', 'M2SL', 'AHETPI']
        
        for indicator in indicators_for_yoy:
            if indicator in df.columns:
                # Calculate YoY change
                df[f'{indicator}_YOY'] = df[indicator].pct_change(12) * 100
        
        # Calculate yield curve indicators
        if 'DGS10' in df.columns and 'DGS2' in df.columns:
            df['YIELD_CURVE_10Y2Y'] = df['DGS10'] - df['DGS2']
        
        if 'DGS10' in df.columns and 'FEDFUNDS' in df.columns:
            df['TERM_PREMIUM'] = df['DGS10'] - df['FEDFUNDS']
        
        # Real interest rates (approximation)
        if 'DGS10' in df.columns and 'CPIAUCSL_YOY' in df.columns:
            df['REAL_10Y_RATE'] = df['DGS10'] - df['CPIAUCSL_YOY']
        
        return df
    
    def create_regime_composites(self, df):
        """Create composite indicators for economic regime classification"""
        df = df.copy()
        
        # Growth composite
        growth_indicators = []
        potential_growth = ['GDPC1_YOY', 'PAYEMS_YOY', 'INDPRO_YOY', 'NAPM', 'UMCSENT']
        
        for indicator in potential_growth:
            if indicator in df.columns:
                growth_indicators.append(indicator)
        
        if growth_indicators:
            # Normalize and combine
            growth_data = df[growth_indicators].copy()
            for col in growth_indicators:
                growth_data[col] = (growth_data[col] - growth_data[col].mean()) / growth_data[col].std()
            df['GROWTH_COMPOSITE'] = growth_data.mean(axis=1, skipna=True)
        
        # Inflation composite
        inflation_indicators = []
        potential_inflation = ['CPIAUCSL_YOY', 'CPILFESL_YOY', 'PPIACO', 'AHETPI_YOY']
        
        for indicator in potential_inflation:
            if indicator in df.columns:
                inflation_indicators.append(indicator)
        
        if inflation_indicators:
            inflation_data = df[inflation_indicators].copy()
            for col in inflation_indicators:
                inflation_data[col] = (inflation_data[col] - inflation_data[col].mean()) / inflation_data[col].std()
            df['INFLATION_COMPOSITE'] = inflation_data.mean(axis=1, skipna=True)
        
        # Economic regime classification
        if 'GROWTH_COMPOSITE' in df.columns and 'INFLATION_COMPOSITE' in df.columns:
            def classify_regime(row):
                growth = row['GROWTH_COMPOSITE']
                inflation = row['INFLATION_COMPOSITE']
                
                if pd.isna(growth) or pd.isna(inflation):
                    return 'Unknown'
                
                if growth > 0 and inflation > 0:
                    return 'Overheating'  # Rising Growth + Rising Inflation
                elif growth > 0 and inflation <= 0:
                    return 'Goldilocks'   # Rising Growth + Falling Inflation
                elif growth <= 0 and inflation > 0:
                    return 'Stagflation'  # Falling Growth + Rising Inflation
                else:
                    return 'Recession'    # Falling Growth + Falling Inflation
            
            df['ECONOMIC_REGIME'] = df.apply(classify_regime, axis=1)
        
        return df
    
    def get_existing_regimes(self, conn):
        """Get existing regime data from economic_regimes table"""
        query = """
        SELECT date, regime_type, risk_score, volatility_percentile, 
               gdp_trend, inflation_trend, employment_trend, market_regime,
               vix_level, vix_percentile, regime_score
        FROM economic_regimes
        ORDER BY date;
        """
        
        try:
            regimes_df = pd.read_sql(query, conn)
            if not regimes_df.empty:
                regimes_df['date'] = pd.to_datetime(regimes_df['date'])
                logger.info(f"Found {len(regimes_df)} existing regime observations")
            return regimes_df
        except Exception as e:
            logger.warning(f"Could not load existing regimes: {e}")
            return pd.DataFrame()
    
    def generate_metadata(self, extraction_log, available_indicators):
        """Generate comprehensive metadata"""
        metadata = {
            'extraction_date': datetime.now().isoformat(),
            'data_source': 'Sharadar Economic Data (FRED)',
            'purpose': 'Economic regime analysis for factor investing',
            'database_structure': {
                'economic_data': 'Time series data with series_id, date, value',
                'economic_indicators': 'Metadata about economic indicators',
                'economic_regimes': 'Pre-computed regime classifications'
            },
            'regime_framework': {
                'description': '4-regime economic framework',
                'regimes': {
                    'Goldilocks': 'Rising Growth + Falling Inflation',
                    'Overheating': 'Rising Growth + Rising Inflation', 
                    'Stagflation': 'Falling Growth + Rising Inflation',
                    'Recession': 'Falling Growth + Falling Inflation'
                }
            },
            'target_indicators': self.key_indicators,
            'extraction_log': extraction_log,
            'successful_extractions': len([k for k, v in extraction_log.items() if v.get('status') == 'success']),
            'total_targets': len(self.key_indicators),
            'available_indicators_count': len(available_indicators),
            'available_indicators': available_indicators.to_dict('records') if not available_indicators.empty else []
        }
        
        return metadata
    
    def run_economic_data_extraction(self, start_date='1990-01-01'):
        """Main execution function"""
        logger.info("🏛️ Starting Sharadar Economic Data Extraction for Regime Analysis")
        logger.info("="*80)
        
        # Extract all economic data
        economic_df, extraction_log, available_indicators = self.extract_all_economic_data(start_date)
        
        if economic_df is None:
            logger.error("Economic data extraction failed")
            return None
        
        # Create derived indicators
        logger.info("Creating derived indicators (YoY changes, yield curves, etc.)...")
        enhanced_df = self.create_derived_indicators(economic_df)
        
        # Create regime composites
        logger.info("Creating economic regime composites...")
        final_df = self.create_regime_composites(enhanced_df)
        
        # Try to get existing regime data
        conn = self.connect_to_database()
        if conn:
            existing_regimes = self.get_existing_regimes(conn)
            if not existing_regimes.empty:
                # Merge with existing regimes
                final_df = pd.merge(final_df, existing_regimes, on='date', how='left', suffixes=('', '_existing'))
            conn.close()
        
        # Save main dataset
        output_file = self.output_dir / 'fred_economic_data.csv'
        final_df.to_csv(output_file, index=False)
        logger.info(f"✅ Economic data saved: {output_file} ({len(final_df)} rows, {len(final_df.columns)} columns)")
        
        # Generate and save metadata
        metadata = self.generate_metadata(extraction_log, available_indicators)
        metadata_file = self.output_dir / 'fred_economic_data_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info(f"✅ Metadata saved: {metadata_file}")
        
        # Print summary
        logger.info("\n" + "="*80)
        logger.info("🎯 ECONOMIC DATA EXTRACTION SUMMARY:")
        logger.info("="*80)
        logger.info(f"📊 Target Indicators: {len(self.key_indicators)}")
        logger.info(f"✅ Successful Extractions: {metadata['successful_extractions']}")
        logger.info(f"📅 Date Range: {final_df['date'].min()} to {final_df['date'].max()}")
        logger.info(f"📈 Total Observations: {len(final_df)}")
        logger.info(f"🏛️ Final Dataset Columns: {len(final_df.columns)}")
        
        successful_series = [k for k, v in extraction_log.items() if v.get('status') == 'success']
        logger.info(f"📋 Successfully Extracted: {successful_series}")
        
        if 'ECONOMIC_REGIME' in final_df.columns:
            regime_counts = final_df['ECONOMIC_REGIME'].value_counts()
            logger.info(f"🎯 Regime Distribution:")
            for regime, count in regime_counts.items():
                pct = count / len(final_df) * 100
                logger.info(f"   {regime}: {count} observations ({pct:.1f}%)")
        
        return final_df, metadata

if __name__ == "__main__":
    # Initialize and run economic data extraction
    extractor = SharadarFREDExtractor()
    economic_data, metadata = extractor.run_economic_data_extraction()
    
    if economic_data is not None:
        print("\n🏆 SHARADAR ECONOMIC DATA EXTRACTION COMPLETE! 🏆")
        print("Ready for economic regime analysis and factor strategy enhancement.")
        print(f"Dataset: {len(economic_data)} observations with {len(economic_data.columns)} variables")
    else:
        print("\n❌ Economic data extraction failed. Check database connection.")
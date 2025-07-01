#!/usr/bin/env python3
"""
FRED Economic Data Extractor for Economic Regime Analysis
Extracts comprehensive economic indicators from Sharadar FRED database for 4-regime framework:
- Rising/Falling Growth × Rising/Falling Inflation

Economic Regime Framework:
1. Rising Growth + Rising Inflation = Overheating
2. Rising Growth + Falling Inflation = Goldilocks
3. Falling Growth + Rising Inflation = Stagflation  
4. Falling Growth + Falling Inflation = Recession

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

class FREDDataExtractor:
    """Extract FRED economic data for regime analysis"""
    
    def __init__(self, output_dir="data/processed"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Database connection parameters
        self.connection_string = "postgresql://futures_user:databento_futures_2025@localhost:5432/sharadar"
        
        # Key FRED indicators for economic regime analysis
        self.fred_indicators = {
            # GROWTH INDICATORS
            'GDP_GROWTH': {
                'series_id': 'GDPC1',
                'name': 'Real GDP (Quarterly)',
                'category': 'growth',
                'frequency': 'quarterly',
                'transformation': 'yoy_change'
            },
            'INDUSTRIAL_PRODUCTION': {
                'series_id': 'INDPRO', 
                'name': 'Industrial Production Index',
                'category': 'growth',
                'frequency': 'monthly',
                'transformation': 'yoy_change'
            },
            'EMPLOYMENT_GROWTH': {
                'series_id': 'PAYEMS',
                'name': 'Nonfarm Payrolls',
                'category': 'growth', 
                'frequency': 'monthly',
                'transformation': 'yoy_change'
            },
            'UNEMPLOYMENT_RATE': {
                'series_id': 'UNRATE',
                'name': 'Unemployment Rate',
                'category': 'growth',
                'frequency': 'monthly',
                'transformation': 'level'
            },
            'ISM_PMI': {
                'series_id': 'NAPM',
                'name': 'ISM Manufacturing PMI',
                'category': 'growth',
                'frequency': 'monthly', 
                'transformation': 'level'
            },
            'ISM_SERVICES': {
                'series_id': 'NAPMSI',
                'name': 'ISM Services PMI',
                'category': 'growth',
                'frequency': 'monthly',
                'transformation': 'level'
            },
            'RETAIL_SALES': {
                'series_id': 'RSAFS',
                'name': 'Retail Sales',
                'category': 'growth',
                'frequency': 'monthly',
                'transformation': 'yoy_change'
            },
            'HOUSING_STARTS': {
                'series_id': 'HOUST',
                'name': 'Housing Starts',
                'category': 'growth',
                'frequency': 'monthly',
                'transformation': 'yoy_change'
            },
            'CONSUMER_CONFIDENCE': {
                'series_id': 'UMCSENT',
                'name': 'Consumer Sentiment',
                'category': 'growth',
                'frequency': 'monthly',
                'transformation': 'level'
            },
            'LEADING_INDEX': {
                'series_id': 'LEADRATIO',
                'name': 'Leading Economic Index',
                'category': 'growth',
                'frequency': 'monthly',
                'transformation': 'yoy_change'
            },
            
            # INFLATION INDICATORS
            'CPI_ALL': {
                'series_id': 'CPIAUCSL',
                'name': 'Consumer Price Index',
                'category': 'inflation',
                'frequency': 'monthly',
                'transformation': 'yoy_change'
            },
            'CPI_CORE': {
                'series_id': 'CPILFESL',
                'name': 'Core CPI (ex food & energy)',
                'category': 'inflation',
                'frequency': 'monthly',
                'transformation': 'yoy_change'
            },
            'PPI': {
                'series_id': 'PPIACO',
                'name': 'Producer Price Index',
                'category': 'inflation',
                'frequency': 'monthly',
                'transformation': 'yoy_change'
            },
            'PCE_CORE': {
                'series_id': 'PCEPILFE',
                'name': 'Core PCE Price Index',
                'category': 'inflation',
                'frequency': 'monthly',
                'transformation': 'yoy_change'
            },
            'WAGES': {
                'series_id': 'AHETPI',
                'name': 'Average Hourly Earnings',
                'category': 'inflation',
                'frequency': 'monthly',
                'transformation': 'yoy_change'
            },
            'UNIT_LABOR_COSTS': {
                'series_id': 'ULCNFB',
                'name': 'Unit Labor Costs',
                'category': 'inflation',
                'frequency': 'quarterly',
                'transformation': 'yoy_change'
            },
            
            # INTEREST RATE INDICATORS
            'FED_FUNDS_RATE': {
                'series_id': 'FEDFUNDS',
                'name': 'Federal Funds Rate',
                'category': 'rates',
                'frequency': 'monthly',
                'transformation': 'level'
            },
            'TREASURY_10Y': {
                'series_id': 'DGS10',
                'name': '10-Year Treasury Rate',
                'category': 'rates',
                'frequency': 'daily',
                'transformation': 'level'
            },
            'TREASURY_2Y': {
                'series_id': 'DGS2',
                'name': '2-Year Treasury Rate',
                'category': 'rates',
                'frequency': 'daily',
                'transformation': 'level'
            },
            'YIELD_CURVE': {
                'series_id': 'T10Y2Y',
                'name': '10Y-2Y Treasury Spread',
                'category': 'rates',
                'frequency': 'daily',
                'transformation': 'level'
            },
            'REAL_RATES': {
                'series_id': 'REAINTRATREARAT10Y',
                'name': '10-Year Real Interest Rate',
                'category': 'rates',
                'frequency': 'daily',
                'transformation': 'level'
            },
            
            # MONETARY INDICATORS
            'MONEY_SUPPLY_M2': {
                'series_id': 'M2SL',
                'name': 'Money Supply M2',
                'category': 'monetary',
                'frequency': 'monthly',
                'transformation': 'yoy_change'
            },
            'CREDIT_GROWTH': {
                'series_id': 'TOTCI',
                'name': 'Total Consumer Credit',
                'category': 'monetary',
                'frequency': 'monthly',
                'transformation': 'yoy_change'
            },
            'BANK_CREDIT': {
                'series_id': 'TOTBKCR',
                'name': 'Bank Credit to Private Sector',
                'category': 'monetary',
                'frequency': 'monthly',
                'transformation': 'yoy_change'
            },
            
            # MARKET INDICATORS
            'TERM_SPREAD': {
                'series_id': 'T10Y3M',
                'name': '10Y-3M Treasury Spread',
                'category': 'market',
                'frequency': 'daily',
                'transformation': 'level'
            },
            'CREDIT_SPREAD': {
                'series_id': 'BAA10Y',
                'name': 'BAA-10Y Treasury Spread',
                'category': 'market', 
                'frequency': 'daily',
                'transformation': 'level'
            },
            'DOLLAR_INDEX': {
                'series_id': 'DTWEXBGS',
                'name': 'Trade Weighted Dollar Index',
                'category': 'market',
                'frequency': 'daily',
                'transformation': 'yoy_change'
            },
            
            # COMMODITY INDICATORS
            'OIL_PRICE': {
                'series_id': 'DCOILWTICO',
                'name': 'WTI Crude Oil Price',
                'category': 'commodity',
                'frequency': 'daily',
                'transformation': 'yoy_change'
            },
            'GOLD_PRICE': {
                'series_id': 'GOLDAMGBD228NLBM',
                'name': 'Gold Price',
                'category': 'commodity',
                'frequency': 'daily',
                'transformation': 'yoy_change'
            },
            'COMMODITY_INDEX': {
                'series_id': 'PPIACO',
                'name': 'Commodity Price Index',
                'category': 'commodity',
                'frequency': 'monthly',
                'transformation': 'yoy_change'
            }
        }
        
    def connect_to_database(self):
        """Connect to Sharadar database"""
        try:
            conn = psycopg2.connect(self.connection_string)
            logger.info("Successfully connected to Sharadar database")
            return conn
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            return None
    
    def get_fred_tables(self, conn):
        """Discover FRED tables in Sharadar database"""
        query = """
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public' 
        AND table_name LIKE '%fred%'
        ORDER BY table_name;
        """
        
        try:
            tables_df = pd.read_sql(query, conn)
            logger.info(f"Found {len(tables_df)} FRED-related tables")
            return tables_df['table_name'].tolist()
        except Exception as e:
            logger.error(f"Error discovering FRED tables: {e}")
            return []
    
    def get_fred_data_structure(self, conn, table_name):
        """Examine structure of FRED tables"""
        query = f"""
        SELECT column_name, data_type 
        FROM information_schema.columns 
        WHERE table_name = '{table_name}'
        ORDER BY ordinal_position;
        """
        
        try:
            structure_df = pd.read_sql(query, conn)
            return structure_df
        except Exception as e:
            logger.error(f"Error getting structure for {table_name}: {e}")
            return pd.DataFrame()
    
    def extract_fred_series(self, conn, series_id, start_date='1990-01-01', end_date='2025-12-31'):
        """Extract a specific FRED series from database"""
        
        # Try multiple possible table structures
        possible_queries = [
            # Query 1: Standard FRED table structure
            f"""
            SELECT date, value 
            FROM sf1_fred 
            WHERE series_id = '{series_id}' 
            AND date >= '{start_date}' 
            AND date <= '{end_date}'
            AND value IS NOT NULL
            ORDER BY date;
            """,
            
            # Query 2: Alternative structure
            f"""
            SELECT date, {series_id.lower()} as value
            FROM fred_data 
            WHERE date >= '{start_date}' 
            AND date <= '{end_date}'
            AND {series_id.lower()} IS NOT NULL
            ORDER BY date;
            """,
            
            # Query 3: Generic economic data table
            f"""
            SELECT date, value
            FROM economic_data 
            WHERE indicator = '{series_id}'
            AND date >= '{start_date}' 
            AND date <= '{end_date}'
            AND value IS NOT NULL
            ORDER BY date;
            """,
            
            # Query 4: Try direct series name
            f"""
            SELECT * FROM {series_id.lower()}
            WHERE date >= '{start_date}' 
            AND date <= '{end_date}'
            ORDER BY date
            LIMIT 5000;
            """
        ]
        
        for i, query in enumerate(possible_queries):
            try:
                df = pd.read_sql(query, conn)
                if not df.empty:
                    logger.info(f"Successfully extracted {series_id} using query method {i+1}")
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.sort_values('date')
                    return df
            except Exception as e:
                logger.debug(f"Query method {i+1} failed for {series_id}: {e}")
                continue
        
        logger.warning(f"Could not extract data for {series_id}")
        return pd.DataFrame()
    
    def transform_series(self, df, transformation):
        """Apply transformations to time series data"""
        if df.empty or 'value' not in df.columns:
            return df
            
        df = df.copy()
        
        if transformation == 'level':
            # No transformation needed
            df['transformed_value'] = df['value']
        elif transformation == 'yoy_change':
            # Year-over-year percentage change
            df['transformed_value'] = df['value'].pct_change(12) * 100
        elif transformation == 'mom_change':
            # Month-over-month percentage change
            df['transformed_value'] = df['value'].pct_change() * 100
        elif transformation == 'diff':
            # First difference
            df['transformed_value'] = df['value'].diff()
        else:
            # Default to level
            df['transformed_value'] = df['value']
        
        return df
    
    def aggregate_to_monthly(self, df, freq_original):
        """Aggregate higher frequency data to monthly"""
        if df.empty:
            return df
            
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        if freq_original == 'daily':
            # Use last value of month for daily data
            monthly_df = df.resample('M').last()
        elif freq_original == 'quarterly':
            # Forward fill quarterly data to monthly
            monthly_df = df.resample('M').fillna(method='ffill')
        else:
            # Already monthly
            monthly_df = df.resample('M').last()
        
        monthly_df.reset_index(inplace=True)
        return monthly_df
    
    def extract_all_fred_data(self):
        """Extract all FRED indicators for regime analysis"""
        logger.info("Starting comprehensive FRED data extraction...")
        
        conn = self.connect_to_database()
        if not conn:
            logger.error("Cannot proceed without database connection")
            return None
        
        try:
            # First, discover available FRED tables
            fred_tables = self.get_fred_tables(conn)
            logger.info(f"Available FRED tables: {fred_tables}")
            
            # Initialize master dataframe
            master_df = pd.DataFrame()
            extraction_log = {}
            
            for indicator_key, indicator_info in self.fred_indicators.items():
                series_id = indicator_info['series_id']
                name = indicator_info['name']
                transformation = indicator_info['transformation']
                frequency = indicator_info['frequency']
                
                logger.info(f"Extracting {indicator_key} ({series_id}): {name}")
                
                # Extract raw data
                raw_df = self.extract_fred_series(conn, series_id)
                
                if not raw_df.empty:
                    # Apply transformation
                    transformed_df = self.transform_series(raw_df, transformation)
                    
                    # Aggregate to monthly if needed
                    monthly_df = self.aggregate_to_monthly(transformed_df, frequency)
                    
                    if not monthly_df.empty and 'transformed_value' in monthly_df.columns:
                        # Prepare for master dataframe
                        monthly_df = monthly_df[['date', 'transformed_value']].copy()
                        monthly_df.columns = ['date', indicator_key]
                        
                        # Merge with master dataframe
                        if master_df.empty:
                            master_df = monthly_df.copy()
                        else:
                            master_df = pd.merge(master_df, monthly_df, on='date', how='outer')
                        
                        extraction_log[indicator_key] = {
                            'status': 'success',
                            'observations': len(monthly_df),
                            'start_date': monthly_df['date'].min().strftime('%Y-%m-%d'),
                            'end_date': monthly_df['date'].max().strftime('%Y-%m-%d')
                        }
                        
                        logger.info(f"✅ {indicator_key}: {len(monthly_df)} observations")
                    else:
                        extraction_log[indicator_key] = {'status': 'transformation_failed'}
                        logger.warning(f"❌ {indicator_key}: Transformation failed")
                else:
                    extraction_log[indicator_key] = {'status': 'extraction_failed'}
                    logger.warning(f"❌ {indicator_key}: Extraction failed")
            
            # Sort by date and clean up
            if not master_df.empty:
                master_df = master_df.sort_values('date')
                master_df.reset_index(drop=True, inplace=True)
                
                logger.info(f"Master dataset created: {len(master_df)} observations, {len(master_df.columns)-1} indicators")
                logger.info(f"Date range: {master_df['date'].min()} to {master_df['date'].max()}")
                
                return master_df, extraction_log
            else:
                logger.error("No data extracted successfully")
                return None, extraction_log
                
        except Exception as e:
            logger.error(f"Error in extraction process: {e}")
            return None, {}
        finally:
            conn.close()
    
    def create_regime_indicators(self, df):
        """Create regime classification indicators"""
        if df.empty:
            return df
            
        df = df.copy()
        
        # Growth indicators (composite)
        growth_indicators = ['GDP_GROWTH', 'INDUSTRIAL_PRODUCTION', 'ISM_PMI', 'EMPLOYMENT_GROWTH']
        available_growth = [col for col in growth_indicators if col in df.columns]
        
        if available_growth:
            # Create growth composite (normalized and averaged)
            growth_data = df[available_growth].copy()
            # Normalize each indicator (z-score)
            for col in available_growth:
                growth_data[col] = (growth_data[col] - growth_data[col].mean()) / growth_data[col].std()
            df['GROWTH_COMPOSITE'] = growth_data.mean(axis=1)
        
        # Inflation indicators (composite)
        inflation_indicators = ['CPI_CORE', 'PPI', 'PCE_CORE']
        available_inflation = [col for col in inflation_indicators if col in df.columns]
        
        if available_inflation:
            inflation_data = df[available_inflation].copy()
            # Normalize each indicator
            for col in available_inflation:
                inflation_data[col] = (inflation_data[col] - inflation_data[col].mean()) / inflation_data[col].std()
            df['INFLATION_COMPOSITE'] = inflation_data.mean(axis=1)
        
        # Regime classification
        if 'GROWTH_COMPOSITE' in df.columns and 'INFLATION_COMPOSITE' in df.columns:
            def classify_regime(row):
                growth = row['GROWTH_COMPOSITE']
                inflation = row['INFLATION_COMPOSITE']
                
                if pd.isna(growth) or pd.isna(inflation):
                    return 'Unknown'
                
                if growth > 0 and inflation > 0:
                    return 'Overheating'
                elif growth > 0 and inflation <= 0:
                    return 'Goldilocks'
                elif growth <= 0 and inflation > 0:
                    return 'Stagflation'
                else:
                    return 'Recession'
            
            df['ECONOMIC_REGIME'] = df.apply(classify_regime, axis=1)
        
        return df
    
    def generate_metadata(self, extraction_log):
        """Generate metadata for the FRED dataset"""
        metadata = {
            'extraction_date': datetime.now().isoformat(),
            'data_source': 'Sharadar FRED Database',
            'purpose': 'Economic regime analysis for factor investing',
            'regime_framework': {
                'description': '4-regime economic framework',
                'regimes': {
                    'Goldilocks': 'Rising Growth + Falling Inflation',
                    'Overheating': 'Rising Growth + Rising Inflation', 
                    'Stagflation': 'Falling Growth + Rising Inflation',
                    'Recession': 'Falling Growth + Falling Inflation'
                }
            },
            'indicators': self.fred_indicators,
            'extraction_log': extraction_log,
            'successful_extractions': len([k for k, v in extraction_log.items() if v.get('status') == 'success']),
            'total_indicators': len(self.fred_indicators)
        }
        
        return metadata
    
    def run_fred_extraction(self):
        """Main execution function"""
        logger.info("🏛️ Starting FRED Economic Data Extraction for Regime Analysis")
        logger.info("="*70)
        
        # Extract all FRED data
        fred_df, extraction_log = self.extract_all_fred_data()
        
        if fred_df is None:
            logger.error("FRED data extraction failed")
            return None
        
        # Create regime indicators
        logger.info("Creating economic regime indicators...")
        enhanced_df = self.create_regime_indicators(fred_df)
        
        # Save main dataset
        output_file = self.output_dir / 'fred_economic_data.csv'
        enhanced_df.to_csv(output_file, index=False)
        logger.info(f"✅ FRED economic data saved: {output_file}")
        
        # Generate and save metadata
        metadata = self.generate_metadata(extraction_log)
        metadata_file = self.output_dir / 'fred_economic_data_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info(f"✅ Metadata saved: {metadata_file}")
        
        # Print summary
        logger.info("\n" + "="*70)
        logger.info("🎯 FRED EXTRACTION SUMMARY:")
        logger.info("="*70)
        logger.info(f"📊 Total Indicators Attempted: {len(self.fred_indicators)}")
        logger.info(f"✅ Successful Extractions: {metadata['successful_extractions']}")
        logger.info(f"📅 Date Range: {enhanced_df['date'].min()} to {enhanced_df['date'].max()}")
        logger.info(f"📈 Total Observations: {len(enhanced_df)}")
        logger.info(f"🏛️ Final Dataset Columns: {len(enhanced_df.columns)}")
        
        successful_indicators = [k for k, v in extraction_log.items() if v.get('status') == 'success']
        logger.info(f"📋 Successfully Extracted: {successful_indicators}")
        
        return enhanced_df, metadata

if __name__ == "__main__":
    # Initialize and run FRED data extraction
    extractor = FREDDataExtractor()
    fred_data, metadata = extractor.run_fred_extraction()
    
    if fred_data is not None:
        print("\n🏆 FRED ECONOMIC DATA EXTRACTION COMPLETE! 🏆")
        print("Ready for economic regime analysis and factor strategy enhancement.")
    else:
        print("\n❌ FRED data extraction failed. Check database connection and table structure.")
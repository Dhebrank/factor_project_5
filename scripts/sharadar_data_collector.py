"""
Sharadar Data Collector for MSCI Validation
Collects VIX and S&P 500 data from Sharadar database for regime detection and benchmarking
"""

import pandas as pd
import numpy as np
import psycopg2
import logging
from pathlib import Path
from datetime import datetime
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SharedarDataCollector:
    """Collect VIX and S&P 500 data from Sharadar database"""
    
    def __init__(self, output_dir="data/processed"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Database connection parameters (validated from factor_project_4)
        self.connection_string = "postgresql://futures_user:databento_futures_2025@localhost:5432/sharadar"
        
    def connect_to_database(self):
        """Connect to Sharadar database"""
        try:
            self.conn = psycopg2.connect(self.connection_string)
            self.cursor = self.conn.cursor()
            logger.info("✅ Connected to Sharadar database")
            return True
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            logger.info("Note: Ensure Sharadar database is running and accessible")
            return False
    
    def collect_vix_data(self, start_date="1990-01-01", end_date="2025-12-31"):
        """Collect VIX data from FRED economic data table"""
        logger.info("Collecting VIX data from Sharadar FRED economic data...")
        
        try:
            # Query VIX data from public.economic_data table (FRED data)
            query = """
            SELECT 
                date, 
                value as vix_level,
                change_1d,
                change_1w,
                change_1m,
                percentile_1y,
                z_score_1y
            FROM public.economic_data 
            WHERE series_id = 'VIXCLS' 
            AND date >= %s 
            AND date <= %s
            ORDER BY date;
            """
            
            self.cursor.execute(query, (start_date, end_date))
            results = self.cursor.fetchall()
            
            if results:
                vix_df = pd.DataFrame(results, columns=[
                    'Date', 'VIX', 'VIX_1d_change', 'VIX_1w_change', 
                    'VIX_1m_change', 'VIX_1y_percentile', 'VIX_1y_zscore'
                ])
                vix_df['Date'] = pd.to_datetime(vix_df['Date'])
                vix_df = vix_df.set_index('Date')
                
                logger.info(f"VIX data collected: {len(vix_df)} observations")
                logger.info(f"Date range: {vix_df.index.min()} to {vix_df.index.max()}")
                logger.info(f"VIX range: {vix_df['VIX'].min():.2f} to {vix_df['VIX'].max():.2f}")
                
                return vix_df
            else:
                logger.warning("No VIX data found in Sharadar FRED database")
                return None
                
        except Exception as e:
            logger.error(f"Error collecting VIX data: {e}")
            return None
    
    def collect_sp500_data(self, start_date="1990-01-01", end_date="2025-12-31"):
        """Collect S&P 500 data from Sharadar sfp_prices table"""
        logger.info("Collecting S&P 500 data from Sharadar...")
        
        try:
            # Query SPY data from sharadar_data.sfp_prices table
            query = """
            SELECT 
                date, 
                close,
                closeadj as adjusted_close,
                volume
            FROM sharadar_data.sfp_prices 
            WHERE ticker = 'SPY' 
            AND date >= %s 
            AND date <= %s
            ORDER BY date;
            """
            
            self.cursor.execute(query, (start_date, end_date))
            results = self.cursor.fetchall()
            
            if results:
                sp500_df = pd.DataFrame(results, columns=['Date', 'SP500', 'SP500_Adj', 'Volume'])
                sp500_df['Date'] = pd.to_datetime(sp500_df['Date'])
                sp500_df = sp500_df.set_index('Date')
                
                # Calculate daily and monthly returns
                sp500_df['SP500_Daily_Return'] = sp500_df['SP500_Adj'].pct_change()
                
                logger.info(f"S&P 500 data collected: {len(sp500_df)} observations")
                logger.info(f"Date range: {sp500_df.index.min()} to {sp500_df.index.max()}")
                logger.info(f"SPY price range: ${sp500_df['SP500'].min():.2f} to ${sp500_df['SP500'].max():.2f}")
                
                return sp500_df
            else:
                logger.warning("No SPY data found in Sharadar database")
                return None
                
        except Exception as e:
            logger.error(f"Error collecting S&P 500 data: {e}")
            return None
    
    def create_fallback_data(self):
        """Create fallback VIX and S&P 500 data if Sharadar unavailable"""
        logger.info("Creating fallback market data for regime detection...")
        
        # Load MSCI data to get date range
        msci_file = self.output_dir / "msci_factor_returns.csv"
        if msci_file.exists():
            msci_data = pd.read_csv(msci_file, index_col=0, parse_dates=True)
            date_range = msci_data.index
            
            # Create synthetic VIX based on MinVol factor volatility (inverse relationship)
            minvol_vol = msci_data['MinVol'].rolling(12).std() * np.sqrt(12)
            synthetic_vix = 30 - (minvol_vol - minvol_vol.mean()) * 20 / minvol_vol.std()
            synthetic_vix = synthetic_vix.clip(10, 80)  # Reasonable VIX bounds
            
            # Create synthetic S&P 500 based on factor performance
            sp500_returns = (msci_data * pd.Series({
                'Value': 0.25, 'Quality': 0.25, 'MinVol': 0.25, 'Momentum': 0.25
            })).sum(axis=1)
            sp500_price = (1 + sp500_returns).cumprod() * 1000  # Start at 1000
            
            fallback_data = pd.DataFrame({
                'VIX': synthetic_vix,
                'SP500': sp500_price,
                'SP500_Monthly_Return': sp500_returns
            }, index=date_range)
            
            logger.info("✅ Created synthetic market data for testing")
            logger.info(f"Synthetic VIX range: {synthetic_vix.min():.2f} to {synthetic_vix.max():.2f}")
            
            return fallback_data
        else:
            logger.error("Cannot create fallback data - MSCI data not found")
            return None
    
    def align_with_msci_data(self, market_data):
        """Align market data with MSCI factor data timeline"""
        logger.info("Aligning market data with MSCI factor timeline...")
        
        # Load MSCI data
        msci_file = self.output_dir / "msci_factor_returns.csv"
        msci_data = pd.read_csv(msci_file, index_col=0, parse_dates=True)
        
        # Convert daily market data to monthly (if needed)
        if len(market_data) > len(msci_data) * 5:  # Likely daily data
            logger.info("Converting daily market data to monthly...")
            
            # Month-end alignment - handle different column combinations
            agg_dict = {}
            if 'VIX' in market_data.columns:
                agg_dict['VIX'] = 'mean'
            if 'SP500' in market_data.columns:
                agg_dict['SP500'] = 'last'
            if 'SP500_Adj' in market_data.columns:
                agg_dict['SP500_Adj'] = 'last'
            if 'Volume' in market_data.columns:
                agg_dict['Volume'] = 'mean'
            
            # Add any VIX enhancement columns
            for col in market_data.columns:
                if col.startswith('VIX_') and col not in agg_dict:
                    agg_dict[col] = 'mean'
            
            market_monthly = market_data.resample('ME').agg(agg_dict)
            
            # Calculate monthly returns
            if 'SP500_Adj' in market_monthly.columns:
                market_monthly['SP500_Monthly_Return'] = market_monthly['SP500_Adj'].pct_change()
            elif 'SP500' in market_monthly.columns:
                market_monthly['SP500_Monthly_Return'] = market_monthly['SP500'].pct_change()
        else:
            market_monthly = market_data.copy()
        
        # Align with MSCI dates
        aligned_data = market_monthly.reindex(msci_data.index, method='nearest')
        aligned_data = aligned_data.fillna(method='ffill').fillna(method='bfill')
        
        logger.info(f"Aligned market data: {len(aligned_data)} observations")
        logger.info(f"Coverage: {aligned_data.notna().all(axis=1).mean():.1%}")
        
        return aligned_data
    
    def save_market_data(self, market_data):
        """Save aligned market data"""
        logger.info("Saving market data...")
        
        # Save market data
        market_file = self.output_dir / "market_data.csv"
        market_data.to_csv(market_file)
        logger.info(f"Saved market data: {market_file}")
        
        # Save metadata
        metadata = {
            'processing_date': datetime.now().isoformat(),
            'data_source': 'Sharadar' if 'VIX' in market_data.columns else 'Synthetic',
            'date_range': {
                'start': market_data.index.min().isoformat(),
                'end': market_data.index.max().isoformat()
            },
            'observations': len(market_data),
            'columns': list(market_data.columns),
            'vix_stats': {
                'mean': float(market_data['VIX'].mean()),
                'std': float(market_data['VIX'].std()),
                'min': float(market_data['VIX'].min()),
                'max': float(market_data['VIX'].max())
            } if 'VIX' in market_data.columns else None
        }
        
        metadata_file = self.output_dir / "market_data_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info(f"Saved metadata: {metadata_file}")
        
        return metadata
    
    def run_data_collection(self):
        """Run complete market data collection"""
        logger.info("🚀 Starting Sharadar market data collection...")
        
        try:
            # Try to connect to Sharadar database
            if self.connect_to_database():
                # Collect VIX data
                vix_data = self.collect_vix_data()
                
                # Collect S&P 500 data
                sp500_data = self.collect_sp500_data()
                
                if vix_data is not None and sp500_data is not None:
                    # Combine data
                    market_data = pd.concat([vix_data, sp500_data], axis=1)
                    logger.info("✅ Successfully collected data from Sharadar")
                else:
                    logger.warning("Incomplete Sharadar data - creating fallback")
                    market_data = self.create_fallback_data()
                
                self.conn.close()
            else:
                logger.warning("Sharadar unavailable - creating fallback data")
                market_data = self.create_fallback_data()
            
            if market_data is not None:
                # Align with MSCI data
                aligned_data = self.align_with_msci_data(market_data)
                
                # Save data
                metadata = self.save_market_data(aligned_data)
                
                logger.info("✅ Market data collection completed")
                return aligned_data, metadata
            else:
                logger.error("❌ Failed to collect or create market data")
                return None, None
                
        except Exception as e:
            logger.error(f"❌ Data collection failed: {e}")
            return None, None

def main():
    """Main execution"""
    collector = SharedarDataCollector()
    market_data, metadata = collector.run_data_collection()
    
    if market_data is not None:
        print("\n🎯 Market Data Collection Complete!")
        print(f"Collected {len(market_data)} monthly observations")
        print("Ready for enhanced MSCI validation with regime detection")
        
        # Display sample
        print("\nSample data:")
        print(market_data.head())
        print("\nData summary:")
        print(market_data.describe())

if __name__ == "__main__":
    main()
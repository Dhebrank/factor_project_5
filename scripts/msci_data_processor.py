"""
MSCI Factor Index Data Processor
Converts MSCI Excel factor index data into analysis-ready format
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MSCIDataProcessor:
    """Process MSCI factor index Excel files into analysis-ready format"""
    
    def __init__(self, data_dir="data/msci_indexes/extracted_indexes", output_dir="data/processed"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # MSCI file mapping
        self.file_mapping = {
            'Value': 'MSCI USA Enhanced Value Index',
            'Quality': 'MSCI USA Sector Neutral Quality Index', 
            'MinVol': 'MSCI USA Minimum Volatility Index (USD)',
            'Momentum': 'MSCI USA Momentum Index'
        }
        
    def read_msci_excel(self, filepath, factor_name):
        """Read MSCI Excel file and extract price data"""
        try:
            # Read Excel file, skipping header rows
            df = pd.read_excel(filepath, skiprows=4)
            df.columns = ['Date', factor_name]
            
            # Skip the header row and clean data
            df = df.iloc[1:].copy()
            df['Date'] = pd.to_datetime(df['Date'])
            df[factor_name] = pd.to_numeric(df[factor_name], errors='coerce')
            
            # Remove any NaN values
            df = df.dropna()
            
            logger.info(f"Loaded {factor_name}: {len(df)} observations from {df['Date'].min()} to {df['Date'].max()}")
            return df.set_index('Date')
            
        except Exception as e:
            logger.error(f"Error reading {filepath}: {e}")
            return None
    
    def process_all_indexes(self):
        """Process all MSCI index files"""
        logger.info("Starting MSCI data processing...")
        
        dfs = {}
        file_info = {}
        
        # Find and process each factor file
        for factor, search_term in self.file_mapping.items():
            matching_files = list(self.data_dir.glob(f"*{search_term}*"))
            
            if matching_files:
                filepath = matching_files[0]
                logger.info(f"Processing {factor}: {filepath.name}")
                
                df = self.read_msci_excel(filepath, factor)
                if df is not None:
                    dfs[factor] = df
                    file_info[factor] = {
                        'file': filepath.name,
                        'start_date': df.index.min(),
                        'end_date': df.index.max(),
                        'observations': len(df)
                    }
            else:
                logger.warning(f"No file found for {factor} (searching for: {search_term})")
        
        return dfs, file_info
    
    def combine_factors(self, dfs):
        """Combine all factor indexes into single DataFrame"""
        logger.info("Combining factor data...")
        
        # Find common date range
        combined_df = pd.concat(dfs.values(), axis=1, join='inner')
        
        logger.info(f"Combined dataset:")
        logger.info(f"  Date range: {combined_df.index.min()} to {combined_df.index.max()}")
        logger.info(f"  Total observations: {len(combined_df)}")
        logger.info(f"  Factors: {list(combined_df.columns)}")
        
        # Check for any missing data
        missing_data = combined_df.isnull().sum()
        if missing_data.any():
            logger.warning(f"Missing data found: {missing_data}")
        else:
            logger.info("✅ No missing data - dataset is complete")
            
        return combined_df
    
    def calculate_returns(self, price_df):
        """Calculate monthly returns from price levels"""
        logger.info("Calculating monthly returns...")
        
        # Calculate simple monthly returns
        returns_df = price_df.pct_change().dropna()
        
        logger.info(f"Returns calculated:")
        logger.info(f"  Period: {returns_df.index.min()} to {returns_df.index.max()}")
        logger.info(f"  Observations: {len(returns_df)}")
        
        # Summary statistics
        stats = returns_df.describe()
        logger.info(f"Return statistics:")
        for col in returns_df.columns:
            logger.info(f"  {col}: Mean={stats.loc['mean', col]:.4f}, Std={stats.loc['std', col]:.4f}")
            
        return returns_df
    
    def save_processed_data(self, price_df, returns_df, file_info):
        """Save processed data to files"""
        logger.info("Saving processed data...")
        
        # Save price levels
        price_file = self.output_dir / "msci_factor_prices.csv"
        price_df.to_csv(price_file)
        logger.info(f"Saved price data: {price_file}")
        
        # Save returns
        returns_file = self.output_dir / "msci_factor_returns.csv"
        returns_df.to_csv(returns_file)
        logger.info(f"Saved returns data: {returns_file}")
        
        # Save metadata
        metadata = {
            'processing_date': datetime.now().isoformat(),
            'price_data': {
                'file': str(price_file),
                'start_date': price_df.index.min().isoformat(),
                'end_date': price_df.index.max().isoformat(),
                'observations': len(price_df),
                'factors': list(price_df.columns)
            },
            'returns_data': {
                'file': str(returns_file),
                'start_date': returns_df.index.min().isoformat(),
                'end_date': returns_df.index.max().isoformat(),
                'observations': len(returns_df),
                'factors': list(returns_df.columns)
            },
            'source_files': file_info
        }
        
        metadata_file = self.output_dir / "msci_data_metadata.json"
        import json
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info(f"Saved metadata: {metadata_file}")
        
        return metadata
    
    def run_full_processing(self):
        """Run complete MSCI data processing pipeline"""
        logger.info("🚀 Starting MSCI Factor Data Processing Pipeline")
        
        try:
            # Process all index files
            dfs, file_info = self.process_all_indexes()
            
            if not dfs:
                logger.error("❌ No data processed - check file paths")
                return None
                
            # Combine factors
            price_df = self.combine_factors(dfs)
            
            # Calculate returns
            returns_df = self.calculate_returns(price_df)
            
            # Save processed data
            metadata = self.save_processed_data(price_df, returns_df, file_info)
            
            logger.info("✅ MSCI data processing completed successfully")
            
            # Summary report
            logger.info("\n📊 PROCESSING SUMMARY:")
            logger.info(f"Factors processed: {len(price_df.columns)}")
            logger.info(f"Date range: {price_df.index.min().strftime('%Y-%m-%d')} to {price_df.index.max().strftime('%Y-%m-%d')}")
            logger.info(f"Total months: {len(price_df)}")
            logger.info(f"Return observations: {len(returns_df)}")
            
            return {
                'prices': price_df,
                'returns': returns_df,
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"❌ Processing failed: {e}")
            raise

def main():
    """Main execution"""
    processor = MSCIDataProcessor()
    result = processor.run_full_processing()
    
    if result:
        print("\n🎯 MSCI Data Processing Complete!")
        print(f"Processed {len(result['prices'].columns)} factors over {len(result['prices'])} months")
        print("Ready for academic validation analysis")

if __name__ == "__main__":
    main()
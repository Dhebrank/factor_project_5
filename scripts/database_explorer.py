#!/usr/bin/env python3
"""
Database Explorer for Sharadar
Explores the structure of the Sharadar database to find economic data tables

Author: Claude Code
Date: July 1, 2025
"""

import pandas as pd
import psycopg2
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatabaseExplorer:
    """Explore Sharadar database structure"""
    
    def __init__(self):
        self.connection_string = "postgresql://futures_user:databento_futures_2025@localhost:5432/sharadar"
    
    def connect_to_database(self):
        """Connect to Sharadar database"""
        try:
            conn = psycopg2.connect(self.connection_string)
            logger.info("Successfully connected to Sharadar database")
            return conn
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            return None
    
    def get_all_tables(self, conn):
        """Get all tables in the database"""
        query = """
        SELECT table_name, table_type
        FROM information_schema.tables 
        WHERE table_schema = 'public'
        ORDER BY table_name;
        """
        
        try:
            tables_df = pd.read_sql(query, conn)
            logger.info(f"Found {len(tables_df)} tables in database")
            return tables_df
        except Exception as e:
            logger.error(f"Error getting tables: {e}")
            return pd.DataFrame()
    
    def get_table_structure(self, conn, table_name):
        """Get structure of a specific table"""
        query = f"""
        SELECT column_name, data_type, is_nullable
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
    
    def explore_table_sample(self, conn, table_name, limit=5):
        """Get sample data from a table"""
        query = f"SELECT * FROM {table_name} LIMIT {limit};"
        
        try:
            sample_df = pd.read_sql(query, conn)
            return sample_df
        except Exception as e:
            logger.debug(f"Could not sample {table_name}: {e}")
            return pd.DataFrame()
    
    def search_for_economic_tables(self, conn):
        """Search for tables that might contain economic data"""
        # Get all tables
        all_tables = self.get_all_tables(conn)
        
        if all_tables.empty:
            return
        
        # Keywords that might indicate economic data
        economic_keywords = ['fred', 'economic', 'macro', 'gdp', 'cpi', 'unemployment', 
                           'inflation', 'interest', 'rate', 'treasury', 'fed', 'indicator']
        
        print("\n🔍 SEARCHING FOR ECONOMIC DATA TABLES")
        print("="*60)
        
        for _, row in all_tables.iterrows():
            table_name = row['table_name'].lower()
            
            # Check if table name contains economic keywords
            contains_keyword = any(keyword in table_name for keyword in economic_keywords)
            
            if contains_keyword:
                print(f"\n📊 TABLE: {row['table_name']}")
                
                # Get structure
                structure = self.get_table_structure(conn, row['table_name'])
                if not structure.empty:
                    print(f"   Columns: {', '.join(structure['column_name'].tolist())}")
                
                # Get sample
                sample = self.explore_table_sample(conn, row['table_name'])
                if not sample.empty:
                    print(f"   Sample rows: {len(sample)}")
                    print(f"   Sample data:\n{sample.head(2)}")
    
    def run_exploration(self):
        """Main exploration function"""
        logger.info("🔍 Starting Sharadar Database Exploration")
        
        conn = self.connect_to_database()
        if not conn:
            return
        
        try:
            # Get all tables
            all_tables = self.get_all_tables(conn)
            
            print(f"\n📋 ALL TABLES IN DATABASE ({len(all_tables)} total):")
            print("="*60)
            for table in all_tables['table_name'].tolist():
                print(f"  • {table}")
            
            # Search for economic data
            self.search_for_economic_tables(conn)
            
            # If no economic tables found, show general structure
            if len(all_tables) > 0:
                print(f"\n📊 SAMPLE TABLE STRUCTURES:")
                print("="*60)
                
                # Show structure of first few tables
                for table_name in all_tables['table_name'].head(5):
                    print(f"\n🔧 {table_name}:")
                    structure = self.get_table_structure(conn, table_name)
                    if not structure.empty:
                        for _, col in structure.iterrows():
                            print(f"   {col['column_name']} ({col['data_type']})")
                    
                    # Show sample
                    sample = self.explore_table_sample(conn, table_name, 2)
                    if not sample.empty:
                        print(f"   Sample: {len(sample.columns)} columns, {len(sample)} rows")
        
        except Exception as e:
            logger.error(f"Error during exploration: {e}")
        finally:
            conn.close()

if __name__ == "__main__":
    explorer = DatabaseExplorer()
    explorer.run_exploration()
"""CLI script for exploratory data analysis of the trade dataset."""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
import numpy as np
from trade_flow_gcn.utils.config import load_config
from trade_flow_gcn.data.preprocessing import preprocess_pipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

def explore_data(csv_path: Path, config_path: str):
    config = load_config(config_path)
    
    logger.info("Loading and preprocessing data...")
    df = preprocess_pipeline(csv_path, config)
    
    print("\n" + "="*50)
    print("      TRADE DATA EXPLORATION SUMMARY")
    print("="*50)
    
    print(f"\n1. DATASET DIMENSIONS")
    print(f"   Total rows: {len(df):,}")
    print(f"   Unique years: {df['year'].nunique()} ({df['year'].min()} to {df['year'].max()})")
    print(f"   Unique exporters: {df['iso3_o'].nunique()}")
    print(f"   Unique importers: {df['iso3_d'].nunique()}")
    
    print(f"\n2. TARGET VARIABLE (Log Trade)")
    print(f"   Mean: {df['target_log_trade'].mean():.4f}")
    print(f"   Median: {df['target_log_trade'].median():.4f}")
    print(f"   Std Dev: {df['target_log_trade'].std():.4f}")
    print(f"   Max: {df['target_log_trade'].max():.4f}")
    
    print(f"\n3. CORRELATIONS WITH TARGET")
    # Identify numeric features
    features = config['data']['node_features'] + config['data']['edge_features']
    # Filter columns that actually exist
    existing_features = [f for f in features if f in df.columns]
    corrs = df[existing_features + ['target_log_trade']].corr()['target_log_trade'].sort_values(ascending=False)
    print(corrs.to_string())
    
    print(f"\n4. TOP TRADING PAIRS (Last available year)")
    last_year = df['year'].max()
    top_pairs = df[df['year'] == last_year].sort_values('target_log_trade', ascending=False).head(10)
    for _, row in top_pairs.iterrows():
        print(f"   {row['iso3_o']} -> {row['iso3_d']}: {row['target_log_trade']:.2f}")
    
    print("\n" + "="*50)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--csv", type=str, help="Path to raw CSV (optional, will find largest if omitted)")
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    if args.csv:
        csv_path = Path(args.csv)
    else:
        # Find largest CSV in raw_dir
        raw_dir = Path(config['data']['raw_dir'])
        csv_candidates = list(raw_dir.glob("*.csv"))
        if not csv_candidates:
            logger.error("No CSV files found in %s", raw_dir)
            return
        csv_path = max(csv_candidates, key=lambda p: p.stat().st_size)
        logger.info("Selected largest CSV: %s", csv_path.name)
        
    explore_data(csv_path, args.config)

if __name__ == "__main__":
    main()

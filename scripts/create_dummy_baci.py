import pandas as pd
import numpy as np
import argparse
from pathlib import Path

def generate_dummy_baci(gravity_csv: Path, out_csv: Path):
    print(f"Reading {gravity_csv} for unique country pairs...")
    df_grav = pd.read_csv(gravity_csv, nrows=5000) # just take first 5000 rows to be fast
    
    pairs = df_grav[['year', 'iso3_o', 'iso3_d']].drop_duplicates()
    
    # We will generate 3 product categories per pair just for dummy testing
    print("Generating dummy product codes (HS6)...")
    # HS6 codes. Assume sections: 10111 (chap 01 -> sect 0), 280110 (chap 28 -> sect 5), 870321 (chap 87 -> sect 16)
    products = [10111, 280110, 870321]
    
    rows = []
    for _, row in pairs.iterrows():
        for p in products:
            # 50% chance they trade this product to maintain sparsity
            if np.random.rand() > 0.5:
                continue
            rows.append({
                't': row['year'],
                'iso3_o': row['iso3_o'],
                'iso3_d': row['iso3_d'],
                'k': p,
                'v': np.random.uniform(10, 10000) # dummy value
            })
            
    df_baci = pd.DataFrame(rows)
    df_baci.to_csv(out_csv, index=False)
    print(f"Saved dummy BACI data with {len(df_baci)} rows to {out_csv}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gravity", type=str, required=True)
    parser.add_argument("--out", type=str, default="data/raw/BACI_dummy.csv")
    args = parser.parse_args()
    generate_dummy_baci(Path(args.gravity), Path(args.out))

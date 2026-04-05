"""CLI script for downloading real BACI Multilayer Dataset from CEPII."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from trade_flow_gcn.data.download import download_baci_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Force redownload")
    args = parser.parse_args()
    
    print("=" * 60)
    print("    DOWNLOADING BACI MULTILAYER PRODUCT TRADE DATASET")
    print("=" * 60)
    print("The BACI dataset is approximately ~500MB compressed.")
    print("It will be extracted directly into data/raw/baci.")
    print("=" * 60)
    
    download_baci_data(force=args.force)

if __name__ == "__main__":
    main()

"""Download the CEPII Gravity dataset to data/raw/.

Usage:
    python scripts/download_data.py [--config configs/default.yaml] [--force]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from trade_flow_gcn.data.download import download_gravity_data
from trade_flow_gcn.utils.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download CEPII Gravity data")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if file exists",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    data_cfg = config.get("data", {})

    csv_path = download_gravity_data(
        url=data_cfg.get(
            "source_url",
            "http://www.cepii.fr/DATA_DOWNLOAD/gravity/data/Gravity_csv_V202211.zip",
        ),
        raw_dir=data_cfg.get("raw_dir", "data/raw"),
        force=args.force,
    )
    print(f"\n✓ Data ready at: {csv_path}")


if __name__ == "__main__":
    main()

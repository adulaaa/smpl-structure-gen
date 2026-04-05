"""Lightweight hyperparameter sweep for tabular baselines.

This script runs a grid search over XGBoost and LightGBM settings using
existing preprocessing and temporal splits, then saves ranked results.

Usage:
    python scripts/experiments/sweep_tabular_baselines.py
    python scripts/experiments/sweep_tabular_baselines.py --max-trials 20
"""

from __future__ import annotations

import argparse
import itertools
import logging
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from trade_flow_gcn.data.dataset import TradeDataModule, build_graphs_from_dataframe
from trade_flow_gcn.data.preprocessing import preprocess_pipeline
from trade_flow_gcn.models.lightgbm_baseline import LightGBMBaseline
from trade_flow_gcn.models.xgboost_baseline import XGBoostBaseline
from trade_flow_gcn.utils.config import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def extract_numpy(data_list):
    x_src, x_dst, edge_attr, y = [], [], [], []
    for graph in data_list:
        src, dst = graph.edge_index
        x_src.append(graph.x[src].numpy())
        x_dst.append(graph.x[dst].numpy())
        edge_attr.append(graph.edge_attr.numpy())
        y.append(graph.y.numpy())
    return (
        np.concatenate(x_src),
        np.concatenate(x_dst),
        np.concatenate(edge_attr),
        np.concatenate(y),
    )


def param_grid_to_dicts(grid: dict[str, Iterable]) -> list[dict]:
    keys = list(grid.keys())
    values = [list(grid[k]) for k in keys]
    combos = []
    for combo in itertools.product(*values):
        combos.append({k: v for k, v in zip(keys, combo)})
    return combos


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep XGBoost and LightGBM baselines")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--max-trials", type=int, default=30, help="Cap number of trials per model")
    parser.add_argument("--output", type=str, default="results/tabular_sweep_results.csv")
    args = parser.parse_args()

    config = load_config(args.config)
    root = Path(".")

    raw_dir = root / config["data"]["raw_dir"]
    csv_candidates = list(raw_dir.glob("*.csv"))
    if not csv_candidates:
        logger.error("No CSV file found in %s", raw_dir)
        return 1

    csv_path = max(csv_candidates, key=lambda p: p.stat().st_size)
    df = preprocess_pipeline(csv_path, config)
    graphs = build_graphs_from_dataframe(df, config["data"]["countries"], config)

    dm = TradeDataModule(
        graphs=graphs,
        train_years=tuple(config["data"]["train_years"]),
        val_years=tuple(config["data"]["val_years"]),
        test_years=tuple(config["data"]["test_years"]),
    )
    dm.setup()

    x_s_train, x_d_train, e_train, y_train = extract_numpy(dm.train_graphs)
    x_s_val, x_d_val, e_val, y_val = extract_numpy(dm.val_graphs)
    x_s_test, x_d_test, e_test, y_test = extract_numpy(dm.test_graphs)
    val_x = np.concatenate([x_s_val, x_d_val, e_val], axis=1)

    xgb_grid = param_grid_to_dicts(
        {
            "n_estimators": [300, 600, 1000],
            "max_depth": [4, 6, 8],
            "learning_rate": [0.03, 0.05, 0.1],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
        }
    )[: args.max_trials]

    lgb_grid = param_grid_to_dicts(
        {
            "n_estimators": [300, 600, 1000],
            "max_depth": [4, 6, 8],
            "learning_rate": [0.03, 0.05, 0.1],
            "num_leaves": [31, 63, 127],
        }
    )[: args.max_trials]

    results: list[dict] = []

    logger.info("Running XGBoost sweep (%d trials)", len(xgb_grid))
    for i, params in enumerate(xgb_grid, start=1):
        logger.info("XGBoost trial %d/%d", i, len(xgb_grid))
        model = XGBoostBaseline(**params)
        model.fit(x_s_train, x_d_train, e_train, y_train, eval_set=[(val_x, y_val)])
        metrics = model.evaluate(x_s_test, x_d_test, e_test, y_test)
        results.append({"family": "xgboost", "params": str(params), **metrics})

    logger.info("Running LightGBM sweep (%d trials)", len(lgb_grid))
    for i, params in enumerate(lgb_grid, start=1):
        logger.info("LightGBM trial %d/%d", i, len(lgb_grid))
        model = LightGBMBaseline(**params)
        model.fit(x_s_train, x_d_train, e_train, y_train, eval_set=[(val_x, y_val)])
        metrics = model.evaluate(x_s_test, x_d_test, e_test, y_test)
        results.append({"family": "lightgbm", "params": str(params), **metrics})

    out_df = pd.DataFrame(results).sort_values("rmse")
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print("\nTop 10 sweep results by RMSE")
    print(out_df.head(10).to_string(index=False))
    print(f"\nSaved full results: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

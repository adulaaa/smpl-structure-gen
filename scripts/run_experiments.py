"""Fast End-to-End Benchmark Orchestrator for Gravity vs BACI Comparison."""

import argparse
import copy
import logging
import pandas as pd
import pytorch_lightning as pl
import os
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from trade_flow_gcn.utils.config import load_config
from trade_flow_gcn.data.preprocessing import preprocess_pipeline
from trade_flow_gcn.data.dataset import build_graphs_from_dataframe, TradeDataModule
from trade_flow_gcn.training.lightning_module import TradeFlowModule
from train import build_model
from evaluate_models import extract_numpy
from trade_flow_gcn.models.xgboost_baseline import XGBoostBaseline
from pytorch_lightning.callbacks import ModelCheckpoint

import shutil
import warnings

# Suppress annoying SKLearn/PyTorch Lightning UX warnings for pure benchmark output
warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# Top 20 Global Economies for fast benchmark prototyping
TOP_20_COUNTRIES = [
    "USA", "CHN", "JPN", "DEU", "GBR", "FRA", "IND", "ITA", "BRA", "CAN",
    "KOR", "RUS", "AUS", "MEX", "IDN", "TUR", "SAU", "ARG", "ZAF", "NLD"
]

def train_and_eval_dl(model_name: str, config: dict, datamodule: TradeDataModule, experiment: str):
    logger.info("   Training %s ...", model_name.upper())
    
    # Dynamically scale model dimensions to match PyG data payloads
    sample_data = datamodule.train_graphs[0]
    n_node = sample_data.x.shape[1]
    e_node = sample_data.edge_attr.shape[1]
    
    if model_name not in config['model']:
        config['model'][model_name] = {}
        
    config['model'][model_name]['node_input_dim'] = n_node
    config['model'][model_name]['edge_input_dim'] = e_node
    config['model'][model_name]['input_dim'] = (n_node * 2) + e_node
    
    # 1. Build Model dynamically
    model = build_model(config, model_name)
    lit_module = TradeFlowModule(
        model=model,
        learning_rate=0.005,
    )
    
    # 2. Checkpoints isolated by experiment phase
    ckpt_dir = Path(f"lightning_logs/{experiment}/{model_name}")
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="best",
    )
    
    # 3. Trainer (Fast Convergence Settings)
    trainer = pl.Trainer(
        max_epochs=20,  # Strict cap for fast turnaround
        callbacks=[checkpoint_callback],
        logger=False,  # Disable TB logs
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    
    # 4. Train
    trainer.fit(lit_module, datamodule=datamodule)
    
    # 5. Evaluate against test set
    metrics = trainer.test(lit_module, datamodule=datamodule, ckpt_path="best", verbose=False)
    
    if metrics and len(metrics) > 0:
        return {
            "rmse": metrics[0].get("test_rmse", 0.0),
            "r2": metrics[0].get("test_r2", 0.0)
        }
    return {"rmse": 0.0, "r2": 0.0}

def extract_nodes(data_list):
    X, Y = [], []
    for g in data_list:
        X.append(g.x.numpy())
        Y.append(g.y.numpy())
    return np.concatenate(X, axis=0), np.concatenate(Y, axis=0)


def run_pipeline(base_config: dict, use_baci: bool, experiment_name: str, use_gravity: bool = True):
    # Deepcopy to selectively override timeline limits
    config = copy.deepcopy(base_config)
    
    config['data']['use_baci'] = use_baci
    if not use_gravity:
        config['data']['edge_features'] = []
    config['data']['countries'] = TOP_20_COUNTRIES
    config['data']['year_start'] = 2005
    config['data']['year_end'] = 2014
    config['data']['train_years'] = [2005, 2012] # Extends historical runway to compensate for Lags
    config['data']['val_years'] = [2013, 2013]   # 1 year val
    config['data']['test_years'] = [2014, 2014]  # 1 year test
    
    raw_dir = Path(config['data']['raw_dir'])
    csv_candidates = list(raw_dir.glob("Gravity_V*.csv"))
    csv_path = max(csv_candidates, key=lambda p: p.stat().st_size)
    
    logger.info("Building data pipeline...")
    df = preprocess_pipeline(csv_path, config)
    graphs = build_graphs_from_dataframe(df, config['data']['countries'], config)
    
    dm = TradeDataModule(
        graphs=graphs,
        train_years=tuple(config['data']['train_years']),
        val_years=tuple(config['data']['val_years']),
        test_years=tuple(config['data']['test_years'])
    )
    dm.setup()
    
    results = {}
    
    # Evaluate Non-DL XGBoost baseline
    logger.info("   Training XGBOOST ...")
    x_train, y_train = extract_nodes(dm.train_graphs)
    x_val, y_val = extract_nodes(dm.val_graphs)
    x_test, y_test = extract_nodes(dm.test_graphs)
    
    xgb = XGBoostBaseline()
    
    # Target reshape for SKLearn interface: multi-output regression natively supported,
    # but sometimes robust fallback needed if output_dim==1
    if y_train.ndim == 2 and y_train.shape[1] == 1:
        y_train = y_train.ravel()
        y_val = y_val.ravel()
    
    xgb.fit(x_train, y_train, eval_set=[(x_val, y_val)])
    
    from trade_flow_gcn.evaluation.metrics import compute_all_metrics
    import torch
    xgb_metrics_list = []
    
    for g in dm.test_graphs:
        y_p = xgb.predict(g.x.numpy())
        if y_p.ndim == 1 and g.y.ndim == 2:
            y_p = y_p.reshape(-1, 1)
            
        m = compute_all_metrics(
            torch.tensor(y_p, dtype=torch.float32), 
            g.y.clone().detach()
        )
        xgb_metrics_list.append(m)
        
    results["XGBoost"] = {
        "rmse": float(np.mean([m["rmse"] for m in xgb_metrics_list])),
        "r2": float(np.mean([m["r2"] for m in xgb_metrics_list]))
    }
    
    # Evaluate Deep Learning Models
    for model_name in ["mlp_baseline", "gcn", "gat", "egnn", "rgcn", "gine"]:
        m = train_and_eval_dl(model_name, config, dm, experiment_name)
        results[model_name.upper().replace("_BASELINE", "")] = m
        
    return results

def main():
    root = Path(".")
    log_dir = root / "lightning_logs"
    if log_dir.exists():
        logger.info("Wiping legacy lightning_logs/ to guarantee clean initialization...")
        shutil.rmtree(log_dir)
        
    config = load_config("configs/default.yaml")
    
    print("\n" + "="*70)
    print("  PHASE 1: GRAVITY (Standard CEPII Macro Features)")
    print("="*70)
    grav_results = run_pipeline(config, use_baci=False, experiment_name="gravity")
    
    print("\n" + "="*70)
    print("  PHASE 2: BACI MULTILAYER (21-Layer Multiplex Network)")
    print("="*70)
    baci_results = run_pipeline(config, use_baci=True, experiment_name="baci")
    
    print("\n" + "="*70)
    print("  PHASE 3: BACI ONLY (No Gravity Edge Features)")
    print("="*70)
    baci_nograv_results = run_pipeline(config, use_baci=True, experiment_name="baci_nograv", use_gravity=False)
    
    # Master comparison processing
    records = []
    for model in grav_results.keys():
        g_rmse = grav_results[model]["rmse"]
        b_rmse = baci_results[model]["rmse"]
        
        improvement = ((g_rmse - b_rmse) / g_rmse) * 100 if g_rmse > 0 else 0.0
        
        records.append({
            "Model": model,
            "Node RMSE (Grav)": f"{g_rmse:.4f}",
            "Node RMSE (BACI)": f"{b_rmse:.4f}",
            "Imp. (%)": f"{improvement:+.1f}%"
        })
        
    df = pd.DataFrame(records)
    print("\n\n" + "="*70)
    print("     FINAL BENCHMARK: NODE VULNERABILITY PREDICTION")
    print("="*70)
    print(df.to_string(index=False))
    print("="*70)

if __name__ == "__main__":
    main()

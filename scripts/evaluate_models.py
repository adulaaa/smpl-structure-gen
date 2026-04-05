"""CLI script for comprehensive model evaluation and comparison."""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Any

from trade_flow_gcn.utils.config import load_config
from trade_flow_gcn.data.preprocessing import preprocess_pipeline
from trade_flow_gcn.data.dataset import build_graphs_from_dataframe, TradeDataModule
from trade_flow_gcn.models.gcn import TradeFlowGCN
from trade_flow_gcn.models.gat import TradeFlowGAT
from trade_flow_gcn.models.egnn import TradeFlowEGNN
from trade_flow_gcn.models.rgcn import TradeFlowRGCN
from trade_flow_gcn.models.mlp_baseline import MLPBaseline
from trade_flow_gcn.models.gravity_baseline import GravityBaseline
from trade_flow_gcn.models.xgboost_baseline import XGBoostBaseline
from trade_flow_gcn.models.lightgbm_baseline import LightGBMBaseline
from trade_flow_gcn.models.hybrid_gae_xgboost import HybridGAEXGBoost
from trade_flow_gcn.training.lightning_module import TradeFlowModule
import torch_geometric

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

def extract_numpy(data_list: List):
    X_src, X_dst, E_attr, Y = [], [], [], []
    for graph in data_list:
        src, dst = graph.edge_index
        X_src.append(graph.x[src].numpy())
        X_dst.append(graph.x[dst].numpy())
        E_attr.append(graph.edge_attr.numpy())
        Y.append(graph.y.numpy())
    return (
        np.concatenate(X_src),
        np.concatenate(X_dst),
        np.concatenate(E_attr),
        np.concatenate(Y)
    )

def evaluate_torch_model(module, graphs):
    device = next(module.parameters()).device
    loader = torch_geometric.loader.DataLoader(graphs, batch_size=1)
    preds, targets = [], []
    module.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = module(batch)
            preds.append(out.cpu().numpy())
            targets.append(batch.y.cpu().numpy())
    
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    return {
        "rmse": float(np.sqrt(mean_squared_error(targets, preds))),
        "mae": float(mean_absolute_error(targets, preds)),
        "r2": float(r2_score(targets, preds))
    }

def load_dl_model(model_type, model_class, base_dir, config):
    ckpt_files = list(base_dir.glob(f"{model_type}/**/checkpoints/*.ckpt"))
    if not ckpt_files:
        ckpt_files = [p for p in base_dir.glob("**/checkpoints/*.ckpt") if model_type in str(p)]
    
    if not ckpt_files:
        logger.warning("No checkpoint found for %s", model_type)
        return None
        
    latest_ckpt = sorted(ckpt_files, key=lambda p: p.stat().st_mtime)[-1]
    
    # Init architecture
    base_node_dim = len(config['data']['node_features'])
    base_edge_dim = len(config['data']['edge_features'])
    use_baci = config.get("data", {}).get("use_baci", False)
    if use_baci:
        base_edge_dim += 21
        
    if model_type == "gcn":
        net = model_class(node_input_dim=base_node_dim, edge_input_dim=base_edge_dim)
    elif model_type in ["gat", "egnn"]:
        net = model_class(node_input_dim=base_node_dim, edge_input_dim=base_edge_dim)
    elif model_type == "rgcn":
        net = model_class(node_input_dim=base_node_dim, edge_input_dim=base_edge_dim, num_relations=21 if use_baci else 3)
    else:
        net = model_class(input_dim=base_node_dim*2 + base_edge_dim)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = TradeFlowModule.load_from_checkpoint(latest_ckpt, model=net).to(device)
    logger.info("Loaded %s from %s", model_type, latest_ckpt.name)
    return module

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--log_dir", type=str, default="lightning_logs")
    args = parser.parse_args()
    
    config = load_config(args.config)
    root = Path(".")
    log_dir = root / args.log_dir
    
    # 1. Load Data
    raw_dir = root / config['data']['raw_dir']
    csv_candidates = list(raw_dir.glob("*.csv"))
    csv_path = max(csv_candidates, key=lambda p: p.stat().st_size)
    
    df = preprocess_pipeline(csv_path, config)
    graphs = build_graphs_from_dataframe(df, config['data']['countries'], config)
    
    dm = TradeDataModule(
        graphs=graphs,
        train_years=tuple(config['data']['train_years']),
        val_years=tuple(config['data']['val_years']),
        test_years=tuple(config['data']['test_years'])
    )
    dm.setup()
    
    results = []
    
    # 2. Evaluate Non-DL Baselines
    logger.info("Evaluating tabular baselines...")
    x_s_train, x_d_train, e_train, y_train = extract_numpy(dm.train_graphs)
    x_s_val, x_d_val, e_val, y_val = extract_numpy(dm.val_graphs)
    x_s_test, x_d_test, e_test, y_test = extract_numpy(dm.test_graphs)
    
    # Gravity
    gravity = GravityBaseline()
    gravity.fit(x_s_train, x_d_train, e_train, y_train)
    results.append({"Model": "Gravity (OLS)", **gravity.evaluate(x_s_test, x_d_test, e_test, y_test)})
    
    # XGBoost
    xgb = XGBoostBaseline()
    xgb_val_X = np.concatenate([x_s_val, x_d_val, e_val], axis=1)
    xgb.fit(x_s_train, x_d_train, e_train, y_train, eval_set=[(xgb_val_X, y_val)])
    results.append({"Model": "XGBoost", **xgb.evaluate(x_s_test, x_d_test, e_test, y_test)})
    
    # LightGBM
    lgb_model = LightGBMBaseline()
    lgb_model.fit(x_s_train, x_d_train, e_train, y_train, eval_set=[(xgb_val_X, y_val)])
    results.append({"Model": "LightGBM", **lgb_model.evaluate(x_s_test, x_d_test, e_test, y_test)})
    
    # 3. Evaluate DL Models
    logger.info("Evaluating Deep Learning models...")
    models_to_test = [
        ("gcn", TradeFlowGCN),
        ("gat", TradeFlowGAT),
        ("egnn", TradeFlowEGNN),
        ("rgcn", TradeFlowRGCN),
        ("mlp_baseline", MLPBaseline)
    ]
    
    for m_name, m_class in models_to_test:
        module = load_dl_model(m_name, m_class, log_dir, config)
        if module:
            pretty_name = m_name.replace("_", " ").upper()
            results.append({"Model": f"TradeFlow {pretty_name}", **evaluate_torch_model(module, dm.test_graphs)})
            
    # 4. Hybrid XGBoost (GAE + Tabular)
    embedding_path = root / "data/processed/node_embeddings.npy"
    if embedding_path.exists():
        logger.info("Evaluating Hybrid XGBoost (GAE + Tabular)...")
        embeddings_dict = np.load(embedding_path, allow_pickle=True).item()
        
        hybrid_model = HybridGAEXGBoost()
        hybrid_model.set_embeddings(embeddings_dict)
        
        # Train & Evaluate
        try:
            hybrid_model.fit(
                dm.train_graphs, 
                dm.val_graphs, 
                train_start_idx=0,
                val_start_idx=len(dm.train_graphs)
            )
            h_metrics = hybrid_model.evaluate(
                dm.test_graphs, 
                start_idx=len(dm.train_graphs) + len(dm.val_graphs)
            )
            results.append({"Model": "HYBRID (GAE + XGBoost)", **h_metrics})
        except Exception as e:
            logger.error("Failed to evaluate Hybrid model: %s", e)
    else:
        logger.warning("No embeddings found at %s. Run scripts/generate_embeddings.py first for Hybrid model.", embedding_path)

    # 5. Show Results
    summary_df = pd.DataFrame(results).sort_values("rmse")
    print("\n" + "="*60)
    print("                MODEL COMPARISON RESULTS")
    print("="*60)
    print(summary_df.to_string(index=False))
    print("="*60)

if __name__ == "__main__":
    main()

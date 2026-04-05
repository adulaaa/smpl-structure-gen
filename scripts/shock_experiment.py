"""Shock Propagation Experiment: Testing GNN Network Memory vs Tabular Baselines."""

import argparse
import copy
import logging
import warnings
import pandas as pd
import pytorch_lightning as pl
import os
import sys
import torch
from pathlib import Path
import numpy as np
import shutil

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from trade_flow_gcn.utils.config import load_config
from trade_flow_gcn.data.preprocessing import preprocess_pipeline
from trade_flow_gcn.data.dataset import build_graphs_from_dataframe, TradeDataModule
from trade_flow_gcn.training.lightning_module import TradeFlowModule
from train import build_model
from evaluate_models import extract_numpy
from trade_flow_gcn.models.xgboost_baseline import XGBoostBaseline
from pytorch_lightning.callbacks import ModelCheckpoint
from torch_geometric.data import Data

# Suppress annoying SKLearn/PyTorch Lightning UX warnings for pure benchmark output
warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

TOP_20_COUNTRIES = [
    "USA", "CHN", "JPN", "DEU", "GBR", "FRA", "IND", "ITA", "BRA", "CAN",
    "KOR", "RUS", "AUS", "MEX", "IDN", "TUR", "SAU", "ARG", "ZAF", "NLD"
]

def train_and_eval_dl(model_name: str, config: dict, datamodule: TradeDataModule):
    logger.info("   Training %s ...", model_name.upper())
    
    sample_data = datamodule.train_graphs[0]
    n_node = sample_data.x.shape[1]
    e_node = sample_data.edge_attr.shape[1]
    
    if model_name not in config['model']:
        config['model'][model_name] = {}
        
    config['model'][model_name]['node_input_dim'] = n_node
    config['model'][model_name]['edge_input_dim'] = e_node
    config['model'][model_name]['input_dim'] = (n_node * 2) + e_node
    
    model = build_model(config, model_name)
    lit_module = TradeFlowModule(
        model=model,
        learning_rate=0.005,
    )
    
    ckpt_dir = Path(f"lightning_logs/shock_experiment/{model_name}")
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="best",
    )
    
    trainer = pl.Trainer(
        max_epochs=20,  # Fast Convergence limit
        callbacks=[checkpoint_callback],
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    
    trainer.fit(lit_module, datamodule=datamodule)
    
    # Load Best Model for manual inference
    best_path = checkpoint_callback.best_model_path
    lit_module = TradeFlowModule.load_from_checkpoint(best_path, model=model)
    lit_module.eval()
    return lit_module

def predict_gnn(lit_module, graph_data: Data):
    with torch.no_grad():
        x = graph_data.x
        edge_index = graph_data.edge_index
        edge_attr = graph_data.edge_attr
        edge_type = getattr(graph_data, 'edge_type', None)
        
        args = [x, edge_index, edge_attr]
        kwargs = {}
        if edge_type is not None:
             kwargs["edge_type"] = edge_type
             
        try:
             y_pred = lit_module.model(*args, **kwargs)
        except TypeError:
             y_pred = lit_module.model(*args)
             
        return y_pred.cpu().numpy()

def create_shocked_graph(data: Data, shock_target: str) -> tuple[Data, np.ndarray]:
    """Drops all edges involving shock_target, returns shocked graph AND the surviving mask."""
    if shock_target not in data.country_list:
        raise ValueError(f"Shock target {shock_target} not in country list!")
        
    shock_idx = data.country_list.index(shock_target)
    
    # Filter edges NOT connected to shock_idx
    src = data.edge_index[0]
    dst = data.edge_index[1]
    
    # Mask of edges that survive the embargo (neither src nor dst are shock target)
    surviving_mask = (src != shock_idx) & (dst != shock_idx)
    
    shocked_data = Data(
        x=data.x.clone(),
        edge_index=data.edge_index[:, surviving_mask].clone(),
        edge_attr=data.edge_attr[surviving_mask].clone(),
        y=data.y[surviving_mask].clone()
    )
    if hasattr(data, 'edge_type'):
         shocked_data.edge_type = data.edge_type[surviving_mask].clone()
         
    shocked_data.year = data.year
    shocked_data.country_list = data.country_list
    
    return shocked_data, surviving_mask.cpu().numpy()

def main():
    root = Path(".")
    log_dir = root / "lightning_logs" / "shock_experiment"
    if log_dir.exists():
        shutil.rmtree(log_dir)
        
    config = load_config("configs/default.yaml")
    config['data']['use_baci'] = True  # We need the full multiplex graph for maximum depth
    config['data']['countries'] = TOP_20_COUNTRIES
    config['data']['year_start'] = 2005
    config['data']['year_end'] = 2014
    config['data']['train_years'] = [2005, 2012] # Extends historical runway to compensate for Lags
    config['data']['val_years'] = [2013, 2013]   # 1 year val
    config['data']['test_years'] = [2014, 2014]  # 1 year test
    
    raw_dir = Path(config['data']['raw_dir'])
    csv_candidates = list(raw_dir.glob("Gravity_V*.csv"))
    csv_path = max(csv_candidates, key=lambda p: p.stat().st_size)
    
    print("\n" + "="*80)
    print("  NETWORK MEMORY EXPERIMENT: SIMULATING A MACROECONOMIC SHOCK (RUSSIA)")
    print("="*80)
    
    logger.info("Building standard data pipelines...")
    df = preprocess_pipeline(csv_path, config)
    graphs = build_graphs_from_dataframe(df, config['data']['countries'], config)
    
    dm = TradeDataModule(
        graphs=graphs,
        train_years=tuple(config['data']['train_years']),
        val_years=tuple(config['data']['val_years']),
        test_years=tuple(config['data']['test_years'])
    )
    dm.setup()
    
    test_graph = dm.test_graphs[0]
    
    print("\n[1] Training Models on Baseline 2010-2013 Data...")
    # XGBoost Baseline
    logger.info("   Training XGBOOST ...")
    x_s_train, x_d_train, e_train, y_train = extract_numpy(dm.train_graphs)
    x_s_val, x_d_val, e_val, y_val = extract_numpy(dm.val_graphs)
    x_s_test, x_d_test, e_test, y_test = extract_numpy([test_graph])
    
    xgb = XGBoostBaseline()
    xgb_val_X = np.concatenate([x_s_val, x_d_val, e_val], axis=1)
    xgb.fit(x_s_train, x_d_train, e_train, y_train, eval_set=[(xgb_val_X, y_val)])
    
    # RGCN Baseline
    lit_rgcn = train_and_eval_dl("rgcn", config, dm)
    
    print("\n[2] Executing Baseline Predictions on 2014 Test Graph...")
    test_X = np.concatenate([x_s_test, x_d_test, e_test], axis=1)
    
    xgb_preds_base = xgb.model.predict(test_X)
    rgcn_preds_base = predict_gnn(lit_rgcn, test_graph)
    
    print("\n[3] INITIATING SHOCK: Securing Embargo and deleting all 'RUS' trade routes...")
    shocked_graph, surviving_mask = create_shocked_graph(test_graph, shock_target="RUS")
    
    edges_removed = test_graph.edge_index.shape[1] - shocked_graph.edge_index.shape[1]
    logger.info(f"   Deleted {edges_removed} routing edges touching 'RUS'.")
    logger.info(f"   Re-evaluating models on the {shocked_graph.edge_index.shape[1]} un-involved routing edges (e.g. USA<->CHN)...")
    
    # Extract Numpy from the SHOCKED graph for XGBoost
    x_s_s, x_d_s, e_s, y_s = extract_numpy([shocked_graph])
    test_X_shocked = np.concatenate([x_s_s, x_d_s, e_s], axis=1)
    
    print("\n[4] Collecting Counterfactual Predictions from the Shocked Topology...")
    
    # XGBoost Prediction on Shocked Graph
    xgb_preds_shocked = xgb.model.predict(test_X_shocked)
    
    # RGCN Prediction on Shocked Graph
    rgcn_preds_shocked = predict_gnn(lit_rgcn, shocked_graph)
    
    # In Baseline, we must isolate the predictions that belong ONLY to the surviving edges
    xgb_preds_base_surviving = xgb_preds_base[surviving_mask]
    rgcn_preds_base_surviving = rgcn_preds_base[surviving_mask]
    
    # Calculate Absolute Absolute Deviation
    xgb_delta = np.mean(np.abs(xgb_preds_shocked - xgb_preds_base_surviving))
    rgcn_delta = np.mean(np.abs(rgcn_preds_shocked - rgcn_preds_base_surviving))
    
    print("\n\n" + "="*80)
    print("      SHOCK PROPAGATION RESULTS: HOW MODELS REACT TO TRADE DIVERSION")
    print("="*80)
    print(f"XGBOOST Prediction Shift: {xgb_delta:.6f} log-volume units")
    print(f"   RGCN Prediction Shift: {rgcn_delta:.6f} log-volume units")
    print("-" * 80)
    
    if xgb_delta == 0.0 and rgcn_delta > 0.0:
        print("VERDICT: SUCCESS! ")
        print("   XGBoost treats all rows as independent. Without Russia, the USA<->CHN row")
        print("   has identical features, meaning XGBoost literally cannot 'see' the shock.")
        print()
        print("   RGCN mathematically detected the global topological shift! Because Russia's")
        print("   economy disappeared from the Neighborhood Message Passing aggregates,")
        print("   RGCN automatically dynamically refactored its forecast for USA<->CHN trade!")
    print("="*80)

if __name__ == "__main__":
    main()

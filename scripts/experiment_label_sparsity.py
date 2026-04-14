"""Experiment script to benchmark PyTorch Geometric (JointSemiSupModule) 
versus Classical Baselines (XGBoost/LightGBM) under varying levels of label sparsity.

Tests necessity of semi-supervised frameworks by showing baseline performance 
degradation when strictly valid targets randomly vanish from the training set.

Usage:
    python scripts/experiment_label_sparsity.py
"""

from __future__ import annotations

import argparse
import logging
import sys
import copy
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl

from torch_geometric.data import Batch

from mol_prop_gnn.data.dataset import MoleculeDataModule
from mol_prop_gnn.data.unified_dataset import build_unified_dataframe, preprocess_unified_dataset
from mol_prop_gnn.data.preprocessing import (
    compute_fingerprint,
    get_node_feature_dim,
    get_edge_feature_dim,
)
from mol_prop_gnn.models.gcn import MolGCN
from mol_prop_gnn.models.gin import MolGIN
from mol_prop_gnn.models.xgboost_baseline import XGBoostBaseline
from mol_prop_gnn.models.lightgbm_baseline import LightGBMBaseline

from mol_prop_gnn.models.factory import build_joint_model
from mol_prop_gnn.training.semi_sup_module import JointSemiSupModule
from mol_prop_gnn.utils.config import apply_config_to_parser, load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


def extract_fingerprints(dataset, n_bits=2048):
    fps, labels = [], []
    for g in dataset:
        if hasattr(g, "smiles") and getattr(g, "smiles", None) is not None:
            fp = compute_fingerprint(g.smiles, n_bits=n_bits)
            if fp is not None:
                fps.append(fp)
                # Labels is a 1D tensor [12] -> append as row
                labels.append(g.y.numpy().flatten())
    return np.array(fps), np.array(labels)


def apply_sparsity_mask(datamodule: MoleculeDataModule, retention_pct: float, seed: int = 42) -> MoleculeDataModule:
    """Creates a deepcopy of the training dataset with artificially injected sparsity."""
    rng = np.random.default_rng(seed)
    
    # We clone the entire datamodule to preserve the exact same structure identically
    dm_corrupted = copy.deepcopy(datamodule)
    
    total_valid = 0
    total_dropped = 0
    
    for g in dm_corrupted.train_dataset:
        y_flat = g.y.view(-1)
        valid_mask = ~torch.isnan(y_flat)
        
        # We process each valid position 
        for idx in range(len(y_flat)):
            if valid_mask[idx]:
                total_valid += 1
                if rng.random() > retention_pct:
                    y_flat[idx] = float('nan')
                    total_dropped += 1
    
    logger.info(f"Sparsity Mask ({retention_pct:.0%}): Dropped {total_dropped}/{total_valid} valid training labels.")
    return dm_corrupted


def evaluate_torch_model(lit_module, datamodule, task_types, target_names, target_to_ds):
    """Evaluate a PyTorch model natively on the uncorrupted test dataset."""
    trainer = pl.Trainer(accelerator="auto", devices=1, logger=False)
    test_metrics = trainer.test(lit_module, datamodule=datamodule, verbose=False)
    metrics_dict = test_metrics[0]
    
    all_aurocs = []
    task_results = {}
    
    for i, (ttype, name) in enumerate(zip(task_types, target_names)):
        if ttype == "classification":
            ds_name = target_to_ds.get(name, "unknown").upper()
            k = f"test/{ds_name} AUROC/{name}"
            # Also occasionally lightning logs it without test/ prefix if configured differently
            # but JointSemiSupModule hardcodes "test/..."
            if k in metrics_dict:
                val = metrics_dict[k]
                all_aurocs.append(val)
                task_results[name] = float(val)
            else:
                task_results[name] = 0.0
            
    if not all_aurocs:
        val = metrics_dict.get("test_auroc_epoch", metrics_dict.get("test_auroc", 0.0))
        return {"auroc": float(val), "tasks": {}}
    else:
        return {"auroc": float(np.mean(all_aurocs)), "tasks": task_results}


def evaluate_tabular_multi_task(model_cls, task_types, target_names, X_train, y_train, X_test, y_test, X_val=None, y_val=None):
    """Fallback loop evaluating a single-task baseline across all tasks iteratively."""
    aurocs = []
    task_results = {}
    
    for i, (ttype, name) in enumerate(zip(task_types, target_names)):
        if ttype != "classification":
            continue
            
        y_train_i = y_train[:, i]
        y_test_i = y_test[:, i]
        
        valid_train = ~np.isnan(y_train_i)
        valid_test = ~np.isnan(y_test_i)
        
        if len(np.unique(y_train_i[valid_train])) < 2 or len(np.unique(y_test_i[valid_test])) < 2:
            task_results[name] = float('nan')
            continue
            
        eval_set = None
        if X_val is not None:
            y_val_i = y_val[:, i]
            eval_set = [(X_val, y_val_i)]
            
        model = model_cls(task_type=ttype)
        model.fit(X_train, y_train_i, eval_set=eval_set)
        
        res = model.evaluate(X_test, y_test_i)
        if "auroc" in res and res["auroc"] > 0:
             aurocs.append(res["auroc"])
             task_results[name] = res["auroc"]
        else:
             task_results[name] = 0.0

    if not aurocs:
        return {"auroc": 0.0, "tasks": {}}
    return {"auroc": float(np.mean(aurocs)), "tasks": task_results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    
    args, _ = parser.parse_known_args()
    if args.config:
        apply_config_to_parser(parser, args.config)
    args = parser.parse_args()
    config = load_config(args.config)
    
    # Base Setup Uncorrupted
    ds_name = config.get("dataset_name", "tox21")
    logger.info(f"=== Preparing base uncorrupted {ds_name} dataset ===")
    
    df, scaling_stats, target_names, task_types, target_to_ds = build_unified_dataframe([ds_name])
    graphs, train_idx, val_idx, test_idx = preprocess_unified_dataset(
        df, 
        target_names=target_names,
        split_type=config.get("split_type", "stratified_butina"),
        similarity_cutoff=config.get("similarity_cutoff", 0.4),
    )
    
    dm_truth = MoleculeDataModule(
        graphs=graphs,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        batch_size=args.batch_size,
    )
    dm_truth.setup()
    
    X_val_fp, y_val_truth = extract_fingerprints(dm_truth.val_dataset)
    X_test_fp, y_test_truth = extract_fingerprints(dm_truth.test_dataset)
    
    node_dim = get_node_feature_dim()
    edge_dim = get_edge_feature_dim()
    num_tasks = len(target_names)
    
    retention_rates = config.get("retention_rates", [1.0, 0.8, 0.5, 0.1])
    experiment_models = config.get("models", ["gcn", "gin", "xgboost", "lightgbm"])
    
    results = []
    
    for pct in retention_rates:
        logger.info(f"\n{'='*60}\nEvaluating Label Retention: {pct:.0%}\n{'='*60}")
        dm_sparse = apply_sparsity_mask(dm_truth, retention_pct=pct)
        X_train_fp, y_train_sparse = extract_fingerprints(dm_sparse.train_dataset)

        for m_name in experiment_models:
            logger.info(f"--- Training {m_name.upper()} @ {pct:.0%} Retention ---")
            
            if m_name in ["gcn", "gin", "rgcn", "pna"]:
                model_config = {
                    "backbone_name": m_name,
                    "node_dim": node_dim,
                    "edge_dim": edge_dim,
                    "num_tasks": num_tasks,
                    "bottleneck_dim": config.get("bottleneck_dim", 256),
                    "hidden_dim": config.get("hidden_dim", 256),
                    "num_layers": config.get("num_layers", 3),
                    "dropout": config.get("dropout", 0.3),
                    "deg": None,
                }
                net = build_joint_model(**model_config)
                lit_module = JointSemiSupModule(
                    model=net,
                    task_types=task_types,
                    dataset_names=target_names,
                    target_to_ds=target_to_ds,
                    learning_rate=args.lr,
                )
                
                trainer = pl.Trainer(
                    accelerator="auto",
                    devices=1,
                    max_epochs=args.epochs,
                    logger=False,
                    enable_progress_bar=False,
                )
                
                trainer.fit(lit_module, datamodule=dm_sparse)
                
                res = evaluate_torch_model(lit_module, dm_truth, task_types, target_names, target_to_ds)
                
                row = {"Retention": f"{pct:.0%}", "Model": m_name.upper(), "Mean AUROC": res["auroc"]}
                row.update(res["tasks"])
                results.append(row)
                
            elif m_name in ["xgboost", "lightgbm"]:
                model_cls = XGBoostBaseline if m_name == "xgboost" else LightGBMBaseline
                
                res = evaluate_tabular_multi_task(
                    model_cls, task_types, target_names,
                    X_train_fp, y_train_sparse, 
                    X_test_fp, y_test_truth, 
                    X_val=X_val_fp, y_val=y_val_truth
                )
                row = {"Retention": f"{pct:.0%}", "Model": m_name.upper(), "Mean AUROC": res["auroc"]}
                row.update(res["tasks"])
                results.append(row)
                
    # Review Final Results
    summary_df = pd.DataFrame(results)
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    
    print("\n" + "=" * 120)
    print("    SEMI-SUPERVISED NECESSITY EXPERIMENT RESULTS (Tox21)")
    print("=" * 120)
    print(summary_df.to_string(index=False, float_format="%.3f"))
    print("=" * 120)

if __name__ == "__main__":
    main()

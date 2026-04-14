"""Matrix execution supervised training script.

Loops over configured datasets and models (both tabular baselines and PyTorch
Lightning graph models) and outputs a comprehensive comparison matrix.

Usage:
    python scripts/train_supervised.py --config configs/supervised.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
import os
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from mol_prop_gnn.data.dataset import MoleculeDataModule
from mol_prop_gnn.data.unified_dataset import preprocess_unified_dataset
from mol_prop_gnn.data.preprocessing import (
    compute_fingerprint,
    compute_descriptors,
    get_node_feature_dim,
    get_edge_feature_dim,
)
from mol_prop_gnn.models.mlp_baseline import MLPBaseline
from mol_prop_gnn.models.gcn import MolGCN
from mol_prop_gnn.models.rgcn import MolRGCN
from mol_prop_gnn.models.gin import MolGIN
from mol_prop_gnn.models.rdkit_baseline import RDKitBaseline
from mol_prop_gnn.models.xgboost_baseline import XGBoostBaseline
from mol_prop_gnn.models.lightgbm_baseline import LightGBMBaseline
from mol_prop_gnn.training.supervised_module import MolPropertyModule
from mol_prop_gnn.utils.config import apply_config_to_parser, load_config
from mol_prop_gnn.data.download import get_dataset_info, download_moleculenet

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


def extract_fingerprints(dataset, n_bits=2048):
    fps, labels = [], []
    for g in dataset:
        if hasattr(g, "smiles") and getattr(g, "smiles", None) is not None:
            fp = compute_fingerprint(g.smiles, n_bits=n_bits)
            if fp is not None:
                fps.append(fp)
                labels.append(g.y.numpy().flatten()[0])
    return np.array(fps), np.array(labels)


def extract_descriptors(dataset):
    descs, labels = [], []
    for g in dataset:
        if hasattr(g, "smiles") and getattr(g, "smiles", None) is not None:
            desc = compute_descriptors(g.smiles)
            if desc is not None:
                descs.append(desc)
                labels.append(g.y.numpy().flatten()[0])
    return np.array(descs), np.array(labels)


def evaluate_torch_model(module, datamodule, task_type):
    """Evaluate a PyTorch model on the test dataset."""
    trainer = pl.Trainer(accelerator="auto", devices=1, logger=False)
    test_metrics = trainer.test(module, datamodule=datamodule, verbose=False)
    metrics_dict = test_metrics[0]
    
    if task_type == "classification":
        acc = metrics_dict.get(f"test_acc_epoch", metrics_dict.get("test_acc", 0.0))
        auroc = metrics_dict.get(f"test_auroc_epoch", metrics_dict.get("test_auroc", 0.0))
        return {"accuracy": float(acc), "auroc": float(auroc)}
    else:
        rmse = metrics_dict.get(f"test_rmse_epoch", metrics_dict.get("test_rmse", 0.0))
        mae = metrics_dict.get(f"test_mae_epoch", metrics_dict.get("test_mae", 0.0))
        r2 = metrics_dict.get(f"test_r2_epoch", metrics_dict.get("test_r2", 0.0))
        return {"rmse": float(rmse), "mae": float(mae), "r2": float(r2)}


def build_single_dataset(ds_name: str) -> tuple[pd.DataFrame, list[str]]:
    csv_path = download_moleculenet(ds_name)
    info = get_dataset_info(ds_name)
    
    df = pd.read_csv(csv_path)
    if "smiles" not in df.columns:
        smiles_col = info.get("smiles_col", "smiles")
        if smiles_col in df.columns:
            df = df.rename(columns={smiles_col: "smiles"})
        else:
            # Fallback
            sc = [c for c in df.columns if c.lower() in ["smiles", "canonical_smiles", "mol"]]
            if sc:
                df = df.rename(columns={sc[0]: "smiles"})
    keep_cols = ["smiles"] + info["target_cols"]
    df = df[keep_cols]
    return df, info["target_cols"]


def main():
    parser = argparse.ArgumentParser(description="Molecules Simple Supervised Evaluator")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--accelerator", type=str, default="auto")
    
    args, _ = parser.parse_known_args()
    if args.config:
        apply_config_to_parser(parser, args.config)
    args = parser.parse_args()
    config = load_config(args.config)

    dataset_names = config.get("datasets", ["bbbp"])
    model_names = config.get("models", ["mlp"])
    split_type = config.get("split_type", "stratified_butina")
    
    results = []

    for ds_name in dataset_names:
        logger.info(f"=== Processing Dataset: {ds_name.upper()} ===")
        df, target_names = build_single_dataset(ds_name)
        info = get_dataset_info(ds_name)
        task_type = info["task_type"]
        main_target = target_names[0]
        
        # Build graphs
        graphs, train_idx, val_idx, test_idx = preprocess_unified_dataset(
            df,
            target_names=[main_target],
            split_type=split_type,
            similarity_cutoff=config.get("similarity_cutoff", 0.4),
        )

        dm = MoleculeDataModule(
            graphs=graphs,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            batch_size=args.batch_size,
        )
        dm.setup()
        
        # Tabular features
        X_train_fp, y_train = None, None
        X_train_desc, y_train_desc = None, None

        for m_name in model_names:
            logger.info(f"--- Training {m_name.upper()} on {ds_name.upper()} ---")
            
            if m_name in ["xgboost", "lightgbm"]:
                if X_train_fp is None:
                    # Lazy extract FPs once
                    X_train_fp, y_train = extract_fingerprints(dm.train_dataset)
                    X_val_fp, y_val = extract_fingerprints(dm.val_dataset)
                    X_test_fp, y_test = extract_fingerprints(dm.test_dataset)
                    
                if m_name == "xgboost":
                    model = XGBoostBaseline(task_type=task_type)
                else:
                    model = LightGBMBaseline(task_type=task_type)
                    
                model.fit(X_train_fp, y_train, eval_set=[(X_val_fp, y_val)])
                res = model.evaluate(X_test_fp, y_test)
                results.append({"Dataset": ds_name.upper(), "Model": m_name.upper(), **res})
                
            elif m_name == "rdkit_rf":
                if X_train_desc is None:
                    X_train_desc, y_train_desc = extract_descriptors(dm.train_dataset)
                    X_test_desc, y_test_desc = extract_descriptors(dm.test_dataset)
                
                model = RDKitBaseline(task_type=task_type)
                model.fit(X_train_desc, y_train_desc)
                res = model.evaluate(X_test_desc, y_test_desc)
                results.append({"Dataset": ds_name.upper(), "Model": "RDKIT_RF", **res})
                
            elif m_name in ["mlp", "gcn", "rgcn", "gin"]:
                node_dim = get_node_feature_dim()
                edge_dim = get_edge_feature_dim()
                
                hidden_dim = config.get("hidden_dim", 64)
                
                if m_name == "mlp":
                    net = MLPBaseline(input_dim=node_dim, hidden_dims=[hidden_dim]*config.get("num_layers", 2), dropout=config.get("dropout", 0.2), output_dim=1)
                elif m_name == "gcn":
                    net = MolGCN(node_input_dim=node_dim, edge_input_dim=edge_dim, hidden_dim=hidden_dim, num_gnn_layers=config.get("num_layers", 3), output_dim=1)
                elif m_name == "rgcn":
                    net = MolRGCN(node_input_dim=node_dim, edge_input_dim=edge_dim, hidden_dim=hidden_dim, num_layers=config.get("num_layers", 3), output_dim=1)
                elif m_name == "gin":
                    net = MolGIN(node_input_dim=node_dim, hidden_dim=hidden_dim, num_gnn_layers=config.get("num_layers", 3), output_dim=1)

                lit_module = MolPropertyModule(
                    model=net,
                    task_type=task_type,
                    learning_rate=args.lr,
                )
                
                tb_logger = pl.loggers.TensorBoardLogger("lightning_logs", name=f"{ds_name}_{m_name}")
                trainer = pl.Trainer(
                    accelerator=args.accelerator,
                    devices=1,
                    max_epochs=args.epochs,
                    logger=tb_logger,
                    enable_progress_bar=False,
                )
                
                trainer.fit(lit_module, datamodule=dm)
                res = evaluate_torch_model(lit_module, dm, task_type)
                results.append({"Dataset": ds_name.upper(), "Model": m_name.upper(), **res})
            else:
                logger.warning(f"Unknown model configured: {m_name}")

    # Display full matrix results
    summary_df = pd.DataFrame(results)
    
    print("\n" + "=" * 80)
    print("                 MATRIX EXECUTION BASELINE RESULTS")
    print(f"      Split config: {split_type}")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print("=" * 80)


if __name__ == "__main__":
    main()

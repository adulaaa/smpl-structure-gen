"""Script to train GAE and generate node embeddings for all countries."""

import argparse
import logging
import sys
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from trade_flow_gcn.utils.config import load_config
from trade_flow_gcn.data.preprocessing import preprocess_pipeline
from trade_flow_gcn.data.dataset import build_graphs_from_dataframe
from trade_flow_gcn.models.gae import create_gae

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

def train_gae(data_list, in_channels, latent_dim=16, epochs=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_gae(in_channels, latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    logger.info("Training GAE on all years to learn structural embeddings...")
    model.train()
    for epoch in tqdm(range(epochs)):
        total_loss = 0
        for data in data_list:
            data = data.to(device)
            optimizer.zero_grad()
            z = model.encode(data.x, data.edge_index)
            loss = model.recon_loss(z, data.edge_index)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
    model.eval()
    return model, device

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument("--output", type=str, default="data/processed/node_embeddings.npy")
    args = parser.parse_args()
    
    config = load_config(args.config)
    root = Path(".")
    
    # 1. Load Data
    raw_dir = root / config['data']['raw_dir']
    csv_candidates = list(raw_dir.glob("*.csv"))
    csv_path = max(csv_candidates, key=lambda p: p.stat().st_size)
    
    df = preprocess_pipeline(csv_path, config)
    graphs = build_graphs_from_dataframe(df, config['data']['countries'], config)
    
    # 2. Train GAE
    in_channels = len(config['data']['node_features'])
    model, device = train_gae(graphs, in_channels, latent_dim=args.latent_dim)
    
    # 3. Generate and Save Embeddings per year
    logger.info("Generating embeddings...")
    embeddings_dict = {}
    with torch.no_grad():
        for i, data in enumerate(graphs):
            year = config['data']['train_years'][0] + i # Simplistic year mapping
            # Correct year mapping:
            # We should probably store the year in the Graph object during construction
            # For now, let's just use the index
            z = model.encode(data.x.to(device), data.edge_index.to(device))
            embeddings_dict[i] = z.cpu().numpy()
            
    output_path = root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, embeddings_dict)
    logger.info("Embeddings saved to %s", output_path)

if __name__ == "__main__":
    main()

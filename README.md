# TradeFlowGNN

**Forecasting Bilateral Trade with Graph Neural Networks**

## Research Question

> Can a graph neural network predict future bilateral trade flows better than classical gravity models and standard tabular ML baselines?
>
> *Supporting question:* Does using the full global trade network improve prediction beyond pairwise country features alone?

## Overview

This project represents the world economy as a **directed weighted graph** where:
- **Nodes** = countries
- **Edges** = bilateral trade flows (exports from country *i* to country *j*)
- **Edge weights** = trade value (log-transformed)

The model predicts **future trade volume** between country pairs using:
- Past trade relations
- Country-level macro features (GDP, population, etc.)
- Pairwise gravity-style features (distance, contiguity, common language, etc.)

## Project Structure

```
trade_flow_gcn/
├── configs/              # YAML configuration files
├── data/                 # Raw and processed data (gitignored)
├── idea/                 # Original project idea documents
├── notebooks/            # Jupyter notebooks for exploration
├── scripts/              # Entry-point scripts
│   ├── download_data.py  # Download CEPII Gravity dataset
│   └── train.py          # Main training script
├── src/trade_flow_gcn/   # Main source package
│   ├── data/             # Data loading & preprocessing
│   ├── models/           # GCN, gravity baseline, MLP baseline
│   ├── training/         # PyTorch Lightning module
│   ├── evaluation/       # Metrics
│   └── utils/            # Configuration utilities
└── tests/                # Unit & smoke tests
```

## Setup

This project uses [uv](https://github.com/astral-sh/uv) for extremely fast Python package and environment management.

### 1. Install uv

**Windows:**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**macOS / Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# Or using Homebrew
brew install uv
```

### 2. Project Setup

```bash
# Clone the repository
git clone https://github.com/tolyaho/TradeFlowGCN.git
cd trade_flow_gcn

# Install dependencies and create environment
uv sync --all-extras
```

## Quick Start

Using `uv run` ensures commands are executed in the project environment:

```bash
# 1. Download CEPII Gravity data
uv run python scripts/download_data.py

# 2. Train the GCN model
uv run python scripts/train.py --config configs/default.yaml

# 3. Run tests
uv run pytest tests/ -v
```

## Data Source

**CEPII Gravity Database** — provides bilateral trade flows at the country-pair-year level with pre-joined gravity variables (distance, contiguity, common language, colonial ties).

## Tech Stack

- **PyTorch** + **PyTorch Geometric** — GNN implementation
- **PyTorch Lightning** — training loop, logging, checkpointing
- **pandas** / **NumPy** — data processing
- **scikit-learn** — baseline models & metrics

## License

MIT

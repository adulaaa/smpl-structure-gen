"""Causal mask visualization for ClearML debug samples.

Generates atom-level importance heatmaps showing which parts of a molecule
the causal model considers pharmacophore (property-driving) vs scaffold
(structural filler). Designed to be understandable for non-chemists.

Colors:
    Gray   → Scaffold (the model ignores this part)
    Yellow → Moderate importance
    Red    → Pharmacophore (the model relies heavily on this part)
"""

from __future__ import annotations

import io
import logging
from typing import Optional

import numpy as np
import torch
from torch_geometric.data import Batch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

import pytorch_lightning as pl

logger = logging.getLogger(__name__)

# ── Custom colormap ──────────────────────────────────────────────────
# Blue (scaffold, ignored) → white (neutral) → red (pharmacophore, critical)
# Diverging scheme maximizes visual contrast for non-chemists
CAUSAL_CMAP = LinearSegmentedColormap.from_list(
    "causal_importance",
    ["#1565C0", "#64B5F6", "#E3F2FD", "#FFCDD2", "#EF5350", "#B71C1C"],
    N=256,
)


# ── Single-molecule rendering ────────────────────────────────────────

def render_mol_with_mask(
    smiles: str,
    mask_values: np.ndarray,
    img_size: tuple[int, int] = (500, 400),
) -> Optional[Image.Image]:
    """Draw a 2D molecule with atoms colored by causal importance.

    Parameters
    ----------
    smiles : str
        SMILES string of the molecule.
    mask_values : ndarray, shape (num_atoms,)
        Per-atom causal mask values in [0, 1].
    img_size : (width, height)
        Output image size in pixels.

    Returns
    -------
    PIL.Image.Image or None
        Rendered molecule image, or None on failure.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError:
        logger.warning("RDKit not available. Skipping molecule visualization.")
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    AllChem.Compute2DCoords(mol)
    n_atoms = mol.GetNumAtoms()
    weights = np.clip(mask_values[:n_atoms], 0.0, 1.0)

    # Enhance contrast: stretch per-molecule mask to fill full [0, 1] range
    # so even small differences between atoms become visually obvious
    w_min, w_max = weights.min(), weights.max()
    if w_max - w_min > 1e-6:
        weights = (weights - w_min) / (w_max - w_min)
    else:
        weights = np.full_like(weights, 0.5)  # uniform mask → neutral color

    # Map mask values → highlight colors and radii
    atom_colors = {}
    atom_radii = {}
    for i, w in enumerate(weights):
        rgba = CAUSAL_CMAP(float(w))
        atom_colors[i] = (rgba[0], rgba[1], rgba[2])
        # Higher importance → larger highlight circle
        atom_radii[i] = 0.35 + 0.25 * float(w)

    try:
        from rdkit.Chem.Draw import rdMolDraw2D
        d = rdMolDraw2D.MolDraw2DCairo(*img_size)
    except Exception:
        logger.warning("Cairo/X11 backend unavailable for RDKit drawing. Skipping.")
        return None

    opts = d.drawOptions()
    opts.useBWAtomPalette()  # Black atom labels for contrast on colored circles

    d.DrawMolecule(
        mol,
        highlightAtoms=list(range(n_atoms)),
        highlightAtomColors=atom_colors,
        highlightAtomRadii=atom_radii,
        highlightBonds=[],
    )
    d.FinishDrawing()

    return Image.open(io.BytesIO(d.GetDrawingText()))


# ── Composite grid report ────────────────────────────────────────────

def create_causal_report(
    smiles_list: list[str],
    mask_list: list[np.ndarray],
    pred_list: list[np.ndarray],
    task_names: list[str],
    task_types: list[str],
    epoch: int,
    ncols: int = 3,
) -> Optional[Image.Image]:
    """Build a grid of molecule visualizations for ClearML debug samples.

    Each cell shows:
    * The molecule with atom-level importance coloring
    * A subtitle with model predictions and mask statistics

    Parameters
    ----------
    smiles_list : list[str]
        SMILES strings for each molecule.
    mask_list : list[ndarray]
        Per-atom mask arrays (one per molecule).
    pred_list : list[ndarray]
        Model predictions (causal head logits), shape (num_tasks,) each.
    task_names, task_types : list[str]
        Parallel lists describing each task.
    epoch : int
        Current training epoch (for the title).
    ncols : int
        Number of columns in the grid.

    Returns
    -------
    PIL.Image.Image or None
    """
    n = len(smiles_list)
    if n == 0:
        return None

    nrows = max(1, (n + ncols - 1) // ncols)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5.5 * ncols, 6.5 * nrows),
        dpi=110,
    )

    # Normalize axes to 2D array
    axes = np.atleast_2d(axes)
    if axes.shape[0] == 1 and nrows > 1:
        axes = axes.T

    fig.suptitle(
        f"Causal Subgraph Analysis — Epoch {epoch}\n"
        "Blue = scaffold (not important)  ·  "
        "Red = pharmacophore (drives prediction)",
        fontsize=11,
        fontweight="bold",
        y=0.99,
    )

    for idx in range(nrows * ncols):
        row, col = idx // ncols, idx % ncols
        ax = axes[row, col] if nrows > 1 else axes[0, col] if ncols > 1 else axes[0, 0]

        if idx >= n:
            ax.axis("off")
            continue

        smiles = smiles_list[idx]
        mask = mask_list[idx]
        preds = pred_list[idx]

        # Render molecule
        mol_img = render_mol_with_mask(smiles, mask, img_size=(450, 330))
        if mol_img is None:
            ax.text(
                0.5, 0.5,
                f"Could not render\n{smiles[:30]}…",
                ha="center", va="center", fontsize=8,
            )
            ax.axis("off")
            continue

        ax.imshow(mol_img)
        ax.axis("off")

        # ── Subtitle: predictions + mask stats ──
        pred_parts = []
        for ti, (tn, tt) in enumerate(zip(task_names, task_types)):
            val = float(preds[ti])
            if tt == "classification":
                prob = torch.sigmoid(torch.tensor(val)).item()
                indicator = "✓" if prob > 0.5 else "✗"
                pred_parts.append(f"{indicator} {tn}: {prob:.0%}")
            else:
                pred_parts.append(f"{tn}: {val:.2f}")

        pred_text = "  ·  ".join(pred_parts)
        mask_pct = float(mask.mean()) * 100
        mask_text = f"Mask activity: {mask_pct:.0f}%"

        short_smiles = smiles if len(smiles) <= 45 else smiles[:42] + "…"
        ax.set_title(
            f"{short_smiles}\n{pred_text}\n{mask_text}",
            fontsize=7.5,
            pad=4,
            linespacing=1.4,
        )

    # ── Colorbar legend at the bottom ──
    cbar_ax = fig.add_axes([0.20, 0.02, 0.60, 0.013])
    sm = plt.cm.ScalarMappable(cmap=CAUSAL_CMAP, norm=plt.Normalize(0, 1))
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label(
        "← Blue: Scaffold  (ignored by the model)                    "
        "Red: Pharmacophore  (drives prediction) →",
        fontsize=9,
    )
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(["0%", "25%", "50%", "75%", "100%"])

    plt.subplots_adjust(left=0.03, right=0.97, top=0.90, bottom=0.07, hspace=0.45, wspace=0.15)

    # Convert matplotlib figure → PIL Image
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


# ── Lightning Callback ───────────────────────────────────────────────

class CausalVisualizationCallback(pl.Callback):
    """Generate causal mask visualizations and report them as ClearML debug samples.

    Selects a fixed set of molecules from the test set and, every *N* epochs,
    runs a forward pass to obtain the learned causal mask.  Each atom is then
    colored on a gray-to-red gradient indicating how important the model
    considers it for predicting molecular properties.

    Parameters
    ----------
    sample_graphs : list
        Pool of PyG Data objects to sample from (must have ``.smiles``).
    task_names : list[str]
        Names of prediction targets.
    task_types : list[str]
        ``"classification"`` or ``"regression"`` per target.
    num_samples : int
        Number of molecules to visualize per report.
    every_n_epochs : int
        Generate a report every *N* epochs (and always on the last epoch).
    """

    def __init__(
        self,
        sample_graphs: list,
        task_names: list[str],
        task_types: list[str],
        num_samples: int = 6,
        every_n_val: int = 1,
    ):
        super().__init__()
        self.task_names = task_names
        self.task_types = task_types
        self.every_n_val = every_n_val
        self._val_count = 0

        # Filter to graphs that actually have SMILES
        candidates = [g for g in sample_graphs if getattr(g, "smiles", None)]
        if not candidates:
            logger.warning(
                "No graphs with SMILES found — causal visualization will be skipped."
            )
            self._sample_graphs = []
            return

        # Select a fixed random subset for consistency across epochs
        rng = np.random.RandomState(42)
        k = min(num_samples, len(candidates))
        indices = rng.choice(len(candidates), size=k, replace=False)
        self._sample_graphs = [candidates[i] for i in sorted(indices)]

    # ──────────────────────────────────────────────────────────────────

    def on_validation_epoch_end(self, trainer, pl_module):
        if not self._sample_graphs:
            return

        self._val_count += 1
        epoch = trainer.current_epoch

        # Fire every N-th validation call, and always on the last epoch.
        # Uses a call counter (not epoch number) to avoid misalignment with
        # check_val_every_n_epoch which makes validation epochs non-sequential.
        is_last = epoch == trainer.max_epochs - 1
        if self._val_count % self.every_n_val != 0 and not is_last:
            return

        # Get ClearML logger (gracefully skip if unavailable)
        try:
            from clearml import Task

            task = Task.current_task()
            if task is None:
                return
            cl_logger = task.get_logger()
        except ImportError:
            return

        # ── Forward pass on sample molecules ──
        pl_module.eval()
        device = pl_module.device
        batch = Batch.from_data_list(self._sample_graphs).to(device)

        with torch.no_grad():
            pred_c, pred_e, mask = pl_module(batch)

        # ── Disaggregate per molecule ──
        smiles_list: list[str] = []
        mask_list: list[np.ndarray] = []
        pred_list: list[np.ndarray] = []

        for mol_idx in range(len(self._sample_graphs)):
            graph = self._sample_graphs[mol_idx]
            smiles = getattr(graph, "smiles", None)
            if not smiles:
                continue

            # Per-atom mask values for this molecule
            atom_mask = mask[batch.batch == mol_idx].cpu().numpy().flatten()
            mol_preds = pred_c[mol_idx].cpu().numpy()

            smiles_list.append(smiles)
            mask_list.append(atom_mask)
            pred_list.append(mol_preds)

        if not smiles_list:
            return

        # ── Create and report composite image ──
        report_img = create_causal_report(
            smiles_list=smiles_list,
            mask_list=mask_list,
            pred_list=pred_list,
            task_names=self.task_names,
            task_types=self.task_types,
            epoch=epoch,
        )

        if report_img is not None:
            cl_logger.report_image(
                title="Causal Subgraph Analysis",
                series="Atom Importance Heatmap",
                image=np.array(report_img),
                iteration=epoch,
            )
            logger.info(
                "Reported causal mask visualization for epoch %d (%d molecules)",
                epoch,
                len(smiles_list),
            )

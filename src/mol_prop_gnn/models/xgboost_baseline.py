"""XGBoost baseline for molecular property prediction on fingerprints.

Operates on dense fingerprint arrays rather than graph structures.
"""

from __future__ import annotations

import logging
import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


class XGBoostBaseline:
    """XGBoost baseline operating on molecular fingerprints.

    Parameters
    ----------
    task_type : str
        'classification' or 'regression'.
    n_estimators : int
        Number of boosting rounds.
    learning_rate : float
        Pacing of learning.
    """

    def __init__(
        self,
        task_type: str = "classification",
        n_estimators: int = 500,
        learning_rate: float = 0.05,
        random_state: int = 42,
    ) -> None:
        self.task_type = task_type
        if task_type == "classification":
            self.model = xgb.XGBClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                random_state=random_state,
                n_jobs=-1,
                eval_metric="auc",
                early_stopping_rounds=50,
            )
        else:
            self.model = xgb.XGBRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                random_state=random_state,
                n_jobs=-1,
                eval_metric="rmse",
                early_stopping_rounds=50,
            )
        self._is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray, eval_set: list[tuple[np.ndarray, np.ndarray]] | None = None) -> "XGBoostBaseline":
        """Fit the XGBoost model.

        Parameters
        ----------
        X : (N, D) molecular descriptors or fingerprints
        y : (N,) targets
        eval_set : List of (X_val, y_val) tuples for early stopping.
        """
        valid = ~np.isnan(y)
        
        if eval_set is not None:
            # Also filter NaNs from eval sets
            filtered_eval_set = []
            for X_val, y_val in eval_set:
                v_valid = ~np.isnan(y_val)
                filtered_eval_set.append((X_val[v_valid], y_val[v_valid]))
            
            self.model.fit(
                X[valid], y[valid], 
                eval_set=filtered_eval_set, 
                verbose=False
            )
        else:
            # We must pass eval_set to use early_stopping_rounds, but if not provided, disable it
            if hasattr(self.model, "early_stopping_rounds"):
                self.model.set_params(early_stopping_rounds=None)
            self.model.fit(X[valid], y[valid], verbose=False)
            
        self._is_fitted = True
        logger.info("XGBoost baseline fitted on %d molecules.", valid.sum())
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.task_type != "classification":
            raise ValueError("predict_proba only available for classification.")
        return self.model.predict_proba(X)[:, 1]

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        """Evaluate predictions against ground truth."""
        valid = ~np.isnan(y)
        X_valid, y_valid = X[valid], y[valid]

        if self.task_type == "classification":
            y_pred = self.predict(X_valid)
            y_proba = self.predict_proba(X_valid)
            metrics = {
                "accuracy": float(accuracy_score(y_valid, y_pred)),
                "auroc": float(roc_auc_score(y_valid, y_proba))
                if len(np.unique(y_valid)) > 1 else 0.0,
            }
        else:
            y_pred = self.predict(X_valid)
            metrics = {
                "rmse": float(np.sqrt(mean_squared_error(y_valid, y_pred))),
                "mae": float(mean_absolute_error(y_valid, y_pred)),
                "r2": float(r2_score(y_valid, y_pred)),
            }

        return metrics

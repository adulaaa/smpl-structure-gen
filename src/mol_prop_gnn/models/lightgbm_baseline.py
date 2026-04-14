"""LightGBM baseline for molecular property prediction on fingerprints."""

from __future__ import annotations

import logging
import numpy as np
import lightgbm as lgb
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


class LightGBMBaseline:
    """LightGBM baseline operating on molecular fingerprints."""

    def __init__(
        self,
        task_type: str = "classification",
        n_estimators: int = 500,
        learning_rate: float = 0.05,
        random_state: int = 42,
    ) -> None:
        self.task_type = task_type
        if task_type == "classification":
            self.model = lgb.LGBMClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                random_state=random_state,
                n_jobs=-1,
                verbose=-1,
            )
        else:
            self.model = lgb.LGBMRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                random_state=random_state,
                n_jobs=-1,
                verbose=-1,
            )
        self._is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray, eval_set: list[tuple[np.ndarray, np.ndarray]] | None = None) -> "LightGBMBaseline":
        valid = ~np.isnan(y)
        
        callbacks = []
        if eval_set is not None:
            filtered_eval_set = []
            for X_val, y_val in eval_set:
                v_valid = ~np.isnan(y_val)
                filtered_eval_set.append((X_val[v_valid], y_val[v_valid]))
            
            callbacks.append(lgb.early_stopping(stopping_rounds=50, verbose=False))
            self.model.fit(
                X[valid], y[valid], 
                eval_set=filtered_eval_set, 
                callbacks=callbacks,
            )
        else:
            self.model.fit(X[valid], y[valid])
            
        self._is_fitted = True
        logger.info("LightGBM baseline fitted on %d molecules.", valid.sum())
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

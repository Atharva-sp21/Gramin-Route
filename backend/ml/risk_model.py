import numpy as np
import pickle
import os
from pathlib import Path
from typing import Optional, Tuple

import xgboost as xgb

from ml.festival_calendar import get_festival_context

# ==========================================
# XGBOOST RISK MODEL
# Replaces the Quantum VQC layer.
#
# 9 input features:
#   Original 6:  stock, sales, lead_time, margin, shelf_life, credit
#   Festival 3:  days_to_festival, spike_factor, product_affinity
#
# Output: risk probability [0.0 – 1.0]
# ==========================================

BASE_DIR  = Path(__file__).parent.parent          # backend/
MODEL_PATH = BASE_DIR / "models" / "xgb_risk_model.pkl"

FEATURE_NAMES = [
    "stock_ratio",
    "sales_ratio",
    "lead_time_ratio",
    "margin_ratio",
    "shelf_life_ratio",
    "credit_ratio",
    "days_to_festival",
    "spike_factor",
    "product_affinity",
]


class XGBoostRiskModel:
    def __init__(self):
        self.model: Optional[xgb.XGBClassifier] = None
        self._load_or_train()

    def _load_or_train(self):
        if MODEL_PATH.exists():
            with open(MODEL_PATH, "rb") as f:
                self.model = pickle.load(f)
            print("✅ XGBoost Risk Model loaded.")
        else:
            print("⚠️  No saved XGBoost risk model — training on synthetic data...")
            self._train_on_synthetic_data()

    def _train_on_synthetic_data(self):
        np.random.seed(42)
        n = 5000

        stock_ratio      = np.random.uniform(0, 1, n)
        sales_ratio      = np.random.uniform(0, 0.5, n)
        lead_time_ratio  = np.random.uniform(0, 1, n)
        margin_ratio     = np.random.uniform(0, 1, n)
        shelf_life_ratio = np.random.uniform(0, 1, n)
        credit_ratio     = np.random.uniform(0.4, 1.0, n)
        days_norm        = np.random.uniform(0, 1, n)
        spike_norm       = np.random.uniform(0.33, 1.0, n)
        affinity         = np.random.randint(0, 2, n).astype(float)

        X = np.column_stack([
            stock_ratio, sales_ratio, lead_time_ratio, margin_ratio,
            shelf_life_ratio, credit_ratio, days_norm, spike_norm, affinity
        ])

        y = self._rule_based_labels(
            stock_ratio, sales_ratio, lead_time_ratio,
            days_norm * 30, spike_norm * 3.0, affinity
        )

        self.model = xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric="logloss", random_state=42,
        )
        self.model.fit(X, y)

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(self.model, f)
        print("✅ XGBoost Risk Model trained and saved.")

    @staticmethod
    def _rule_based_labels(stock_ratio, sales_ratio, lead_time_ratio,
                            days_to_festival, spike_factor, affinity) -> np.ndarray:
        effective_sales = sales_ratio * 50 * np.where(affinity > 0.5, spike_factor, 1.0)
        days_until_out  = np.where(
            effective_sales > 0,
            (stock_ratio * 200) / (effective_sales + 1e-6),
            30.0
        )
        return (
            (days_until_out < lead_time_ratio * 7) |
            ((affinity > 0.5) & (days_until_out < days_to_festival)) |
            (stock_ratio < 0.10)
        ).astype(int)

    def predict_risk(self, features: np.ndarray) -> float:
        if self.model is None:
            return 0.5
        return float(np.clip(
            self.model.predict_proba(features.reshape(1, -1))[0][1], 0.0, 1.0
        ))

    def feature_importance(self) -> dict:
        if self.model is None:
            return {}
        return {name: round(float(s), 4)
                for name, s in zip(FEATURE_NAMES, self.model.feature_importances_)}


def build_features(
    current_stock: int,
    daily_sales: int,
    lead_time_days: int,
    profit_margin: float,
    shelf_life: int,
    credit_score: int,
    product_name: str = "Rice (50kg)",
) -> Tuple[np.ndarray, dict]:
    festival_ctx = get_festival_context(product_name)

    features = np.array([
        current_stock  / 200.0,
        daily_sales    / 50.0,
        lead_time_days / 7.0,
        profit_margin  / 60.0,
        shelf_life     / 365.0,
        credit_score   / 900.0,
        min(festival_ctx["days_to_festival"] / 30.0, 1.0),
        festival_ctx["spike_factor"] / 3.0,
        festival_ctx["product_affinity"],
    ], dtype=np.float32)

    return features, festival_ctx


_instance: Optional[XGBoostRiskModel] = None

def get_risk_model() -> XGBoostRiskModel:
    global _instance
    if _instance is None:
        _instance = XGBoostRiskModel()
    return _instance

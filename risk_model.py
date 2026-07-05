import numpy as np
import pickle
import os
from typing import Optional, Tuple

import xgboost as xgb

from festival_calendar import get_festival_context

# ==========================================
# XGBOOST RISK MODEL
# Replaces the Quantum VQC layer.
#
# 9 input features (vs. 7 before):
#   Original 6:  stock, sales, lead_time, margin, shelf_life, credit
#   Festival 3:  days_to_festival, spike_factor, product_affinity
#
# Output: risk probability [0.0 – 1.0]
#
# Trained on synthetic labels generated from the existing rule-based
# logic ("knowledge distillation"). Replace labels with real outcome
# data as it accumulates.
# ==========================================

MODEL_PATH = "model/xgb_risk_model.pkl"

FEATURE_NAMES = [
    "stock_ratio",         # current_stock / 200
    "sales_ratio",         # daily_sales / 50
    "lead_time_ratio",     # lead_time_days / 7
    "margin_ratio",        # profit_margin / 60
    "shelf_life_ratio",    # shelf_life / 365
    "credit_ratio",        # credit_score / 900
    "days_to_festival",    # min(days / 30, 1.0)   — closer = higher pressure
    "spike_factor",        # festival_spike / 3.0
    "product_affinity",    # 1.0 if product affected by festival, else 0.0
]


class XGBoostRiskModel:
    """
    Per-shop risk scorer. Runs before GATv2 GNN —
    its output (raw_risk_score) is appended to the node
    feature matrix that GATv2 receives as input.
    """

    def __init__(self):
        self.model: Optional[xgb.XGBClassifier] = None
        self._load_or_train()

    # --------------------------------------------------
    # Load / Train
    # --------------------------------------------------

    def _load_or_train(self):
        if os.path.exists(MODEL_PATH):
            with open(MODEL_PATH, "rb") as f:
                self.model = pickle.load(f)
            print("✅ XGBoost Risk Model loaded.")
        else:
            print("⚠️  No saved XGBoost risk model — training on synthetic data...")
            self._train_on_synthetic_data()

    def _train_on_synthetic_data(self):
        """
        Generates 5,000 synthetic samples and labels them using
        the existing rule-based business logic as a 'teacher'.
        XGBoost generalises these rules to edge cases the
        hard-coded matrix couldn't handle.
        """
        np.random.seed(42)
        n = 5000

        stock_ratio      = np.random.uniform(0, 1, n)
        sales_ratio      = np.random.uniform(0, 0.5, n)
        lead_time_ratio  = np.random.uniform(0, 1, n)
        margin_ratio     = np.random.uniform(0, 1, n)
        shelf_life_ratio = np.random.uniform(0, 1, n)
        credit_ratio     = np.random.uniform(0.4, 1.0, n)
        days_norm        = np.random.uniform(0, 1, n)       # days_to_festival normalized
        spike_norm       = np.random.uniform(0.33, 1.0, n)  # spike_factor / 3.0
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
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
        )
        self.model.fit(X, y)

        os.makedirs("model", exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(self.model, f)
        print("✅ XGBoost Risk Model trained and saved.")

    @staticmethod
    def _rule_based_labels(
        stock_ratio, sales_ratio, lead_time_ratio,
        days_to_festival, spike_factor, affinity
    ) -> np.ndarray:
        """
        Existing business logic used as training labels:
        HIGH RISK (1) if:
          - Stock will run out before the lead time, OR
          - Stock will run out before the festival (for affected products), OR
          - Raw stock is critically low (< 10% of max)
        """
        effective_sales = sales_ratio * 50 * np.where(affinity > 0.5, spike_factor, 1.0)
        days_until_out  = np.where(
            effective_sales > 0,
            (stock_ratio * 200) / (effective_sales + 1e-6),
            30.0
        )
        risk = (
            (days_until_out < lead_time_ratio * 7) |
            ((affinity > 0.5) & (days_until_out < days_to_festival)) |
            (stock_ratio < 0.10)
        ).astype(int)
        return risk

    # --------------------------------------------------
    # Inference
    # --------------------------------------------------

    def predict_risk(self, features: np.ndarray) -> float:
        """
        features: 1D array of length 9 (already normalized).
        Returns risk probability in [0.0, 1.0].
        """
        if self.model is None:
            return 0.5
        prob = self.model.predict_proba(features.reshape(1, -1))[0][1]
        return float(np.clip(prob, 0.0, 1.0))

    def feature_importance(self) -> dict:
        """Returns feature → importance score for interpretability / demo."""
        if self.model is None:
            return {}
        scores = self.model.feature_importances_
        return {name: round(float(s), 4) for name, s in zip(FEATURE_NAMES, scores)}


# --------------------------------------------------
# Feature Builder
# --------------------------------------------------

def build_features(
    current_stock: int,
    daily_sales: int,
    lead_time_days: int,
    profit_margin: float,
    shelf_life: int,
    credit_score: int,
    product_name: str = "Rice (50kg)",
) -> Tuple[np.ndarray, dict]:
    """
    Builds the normalized 9-feature vector for XGBoost
    and returns the festival context alongside it.

    Returns:
        features:     np.ndarray shape (9,)
        festival_ctx: dict from festival_calendar.get_festival_context()
    """
    festival_ctx = get_festival_context(product_name)

    features = np.array([
        current_stock / 200.0,
        daily_sales   / 50.0,
        lead_time_days / 7.0,
        profit_margin / 60.0,
        shelf_life    / 365.0,
        credit_score  / 900.0,
        min(festival_ctx["days_to_festival"] / 30.0, 1.0),
        festival_ctx["spike_factor"] / 3.0,
        festival_ctx["product_affinity"],
    ], dtype=np.float32)

    return features, festival_ctx


# --------------------------------------------------
# Singleton
# --------------------------------------------------

_instance: Optional[XGBoostRiskModel] = None


def get_risk_model() -> XGBoostRiskModel:
    global _instance
    if _instance is None:
        _instance = XGBoostRiskModel()
    return _instance

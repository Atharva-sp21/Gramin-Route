import numpy as np
import pickle
import os
from typing import List, Dict, Optional

import xgboost as xgb

# ==========================================
# XGBOOST DISTRIBUTOR RECOMMENDER
# Replaces the hardcoded rule-based scoring matrix in main.py.
#
# Problem framing: Contextual Bandit
#   Context = retailer state (risk score, urgency, festival, etc.)
#   Arms    = 3 distributors (FastTrack, GraminRoute Hub, Budget Movers)
#   Reward  = delivery success × cost efficiency
#
# Current implementation: XGBoost multi-class classifier trained on
# synthetic labels from the existing rule logic. Designed to ingest
# real reward data as it accumulates (swap labels → retrain).
# ==========================================

MODEL_PATH = "model/xgb_recommender.pkl"

DISTRIBUTORS = [
    {
        "id": 0,
        "name": "FastTrack Logistics",
        "cost": 100,
        "speed_hrs": 4,
        "reliability": 0.99,
        "tier": "PREMIUM",
    },
    {
        "id": 1,
        "name": "GraminRoute Hub",
        "cost": 75,
        "speed_hrs": 12,
        "reliability": 0.95,
        "tier": "BALANCED",
    },
    {
        "id": 2,
        "name": "Budget Movers",
        "cost": 60,
        "speed_hrs": 24,
        "reliability": 0.85,
        "tier": "ECONOMY",
    },
]

FEATURE_NAMES = [
    "spatial_risk_score",   # output of SpatialGNN (0–1)
    "days_until_stockout",  # from festival_predictor (normalized /30)
    "days_to_festival",     # from festival context (normalized /30)
    "credit_ratio",         # credit_score / 900
    "qty_needed_ratio",     # recommended_order_qty / 100
]


def _reason(dist_id: int, risk: float, days_stockout: float) -> str:
    if dist_id == 0:
        return f"URGENT: AI risk {int(risk * 100)}% — speed prioritized over cost"
    elif dist_id == 1:
        return "BALANCED: reliable delivery at moderate cost for your risk level"
    else:
        return "COST SAVING: low urgency — optimise for delivery cost"


class DistributorRecommender:
    """
    Multi-class XGBoost classifier:
      class 0 → FastTrack Logistics
      class 1 → GraminRoute Hub
      class 2 → Budget Movers

    predict_proba gives a confidence score per distributor,
    which we return as the ranked recommendation list.
    """

    def __init__(self):
        self.model: Optional[xgb.XGBClassifier] = None
        self._load_or_train()

    def _load_or_train(self):
        if os.path.exists(MODEL_PATH):
            with open(MODEL_PATH, "rb") as f:
                self.model = pickle.load(f)
            print("✅ Distributor Recommender loaded.")
        else:
            print("⚠️  No saved recommender — training on synthetic data...")
            self._train_on_synthetic_data()

    def _train_on_synthetic_data(self):
        np.random.seed(42)
        n = 4000

        risk           = np.random.uniform(0, 1, n)
        days_stockout  = np.random.uniform(0.1, 1.0, n)   # normalized /30
        days_festival  = np.random.uniform(0, 1, n)        # normalized /30
        credit         = np.random.uniform(0.4, 1.0, n)
        qty            = np.random.uniform(0, 1, n)

        X = np.column_stack([risk, days_stockout, days_festival, credit, qty])

        # Labels using existing rule logic:
        # 0 = FastTrack  (high risk or running out fast)
        # 1 = GraminRoute Hub  (medium)
        # 2 = Budget Movers  (low risk, routine)
        y = np.where(
            risk > 0.7,  0,
            np.where(
                days_stockout < 0.17,  0,    # < 5 days → urgent
                np.where(
                    risk > 0.4,  1,
                    2
                )
            )
        )

        self.model = xgb.XGBClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=42,
        )
        self.model.fit(X, y)

        os.makedirs("model", exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(self.model, f)
        print("✅ Distributor Recommender trained and saved.")

    def rank(
        self,
        spatial_risk_score: float,
        days_until_stockout: float,
        days_to_festival: int,
        credit_score: int,
        recommended_qty: int,
    ) -> List[Dict]:
        """
        Returns distributors ranked by confidence (highest first).

        Each entry:
            distributor, tier, confidence, cost, eta, reliability, reason
        """
        features = np.array([[
            spatial_risk_score,
            min(days_until_stockout / 30.0, 1.0),
            min(days_to_festival    / 30.0, 1.0),
            credit_score   / 900.0,
            min(recommended_qty / 100.0, 1.0),
        ]])

        probas = self.model.predict_proba(features)[0]   # [p0, p1, p2]

        ranked = []
        for dist in DISTRIBUTORS:
            ranked.append({
                "distributor": dist["name"],
                "tier":        dist["tier"],
                "confidence":  round(float(probas[dist["id"]]), 3),
                "cost":        dist["cost"],
                "eta":         f"{dist['speed_hrs']} Hours",
                "reliability": dist["reliability"],
                "reason":      _reason(dist["id"], spatial_risk_score, days_until_stockout),
            })

        return sorted(ranked, key=lambda x: x["confidence"], reverse=True)


# --------------------------------------------------
# Singleton
# --------------------------------------------------

_instance: Optional[DistributorRecommender] = None


def get_recommender() -> DistributorRecommender:
    global _instance
    if _instance is None:
        _instance = DistributorRecommender()
    return _instance

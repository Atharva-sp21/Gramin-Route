import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Optional

import xgboost as xgb

# ==========================================
# XGBOOST DISTRIBUTOR RECOMMENDER
# Replaces the hardcoded rule-based scoring matrix.
#
# Contextual Bandit framing:
#   Context = retailer state  (risk, urgency, festival, credit, qty)
#   Arms    = 3 distributors  (FastTrack, GraminRoute Hub, Budget Movers)
#   Reward  = delivery success × cost efficiency
# ==========================================

BASE_DIR = Path(__file__).parent
# Check local (Docker-staged) models/ first, then the shared repo-root models/
_candidates = [
    BASE_DIR / "models" / "xgb_recommender.pkl",                 # local to service (Docker COPY)
    BASE_DIR.parent.parent / "models" / "xgb_recommender.pkl",   # shared repo-root models/
]
MODEL_PATH = next((p for p in _candidates if p.exists()), _candidates[0])

DISTRIBUTORS = [
    {"id": 0, "name": "FastTrack Logistics", "cost": 100, "speed_hrs": 4,  "reliability": 0.99, "tier": "PREMIUM"},
    {"id": 1, "name": "GraminRoute Hub",     "cost": 75,  "speed_hrs": 12, "reliability": 0.95, "tier": "BALANCED"},
    {"id": 2, "name": "Budget Movers",       "cost": 60,  "speed_hrs": 24, "reliability": 0.85, "tier": "ECONOMY"},
]


def _reason(dist_id: int, risk: float, days_stockout: float) -> str:
    if dist_id == 0:
        return f"URGENT: AI risk {int(risk * 100)}% — speed prioritized over cost"
    elif dist_id == 1:
        return "BALANCED: reliable delivery at moderate cost for your risk level"
    return "COST SAVING: low urgency — optimise for delivery cost"


class DistributorRecommender:
    def __init__(self):
        self.model: Optional[xgb.XGBClassifier] = None
        self._load_or_train()

    def _load_or_train(self):
        if MODEL_PATH.exists():
            with open(MODEL_PATH, "rb") as f:
                self.model = pickle.load(f)
            print("✅ Distributor Recommender loaded.")
        else:
            print("⚠️  No saved recommender — training on synthetic data...")
            self._train_on_synthetic_data()

    def _train_on_synthetic_data(self):
        np.random.seed(42)
        n = 4000

        risk          = np.random.uniform(0, 1, n)
        days_stockout = np.random.uniform(0.1, 1.0, n)
        days_festival = np.random.uniform(0, 1, n)
        credit        = np.random.uniform(0.4, 1.0, n)
        qty           = np.random.uniform(0, 1, n)

        X = np.column_stack([risk, days_stockout, days_festival, credit, qty])

        y = np.where(
            risk > 0.7, 0,
            np.where(days_stockout < 0.17, 0,
                     np.where(risk > 0.4, 1, 2))
        )

        self.model = xgb.XGBClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.1,
            use_label_encoder=False, eval_metric="mlogloss", random_state=42,
        )
        self.model.fit(X, y)

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
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
        features = np.array([[
            spatial_risk_score,
            min(days_until_stockout / 30.0, 1.0),
            min(days_to_festival    / 30.0, 1.0),
            credit_score   / 900.0,
            min(recommended_qty / 100.0, 1.0),
        ]])

        probas = self.model.predict_proba(features)[0]

        return sorted([
            {
                "distributor": d["name"],
                "tier":        d["tier"],
                "confidence":  round(float(probas[d["id"]]), 3),
                "cost":        d["cost"],
                "eta":         f"{d['speed_hrs']} Hours",
                "reliability": d["reliability"],
                "reason":      _reason(d["id"], spatial_risk_score, days_until_stockout),
            }
            for d in DISTRIBUTORS
        ], key=lambda x: x["confidence"], reverse=True)


_instance: Optional[DistributorRecommender] = None

def get_recommender() -> DistributorRecommender:
    global _instance
    if _instance is None:
        _instance = DistributorRecommender()
    return _instance

<div align="center">

# 🚛 GraminRoute

### B2B Rural Last-Mile Logistics Platform for Jangaon District, Telangana

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch_Geometric-GATv2-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch-geometric.readthedocs.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-Risk_Model-FF6600?style=flat-square)](https://xgboost.readthedocs.io)
[![Kafka](https://img.shields.io/badge/Kafka-Event_Streaming-231F20?style=flat-square&logo=apachekafka&logoColor=white)](https://kafka.apache.org)
[![Redis](https://img.shields.io/badge/Redis-Pub%2FSub-DC382D?style=flat-square&logo=redis&logoColor=white)](https://redis.io)
[![Docker](https://img.shields.io/badge/Docker-10_Services-2496ED?style=flat-square&logo=docker&logoColor=white)](https://docker.com)
[![React](https://img.shields.io/badge/React_+_Vite-Dashboard-61DAFB?style=flat-square&logo=react&logoColor=black)](https://vitejs.dev)

**Predicts stockouts before they happen. Pools orders across villages. Routes deliveries optimally.**

[ML Pipeline](#-ml-pipeline) · [Architecture](#-system-architecture) · [Services](#-microservices) · [Setup](#-setup) · [API](#-api-reference)

</div>

---

## 📌 Problem

Rural kirana shops in Jangaon District (500+ villages) face:
- **Stockouts during festivals** — demand spikes 2–3× with no early warning
- **Fragmented deliveries** — each shop orders separately, distributors make 10 trips instead of 1
- **No credit visibility** — distributors can't assess shop risk before extending credit

GraminRoute solves all three with a spatial ML pipeline, cooperative order pooling, and real-time event streaming.

---

## 🧠 ML Pipeline

The core of GraminRoute is a **5-stage inference pipeline** that runs on every stock update.

```
┌─────────────────────────────────────────────────────────────────┐
│                    5-STAGE INFERENCE PIPELINE                   │
│                                                                 │
│  Shop Data ──► [1] Feature Engineering                         │
│                      │                                         │
│                      ▼                                         │
│               [2] XGBoost Risk Scorer                          │
│                    9 features → risk ∈ [0,1]                   │
│                    AUROC: 0.705 | Sensitivity: 94.4%           │
│                      │                                         │
│                      ▼                                         │
│               [3] GATv2 SpatialGNN                             │
│                    500 nodes · 1,600 edges                      │
│                    Propagates risk through road network         │
│                      │                                         │
│                      ▼                                         │
│               [4] Festival Predictor                           │
│                    5 Telangana festivals                        │
│                    Days-to-stockout + reorder qty              │
│                      │                                         │
│                      ▼                                         │
│               [5] Contextual Bandit Recommender                │
│                    XGBoost multi-class                         │
│                    FastTrack · Hub · Budget Movers             │
└─────────────────────────────────────────────────────────────────┘
```

### Stage Details

| Stage | Model | Input | Output |
|-------|-------|-------|--------|
| Feature Engineering | Rule-based normalisation | Stock, Sales, Lead time, Margin, Shelf life, Credit | 9 normalised features |
| XGBoost Risk | XGBClassifier (100 trees, depth=3) | 9 features | Risk score [0–1] |
| SpatialGNN | GATv2Conv (4 heads → 1 head, hidden=64) | 10 features + graph edges | Spatial risk [0–1] |
| Festival Predictor | Calendar + arithmetic | Risk + stock + festival window | Days to stockout, urgency |
| Distributor Recommender | XGBoost multi-class (Contextual Bandit) | Risk + urgency + credit | Ranked distributor list |

### Training Data

- **Dataset:** DataCo Supply Chain Dataset — 180,519 real delivery orders
- **Adapted for:** Jangaon District geography (17.65–18.15°N, 79.05–79.65°E)
- **Graph:** 500 village nodes, 1,600 edges (KNN k=5, Haversine metric)
- **Class balance:** 78% high-risk shops → handled with `scale_pos_weight`
- **Festival calendar:** Sankranti, Ugadi, Eid, Dussehra, Diwali

---

## 🏗 System Architecture

```
                         ┌──────────────────────┐
                         │    React + Vite       │
                         │     Dashboard         │
                         │  Retailer  |  Hub     │
                         └──────────┬───────────┘
                                    │ HTTP / WebSocket
                         ┌──────────▼───────────┐
                         │     API Gateway       │
                         │  JWT · CORS · Proxy   │
                         │       :8080           │
                         └───┬────────┬──────────┘
                             │        │
              ┌──────────────┘        └──────────────┐
              │                                      │
   ┌──────────▼──────────┐              ┌────────────▼────────┐
   │  Inventory Service  │              │    ML Service        │
   │  Stock updates      │              │  5-stage AI pipeline │
   │  XGBoost risk check │              │  Kafka consumer EWA  │
   │       :8001         │              │       :8004          │
   └──────────┬──────────┘              └────────────┬────────┘
              │                                      │
              └──────────────┬───────────────────────┘
                             │
              ┌──────────────▼───────────────────────┐
              │           Apache Kafka                │
              │                                      │
              │  inventory.updated → ML analytics    │
              │  pool.formed       → Route optimizer │
              │  dispatch.sent     → Notify service  │
              └──────┬───────────────────────────────┘
                     │
      ┌──────────────┼──────────────────┐
      │              │                  │
┌─────▼──────┐ ┌─────▼──────┐ ┌────────▼──────────┐
│   Route    │ │   Notify   │ │  WebSocket Server  │
│ Optimizer  │ │  Service   │ │   Redis pub/sub    │
│ TSP solver │ │   Alerts   │ │      :8002         │
└────────────┘ └────────────┘ └────────────────────┘
      │              │                  │
      └──────────────▼──────────────────┘
                     │
      ┌──────────────▼──────────────────┐
      │          Infrastructure         │
      │                                 │
      │  PostgreSQL 15 + PostGIS 3.3    │
      │  Shops · Inventory · Pools      │
      │  Orders · Routes                │
      │                                 │
      │  Redis 7                        │
      │  Demand EWA · Pub/Sub alerts    │
      └─────────────────────────────────┘
```

---

## 📦 Microservices

| Service | Port | Responsibility |
|---------|------|----------------|
| `gateway` | 8080 | JWT auth, request routing, rate limiting |
| `inventory` | 8001 | Stock updates, XGBoost risk check, Kafka publish |
| `ml` | 8004 | Full 5-stage AI pipeline, EWA demand stats via Kafka |
| `pool_formation` | — | PostGIS ST_DWithin order clustering (2km radius) |
| `route_optimizer` | — | TSP route solving for dispatched pools |
| `websocket` | 8002 | Redis pub/sub → real-time browser push |
| `notify` | — | Push restock alerts to shops |
| `postgres` | 5432 | PostgreSQL + PostGIS spatial database |
| `redis` | 6379 | Pub/sub channels, demand EWA cache |
| `kafka` | 9092 | Event streaming backbone (Confluent 7.5) |

---

## 🗂 Repository Structure

```
Gramin-Route/
├── backend/                      # FastAPI ML server (standalone mode)
│   ├── api/
│   │   ├── main.py               # 4 endpoints: risk, pool, simulate, model_info
│   │   └── schemas.py            # Pydantic request/response models
│   ├── ml/
│   │   ├── model_def.py          # GATv2 SpatialGNN architecture
│   │   ├── risk_model.py         # XGBoost risk scorer + SHAP
│   │   ├── recommender.py        # Contextual bandit distributor ranker
│   │   ├── festival_calendar.py  # 5 Telangana festival spike map
│   │   └── festival_predictor.py # Stockout forecast engine
│   └── services/
│       └── pooling.py            # DBSCAN order-pooling engine
│
├── services/                     # Dockerised microservices
│   ├── gateway/                  # API gateway + JWT auth
│   ├── ml/                       # ML service + Kafka consumer
│   ├── inventory/                # Stock management + risk trigger
│   ├── pool_formation/           # PostGIS spatial clustering
│   ├── route_optimizer/          # TSP delivery routing
│   ├── websocket/                # Real-time WebSocket server
│   ├── notify/                   # Push notification service
│   └── db/                       # PostgreSQL migrations + seed
│
├── frontend/                     # React + Vite dashboard
│   └── src/
│       ├── pages/                # RetailerDashboard, DistributorDashboard, Login
│       ├── components/           # LogisticsMap, SimulationChart
│       └── services/api.jsx      # API client
│
├── models/                       # Trained model artifacts
│   ├── xgb_risk_model.pkl        # XGBoost risk classifier
│   ├── xgb_recommender.pkl       # Contextual bandit recommender
│   └── spatial_gnn.pth           # GATv2 GNN weights
│
├── notebook/
│   └── GraminRoute_Training.ipynb  # Full training pipeline with outputs
│
├── data/
│   └── jangaon_shops.csv         # 500 Jangaon shops (from DataCo 180k orders)
│
└── docker-compose.yml            # 10-service orchestration
```

---

## 🚀 Setup

### Option A — Standalone (no Docker)

```bash
# Backend
cd backend
pip install -r requirements.txt
uvicorn api.main:app --reload
# → http://localhost:8000
# → http://localhost:8000/docs

# Frontend (separate terminal)
cd frontend
npm install
npm run dev
# → http://localhost:5173
```

### Option B — Full Microservices Stack

```bash
# Start all 10 services
docker compose up -d

# Seed the database (500 shops + 4,000 inventory rows)
docker compose build seed && docker compose run seed

# Default credentials
# Retailer:     R001 / sharma123
# Distributor:  D001 / dist123
```

| Service | URL |
|---------|-----|
| Gateway API | http://localhost:8080 |
| ML Service | http://localhost:8004 |
| WebSocket | http://localhost:8002 |
| Frontend | http://localhost:5173 |

---

## 📡 API Reference

### `POST /recommend_distributor`
Full 5-stage ML pipeline for a single shop.

```json
// Request
{
  "shop_id": "GHPR-042",
  "lat": 17.73,
  "lon": 79.16,
  "current_stock": 18,
  "daily_sales": 8,
  "lead_time_days": 3,
  "profit_margin": 22.0,
  "shelf_life": 365,
  "credit_score": 720,
  "product_name": "Rice (50kg)"
}

// Response
{
  "xgb_risk_score": 0.29,
  "spatial_risk_score": 0.09,
  "shop_status": "STABLE",
  "days_until_stockout": 2.2,
  "restock_urgency": "IMMEDIATE",
  "festival_alert": {
    "festival_name": "Dussehra",
    "days_away": 89,
    "demand_multiplier": 1.0
  },
  "top_pick": {
    "name": "FastTrack Logistics",
    "confidence": 0.998,
    "cost_rs": 100,
    "eta_hours": 4
  }
}
```

### `POST /pool_orders`
DBSCAN spatial clustering of pending orders into delivery pools.

### `GET /simulate_savings`
60-day Monte Carlo simulation: Traditional vs GraminRoute portfolio.

### `GET /model_info`
Model metadata, feature importance, and architecture details.

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| AUROC | 0.705 |
| F1 Score | 0.776 |
| Sensitivity (Recall) | 94.4% |
| Precision | 66.0% |
| Training samples | 180,519 real orders |
| Graph nodes | 500 villages |
| Graph edges | 1,600 road connections |
| Avg node degree | 6.4 |

> **High sensitivity (94.4%)** is intentional — missing a real stockout costs more than a false alarm in rural logistics.

---

## 🧪 Training Notebook

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Atharva-sp21/Gramin-Route/blob/main/notebook/GraminRoute_Training.ipynb)

The notebook covers:
1. EDA on 180k DataCo supply chain orders
2. Festival feature engineering (5 Telangana festivals)
3. XGBoost risk model training + SHAP explainability
4. Village graph construction (KNN k=5, Haversine metric)
5. GATv2 SpatialGNN training — 200 epochs
6. Contextual bandit distributor recommender
7. End-to-end demo — Ramesh's kirana shop in Ghanpur

---

## 🛠 Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Models | XGBoost, PyTorch Geometric (GATv2), SHAP |
| MLOps | Kafka event streaming, Redis EWA cache, asyncpg |
| API | FastAPI, Pydantic v2, JWT (python-jose) |
| Database | PostgreSQL 15 + PostGIS 3.3 |
| Frontend | React 18 + Vite, Leaflet maps |
| Infrastructure | Docker Compose, Redis 7, Kafka 7.5 (Confluent) |
| Training | Google Colab, scikit-learn, nbformat |

---

## 👥 Contributors

| Role | Contribution |
|------|-------------|
| **Atharva** | ML pipeline (XGBoost + GATv2 GNN), FastAPI backend, training notebook |
| **Partner** | Microservices architecture, Kafka/Redis/PostgreSQL infra, Docker orchestration |

---

<div align="center">
Built for rural India 🇮🇳 · Jangaon District, Telangana
</div>

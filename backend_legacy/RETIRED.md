# ⚠️ RETIRED — do not run this

This standalone backend has been superseded by the microservices stack in
`services/`, orchestrated via the root `docker-compose.yml`.

Its logic was ported to `services/ml/` on 2026-07-07:

| Old file                              | New location                          |
|----------------------------------------|----------------------------------------|
| `backend/api/main.py` (`/recommend_distributor`) | `services/ml/main.py`        |
| `backend/ml/risk_model.py`             | `services/ml/risk.py` (and `services/inventory/risk.py`) |
| `backend/ml/festival_calendar.py`      | `services/ml/festival.py`              |
| `backend/ml/festival_predictor.py`     | `services/ml/festival_predictor.py`    |
| `backend/ml/model_def.py` (SpatialGNN) | `services/ml/model_def.py`             |
| `backend/ml/recommender.py`            | `services/ml/recommender.py`           |
| `backend/models/*.pth, *.pkl`          | repo-root `models/` (shared by `services/ml` and `services/inventory`) |

`backend/services/pooling.py` (DBSCAN order pooling) was **not** ported —
it's superseded by the event-driven, PostGIS-based `services/pool_formation`
service, which works differently (Kafka events + `ST_DWithin`, not a
synchronous HTTP endpoint). The frontend's `getOptimizedPools()` call to
`/pool_orders` on port 8000 is now dead code and still needs a fix — see the
`pool_formation` service and gateway for the equivalent flow.

This folder is kept only for reference. Safe to delete once you've confirmed
`services/ml` behaves as expected.

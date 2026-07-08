FROM python:3.11-slim
WORKDIR /app
COPY services/ml/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY services/ml/ .
# Copy pre-trained models from repo root (shared across services)
COPY models/ ./models/
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8004"]

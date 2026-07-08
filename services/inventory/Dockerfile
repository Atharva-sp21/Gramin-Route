FROM python:3.11-slim
WORKDIR /app
COPY services/inventory/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY services/inventory/ .
# Copy pre-trained models from repo root so risk.py can find them
COPY models/ ./models/
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]

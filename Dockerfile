FROM python:3.13-slim
WORKDIR /app

# Pinned to the versions that trained the model (see requirements-api.txt)
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY src/__init__.py src/api.py src/
COPY models/model_bundle.joblib models/

ENV MODEL_PATH=/app/models/model_bundle.joblib
EXPOSE 8000
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]

FROM python:3.12-slim

WORKDIR /app

# Copy requirements first for better caching
COPY MLProject/conda.yaml requirements.txt* ./

# Install dependencies
RUN pip install --no-cache-dir \
    pandas==2.2.2 \
    numpy==1.26.4 \
    scikit-learn==1.5.0 \
    mlflow==2.19.0 \
    joblib==1.4.2 \
    dagshub==0.3.0

# Copy application code
COPY MLProject/modelling.py .
COPY MLProject/model_artifacts/ ./model_artifacts/

EXPOSE 5000

CMD ["mlflow", "models", "serve", "-m", "model_artifacts/model", "-p", "5000", "--no-conda"]

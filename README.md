# MLflow CI/CD Pipeline

Rudy Peter Agung Chendra (M208D5Y1771)

Automated training pipeline for machine learning models using MLflow and GitHub Actions.

## Overview

This repository contains the CI/CD infrastructure for automated model training and deployment. The pipeline handles model training, artifact management, and optional Docker image publishing.

## Structure

```
MLProject/
├── MLProject              # MLflow project configuration
├── conda.yaml            # Conda environment specification
├── modelling.py          # Training script
└── iris_preprocessing/   # Preprocessed data
```

## Usage

### Local Training

```bash
cd MLProject
python modelling.py --n_estimators 100 --max_depth 5
```

### MLflow Project

```bash
mlflow run . -P n_estimators=100 -P max_depth=5
```

### GitHub Actions

The workflow triggers automatically on push to main branch. Manual triggers are also supported through workflow_dispatch.

Parameters:
- n_estimators: Number of trees in the random forest (default: 100)
- max_depth: Maximum depth of trees (default: 5)

## CI Pipeline

The pipeline consists of three main jobs:

1. Training - Runs the ML model training script
2. Artifact Upload - Saves model artifacts to repository
3. Docker Build - Creates and pushes Docker image (requires secrets)

## Requirements

- Python 3.12
- Dependencies: pandas, numpy, scikit-learn, mlflow, joblib

## Configuration

For Docker Hub publishing, configure repository secrets:
- DOCKER_USERNAME
- DOCKER_PASSWORD

## Related Repository

Main experimentation: https://github.com/PeterChen712/Eksperimen_SML_Rudy-Peter-Agung-Chendra

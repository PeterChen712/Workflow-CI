import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import json
import os
import sys
import argparse
from datetime import datetime


def prepare_data():
    train_path = "iris_preprocessing/iris_train_preprocessed.csv"
    test_path = "iris_preprocessing/iris_test_preprocessed.csv"
    
    if os.path.exists(train_path) and os.path.exists(test_path):
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        X_train = train_df.drop('target', axis=1)
        y_train = train_df['target']
        X_test = test_df.drop('target', axis=1)
        y_test = test_df['target']
    else:
        os.makedirs("iris_preprocessing", exist_ok=True)
        
        iris = load_iris()
        X = pd.DataFrame(iris.data, columns=iris.feature_names)
        y = iris.target
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        scaler = StandardScaler()
        X_train = pd.DataFrame(
            scaler.fit_transform(X_train), columns=iris.feature_names
        )
        X_test = pd.DataFrame(
            scaler.transform(X_test), columns=iris.feature_names
        )
        
        train_df = X_train.copy()
        train_df['target'] = y_train.values
        test_df = X_test.copy()
        test_df['target'] = y_test.values
        
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        joblib.dump(scaler, "iris_preprocessing/scaler.pkl")
    
    return X_train, X_test, y_train, y_test


def train_model(n_estimators=100, max_depth=5, min_samples_split=2):
    X_train, X_test, y_train, y_test = prepare_data()
    
    dagshub_token = os.getenv('DAGSHUB_TOKEN')
    if dagshub_token:
        try:
            import dagshub
            dagshub.init(
                repo_owner='PeterChen712',
                repo_name='Eksperimen_SML_Rudy-Peter-Agung-Chendra',
                mlflow=True
            )
            print("Using DagsHub remote tracking")
        except Exception as e:
            print(f"DagsHub init failed: {e}, using in-memory tracking")
            mlflow.set_tracking_uri("file:///tmp/mlruns")
    else:
        print("No DagsHub token, using in-memory tracking")
        mlflow.set_tracking_uri("file:///tmp/mlruns")
    
    mlflow.set_experiment("iris-classification")
    
    with mlflow.start_run():
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("min_samples_split", min_samples_split)
        mlflow.log_param("model_type", "RandomForestClassifier")
        
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        
        mlflow.sklearn.log_model(model, "model")
        
        os.makedirs("model_artifacts", exist_ok=True)
        model_path = "model_artifacts/model.pkl"
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path)
        
        metrics = {
            "accuracy": accuracy,
            "f1_score": f1,
            "precision": precision,
            "recall": recall,
            "timestamp": datetime.now().isoformat()
        }
        metrics_path = "model_artifacts/metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        mlflow.log_artifact(metrics_path)
        
        mlflow.set_tag("author", "Rudy Peter Agung Chendra")
        mlflow.set_tag("project", "iris-classification")
        
        return model, metrics


def main():
    parser = argparse.ArgumentParser(description='MLflow Project Training')
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--max_depth', type=int, default=5)
    parser.add_argument('--min_samples_split', type=int, default=2)
    
    args = parser.parse_args()
    
    model, metrics = train_model(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split
    )


if __name__ == "__main__":
    main()

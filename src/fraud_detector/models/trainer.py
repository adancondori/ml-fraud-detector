"""
Model training utilities with MLflow tracking.
Supports multiple algorithms and hyperparameter tuning.
"""
from typing import Any, Dict, Optional

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

from config.config import settings
from fraud_detector.utils.logger import log_model_metrics, logger


class ModelTrainer:
    """
    Train and evaluate fraud detection models.

    Features:
    - Multiple algorithm support (XGBoost, LightGBM, Random Forest, Logistic Regression)
    - MLflow experiment tracking
    - Automatic metric computation
    - Model persistence
    """

    MODEL_REGISTRY = {
        "xgboost": XGBClassifier,
        "lightgbm": LGBMClassifier,
        "random_forest": RandomForestClassifier,
        "logistic": LogisticRegression,
    }

    def __init__(
        self,
        model_type: str = "xgboost",
        model_params: Optional[Dict[str, Any]] = None,
        experiment_name: Optional[str] = None,
    ):
        """
        Initialize model trainer.

        Args:
            model_type: Type of model to train
            model_params: Model hyperparameters
            experiment_name: MLflow experiment name
        """
        if model_type not in self.MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model type: {model_type}. "
                f"Available: {list(self.MODEL_REGISTRY.keys())}"
            )

        self.model_type = model_type
        self.model_params = model_params or {}
        self.experiment_name = experiment_name or settings.mlflow_experiment_name
        self.model = None

        # Setup MLflow
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(self.experiment_name)

        logger.info(f"Initialized trainer with {model_type} model")

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        """
        Train model with MLflow tracking.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
        """
        logger.info(f"Training {self.model_type} model...")

        with mlflow.start_run(run_name=f"{self.model_type}_training"):
            # Log parameters
            mlflow.log_params(self.model_params)
            mlflow.log_param("model_type", self.model_type)
            mlflow.log_param("n_samples_train", len(X_train))

            # Initialize model
            model_class = self.MODEL_REGISTRY[self.model_type]
            self.model = model_class(
                random_state=settings.random_seed, **self.model_params
            )

            # Train model
            if X_val is not None and y_val is not None:
                # Train with validation set if provided
                if self.model_type in ["xgboost", "lightgbm"]:
                    eval_set = [(X_val, y_val)]
                    self.model.fit(
                        X_train,
                        y_train,
                        eval_set=eval_set,
                        verbose=False,
                    )
                else:
                    self.model.fit(X_train, y_train)
            else:
                self.model.fit(X_train, y_train)

            # Evaluate on training set
            train_metrics = self.evaluate(X_train, y_train, dataset_name="train")

            # Evaluate on validation set if provided
            if X_val is not None and y_val is not None:
                val_metrics = self.evaluate(X_val, y_val, dataset_name="val")

            # Log model
            mlflow.sklearn.log_model(self.model, "model")

            logger.info("Model training completed")

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        dataset_name: str = "test",
    ) -> Dict[str, float]:
        """
        Evaluate model and compute metrics.

        Args:
            X: Features
            y: True labels
            dataset_name: Name of dataset (for logging)

        Returns:
            Dictionary of metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        logger.info(f"Evaluating model on {dataset_name} set...")

        # Predictions
        y_pred = self.model.predict(X)
        y_pred_proba = self.model.predict_proba(X)[:, 1]

        # Compute metrics
        metrics = {
            f"{dataset_name}_accuracy": accuracy_score(y, y_pred),
            f"{dataset_name}_precision": precision_score(y, y_pred, zero_division=0),
            f"{dataset_name}_recall": recall_score(y, y_pred, zero_division=0),
            f"{dataset_name}_f1": f1_score(y, y_pred, zero_division=0),
            f"{dataset_name}_roc_auc": roc_auc_score(y, y_pred_proba),
        }

        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        tn, fp, fn, tp = cm.ravel()

        metrics.update(
            {
                f"{dataset_name}_true_negatives": int(tn),
                f"{dataset_name}_false_positives": int(fp),
                f"{dataset_name}_false_negatives": int(fn),
                f"{dataset_name}_true_positives": int(tp),
            }
        )

        # Log to MLflow
        mlflow.log_metrics(metrics)

        # Log metrics
        log_model_metrics(metrics)

        return metrics

    def save_model(self, output_path: str) -> None:
        """
        Save trained model to disk.

        Args:
            output_path: Path to save model
        """
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")

        logger.info(f"Saving model to: {output_path}")
        joblib.dump(self.model, output_path)
        logger.info("Model saved successfully")

    def load_model(self, model_path: str) -> None:
        """
        Load trained model from disk.

        Args:
            model_path: Path to model file
        """
        logger.info(f"Loading model from: {model_path}")
        self.model = joblib.load(model_path)
        logger.info("Model loaded successfully")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with trained model.

        Args:
            X: Features

        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() or load_model() first.")

        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Features

        Returns:
            Probability predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() or load_model() first.")

        return self.model.predict_proba(X)

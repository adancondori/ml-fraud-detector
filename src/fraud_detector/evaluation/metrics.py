"""
Métricas específicas para evaluación de modelos de detección de fraude.

En fraude, Accuracy NO es útil. Necesitamos métricas que consideren:
1. Costo de falsos positivos vs costo de fraude
2. Capacidad de revisión manual
3. Precision y Recall en top K%
4. PR-AUC (más importante que ROC-AUC en datos desbalanceados)
"""
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from config.config import settings
from fraud_detector.utils.logger import logger


class FraudMetrics:
    """
    Métricas específicas para detección de fraude.

    Métricas implementadas:
    1. Precision/Recall @ K (top K% predictions)
    2. PR-AUC (Precision-Recall AUC)
    3. Cost-based metrics
    4. Expected value
    5. Confusion matrix con interpretación de negocio
    """

    def __init__(
        self,
        fraud_cost: Optional[float] = None,
        false_positive_cost: Optional[float] = None,
        review_capacity: Optional[int] = None,
    ):
        """
        Inicializa calculador de métricas.

        Args:
            fraud_cost: Costo promedio de una transacción fraudulenta
            false_positive_cost: Costo de revisar un falso positivo
            review_capacity: Número máximo de transacciones que se pueden revisar
        """
        self.fraud_cost = fraud_cost or settings.fraud_cost_per_transaction
        self.false_positive_cost = (
            false_positive_cost or settings.false_positive_cost
        )
        self.review_capacity = review_capacity or settings.review_capacity_per_day

        logger.info("FraudMetrics inicializado")
        logger.info(f"  Costo por fraude: ${self.fraud_cost}")
        logger.info(f"  Costo por falso positivo: ${self.false_positive_cost}")
        logger.info(f"  Capacidad de revisión: {self.review_capacity} casos/día")

    def compute_all_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Calcula todas las métricas relevantes.

        Args:
            y_true: Labels verdaderos (0/1)
            y_pred: Predicciones binarias (0/1)
            y_pred_proba: Probabilidades predichas (opcional)

        Returns:
            Dict con todas las métricas
        """
        logger.info("Calculando métricas de fraude...")

        metrics = {}

        # Métricas básicas
        metrics.update(self._compute_basic_metrics(y_true, y_pred))

        # Confusion matrix
        metrics.update(self._compute_confusion_matrix_metrics(y_true, y_pred))

        # Métricas basadas en probabilidades
        if y_pred_proba is not None:
            metrics.update(self._compute_probability_metrics(y_true, y_pred_proba))

            # Precision/Recall @ K
            metrics.update(self._compute_precision_recall_at_k(y_true, y_pred_proba))

        # Métricas de costo
        metrics.update(self._compute_cost_metrics(y_true, y_pred))

        logger.info(f"✅ Calculadas {len(metrics)} métricas")

        return metrics

    def _compute_basic_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calcula métricas básicas."""
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0),
        }

    def _compute_confusion_matrix_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calcula métricas de confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)

        # Puede ser 2x2 o menos si falta alguna clase
        tn = cm[0, 0] if cm.shape[0] > 0 and cm.shape[1] > 0 else 0
        fp = cm[0, 1] if cm.shape[0] > 0 and cm.shape[1] > 1 else 0
        fn = cm[1, 0] if cm.shape[0] > 1 and cm.shape[1] > 0 else 0
        tp = cm[1, 1] if cm.shape[0] > 1 and cm.shape[1] > 1 else 0

        total = tn + fp + fn + tp

        return {
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn),
            "false_positive_rate": fp / (fp + tn) if (fp + tn) > 0 else 0,
            "false_negative_rate": fn / (fn + tp) if (fn + tp) > 0 else 0,
            "true_positive_rate": tp / (tp + fn) if (tp + fn) > 0 else 0,  # = recall
            "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
        }

    def _compute_probability_metrics(
        self, y_true: np.ndarray, y_pred_proba: np.ndarray
    ) -> Dict[str, float]:
        """Calcula métricas basadas en probabilidades."""
        try:
            roc_auc = roc_auc_score(y_true, y_pred_proba)
        except Exception:
            roc_auc = 0.0
            logger.warning("No se pudo calcular ROC-AUC")

        try:
            pr_auc = average_precision_score(y_true, y_pred_proba)
        except Exception:
            pr_auc = 0.0
            logger.warning("No se pudo calcular PR-AUC")

        return {
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,  # MÁS IMPORTANTE que ROC-AUC en datos desbalanceados
        }

    def _compute_precision_recall_at_k(
        self, y_true: np.ndarray, y_pred_proba: np.ndarray
    ) -> Dict[str, float]:
        """
        Calcula Precision y Recall en top K%.

        Esencial para fraude: ¿qué tan bien detectamos fraude en los top K
        casos más sospechosos?
        """
        metrics = {}

        # Calcular para diferentes K
        k_values = [1, 5, 10, 20]  # Top 1%, 5%, 10%, 20%

        for k_pct in k_values:
            k = max(1, int(len(y_true) * k_pct / 100))

            # Índices de top K predicciones
            top_k_indices = np.argsort(y_pred_proba)[-k:]

            # Predicciones y verdaderos en top K
            y_true_top_k = y_true[top_k_indices]
            y_pred_top_k = np.ones(k)  # Asumimos que predecimos fraude para top K

            # Precision @ K
            precision_at_k = precision_score(
                y_true_top_k, y_pred_top_k, zero_division=0
            )

            # Recall @ K
            total_fraud = np.sum(y_true)
            fraud_in_top_k = np.sum(y_true_top_k)
            recall_at_k = fraud_in_top_k / total_fraud if total_fraud > 0 else 0

            metrics[f"precision_at_{k_pct}"] = precision_at_k
            metrics[f"recall_at_{k_pct}"] = recall_at_k
            metrics[f"fraud_detected_at_{k_pct}"] = int(fraud_in_top_k)

        return metrics

    def _compute_cost_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Calcula métricas basadas en costos de negocio.

        Costo total = (Fraude no detectado * costo_fraude) +
                      (Falsos positivos * costo_falso_positivo)
        """
        cm = confusion_matrix(y_true, y_pred)

        tn = cm[0, 0] if cm.shape[0] > 0 and cm.shape[1] > 0 else 0
        fp = cm[0, 1] if cm.shape[0] > 0 and cm.shape[1] > 1 else 0
        fn = cm[1, 0] if cm.shape[0] > 1 and cm.shape[1] > 0 else 0
        tp = cm[1, 1] if cm.shape[0] > 1 and cm.shape[1] > 1 else 0

        # Costos
        cost_missed_fraud = fn * self.fraud_cost
        cost_false_positives = fp * self.false_positive_cost
        total_cost = cost_missed_fraud + cost_false_positives

        # Savings (fraude detectado correctamente)
        savings = tp * self.fraud_cost

        # Net value
        net_value = savings - total_cost

        return {
            "cost_missed_fraud": cost_missed_fraud,
            "cost_false_positives": cost_false_positives,
            "total_cost": total_cost,
            "savings_from_detected_fraud": savings,
            "net_value": net_value,
        }

    def print_metrics_report(self, metrics: Dict[str, float]) -> None:
        """
        Imprime reporte de métricas formateado.

        Args:
            metrics: Dict con métricas calculadas
        """
        logger.info("\n" + "=" * 70)
        logger.info("REPORTE DE MÉTRICAS - DETECCIÓN DE FRAUDE")
        logger.info("=" * 70)

        # Métricas básicas
        logger.info("\n📊 MÉTRICAS BÁSICAS:")
        logger.info(f"  Accuracy:  {metrics.get('accuracy', 0):.4f}")
        logger.info(f"  Precision: {metrics.get('precision', 0):.4f}")
        logger.info(f"  Recall:    {metrics.get('recall', 0):.4f}")
        logger.info(f"  F1-Score:  {metrics.get('f1_score', 0):.4f}")

        # Confusion Matrix
        logger.info("\n🔍 CONFUSION MATRIX:")
        logger.info(f"  True Positives (TP):  {metrics.get('true_positives', 0):,}")
        logger.info(f"  False Positives (FP): {metrics.get('false_positives', 0):,}")
        logger.info(f"  True Negatives (TN):  {metrics.get('true_negatives', 0):,}")
        logger.info(f"  False Negatives (FN): {metrics.get('false_negatives', 0):,}")

        # Métricas avanzadas
        if "roc_auc" in metrics:
            logger.info("\n📈 MÉTRICAS AVANZADAS:")
            logger.info(f"  ROC-AUC: {metrics.get('roc_auc', 0):.4f}")
            logger.info(f"  PR-AUC:  {metrics.get('pr_auc', 0):.4f} ⭐ (MÁS IMPORTANTE)")

        # Precision/Recall @ K
        if "precision_at_1" in metrics:
            logger.info("\n🎯 PRECISION & RECALL @ K (Top K% más sospechosos):")
            for k in [1, 5, 10, 20]:
                if f"precision_at_{k}" in metrics:
                    p = metrics.get(f"precision_at_{k}", 0)
                    r = metrics.get(f"recall_at_{k}", 0)
                    detected = metrics.get(f"fraud_detected_at_{k}", 0)
                    logger.info(
                        f"  Top {k:2}%: Precision={p:.4f} | Recall={r:.4f} | "
                        f"Fraudes detectados={detected}"
                    )

        # Costos
        if "total_cost" in metrics:
            logger.info("\n💰 ANÁLISIS DE COSTOS:")
            logger.info(
                f"  Costo de fraude no detectado:  ${metrics.get('cost_missed_fraud', 0):,.2f}"
            )
            logger.info(
                f"  Costo de falsos positivos:     ${metrics.get('cost_false_positives', 0):,.2f}"
            )
            logger.info(f"  Costo total:                    ${metrics.get('total_cost', 0):,.2f}")
            logger.info(
                f"  Ahorros (fraude detectado):     ${metrics.get('savings_from_detected_fraud', 0):,.2f}"
            )

            net_value = metrics.get("net_value", 0)
            logger.info(f"  VALOR NETO:                     ${net_value:,.2f}")

            if net_value > 0:
                logger.info("  ✅ El modelo genera VALOR positivo")
            else:
                logger.warning("  ⚠️  El modelo genera PÉRDIDAS")

        logger.info("\n" + "=" * 70)


def evaluate_fraud_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: Optional[np.ndarray] = None,
    print_report: bool = True,
) -> Dict[str, float]:
    """
    Función helper para evaluar un modelo de fraude.

    Args:
        y_true: Labels verdaderos
        y_pred: Predicciones binarias
        y_pred_proba: Probabilidades (opcional)
        print_report: Si imprimir reporte

    Returns:
        Dict con métricas
    """
    evaluator = FraudMetrics()
    metrics = evaluator.compute_all_metrics(y_true, y_pred, y_pred_proba)

    if print_report:
        evaluator.print_metrics_report(metrics)

    return metrics


def find_optimal_threshold(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    metric: str = "f1",
) -> Tuple[float, float]:
    """
    Encuentra el threshold óptimo basado en una métrica.

    Args:
        y_true: Labels verdaderos
        y_pred_proba: Probabilidades predichas
        metric: Métrica a optimizar ('f1', 'precision', 'recall')

    Returns:
        (threshold_optimo, valor_metrica)
    """
    logger.info(f"Buscando threshold óptimo para maximizar {metric}...")

    # Probar diferentes thresholds
    thresholds = np.arange(0.1, 1.0, 0.01)

    best_threshold = 0.5
    best_score = 0.0

    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)

        if metric == "f1":
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == "precision":
            score = precision_score(y_true, y_pred, zero_division=0)
        elif metric == "recall":
            score = recall_score(y_true, y_pred, zero_division=0)
        else:
            raise ValueError(f"Métrica '{metric}' no soportada")

        if score > best_score:
            best_score = score
            best_threshold = threshold

    logger.info(f"✅ Threshold óptimo: {best_threshold:.3f}")
    logger.info(f"   {metric.capitalize()}: {best_score:.4f}")

    return best_threshold, best_score

"""
Módulo para manejo de datos desbalanceados en detección de fraude.

En fraude, típicamente menos del 1% de transacciones son fraudulentas.
Este módulo implementa técnicas para balancear las clases.
"""
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN, SMOTETomek
from sklearn.utils.class_weight import compute_class_weight

from config.config import settings
from fraud_detector.utils.logger import logger


class DataBalancer:
    """
    Maneja el desbalanceo de clases en datos de fraude.

    Técnicas implementadas:
    1. SMOTE: Synthetic Minority Over-sampling Technique
    2. ADASYN: Adaptive Synthetic Sampling
    3. Random Over-sampling
    4. Random Under-sampling
    5. SMOTEENN: SMOTE + Edited Nearest Neighbors
    6. SMOTETomek: SMOTE + Tomek links
    7. Class weights: Para usar con modelos
    """

    AVAILABLE_STRATEGIES = {
        "smote": SMOTE,
        "adasyn": ADASYN,
        "random_over": RandomOverSampler,
        "random_under": RandomUnderSampler,
        "smoteenn": SMOTEENN,
        "smotetomek": SMOTETomek,
    }

    def __init__(
        self,
        strategy: str = "smote",
        sampling_strategy: Optional[float] = None,
        random_state: Optional[int] = None,
    ):
        """
        Inicializa el balanceador.

        Args:
            strategy: Estrategia de balanceo ('smote', 'adasyn', etc.)
            sampling_strategy: Ratio deseado minority/majority (default: 0.5)
            random_state: Semilla para reproducibilidad
        """
        if strategy not in self.AVAILABLE_STRATEGIES:
            raise ValueError(
                f"Estrategia '{strategy}' no válida. "
                f"Disponibles: {list(self.AVAILABLE_STRATEGIES.keys())}"
            )

        self.strategy = strategy
        self.sampling_strategy = sampling_strategy or settings.smote_sampling_strategy
        self.random_state = random_state or settings.random_seed

        self.sampler = None
        self.class_weights = None

        logger.info(f"DataBalancer inicializado con estrategia: {strategy}")
        logger.info(f"  Sampling strategy: {self.sampling_strategy}")

    def fit_resample(
        self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Aplica balanceo a los datos.

        Args:
            X: Features
            y: Target (0/1)

        Returns:
            X_resampled, y_resampled
        """
        logger.info(f"Aplicando balanceo con {self.strategy}...")

        # Verificar distribución original
        unique, counts = np.unique(y, return_counts=True)
        original_dist = dict(zip(unique, counts))

        logger.info(f"Distribución original: {original_dist}")
        logger.info(
            f"  Ratio fraude: {original_dist.get(1, 0) / sum(counts):.2%}"
        )

        # Convertir a numpy si es necesario
        if isinstance(X, pd.DataFrame):
            X_array = X.values
            feature_names = X.columns
        else:
            X_array = X
            feature_names = None

        if isinstance(y, pd.Series):
            y_array = y.values
        else:
            y_array = y

        # Crear sampler
        sampler_class = self.AVAILABLE_STRATEGIES[self.strategy]

        try:
            if self.strategy in ["smote", "adasyn"]:
                self.sampler = sampler_class(
                    sampling_strategy=self.sampling_strategy,
                    random_state=self.random_state,
                )
            elif self.strategy in ["smoteenn", "smotetomek"]:
                self.sampler = sampler_class(
                    sampling_strategy=self.sampling_strategy,
                    random_state=self.random_state,
                )
            else:
                self.sampler = sampler_class(
                    sampling_strategy=self.sampling_strategy,
                    random_state=self.random_state,
                )

            # Aplicar resampling
            X_resampled, y_resampled = self.sampler.fit_resample(X_array, y_array)

        except Exception as e:
            logger.error(f"Error durante resampling: {e}")
            logger.warning("Retornando datos originales sin balanceo")
            return X_array, y_array

        # Verificar distribución resultante
        unique_new, counts_new = np.unique(y_resampled, return_counts=True)
        new_dist = dict(zip(unique_new, counts_new))

        logger.info(f"Distribución después de balanceo: {new_dist}")
        logger.info(
            f"  Ratio fraude: {new_dist.get(1, 0) / sum(counts_new):.2%}"
        )
        logger.info(
            f"  Nuevas muestras sintéticas: {len(y_resampled) - len(y_array)}"
        )

        logger.info("✅ Balanceo completado exitosamente")

        return X_resampled, y_resampled

    def compute_class_weights(
        self, y: np.ndarray | pd.Series
    ) -> dict:
        """
        Calcula pesos de clases para usar con modelos.

        Útil cuando no quieres modificar los datos pero sí compensar el desbalanceo.

        Args:
            y: Target (0/1)

        Returns:
            Dict con pesos por clase {0: weight0, 1: weight1}
        """
        logger.info("Calculando class weights...")

        if isinstance(y, pd.Series):
            y_array = y.values
        else:
            y_array = y

        # Calcular pesos
        classes = np.unique(y_array)
        weights = compute_class_weight(
            class_weight="balanced", classes=classes, y=y_array
        )

        self.class_weights = dict(zip(classes, weights))

        logger.info(f"Class weights calculados: {self.class_weights}")

        # Interpretación
        fraud_weight = self.class_weights.get(1, 1.0)
        normal_weight = self.class_weights.get(0, 1.0)

        logger.info(
            f"  Transacciones fraudulentas tienen peso {fraud_weight:.2f}x"
        )
        logger.info(
            f"  Transacciones normales tienen peso {normal_weight:.2f}x"
        )

        return self.class_weights

    def get_sample_weight(
        self, y: np.ndarray | pd.Series
    ) -> np.ndarray:
        """
        Obtiene array de pesos individuales para cada muestra.

        Útil para pasar a fit(sample_weight=...) de scikit-learn.

        Args:
            y: Target (0/1)

        Returns:
            Array de pesos del mismo tamaño que y
        """
        if self.class_weights is None:
            self.compute_class_weights(y)

        if isinstance(y, pd.Series):
            y_array = y.values
        else:
            y_array = y

        # Mapear pesos a cada muestra
        sample_weights = np.array([self.class_weights[label] for label in y_array])

        logger.info(f"Sample weights generados para {len(sample_weights)} muestras")

        return sample_weights

    @staticmethod
    def analyze_imbalance(y: np.ndarray | pd.Series) -> dict:
        """
        Analiza el grado de desbalanceo en los datos.

        Args:
            y: Target (0/1)

        Returns:
            Dict con estadísticas de desbalanceo
        """
        if isinstance(y, pd.Series):
            y_array = y.values
        else:
            y_array = y

        unique, counts = np.unique(y_array, return_counts=True)
        dist = dict(zip(unique, counts))

        total = len(y_array)
        fraud_count = dist.get(1, 0)
        normal_count = dist.get(0, 0)

        fraud_ratio = fraud_count / total if total > 0 else 0
        imbalance_ratio = normal_count / fraud_count if fraud_count > 0 else float('inf')

        analysis = {
            "total_samples": total,
            "fraud_samples": fraud_count,
            "normal_samples": normal_count,
            "fraud_ratio": fraud_ratio,
            "imbalance_ratio": imbalance_ratio,
            "distribution": dist,
        }

        # Log del análisis
        logger.info("=" * 60)
        logger.info("ANÁLISIS DE DESBALANCEO")
        logger.info("=" * 60)
        logger.info(f"Total muestras: {total:,}")
        logger.info(f"Fraude: {fraud_count:,} ({fraud_ratio:.2%})")
        logger.info(f"Normal: {normal_count:,} ({1-fraud_ratio:.2%})")
        logger.info(f"Ratio de desbalanceo: 1:{imbalance_ratio:.1f}")

        if imbalance_ratio > 100:
            logger.warning("⚠️  Desbalanceo EXTREMO (>100:1)")
            logger.warning("   Recomendación: Usar SMOTE + class weights")
        elif imbalance_ratio > 10:
            logger.warning("⚠️  Desbalanceo ALTO (>10:1)")
            logger.warning("   Recomendación: Usar SMOTE o class weights")
        else:
            logger.info("✅ Desbalanceo moderado (<10:1)")

        logger.info("=" * 60)

        return analysis


def apply_balancing(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    strategy: Optional[str] = None,
    use_smote: Optional[bool] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[dict]]:
    """
    Función helper para aplicar balanceo según configuración.

    Args:
        X: Features
        y: Target
        strategy: Estrategia de balanceo (si None, usa config)
        use_smote: Si usar SMOTE (si None, usa config)

    Returns:
        X_balanced, y_balanced, class_weights
    """
    use_smote = use_smote if use_smote is not None else settings.use_smote

    if not use_smote:
        logger.info("SMOTE deshabilitado en configuración")

        if settings.use_class_weights:
            balancer = DataBalancer()
            class_weights = balancer.compute_class_weights(y)
            return X, y, class_weights
        else:
            logger.warning("⚠️  Datos no balanceados y class weights deshabilitados")
            return X, y, None

    # Aplicar SMOTE
    strategy = strategy or "smote"
    balancer = DataBalancer(strategy=strategy)

    # Analizar desbalanceo
    balancer.analyze_imbalance(y)

    # Aplicar resampling
    X_balanced, y_balanced = balancer.fit_resample(X, y)

    # Calcular class weights también (por si acaso)
    class_weights = None
    if settings.use_class_weights:
        class_weights = balancer.compute_class_weights(y_balanced)

    return X_balanced, y_balanced, class_weights

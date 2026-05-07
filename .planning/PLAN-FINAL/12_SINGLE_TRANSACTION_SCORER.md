# Fase 12: Single Transaction Scorer

> **Prerequisitos:** Fases 5-7 completadas (modelo entrenado, scaler ajustado, threshold calibrado).
> **Alcance:** Transformar los artefactos del pipeline batch (`.joblib`) en un scorer capaz de evaluar UNA transaccion individual y devolver: score continuo, clasificacion binaria (anomalo si/no), nivel de riesgo y factores explicativos.
> **Naturaleza:** Post-tesis. No afecta las hipotesis HE1-HE4 ni los artefactos academicos. Es el puente entre la investigacion y la operacion.

---

## Problema

El pipeline de Fases 0-11 produce artefactos batch:

```
FeatureEngineer.transform(df_millones_de_filas) → DataFrame con 31 features
scaler.transform(X_array) → X_scaled
model.score_samples(X_scaled) → scores[]
```

Para uso operativo se necesita:

```
score_transaction(payment_dict) → { score, is_anomaly, risk_level, factors }
```

La brecha esta en el **calculo de features para una sola transaccion**, porque 21 de 31 features requieren contexto historico del usuario (ventanas rolling de 1h, 24h, 30d).

---

## Arquitectura

```
                    Transaccion nueva (dict)
                            │
                            ▼
              ┌──────────────────────────────┐
              │   SingleTransactionScorer    │
              │                              │
              │  1. UserContextProvider       │──→ ClickHouse (historial usuario)
              │     └─ cache opcional Redis   │
              │                              │
              │  2. SingleFeatureCalculator   │──→ feature_engineer.joblib (parametros)
              │     └─ 31 features            │
              │                              │
              │  3. scaler.transform([X])     │──→ scaler.joblib
              │                              │
              │  4. model.score_samples([X])  │──→ isolation_forest.joblib
              │                              │
              │  5. ThresholdClassifier       │──→ thresholds.json (de Fase 7)
              │     └─ score → nivel + si/no  │
              │                              │
              │  6. FactorExplainer           │──→ SHAP o feature contribution
              │     └─ top features            │
              └──────────┬───────────────────┘
                         │
                         ▼
              { score: 0.78, is_anomaly: true,
                risk_level: "high",
                factors: ["velocity_24h: 12 txns",
                          "reversal_ratio: 0.45"] }
```

---

## Componentes

### 1. `UserContextProvider` — historial del usuario desde ClickHouse

Calcula los agregados rolling que las features de grupos C, D, F, H necesitan para una transaccion individual.

```python
# src/fraud_detector/scoring/context.py

class UserContextProvider:
    """Obtiene contexto historico de un usuario desde ClickHouse
    para calcular features de una sola transaccion."""

    def __init__(self, ch_connector: ClickHouseConnector):
        self._ch = ch_connector

    def get_context(self, user_id: int, facility_id: int,
                    timestamp: datetime) -> UserContext:
        """Ejecuta queries ligeros contra ClickHouse para obtener
        el contexto rolling del usuario.

        Returns:
            UserContext con todos los agregados necesarios.
        """
        results = {}

        # --- Grupo C: Velocidad (1h, 24h) ---
        velocity = self._ch.query_df("""
            SELECT
                countIf(created_at >= %(ts)s - INTERVAL 1 HOUR
                        AND created_at < %(ts)s) AS txn_count_1h,
                countIf(created_at >= %(ts)s - INTERVAL 24 HOUR
                        AND created_at < %(ts)s) AS txn_count_24h,
                sumIf(reservation_paid_out,
                      created_at >= %(ts)s - INTERVAL 24 HOUR
                      AND created_at < %(ts)s) AS amount_24h,
                max(CASE WHEN created_at < %(ts)s
                         THEN created_at ELSE NULL END) AS last_txn_at
            FROM pbp_productionDB_optimized.payments FINAL
            WHERE user_id = %(uid)s
              AND _peerdb_is_deleted = 0
              AND payment_method NOT IN ('reversal', 'free')
        """, {"uid": user_id, "ts": timestamp})
        results.update(velocity.iloc[0].to_dict())

        # --- Grupo D: Comportamiento 30d ---
        behavior = self._ch.query_df("""
            SELECT
                count(DISTINCT facility_id) AS distinct_facilities_30d,
                count(DISTINCT payment_method) AS distinct_methods,
                countIf(status IN ('totally_refunded', 'refunded_to_credit'))
                    * 1.0 / greatest(count(), 1) AS reversal_ratio_30d,
                sumIf(discount, 1=1) / greatest(sumIf(reservation_paid_out, 1=1), 0.01)
                    AS discount_ratio_30d,
                count() AS txn_count_30d
            FROM pbp_productionDB_optimized.payments FINAL
            WHERE user_id = %(uid)s
              AND created_at >= %(ts)s - INTERVAL 30 DAY
              AND created_at < %(ts)s
              AND _peerdb_is_deleted = 0
              AND payment_method NOT IN ('reversal', 'free')
        """, {"uid": user_id, "ts": timestamp})
        results.update(behavior.iloc[0].to_dict())

        # --- Grupo F: Credito/Flujo 30d ---
        credit = self._ch.query_df("""
            SELECT
                countIf(category = 'debit') AS debit_count_30d,
                sumIf(reservation_paid_out, category = 'debit') AS debit_amount_30d,
                sumIf(reservation_paid_out, payment_method = 'prepaid') AS prepaid_spend_30d
            FROM pbp_productionDB_optimized.payments FINAL
            WHERE user_id = %(uid)s
              AND created_at >= %(ts)s - INTERVAL 30 DAY
              AND created_at < %(ts)s
              AND _peerdb_is_deleted = 0
        """, {"uid": user_id, "ts": timestamp})
        results.update(credit.iloc[0].to_dict())

        # --- Grupo H: Diversidad 30d ---
        diversity = self._ch.query_df("""
            SELECT
                groupArray(category) AS categories_30d,
                countIf(status IN ('totally_refunded', 'refunded_to_credit'))
                    AS reversal_count_30d,
                countIf(category = 'merchandise') * 1.0
                    / greatest(count(), 1) AS merchandise_ratio_30d
            FROM pbp_productionDB_optimized.payments FINAL
            WHERE user_id = %(uid)s
              AND created_at >= %(ts)s - INTERVAL 30 DAY
              AND created_at < %(ts)s
              AND _peerdb_is_deleted = 0
              AND payment_method NOT IN ('reversal', 'free')
        """, {"uid": user_id, "ts": timestamp})
        results.update(diversity.iloc[0].to_dict())

        # --- user_account_age_days ---
        user_info = self._ch.query_df("""
            SELECT created_at AS user_created_at
            FROM pbp_productionDB_optimized.users FINAL
            WHERE id = %(uid)s AND _peerdb_is_deleted = 0
            LIMIT 1
        """, {"uid": user_id})
        if not user_info.empty:
            results["user_created_at"] = user_info.iloc[0]["user_created_at"]

        # --- is_staff, user_role ---
        role_info = self._ch.query_df("""
            SELECT role
            FROM pbp_productionDB_optimized.facilities_users FINAL
            WHERE user_id = %(uid)s AND facility_id = %(fid)s
              AND _peerdb_is_deleted = 0
            LIMIT 1
        """, {"uid": user_id, "fid": facility_id})
        if not role_info.empty:
            results["user_role"] = role_info.iloc[0]["role"]
        else:
            results["user_role"] = "player"

        return UserContext(**results)
```

**Nota sobre performance:** Los 5 queries contra ClickHouse son ligeros (filtran por `user_id` con indice). Latencia esperada: 50-200ms total. Si se necesita < 10ms, implementar un cache con `UserContextCache` (ver seccion de optimizacion).

### 2. `SingleFeatureCalculator` — calcula 31 features para 1 transaccion

```python
# src/fraud_detector/scoring/features.py

class SingleFeatureCalculator:
    """Calcula las 31 features para una transaccion individual,
    usando parametros aprendidos del FeatureEngineer (fit en train)
    y contexto del usuario desde UserContextProvider.

    Reutiliza los mismos parametros (global_avg_amount, facility_avg_amount,
    staff_stats) que el FeatureEngineer batch aprendio en fit().
    """

    def __init__(self, feature_engineer_path: str = "output/models/feature_engineer.joblib"):
        self._fe = joblib.load(feature_engineer_path)
        # Extraer parametros aprendidos de los FeatureGroups
        self._global_avg_amount = self._fe._groups[0]._global_avg_amount  # TransactionalFeatures
        self._facility_avgs = self._fe._groups[4]._facility_avg  # ContextualFeatures
        self._staff_stats = self._fe._groups[6]._staff_stats  # StaffFeatures

    def calculate(self, payment: dict, context: UserContext) -> np.ndarray:
        """Calcula las 31 features en el orden canonico de FEATURE_NAMES.

        Args:
            payment: Dict con campos de la transaccion
                     (reservation_paid_out, discount, tip, created_at,
                      payment_method, category, club_credit_flag,
                      paid_by_manager, facility_id, currency)
            context: UserContext con agregados historicos del usuario

        Returns:
            np.ndarray de shape (31,) con las features en orden FEATURE_NAMES.
        """
        amount = float(payment["reservation_paid_out"])
        discount = float(payment.get("discount", 0))
        tip = float(payment.get("tip", 0))
        ts = pd.Timestamp(payment["created_at"])
        hour = ts.hour
        dow = ts.dayofweek + 1  # 1=Lun, 7=Dom
        fid = payment["facility_id"]

        facility_avg = self._facility_avgs.get(fid, self._global_avg_amount)
        is_staff = context.user_role in ("court_manager", "court_operator", "teacher")
        role_key = context.user_role if is_staff else "player"
        staff_mean = self._staff_stats.get(role_key, {}).get("mean", self._global_avg_amount)
        staff_std = self._staff_stats.get(role_key, {}).get("std", 1.0)

        # Calcular time_since_last_txn
        if context.last_txn_at is not None:
            time_since = (ts - pd.Timestamp(context.last_txn_at)).total_seconds()
        else:
            time_since = 0.0

        # Calcular category_entropy_30d
        cat_entropy = self._shannon_entropy(context.categories_30d)

        # Calcular credit_flow_ratio
        prepaid_spend = max(float(context.prepaid_spend_30d or 0), 0.01)
        credit_flow = float(context.debit_amount_30d or 0) / prepaid_spend

        # user_account_age_days
        if context.user_created_at is not None:
            account_age = (ts - pd.Timestamp(context.user_created_at)).days
        else:
            account_age = 0

        features = np.array([
            # Grupo A: Transaccionales
            amount,                                              # F01 amount
            np.log1p(amount),                                    # F02 log_amount
            amount / max(self._global_avg_amount, 1e-8),         # F03 amount_usd_ratio
            discount / max(amount, 0.01),                        # F04 discount_ratio
            1.0 if tip > 0 else 0.0,                             # F05 has_tip

            # Grupo B: Temporales
            np.sin(2 * np.pi * hour / 24),                      # F07 hour_sin
            np.cos(2 * np.pi * hour / 24),                      # F08 hour_cos
            float(dow),                                          # F09 day_of_week
            1.0 if dow >= 6 else 0.0,                            # F10 is_weekend
            1.0 if hour >= 23 or hour <= 6 else 0.0,            # F11 is_off_hours

            # Grupo C: Velocidad
            float(context.txn_count_1h or 0),                    # F12 user_txn_count_1h
            float(context.txn_count_24h or 0),                   # F13 user_txn_count_24h
            time_since,                                          # F14 time_since_last_txn
            float(context.amount_24h or 0),                      # F15 user_amount_24h

            # Grupo D: Comportamiento
            float(context.distinct_facilities_30d or 0),         # F16 user_distinct_facilities_30d
            float(context.distinct_methods or 0),                # F17 user_distinct_methods
            float(context.reversal_ratio_30d or 0),              # F18 user_reversal_ratio_30d
            float(account_age),                                  # F19 user_account_age_days
            float(context.discount_ratio_30d or 0),              # F20 user_discount_ratio_30d

            # Grupo E: Contextuales
            facility_avg,                                        # F22 facility_avg_amount
            amount / max(facility_avg, 1e-8),                    # F23 amount_facility_ratio

            # Grupo F: Credito/Flujo
            1.0 if payment.get("club_credit_flag") else 0.0,     # F24 is_club_credit
            float(context.debit_count_30d or 0),                 # F25 user_debit_count_30d
            float(context.debit_amount_30d or 0),                # F26 user_debit_amount_30d
            credit_flow,                                         # F27 credit_flow_ratio

            # Grupo G: Rol/Staff
            1.0 if is_staff else 0.0,                            # F28 is_staff
            1.0 if payment.get("paid_by_manager") else 0.0,      # F29 paid_by_manager
            (amount - staff_mean) / max(staff_std, 1e-8),        # F30 staff_amount_zscore

            # Grupo H: Diversidad
            cat_entropy,                                         # F31 category_entropy_30d
            float(context.reversal_count_30d or 0),              # F32 user_reversal_count_30d
            float(context.merchandise_ratio_30d or 0),           # F33 user_merchandise_ratio_30d
        ], dtype=np.float32)

        return features

    @staticmethod
    def _shannon_entropy(categories: list) -> float:
        if not categories:
            return 0.0
        from collections import Counter
        counts = Counter(categories)
        total = sum(counts.values())
        return -sum((c / total) * np.log2(c / total) for c in counts.values() if c > 0)
```

### 3. `ThresholdClassifier` — score a decision binaria

```python
# src/fraud_detector/scoring/classifier.py

@dataclass
class ScoringResult:
    """Resultado del scoring de una transaccion individual."""
    score: float
    is_anomaly: bool
    risk_level: str      # minimal | low | medium | high | critical
    percentile: float    # Posicion en la distribucion de scores del test set
    factors: list[dict]  # Top features que contribuyen al score

RISK_LEVELS = {
    "minimal":  (0.0, 0.50),
    "low":      (0.50, 0.70),
    "medium":   (0.70, 0.85),
    "high":     (0.85, 0.95),
    "critical": (0.95, 1.01),
}

class ThresholdClassifier:
    """Convierte un score continuo de anomalia en una decision binaria
    y un nivel de riesgo, usando el threshold calibrado en Fase 7.

    El threshold primario se deriva de Precision@5% del test set.
    Los percentiles se calculan contra la distribucion de scores del test set.
    """

    def __init__(self, thresholds_path: str = "output/models/thresholds.json"):
        with open(thresholds_path) as f:
            config = json.load(f)
        self._threshold = config["binary_threshold"]
        self._score_percentiles = np.array(config["score_percentiles"])

    def classify(self, score: float) -> tuple[bool, str, float]:
        """Clasifica un score en anomalo/normal + nivel de riesgo.

        Returns:
            (is_anomaly, risk_level, percentile)
        """
        is_anomaly = score > self._threshold
        percentile = self._compute_percentile(score)
        risk_level = self._assign_risk_level(percentile)
        return is_anomaly, risk_level, percentile

    def _compute_percentile(self, score: float) -> float:
        idx = np.searchsorted(self._score_percentiles, score)
        return min(idx / len(self._score_percentiles), 1.0)

    def _assign_risk_level(self, percentile: float) -> str:
        for level, (low, high) in RISK_LEVELS.items():
            if low <= percentile < high:
                return level
        return "critical"
```

### 4. `SingleTransactionScorer` — facade que integra todo

```python
# src/fraud_detector/scoring/scorer.py

class SingleTransactionScorer:
    """Facade: recibe un dict de pago y devuelve score + decision.

    Carga una sola vez los artefactos de Fases 5-7 (.joblib, .json)
    y reutiliza la conexion a ClickHouse para queries de contexto.

    Uso:
        scorer = SingleTransactionScorer()
        result = scorer.score({
            "user_id": 12345,
            "facility_id": 67,
            "reservation_paid_out": 150.00,
            "discount": 0, "tip": 5.0, "tax": 12.50,
            "payment_method": "card",
            "category": "reservation",
            "club_credit_flag": False,
            "paid_by_manager": False,
            "currency": "USD",
            "created_at": "2026-04-28T14:30:00",
        })
        # result.score = 0.78
        # result.is_anomaly = True
        # result.risk_level = "high"
        # result.factors = [{"feature": "user_txn_count_24h", ...}, ...]
    """

    def __init__(
        self,
        model_path: str = "output/models/isolation_forest.joblib",
        scaler_path: str = "output/models/scaler.joblib",
        feature_engineer_path: str = "output/models/feature_engineer.joblib",
        thresholds_path: str = "output/models/thresholds.json",
        ch_connector: ClickHouseConnector | None = None,
    ):
        self._model = joblib.load(model_path)
        self._scaler = joblib.load(scaler_path)
        self._feature_calc = SingleFeatureCalculator(feature_engineer_path)
        self._classifier = ThresholdClassifier(thresholds_path)

        if ch_connector is None:
            from config.config import get_settings
            settings = get_settings()
            ch_connector = ClickHouseConnector(
                host=settings.clickhouse_host,
                port=settings.clickhouse_port,
                user=settings.clickhouse_user,
                password=settings.clickhouse_password,
                database=settings.clickhouse_database,
                secure=settings.clickhouse_secure,
            ).connect()
        self._context_provider = UserContextProvider(ch_connector)

    def score(self, payment: dict) -> ScoringResult:
        """Scorea una transaccion individual.

        Args:
            payment: Dict con campos minimos:
                user_id, facility_id, reservation_paid_out,
                payment_method, category, created_at.
                Opcionales: discount, tip, tax, club_credit_flag,
                paid_by_manager, currency.

        Returns:
            ScoringResult con score, is_anomaly, risk_level, factors.
        """
        # 1. Obtener contexto historico del usuario
        context = self._context_provider.get_context(
            user_id=payment["user_id"],
            facility_id=payment["facility_id"],
            timestamp=pd.Timestamp(payment["created_at"]),
        )

        # 2. Calcular 31 features
        features = self._feature_calc.calculate(payment, context)

        # 3. Escalar
        X_scaled = self._scaler.transform(features.reshape(1, -1))

        # 4. Obtener anomaly score (higher = mas anomalo)
        raw_score = -self._model.score_samples(X_scaled)[0]

        # 5. Clasificar
        is_anomaly, risk_level, percentile = self._classifier.classify(raw_score)

        # 6. Explicar (top features por magnitud del z-score escalado)
        factors = self._explain_top_factors(features, X_scaled[0], top_n=5)

        return ScoringResult(
            score=float(raw_score),
            is_anomaly=is_anomaly,
            risk_level=risk_level,
            percentile=percentile,
            factors=factors,
        )

    def _explain_top_factors(self, raw_features: np.ndarray,
                             scaled_features: np.ndarray,
                             top_n: int = 5) -> list[dict]:
        """Explica el score mostrando las features mas desviadas de lo normal.

        Usa la magnitud absoluta del valor escalado (z-score) como proxy
        de contribucion. No es SHAP, pero es rapido y razonable para
        explicabilidad operativa.
        """
        from fraud_detector.features.engineering import FEATURE_NAMES
        abs_scaled = np.abs(scaled_features)
        top_indices = np.argsort(abs_scaled)[-top_n:][::-1]

        factors = []
        for idx in top_indices:
            factors.append({
                "feature": FEATURE_NAMES[idx],
                "value": float(raw_features[idx]),
                "z_score": float(scaled_features[idx]),
                "direction": "high" if scaled_features[idx] > 0 else "low",
            })
        return factors
```

---

## Artefacto adicional: `thresholds.json`

Generado al final de Fase 7 (Evaluacion). Contiene los puntos de corte calibrados:

```json
{
    "binary_threshold": 0.0,
    "threshold_source": "precision_at_5pct_test_set",
    "score_percentiles": [],
    "calibration_date": "2026-XX-XX",
    "model": "isolation_forest",
    "model_params_path": "output/models/best_params_if.json",
    "notes": "Threshold derivado del score que corresponde al top-5% del test set"
}
```

**Logica para generar `thresholds.json` en Fase 7:**

```python
def export_thresholds(scores_test: np.ndarray, output_path: str):
    """Genera thresholds.json a partir de los scores del test set.

    El binary_threshold se establece en el percentil 95 de los scores
    (top-5%), alineado con el Enrichment Factor de HE3.
    """
    threshold = float(np.percentile(scores_test, 95))
    percentiles = np.sort(scores_test).tolist()

    result = {
        "binary_threshold": threshold,
        "threshold_source": "percentile_95_test_set",
        "score_percentiles": percentiles,
        "calibration_date": datetime.now().isoformat(),
    }
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
```

**Nota:** `score_percentiles` contiene los 2.5M scores del test ordenados. Ocupa ~20MB en JSON. Si es excesivo, se puede reducir a 1000 percentiles equidistantes (~8KB).

---

## Contratos de test (`test_scoring.py`)

```python
def test_scorer_returns_scoring_result():
    """score() devuelve un ScoringResult con todos los campos."""
    scorer = SingleTransactionScorer(...)
    result = scorer.score(SAMPLE_PAYMENT)
    assert isinstance(result.score, float)
    assert isinstance(result.is_anomaly, bool)
    assert result.risk_level in ("minimal", "low", "medium", "high", "critical")
    assert 0.0 <= result.percentile <= 1.0
    assert len(result.factors) <= 5

def test_feature_count_is_31():
    """SingleFeatureCalculator produce exactamente 31 features."""
    calc = SingleFeatureCalculator(...)
    features = calc.calculate(SAMPLE_PAYMENT, SAMPLE_CONTEXT)
    assert features.shape == (31,)
    assert not np.any(np.isnan(features))

def test_feature_order_matches_batch():
    """Las features del scorer individual coinciden con el pipeline batch
    para la misma transaccion."""
    # Extraer una transaccion del test_features.parquet
    batch_features = pd.read_parquet("data/processed/test_features.parquet")
    row = batch_features.iloc[0]
    payment = _row_to_payment_dict(row)
    context = _row_to_context(row)
    single_features = calc.calculate(payment, context)
    batch_values = row[FEATURE_NAMES].values.astype(np.float32)
    np.testing.assert_allclose(single_features, batch_values, rtol=0.05)

def test_threshold_classifier_boundaries():
    """Scores por encima del threshold son anomalos, por debajo no."""
    classifier = ThresholdClassifier(...)
    is_anom_high, _, _ = classifier.classify(999.0)
    is_anom_low, _, _ = classifier.classify(-999.0)
    assert is_anom_high is True
    assert is_anom_low is False

def test_context_provider_handles_new_user():
    """Un usuario sin historial devuelve contexto con zeros."""
    ctx = provider.get_context(user_id=999999999, facility_id=1,
                                timestamp=datetime.now())
    assert ctx.txn_count_1h == 0
    assert ctx.txn_count_24h == 0

def test_explain_factors_sorted_by_importance():
    """Los factores estan ordenados de mayor a menor z_score absoluto."""
    result = scorer.score(SAMPLE_PAYMENT)
    z_scores = [abs(f["z_score"]) for f in result.factors]
    assert z_scores == sorted(z_scores, reverse=True)
```

---

## Validacion critica: batch vs single

Antes de declarar la Fase 12 completa, se debe verificar que el scorer individual produce **los mismos resultados** que el pipeline batch para transacciones conocidas.

```python
def validate_batch_vs_single(n_samples: int = 100):
    """Compara scores del pipeline batch con el scorer individual.

    Toma n transacciones aleatorias del test set, las pasa por ambos
    caminos, y verifica que los scores sean equivalentes (rtol=5%).
    """
    test_scores = pd.read_parquet("output/scores/test_scores.parquet")
    test_raw = pd.read_parquet("data/processed/test_raw.parquet")
    sample = test_scores.sample(n_samples, random_state=42)

    scorer = SingleTransactionScorer()
    mismatches = 0

    for _, row in sample.iterrows():
        payment_row = test_raw.loc[test_raw["id"] == row["id"]].iloc[0]
        payment_dict = _row_to_payment_dict(payment_row)
        result = scorer.score(payment_dict)

        batch_score = row["if_score"]
        if abs(result.score - batch_score) / max(abs(batch_score), 1e-8) > 0.05:
            mismatches += 1

    assert mismatches / n_samples < 0.05, \
        f"{mismatches}/{n_samples} transacciones difieren > 5%"
```

**Tolerancia:** Se acepta hasta 5% de diferencia por:
- Los queries de ClickHouse para contexto usan `< timestamp` (exclusivo), mientras el batch usa `shift(1)` dentro del DataFrame — diferencia minima por granularidad de timestamp.
- Valores float32 vs float64 en intermedios.

---

## Optimizacion: UserContextCache (opcional)

Para latencia < 10ms, pre-calcular los agregados rolling en una tabla materializada o Redis:

```sql
-- ClickHouse materialized view (actualizada por MergeTree)
CREATE MATERIALIZED VIEW user_context_mv
ENGINE = AggregatingMergeTree()
ORDER BY (user_id, window_end)
AS SELECT
    user_id,
    toStartOfHour(created_at) AS window_end,
    countState() AS txn_count_24h,
    sumState(reservation_paid_out) AS amount_24h,
    ...
FROM pbp_productionDB_optimized.payments FINAL
WHERE _peerdb_is_deleted = 0
GROUP BY user_id, window_end;
```

Alternativa Redis:
```python
class UserContextCache:
    """Cache Redis con TTL de 5 minutos para contexto de usuario.
    Se invalida automaticamente; si falta, cae al query ClickHouse."""

    def get_or_fetch(self, user_id, facility_id, timestamp):
        key = f"uctx:{user_id}:{facility_id}"
        cached = self._redis.get(key)
        if cached:
            return UserContext.from_json(cached)
        ctx = self._provider.get_context(user_id, facility_id, timestamp)
        self._redis.setex(key, 300, ctx.to_json())
        return ctx
```

---

## Estructura de archivos nuevos

```
src/fraud_detector/
├── scoring/                           # NUEVO — Fase 12
│   ├── __init__.py
│   ├── context.py                    # UserContextProvider + UserContext dataclass
│   ├── features.py                   # SingleFeatureCalculator
│   ├── classifier.py                 # ThresholdClassifier + ScoringResult + RISK_LEVELS
│   └── scorer.py                     # SingleTransactionScorer (facade)
output/models/
│   └── thresholds.json               # NUEVO — generado en Fase 7, consumido aqui
tests/
│   └── test_scoring.py               # NUEVO — tests de Fase 12
```

---

## Dependencia con Fase 7

La Fase 12 depende de un artefacto que la Fase 7 debe generar: `thresholds.json`.

**Accion requerida en Fase 7:** Agregar al final de la evaluacion de hipotesis la exportacion de `thresholds.json` usando `export_thresholds()`. Esto no cambia ninguna logica de evaluacion ni afecta los resultados academicos.

---

## Gate de salida — Fase 12

**No declarar completa hasta que:**

1. `SingleTransactionScorer.score(payment)` devuelve `ScoringResult` sin errores para pagos validos.
2. Las 31 features del scorer individual coinciden con el batch (test `test_feature_order_matches_batch`, tolerancia 5%).
3. Los scores del scorer individual coinciden con los del batch (test `validate_batch_vs_single`, tolerancia 5%).
4. Usuarios sin historial devuelven score valido (contexto con zeros).
5. Tests en `test_scoring.py` pasan.
6. `thresholds.json` existe y contiene un threshold calibrado.

---

## Entregables Fase 12

- [ ] `src/fraud_detector/scoring/context.py` — UserContextProvider + UserContext
- [ ] `src/fraud_detector/scoring/features.py` — SingleFeatureCalculator
- [ ] `src/fraud_detector/scoring/classifier.py` — ThresholdClassifier + ScoringResult
- [ ] `src/fraud_detector/scoring/scorer.py` — SingleTransactionScorer (facade)
- [ ] `output/models/thresholds.json` — generado por Fase 7
- [ ] `tests/test_scoring.py` — tests unitarios + validacion batch vs single
- [ ] Validacion de equivalencia batch/single sobre 100 transacciones del test set

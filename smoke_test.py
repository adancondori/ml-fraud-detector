"""
Phase 1 smoke test: sklearn pickle compatibility (Python 3.9 -> 3.12).

Uses SingleTransactionScorer to exercise the FULL import chain:
- SingleTransactionScorer.__init__ loads isolation_forest, scaler, feature_engineer, thresholds
- SingleFeatureCalculator.__init__ accesses fe._groups[0]._global_avg_amount (pickle risk)
- ThresholdClassifier.__init__ reads thresholds.json
- scorer.score() runs end-to-end with synthetic payment + empty UserContext

Verifies:
- No ModuleNotFoundError (fraud_detector package must be importable)
- No AttributeError (feature_engineer internal attributes must survive 3.12 pickle)
- No sklearn UserWarning about version mismatch
- Successful end-to-end scoring with valid ScoringResult

Exit codes:
  0 = all checks passed
  1 = critical failure (model cannot be used in 3.12 container)
"""
import os
import sys
import warnings


def run_smoke_test():
    models_dir = os.environ.get("MODEL_DIR", "output/models")

    print(f"Python version: {sys.version}")
    print(f"Models directory: {models_dir}")
    print()

    # --- Check 1: sklearn version ---
    import sklearn
    print(f"sklearn version: {sklearn.__version__}")
    if sklearn.__version__ != "1.6.1":
        print(f"FAIL: Expected sklearn 1.6.1, got {sklearn.__version__}")
        sys.exit(1)
    print("OK: sklearn version matches training environment (1.6.1)")
    print()

    # --- Check 2: Verify model files exist ---
    model_files = {
        "isolation_forest": f"{models_dir}/isolation_forest.joblib",
        "scaler": f"{models_dir}/scaler.joblib",
        "feature_engineer": f"{models_dir}/feature_engineer.joblib",
        "thresholds": f"{models_dir}/thresholds.json",
    }
    for name, path in model_files.items():
        if not os.path.exists(path):
            print(f"FAIL: Model file not found: {path}")
            sys.exit(1)
        print(f"OK: Found {name} at {path}")
    print()

    # --- Check 3: Load via SingleTransactionScorer with warning capture ---
    # This exercises the full import chain:
    # - joblib.load(isolation_forest.joblib) -> IsolationForest
    # - joblib.load(scaler.joblib) -> UnsupervisedPreprocessor (custom class)
    # - SingleFeatureCalculator(fe_path) -> loads feature_engineer.joblib,
    #   accesses fe._groups[0]._global_avg_amount, fe._groups[4]._facility_avg,
    #   fe._groups[6]._staff_stats (PICKLE RISK: private attribute access)
    # - ThresholdClassifier(thresholds_path) -> reads thresholds.json

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        from fraud_detector.scoring.scorer import SingleTransactionScorer
        from fraud_detector.scoring.context import UserContext

        print("Loading models via SingleTransactionScorer...")
        scorer = SingleTransactionScorer(
            model_path=model_files["isolation_forest"],
            scaler_path=model_files["scaler"],
            feature_engineer_path=model_files["feature_engineer"],
            thresholds_path=model_files["thresholds"],
            ch_connector=None,
        )
        print("OK: SingleTransactionScorer initialized successfully")

    # Check for sklearn version mismatch warnings
    sklearn_warnings = [
        w for w in caught
        if issubclass(w.category, UserWarning)
        and ("unpickle" in str(w.message).lower()
             or ("version" in str(w.message).lower()
                 and "sklearn" in str(w.message).lower()))
    ]
    if sklearn_warnings:
        print()
        print("FAIL: sklearn version mismatch warnings detected:")
        for w in sklearn_warnings:
            print(f"  WARNING: {w.message}")
        sys.exit(1)

    print("OK: No sklearn version mismatch warnings")
    print()

    # --- Check 4: Verify internal components loaded correctly ---
    # IsolationForest model type
    assert type(scorer._model).__name__ == "IsolationForest", (
        f"FAIL: Expected IsolationForest, got {type(scorer._model).__name__}"
    )
    print("OK: Model is IsolationForest")

    # UnsupervisedPreprocessor (custom class from fraud_detector package)
    assert type(scorer._scaler).__name__ == "UnsupervisedPreprocessor", (
        f"FAIL: Expected UnsupervisedPreprocessor, got {type(scorer._scaler).__name__}"
    )
    print("OK: Scaler is UnsupervisedPreprocessor")

    # SingleFeatureCalculator loaded learned parameters (the key pickle risk)
    fc = scorer._feature_calc
    assert fc._global_avg_amount is not None and fc._global_avg_amount > 0, (
        f"FAIL: _global_avg_amount is {fc._global_avg_amount} (expected positive float)"
    )
    print(f"OK: SingleFeatureCalculator._global_avg_amount = {fc._global_avg_amount:.2f}")
    print(f"OK: SingleFeatureCalculator._facility_avgs has {len(fc._facility_avgs)} entries")
    print(f"OK: SingleFeatureCalculator._staff_stats has {len(fc._staff_stats)} entries")

    # ThresholdClassifier loaded threshold
    assert scorer._classifier.threshold > 0, (
        f"FAIL: threshold is {scorer._classifier.threshold} (expected positive float)"
    )
    print(f"OK: ThresholdClassifier.threshold = {scorer._classifier.threshold:.6f}")
    print()

    # --- Check 5: Run end-to-end scoring ---
    payment = {
        "user_id": 1, "facility_id": 1,
        "reservation_paid_out": 100.0, "discount": 0, "tip": 0,
        "payment_method": "card", "category": "reservation",
        "club_credit_flag": False, "paid_by_manager": False,
        "currency": "USD", "created_at": "2025-10-15T14:30:00",
    }
    context = UserContext()
    result = scorer.score(payment, context=context)

    # Validate ScoringResult fields
    assert isinstance(result.score, float), f"FAIL: score must be float, got {type(result.score)}"
    assert result.risk_level in ("minimal", "low", "medium", "high", "critical"), (
        f"FAIL: unexpected risk_level: {result.risk_level}"
    )
    assert isinstance(result.is_anomaly, bool), f"FAIL: is_anomaly must be bool, got {type(result.is_anomaly)}"
    assert isinstance(result.percentile, float), f"FAIL: percentile must be float, got {type(result.percentile)}"
    assert len(result.factors) > 0, "FAIL: factors must not be empty"

    # Validate factor structure
    first_factor = result.factors[0]
    assert "feature" in first_factor, "FAIL: factor missing 'feature' key"
    assert "value" in first_factor, "FAIL: factor missing 'value' key"
    assert "z_score" in first_factor, "FAIL: factor missing 'z_score' key"
    assert "direction" in first_factor, "FAIL: factor missing 'direction' key"

    print(f"OK: score = {result.score:.6f}")
    print(f"OK: risk_level = {result.risk_level}")
    print(f"OK: is_anomaly = {result.is_anomaly}")
    print(f"OK: percentile = {result.percentile:.4f}")
    print(f"OK: factors count = {len(result.factors)}")
    print(f"OK: top factor = {result.factors[0]}")
    print()

    # --- Summary ---
    print("=" * 60)
    print("SMOKE TEST PASSED")
    print("=" * 60)
    print()
    print("Summary:")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  sklearn: {sklearn.__version__}")
    print(f"  Model: IsolationForest (via SingleTransactionScorer)")
    print(f"  Scaler: UnsupervisedPreprocessor")
    print(f"  FeatureCalc: global_avg={fc._global_avg_amount:.2f}, "
          f"facilities={len(fc._facility_avgs)}, staff_roles={len(fc._staff_stats)}")
    print(f"  Threshold: {scorer._classifier.threshold:.6f}")
    print(f"  Test score: {result.score:.6f} ({result.risk_level})")
    print()
    print("The 3.9-trained models are compatible with Python 3.12.")
    print("The full SingleTransactionScorer import chain works correctly.")


if __name__ == "__main__":
    try:
        run_smoke_test()
    except Exception as e:
        print(f"FAIL: Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

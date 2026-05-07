#!/usr/bin/env python
"""Anomaly detection pipeline orchestrator.

Single entry point for the full thesis pipeline: extraction to reporting.
Supports partial execution, prerequisite validation, and dry-run mode.

Usage:
    python run_pipeline.py                    # All steps
    python run_pipeline.py --step 3           # Single step
    python run_pipeline.py --from-step 5      # From step 5 onward
    python run_pipeline.py --fast             # Reduced bootstrap, skip grid search
    python run_pipeline.py --dry-run          # Validate prerequisites only
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Callable

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config.config import settings
from fraud_detector.utils.logger import logger

STEP_INPUTS = {
    1: [],
    2: [
        "data/processed/train_raw.parquet",
        "data/processed/val_raw.parquet",
        "data/processed/test_raw.parquet",
        "data/processed/warm_raw.parquet",
    ],
    3: [
        "data/processed/train_features.parquet",
        "data/processed/val_features.parquet",
        "data/processed/test_features.parquet",
    ],
    4: [
        "output/scores/X_train.npy",
        "output/scores/X_val.npy",
    ],
    5: [
        "output/models/isolation_forest.joblib",
        "output/models/lof.joblib",
        "output/models/ocsvm.joblib",
        "output/scores/X_test.npy",
    ],
    6: [
        "output/scores/test_scores.parquet",
        "data/processed/test_features.parquet",
    ],
    7: [
        "output/results.json",
        "output/models/isolation_forest.joblib",
        "output/scores/X_test.npy",
    ],
    8: [
        "output/results.json",
        "output/results_sensitivity.json",
        "output/results_posthoc.json",
    ],
}

STEP_NAMES = {
    1: "Data Extraction",
    2: "Feature Engineering",
    3: "Preprocessing",
    4: "Model Training",
    5: "Test Set Scoring",
    6: "Evaluation (HE1-HE4)",
    7: "Sensitivity + SHAP + Post-Hoc",
    8: "Reports (tables + figures)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline de deteccion de anomalias")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--step", type=int, choices=range(1, 9), help="Run only this step")
    group.add_argument("--from-step", type=int, choices=range(1, 9), help="Run from this step onward")
    parser.add_argument("--fast", action="store_true", help="Reduced bootstrap, skip grid search")
    parser.add_argument("--dry-run", action="store_true", help="Validate prerequisites only")
    return parser.parse_args()


def should_run(step_num: int, args: argparse.Namespace) -> bool:
    if args.step is not None:
        return step_num == args.step
    if args.from_step is not None:
        return step_num >= args.from_step
    return True


def validate_prerequisites(step_num: int, base_dir: Path) -> None:
    for path_str in STEP_INPUTS.get(step_num, []):
        full_path = base_dir / path_str
        if not full_path.exists():
            raise FileNotFoundError(
                f"Missing prerequisite for step {step_num}: {path_str}. "
                f"Run the previous step first."
            )


def run_step(step_num: int, name: str, fn: Callable[[], None], dry_run: bool = False) -> None:
    logger.info("=" * 60)
    logger.info(f"Step {step_num}/8: {name}")
    logger.info("=" * 60)

    validate_prerequisites(step_num, PROJECT_ROOT)

    if dry_run:
        logger.info(f"  [DRY-RUN] Prerequisites OK — skipping execution")
        return

    t0 = time.perf_counter()
    fn()
    elapsed = time.perf_counter() - t0
    minutes, seconds = divmod(int(elapsed), 60)
    logger.info(f"Step {step_num} completed in {minutes:02d}:{seconds:02d}")
    gc.collect()


# ── Step implementations (lazy imports) ─────────────────────────


def step1_extract():
    from fraud_detector.data.loader import DataManager
    dm = DataManager(settings)
    dm.extract_from_clickhouse()


def step2_engineer():
    import pandas as pd
    from fraud_detector.features.engineering import FeatureEngineer
    fe = FeatureEngineer()
    df_warm = pd.read_parquet(settings.processed_dir / "warm_raw.parquet")
    for split in ["train", "val", "test"]:
        df = pd.read_parquet(settings.processed_dir / f"{split}_raw.parquet")
        if split == "train":
            df_combined = pd.concat([df_warm, df], ignore_index=True)
            fe.fit(df_combined)
            result = fe.transform(df_combined)
            result = result.iloc[len(df_warm):]
        else:
            result = fe.transform(df)
        result.to_parquet(settings.processed_dir / f"{split}_features.parquet", index=False)
        logger.info(f"  {split}: {len(result):,} rows")
        del df, result
    fe.save(str(settings.models_output_dir / "feature_engineer.joblib"))


def step3_preprocess():
    import numpy as np
    import pandas as pd
    from fraud_detector.features.engineering import FEATURE_NAMES
    from fraud_detector.features.preprocessor import UnsupervisedPreprocessor
    prep = UnsupervisedPreprocessor(variant="full")
    df_train = pd.read_parquet(settings.processed_dir / "train_features.parquet")
    X_train = prep.fit_transform(df_train)
    np.save(settings.scores_dir / "X_train.npy", X_train)
    for split in ["val", "test"]:
        df = pd.read_parquet(settings.processed_dir / f"{split}_features.parquet")
        X = prep.transform(df)
        np.save(settings.scores_dir / f"X_{split}.npy", X)
        del df, X
    prep.save(str(settings.models_output_dir / "scaler.joblib"))
    del df_train, X_train


def step4_train(fast: bool = False):
    """Runs grid search + trains final models. With --fast, skips grid search."""
    import subprocess
    cmd = [sys.executable, "scripts/run_fase6_modeling.py"]
    if fast:
        logger.info("  [FAST MODE] Skipping grid search — using default params")
        cmd.append("--step")
        cmd.append("train")
    subprocess.run(cmd, check=True)


def step5_score():
    """Score test set with all models + save thresholds."""
    import subprocess
    # Reuse the evaluation script's scoring portion
    subprocess.run([sys.executable, "scripts/run_fase7_evaluation.py"], check=True)


def step6_evaluate():
    """HE1-HE4 already computed in step5 (combined in run_fase7_evaluation.py)."""
    logger.info("  Evaluation completed in Step 5 (combined scoring + evaluation)")
    results_path = PROJECT_ROOT / "output" / "results.json"
    if results_path.exists():
        results = json.loads(results_path.read_text())
        he1 = results.get("isolation_forest", {}).get("he1", {})
        he2 = results.get("isolation_forest", {}).get("he2", {})
        he3 = results.get("isolation_forest", {}).get("he3", {})
        he4 = results.get("he4", {})
        logger.info(f"  HE1: {'PASS' if he1.get('he1_pass') else 'FAIL'} (r={he1.get('rank_biserial_r', 0):.4f})")
        logger.info(f"  HE2: {'PASS' if he2.get('he2_pass') else 'FAIL'} (AUC={he2.get('auc_roc', 0):.4f})")
        logger.info(f"  HE3: {'PASS' if he3.get('he3_pass') else 'FAIL'} (EF@5%={he3.get('ef_at_5pct', 0):.4f})")
        logger.info(f"  HE4: {'PASS' if he4.get('he4_pass') else 'FAIL'} (IF wins {he4.get('if_wins', 0)}/4)")


def step7_sensitivity():
    import subprocess
    subprocess.run([sys.executable, "scripts/run_fase8_sensitivity.py"], check=True)


def step8_reports():
    import subprocess
    subprocess.run([sys.executable, "scripts/run_fase9_reporting.py"], check=True)


def print_summary():
    """Print final pipeline summary with key metrics."""
    results_path = PROJECT_ROOT / "output" / "results.json"
    if not results_path.exists():
        return

    results = json.loads(results_path.read_text())
    he1 = results.get("isolation_forest", {}).get("he1", {})
    he2 = results.get("isolation_forest", {}).get("he2", {})
    he3 = results.get("isolation_forest", {}).get("he3", {})
    he4 = results.get("he4", {})

    logger.info("")
    logger.info("=" * 60)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Dataset:   {results.get('test_set_size', '?'):,} test transactions")
    logger.info(f"  Proxy:     {results.get('proxy_used', '?')} ({results.get('proxy_unified_base_rate', 0)*100:.2f}%)")
    logger.info(f"  IF AUC:    {he2.get('auc_roc', 0):.4f}")
    logger.info(f"  IF AP:     {he2.get('average_precision', 0):.4f}")
    logger.info(f"  IF EF@5%:  {he3.get('ef_at_5pct', 0):.4f}")
    logger.info("")

    for name, he, key in [("HE1", he1, "he1_pass"), ("HE2", he2, "he2_pass"),
                           ("HE3", he3, "he3_pass"), ("HE4", he4, "he4_pass")]:
        status = "PASS" if he.get(key) else "FAIL"
        logger.info(f"  {name}: {status}")
    logger.info("=" * 60)


def main():
    args = parse_args()
    settings.ensure_directories()

    t_total = time.perf_counter()
    fast = args.fast

    steps = {
        1: ("Data Extraction", step1_extract),
        2: ("Feature Engineering", step2_engineer),
        3: ("Preprocessing", step3_preprocess),
        4: ("Model Training", lambda: step4_train(fast)),
        5: ("Scoring + Evaluation", step5_score),
        6: ("Hypothesis Summary", step6_evaluate),
        7: ("Sensitivity + SHAP", step7_sensitivity),
        8: ("Reports", step8_reports),
    }

    for step_num, (name, fn) in steps.items():
        if should_run(step_num, args):
            run_step(step_num, name, fn, dry_run=args.dry_run)

    if not args.dry_run:
        print_summary()

    elapsed = time.perf_counter() - t_total
    minutes, seconds = divmod(int(elapsed), 60)
    logger.info(f"Total pipeline time: {minutes:02d}:{seconds:02d}")


if __name__ == "__main__":
    main()

"""Análisis de sensibilidad inferencial de HE2 (dual bootstrap + composición).

Complementa el confirmatorio V2 (``v2_confirmatory_scoreboard.json``, clustered
por usuario) con la variante de bootstrap POR TRANSACCIÓN pre-registrada en el
contrato de resultados (fila A16: "por transacción y clustered por usuario") y
que el artefacto b564012 no reportó. NO modifica el criterio congelado de HE2:
produce la segunda inferencia pre-registrada y diagnósticos de apoyo.

Salidas (``output/revision/he2_sensitivity.json``):

1. ``per_txn_bootstrap``: EF@1%/EF@5% con IC95 bootstrap por transacción
   (ruta legacy de ``bootstrap_ci``, ``user_ids=None``, random_seed=42) para
   los 3 tipos, la unión tipificada y el control negativo refund.
2. ``cluster_diagnostics``: nº de usuarios únicos entre los positivos de cada
   proxy y participación del usuario dominante — documenta por qué el IC
   clustered de ``velocity_extreme`` es inutilizable (pocos clusters efectivos).
3. ``top1pct_composition``: composición del top-1% del ranking por tipo
   (soporte del reencuadre a nivel de la variable criterio).
4. ``card_testing_per_seed``: EF@1% + IC por transacción para las semillas
   43/44 — verifica que el veredicto de card_testing es invariante entre
   semillas aunque el gate de estabilidad (rango relativo 16,8%) falle.
5. ``he2_verdicts``: conteo de tipos que satisfacen EF@1%>=2 con LI>1 bajo
   cada inferencia (clustered = artefacto confirmatorio; per_txn = este script).

Uso::

    python scripts/eval_he2_sensitivity.py \\
        --scores output/revision/test_scores_v2.parquet \\
        --bootstrap 1000 \\
        --output output/revision/he2_sensitivity.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from fraud_detector.evaluation.metrics import (  # noqa: E402
    bootstrap_ci,
    enrichment_factor,
)
from fraud_detector.scoring import rules  # noqa: E402

TYPED_PROXIES = ["card_testing", "velocity_extreme", "new_user_burst"]
UNION_KEY = "typed_union"
REFUND_KEY = "refund_negative_control"

DEFAULT_SCORES = ROOT / "output" / "revision" / "test_scores_v2.parquet"
DEFAULT_OUTPUT = ROOT / "output" / "revision" / "he2_sensitivity.json"
DEFAULT_BOOTSTRAP = 1000

# Gate HE2 congelado (pre-registro): EF@1% >= 2 con LI IC95 > 1 por tipo.
HE2_EF_MIN = 2.0
HE2_CI_LOW_MIN = 1.0


def build_proxies(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Variable criterio tipificada + control negativo desde rules.py (idéntico
    a eval_scoreboard.build_proxies)."""
    return {
        "card_testing": rules.card_testing(df).astype(np.int8),
        "velocity_extreme": rules.velocity_extreme(df).astype(np.int8),
        "new_user_burst": rules.new_user_burst(df).astype(np.int8),
        UNION_KEY: rules.typed_union(df).astype(np.int8),
        REFUND_KEY: rules.refund_negative_control(df).astype(np.int8),
    }


def _per_txn_ef_ci(
    y: np.ndarray, s: np.ndarray, k_pct: float, n_iterations: int
) -> Dict[str, float]:
    """IC95 bootstrap POR TRANSACCIÓN de EF@k (ruta legacy de bootstrap_ci,
    mismo random_seed=42 que el confirmatorio)."""

    def ef_metric(y_true, y_scores):
        return enrichment_factor(y_true, y_scores, k_pct=k_pct)

    return bootstrap_ci(y, s, ef_metric, n_iterations=n_iterations, random_seed=42)


def evaluate_per_txn(
    scores: np.ndarray, y: np.ndarray, n_bootstrap: int
) -> Dict[str, Optional[float]]:
    """EF@1%/EF@5% puntuales + IC95 por transacción para un proxy."""
    if y.sum() == 0:
        return {k: None for k in (
            "ef_at_1pct", "ef_at_1pct_ci_low", "ef_at_1pct_ci_high",
            "ef_at_5pct", "ef_at_5pct_ci_low", "ef_at_5pct_ci_high",
        )}
    ci1 = _per_txn_ef_ci(y, scores, 0.01, n_bootstrap)
    ci5 = _per_txn_ef_ci(y, scores, 0.05, n_bootstrap)
    return {
        "ef_at_1pct": float(enrichment_factor(y, scores, k_pct=0.01)),
        "ef_at_1pct_ci_low": ci1["lower"],
        "ef_at_1pct_ci_high": ci1["upper"],
        "ef_at_5pct": float(enrichment_factor(y, scores, k_pct=0.05)),
        "ef_at_5pct_ci_low": ci5["lower"],
        "ef_at_5pct_ci_high": ci5["upper"],
    }


def cluster_diagnostics(
    df: pd.DataFrame, proxies: Dict[str, np.ndarray]
) -> Dict[str, Dict[str, float]]:
    """Usuarios únicos entre los positivos de cada proxy y concentración.

    Un IC clustered explota cuando los positivos pertenecen a pocos usuarios:
    el tamaño muestral efectivo es el nº de clusters con señal, no el nº de
    transacciones.
    """
    out: Dict[str, Dict[str, float]] = {}
    user = df["user_id"].to_numpy()
    n_users_total = int(pd.unique(user).size)
    for label, y in proxies.items():
        mask = y.astype(bool)
        pos_users = pd.Series(user[mask])
        counts = pos_users.value_counts()
        out[label] = {
            "n_positives": int(mask.sum()),
            "n_unique_users_positive": int(counts.size),
            "top_user_share_of_positives": (
                float(counts.iloc[0] / mask.sum()) if counts.size else 0.0
            ),
            "top5_users_share_of_positives": (
                float(counts.iloc[:5].sum() / mask.sum()) if counts.size else 0.0
            ),
            "n_users_total": n_users_total,
        }
    return out


def top1pct_composition(
    scores: np.ndarray, proxies: Dict[str, np.ndarray]
) -> Dict[str, Dict[str, float]]:
    """Composición del top-1% del ranking por tipo (mismo corte que EF@1%)."""
    k = max(1, int(np.ceil(len(scores) * 0.01)))
    top_idx = np.argsort(scores)[-k:]
    out: Dict[str, Dict[str, float]] = {"_meta": {"k": float(k)}}
    for label, y in proxies.items():
        hits = int(y[top_idx].sum())
        out[label] = {
            "hits_in_top1pct": float(hits),
            "pct_of_top1pct": float(hits / k),
            "pct_of_type_captured": (
                float(hits / y.sum()) if y.sum() else 0.0
            ),
        }
    return out


def he2_verdict(blocks: Dict[str, Dict[str, Optional[float]]]) -> Dict[str, object]:
    """Tipos que satisfacen el gate congelado bajo la inferencia dada."""
    passing = [
        t
        for t in TYPED_PROXIES
        if blocks[t]["ef_at_1pct"] is not None
        and blocks[t]["ef_at_1pct"] >= HE2_EF_MIN
        and blocks[t]["ef_at_1pct_ci_low"] is not None
        and blocks[t]["ef_at_1pct_ci_low"] > HE2_CI_LOW_MIN
    ]
    return {
        "types_passing": passing,
        "n_passing": len(passing),
        "gate_satisfied": len(passing) >= 2,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP)
    p.add_argument(
        "--skip-seeds",
        action="store_true",
        help="Omite los IC por semilla 43/44 de card_testing (más rápido).",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    t0 = time.time()

    df = pd.read_parquet(args.scores)
    scores = df["score"].to_numpy(dtype=np.float64)
    proxies = build_proxies(df)
    print(f"[{time.time() - t0:6.1f}s] parquet cargado: n={len(df):,}", flush=True)

    result: Dict[str, object] = {
        "meta": {
            "scores_path": str(args.scores),
            "n": int(len(df)),
            "n_bootstrap": int(args.bootstrap),
            "clustered_bootstrap": False,
            "random_seed": 42,
            "note": (
                "Variante por transacción pre-registrada (contrato A16); "
                "complementa el clustered de v2_confirmatory_scoreboard.json"
            ),
        },
        "cluster_diagnostics": cluster_diagnostics(df, proxies),
        "top1pct_composition": top1pct_composition(scores, proxies),
    }
    _dump(result, args.output)
    print(f"[{time.time() - t0:6.1f}s] diagnósticos de cluster y composición listos", flush=True)

    per_txn: Dict[str, Dict[str, Optional[float]]] = {}
    for label, y in proxies.items():
        per_txn[label] = evaluate_per_txn(scores, y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y), args.bootstrap)
        result["per_txn_bootstrap"] = per_txn
        _dump(result, args.output)
        print(
            f"[{time.time() - t0:6.1f}s] {label}: EF@1%={per_txn[label]['ef_at_1pct']:.2f} "
            f"IC_txn[{per_txn[label]['ef_at_1pct_ci_low']:.2f}, "
            f"{per_txn[label]['ef_at_1pct_ci_high']:.2f}]",
            flush=True,
        )

    result["he2_verdicts"] = {"per_txn": he2_verdict(per_txn)}

    if not args.skip_seeds:
        y_ct = np.asarray(proxies["card_testing"])
        per_seed: Dict[str, Dict[str, Optional[float]]] = {}
        for seed_col in ("score_seed43", "score_seed44"):
            s_seed = df[seed_col].to_numpy(dtype=np.float64)
            ci1 = _per_txn_ef_ci(y_ct, s_seed, 0.01, args.bootstrap)
            per_seed[seed_col] = {
                "ef_at_1pct": float(enrichment_factor(y_ct, s_seed, k_pct=0.01)),
                "ef_at_1pct_ci_low": ci1["lower"],
                "ef_at_1pct_ci_high": ci1["upper"],
            }
            result["card_testing_per_seed"] = per_seed
            _dump(result, args.output)
            print(
                f"[{time.time() - t0:6.1f}s] card_testing {seed_col}: "
                f"EF@1%={per_seed[seed_col]['ef_at_1pct']:.2f} "
                f"IC_txn[{ci1['lower']:.2f}, {ci1['upper']:.2f}]",
                flush=True,
            )

    result["elapsed_seconds"] = round(time.time() - t0, 1)
    _dump(result, args.output)
    print(f"[{time.time() - t0:6.1f}s] escrito {args.output}", flush=True)
    return 0


def _dump(obj: Dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    raise SystemExit(main())

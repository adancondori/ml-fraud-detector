"""Juez automático del confirmatorio V2 — scoreboard de gates pre-registrados.

Dado (a) los scores del modelo, (b) los proxies de ``rules.py`` (variable
criterio tipificada NO circular + control negativo de reembolso) y (c) la lista
de features del modelo, produce el **scoreboard JSON** con el esquema que consume
``fraud_detector.reporting.latex_tables.generate_v2_tables`` (Fase 5A).

Métricas titulares (plan §7 / RESUMEN-CADENA-METODOLOGICA-V2-PIVOTE.md §7):
  - HE1: Mann-Whitney U (α=0,05) y rank-biserial r_rb > 0,10 sobre typed_union.
  - HE2: EF@1% >= 2 en >= 2/3 tipos, con IC bootstrap 95% cuyo LI > 1 (titular);
    AUC/AP secundarias, sin gate.
  - HE3: control negativo refund con EF@1% ∈ [0,8; 1,3] y AUC ∈ [0,45; 0,55].
  - HE4: IF >= LOF y OC-SVM en >= 3/4 (EF@1%, EF@5%, AP, P@1%).
  - Multiseed: rango relativo EF@1% < 15% en tipos con gate.

Reutiliza helpers existentes (NO se duplica EF@k):
  - ``fraud_detector.evaluation.metrics.precision_at_k`` / ``enrichment_factor``
  - ``fraud_detector.evaluation.metrics.bootstrap_ci``
  - ``fraud_detector.evaluation.hypothesis.run_mann_whitney`` (MWU + rank-biserial)
  - ``fraud_detector.scoring.rules.assert_disjoint_from_features`` (disyunción)

Formato de entrada esperado por la CLI (parquet de scores + proxies):
  Un parquet con, como mínimo, las columnas:
    - ``score``            : float, score de anomalía (mayor = más anómalo).
    - Campos EXTERNOS de las reglas para construir los proxies en vivo:
        ``same_amount_count_1h``, ``failed_count_1h``, ``user_txn_count_24h``,
        ``user_account_age_days``, ``user_txn_count_1h``, ``status``.
    - Opcional, para HE4: ``score_lof`` y ``score_ocsvm`` (scores de los
      comparadores sobre las mismas filas).
    - Opcional, para multiseed: ``score_seed42``, ``score_seed43``,
      ``score_seed44`` (scores por semilla; si faltan, se omite ``multiseed``).
    - Opcional: ``user_id`` para bootstrap agrupado por usuario (clustered).

Uso (NO ejecutar contra datos reales aquí — eso es la corrida 5B):
    ./venv/bin/python scripts/eval_scoreboard.py \\
        --scores data/processed/test_scores_v2.parquet \\
        --features frame-v1 --proxy-set typed_v2 --negative-control refund \\
        --split test --seeds 42,43,44 --bootstrap 1000 \\
        --output output/revision/v2_confirmatory_scoreboard.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from fraud_detector.evaluation.hypothesis import run_mann_whitney  # noqa: E402
from fraud_detector.evaluation.metrics import (  # noqa: E402
    bootstrap_ci,
    enrichment_factor,
    precision_at_k,
)
from fraud_detector.scoring import rules  # noqa: E402
from fraud_detector.scoring.features_frame_v1 import (  # noqa: E402
    FRAME_V1_FEATURE_NAMES,
)

# Tipos tipificados NO circulares que participan del gate HE2/multiseed.
TYPED_PROXIES = ["card_testing", "velocity_extreme", "new_user_burst"]
UNION_KEY = "typed_union"
REFUND_KEY = "refund_negative_control"

# new_user_burst se EXCLUYE del gate de estabilidad multiseed: IF es casi ciego
# a él (EF ~1-2 con base rate mínima -> varianza estructural alta). Su cobertura
# es responsabilidad de las reglas, no del modelo (plan §5, re-especificación).
MULTISEED_GATED = {"card_testing", "velocity_extreme", UNION_KEY}

DEFAULT_BOOTSTRAP = 1000


# ---------------------------------------------------------------------------
# Construcción de proxies desde rules.py (variable criterio, NUNCA entrenamiento)
# ---------------------------------------------------------------------------


def build_proxies(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Construye la variable criterio tipificada + control negativo desde rules.py.

    Todos los proxies se derivan de campos EXTERNOS al feature set (disyunción
    garantizada por :func:`rules.assert_disjoint_from_features`).
    """
    return {
        "card_testing": rules.card_testing(df).astype(np.int8),
        "velocity_extreme": rules.velocity_extreme(df).astype(np.int8),
        "new_user_burst": rules.new_user_burst(df).astype(np.int8),
        UNION_KEY: rules.typed_union(df).astype(np.int8),
        REFUND_KEY: rules.refund_negative_control(df).astype(np.int8),
    }


# ---------------------------------------------------------------------------
# Métricas por proxy (EF@k, P@k, AUC, AP + IC bootstrap) — reutiliza metrics.py
# ---------------------------------------------------------------------------


def _ef_at(y: np.ndarray, s: np.ndarray, k_pct: float) -> float:
    """EF@k = precision@k / base_rate (reutiliza enrichment_factor de metrics.py)."""
    return float(enrichment_factor(y, s, k_pct=k_pct))


def _bootstrap_ef_ci(
    y: np.ndarray,
    s: np.ndarray,
    k_pct: float,
    n_iterations: int,
    user_ids: Optional[np.ndarray] = None,
    random_seed: int = 42,
) -> Dict[str, float]:
    """IC95 bootstrap de EF@k. Reutiliza ``bootstrap_ci`` con metric_fn=EF@k.

    Si ``user_ids`` se provee, hace bootstrap agrupado por usuario (clustered,
    ruta ``concatenate`` — EF@k no soporta sample_weight).
    """

    def ef_metric(y_true, y_scores):
        return enrichment_factor(y_true, y_scores, k_pct=k_pct)

    return bootstrap_ci(
        y,
        s,
        ef_metric,
        n_iterations=n_iterations,
        random_seed=random_seed,
        user_ids=user_ids,
        method="concatenate" if user_ids is not None else "auto",
    )


def evaluate_proxy(
    scores: np.ndarray,
    y: np.ndarray,
    n_bootstrap: int,
    user_ids: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Bloque de métricas de un tipo para el esquema del scoreboard.

    Claves emitidas (exactas para latex_tables): ``base_rate``, ``auc``, ``ap``,
    ``ef_at_1pct`` (+ci_low/ci_high), ``ef_at_5pct`` (+ci_low/ci_high),
    ``precision_at_1pct``.
    """
    base_rate = float(y.mean())
    block: Dict[str, float] = {"base_rate": base_rate}

    if base_rate == 0.0 or base_rate == 1.0:
        # Sin ambas clases no hay métrica discriminativa definida.
        block.update(
            {
                "auc": None,
                "ap": None,
                "ef_at_1pct": None,
                "ef_at_1pct_ci_low": None,
                "ef_at_1pct_ci_high": None,
                "ef_at_5pct": None,
                "ef_at_5pct_ci_low": None,
                "ef_at_5pct_ci_high": None,
                "precision_at_1pct": None,
                "n_positives": int(y.sum()),
            }
        )
        return block

    block["auc"] = float(roc_auc_score(y, scores))
    block["ap"] = float(average_precision_score(y, scores))
    block["ef_at_1pct"] = _ef_at(y, scores, 0.01)
    block["ef_at_5pct"] = _ef_at(y, scores, 0.05)
    block["precision_at_1pct"] = float(precision_at_k(y, scores, k_pct=0.01))
    block["n_positives"] = int(y.sum())

    ci1 = _bootstrap_ef_ci(y, scores, 0.01, n_bootstrap, user_ids)
    ci5 = _bootstrap_ef_ci(y, scores, 0.05, n_bootstrap, user_ids)
    block["ef_at_1pct_ci_low"] = ci1["lower"]
    block["ef_at_1pct_ci_high"] = ci1["upper"]
    block["ef_at_5pct_ci_low"] = ci5["lower"]
    block["ef_at_5pct_ci_high"] = ci5["upper"]
    return block


# ---------------------------------------------------------------------------
# Bloque global (HE1: MWU + rank-biserial sobre typed_union)
# ---------------------------------------------------------------------------


def evaluate_global(scores: np.ndarray, union: np.ndarray) -> Dict[str, float]:
    """HE1: Mann-Whitney U + rank-biserial sobre la unión tipificada.

    Reutiliza ``run_mann_whitney`` (una cola 'greater') y remapea sus claves al
    esquema del scoreboard: ``mwu_u``, ``mwu_p``, ``r_rb``.
    """
    mwu = run_mann_whitney(scores, union)
    return {
        "mwu_u": float(mwu["U_statistic"]),
        "mwu_p": float(mwu["p_value"]),
        "r_rb": float(mwu["rank_biserial_r"]),
    }


# ---------------------------------------------------------------------------
# Bloque comparadores (HE4: IF vs LOF vs OC-SVM en las 4 métricas)
# ---------------------------------------------------------------------------


def evaluate_comparators(
    y: np.ndarray, model_scores: Dict[str, np.ndarray]
) -> Dict[str, Dict[str, float]]:
    """HE4: 4 métricas (EF@1%, EF@5%, AP, P@1%) por modelo sobre la unión tipificada."""
    out: Dict[str, Dict[str, float]] = {}
    for name, s in model_scores.items():
        s = np.asarray(s, dtype=np.float64)
        out[name] = {
            "ef_at_1pct": _ef_at(y, s, 0.01),
            "ef_at_5pct": _ef_at(y, s, 0.05),
            "ap": float(average_precision_score(y, s)),
            "precision_at_1pct": float(precision_at_k(y, s, k_pct=0.01)),
        }
    return out


# ---------------------------------------------------------------------------
# Bloque multiseed (rango relativo EF@1% por tipo entre semillas)
# ---------------------------------------------------------------------------


def _relative_range(values: Sequence[float]) -> float:
    """Rango relativo: (max - min) / mediana (patrón verify_multiseed_stability)."""
    med = float(np.median(values))
    if med == 0:
        return float("nan")
    return (max(values) - min(values)) / med


def evaluate_multiseed(
    seed_scores: Dict[int, np.ndarray], proxies: Dict[str, np.ndarray]
) -> Dict[str, Dict[str, object]]:
    """Rango relativo de EF@1% por tipo entre semillas + flag ``gated``.

    Emite las claves que consume ``table_v2_multiseed_stability``:
    ``ef1_relative_range`` y ``gated``.
    """
    out: Dict[str, Dict[str, object]] = {}
    for label in TYPED_PROXIES + [UNION_KEY]:
        y = proxies[label]
        if float(y.mean()) == 0.0:
            continue
        ef1_per_seed = [
            _ef_at(y, np.asarray(s, dtype=np.float64), 0.01) for s in seed_scores.values()
        ]
        out[label] = {
            "ef1_relative_range": _relative_range(ef1_per_seed),
            "ef1_per_seed": ef1_per_seed,
            "gated": label in MULTISEED_GATED,
        }
    return out


# ---------------------------------------------------------------------------
# Orquestador puro (testeable sin I/O)
# ---------------------------------------------------------------------------


def build_scoreboard(
    scores: np.ndarray,
    proxies: Dict[str, np.ndarray],
    feature_names: Sequence[str],
    *,
    comparator_scores: Optional[Dict[str, np.ndarray]] = None,
    seed_scores: Optional[Dict[int, np.ndarray]] = None,
    n_bootstrap: int = DEFAULT_BOOTSTRAP,
    user_ids: Optional[np.ndarray] = None,
    split: str = "test",
) -> Dict[str, object]:
    """Ensambla el scoreboard JSON con el esquema de ``generate_v2_tables``.

    Verifica la disyunción feature↔proxy ANTES de calcular nada (falla ruidoso
    si un campo de regla es feature del modelo) y registra el reporte en el JSON.

    Args:
        scores: score de anomalía del modelo (mayor = más anómalo).
        proxies: mapa tipo -> etiqueta binaria (de :func:`build_proxies`).
        feature_names: features del modelo (para la verificación de disyunción).
        comparator_scores: opcional, ``{isolation_forest, lof, ocsvm}`` para HE4.
        seed_scores: opcional, ``{42: scores, 43: ..., 44: ...}`` para multiseed.
        n_bootstrap: iteraciones de bootstrap para los IC (reducir en tests).
        user_ids: opcional, para bootstrap agrupado por usuario.
        split: nombre del split (metadato).

    Returns:
        Dict con claves ``proxies``, ``global``, ``multiseed``, ``comparators``,
        ``disjointness`` y ``meta``.
    """
    scores = np.asarray(scores, dtype=np.float64)

    # 1. Disyunción feature↔proxy — invariante de no-circularidad (falla ruidoso).
    disjoint_report = rules.assert_disjoint_from_features(feature_names)

    # 2. Métricas por tipo + control negativo.
    proxy_blocks: Dict[str, Dict[str, float]] = {}
    for label, y in proxies.items():
        proxy_blocks[label] = evaluate_proxy(scores, y, n_bootstrap, user_ids)

    # 3. Global (HE1) sobre la unión tipificada.
    union = proxies[UNION_KEY]
    global_block = (
        evaluate_global(scores, union)
        if 0 < float(union.mean()) < 1
        else {"mwu_u": None, "mwu_p": None, "r_rb": None}
    )

    scoreboard: Dict[str, object] = {
        "meta": {
            "split": split,
            "n_rows": int(len(scores)),
            "n_bootstrap": int(n_bootstrap),
            "feature_set_size": len(list(feature_names)),
            "clustered_bootstrap": user_ids is not None,
        },
        "disjointness": disjoint_report,
        "proxies": proxy_blocks,
        "global": global_block,
        "multiseed": {},
        "comparators": {},
    }

    # 4. Comparadores (HE4) sobre la unión tipificada.
    if comparator_scores:
        scoreboard["comparators"] = evaluate_comparators(union, comparator_scores)

    # 5. Multiseed.
    if seed_scores:
        scoreboard["multiseed"] = evaluate_multiseed(seed_scores, proxies)

    return scoreboard


# ---------------------------------------------------------------------------
# CLI (carga parquet de scores + proxies) — NO ejecutar contra datos reales aquí
# ---------------------------------------------------------------------------

_FEATURE_SETS = {"frame-v1": list(FRAME_V1_FEATURE_NAMES)}


def _parse_seeds(raw: str) -> List[int]:
    return [int(s.strip()) for s in raw.split(",") if s.strip()]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--scores",
        required=True,
        help="Parquet con columna 'score' + campos externos de reglas (ver docstring).",
    )
    p.add_argument(
        "--features",
        default="frame-v1",
        choices=sorted(_FEATURE_SETS),
        help="Feature set del modelo (para verificar disyunción).",
    )
    p.add_argument(
        "--proxy-set",
        default="typed_v2",
        choices=["typed_v2"],
        help="Conjunto de proxies (variable criterio). Solo typed_v2 en V2.",
    )
    p.add_argument(
        "--negative-control",
        default="refund",
        choices=["refund"],
        help="Control negativo (HE3).",
    )
    p.add_argument("--split", default="test")
    p.add_argument("--seeds", default="42,43,44", help="Semillas para multiseed.")
    p.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP)
    p.add_argument(
        "--output",
        default=str(ROOT / "output" / "revision" / "v2_confirmatory_scoreboard.json"),
    )
    return p.parse_args(argv)


def _load_from_parquet(path: Path, seeds: List[int]) -> Dict[str, object]:
    """Carga scores, comparadores y scores por semilla desde el parquet de entrada."""
    df = pd.read_parquet(path)
    if "score" not in df.columns:
        raise ValueError(f"El parquet {path} no tiene columna 'score'.")

    proxies = build_proxies(df)
    scores = df["score"].to_numpy(dtype=np.float64)

    comparator_scores: Dict[str, np.ndarray] = {}
    col_map = {"score": "isolation_forest", "score_lof": "lof", "score_ocsvm": "ocsvm"}
    for col, model in col_map.items():
        if col in df.columns:
            comparator_scores[model] = df[col].to_numpy(dtype=np.float64)
    # HE4 solo se emite si están los tres modelos.
    if not {"lof", "ocsvm"}.issubset(comparator_scores):
        comparator_scores = {}

    seed_scores: Dict[int, np.ndarray] = {}
    for seed in seeds:
        col = f"score_seed{seed}"
        if col in df.columns:
            seed_scores[seed] = df[col].to_numpy(dtype=np.float64)

    user_ids = df["user_id"].to_numpy() if "user_id" in df.columns else None
    return {
        "scores": scores,
        "proxies": proxies,
        "comparator_scores": comparator_scores or None,
        "seed_scores": seed_scores or None,
        "user_ids": user_ids,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    seeds = _parse_seeds(args.seeds)
    loaded = _load_from_parquet(Path(args.scores), seeds)

    scoreboard = build_scoreboard(
        loaded["scores"],
        loaded["proxies"],
        _FEATURE_SETS[args.features],
        comparator_scores=loaded["comparator_scores"],
        seed_scores=loaded["seed_scores"],
        n_bootstrap=args.bootstrap,
        user_ids=loaded["user_ids"],
        split=args.split,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(scoreboard, indent=2, ensure_ascii=False))
    print(json.dumps({"output": str(out_path), "meta": scoreboard["meta"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

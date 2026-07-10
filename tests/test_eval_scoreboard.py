"""Tests para el juez automático del confirmatorio V2 (scripts/eval_scoreboard.py).

Verifica, con arrays sintéticos de señal conocida:
  - EF@k / P@k / AUC correctos.
  - El scoreboard tiene el esquema EXACTO que latex_tables.generate_v2_tables espera.
  - generate_v2_tables(scoreboard) produce las 6 tablas sin error (import cruzado).
  - MWU + rank-biserial, bootstrap con pocas iteraciones, verificación de disyunción.

Bootstrap con pocas iteraciones (n_bootstrap pequeño) para velocidad.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import eval_scoreboard as esb  # noqa: E402

from fraud_detector.reporting import latex_tables as tbl  # noqa: E402
from fraud_detector.scoring.features_frame_v1 import FRAME_V1_FEATURE_NAMES  # noqa: E402

V2_TABLE_FILENAMES = [
    "table_v2_he1_mwu.tex",
    "table_v2_he2_ef_by_type.tex",
    "table_v2_he3_negative_control.tex",
    "table_v2_he4_comparison.tex",
    "table_v2_secondary_metrics.tex",
    "table_v2_multiseed_stability.tex",
]


# ---------------------------------------------------------------------------
# Fixtures sintéticos con señal conocida
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_df():
    """DataFrame donde las primeras filas son 'anómalas' con señal fuerte.

    Construcción: n=1000. Los primeros 20 registros (2%) disparan card_testing
    y typed_union, y tienen los scores más altos -> EF alto y determinista.
    """
    n = 1000
    df = pd.DataFrame(
        {
            "same_amount_count_1h": np.zeros(n, dtype=int),
            "failed_count_1h": np.zeros(n, dtype=int),
            "user_txn_count_24h": np.full(n, 5, dtype=int),
            "user_account_age_days": np.full(n, 400, dtype=int),
            "user_txn_count_1h": np.zeros(n, dtype=int),
            "status": np.array(["captured"] * n, dtype=object),
        }
    )
    # 20 card_testing positivos (2%).
    df.loc[:19, "same_amount_count_1h"] = 6
    # 10 velocity_extreme positivos, disjuntos (índices 500-509).
    df.loc[500:509, "user_txn_count_24h"] = 150
    # refund control negativo: 60 filas (~6%), esparcidas.
    df.loc[900:959, "status"] = "totally_refunded"

    # Scores: monotónicos descendentes -> los card_testing (índices 0-19) tienen
    # los scores más altos, así el top-1% (10 filas) es 100% card_testing.
    scores = np.linspace(1.0, 0.0, n)
    return df, scores


@pytest.fixture
def scoreboard(synthetic_df):
    df, scores = synthetic_df
    proxies = esb.build_proxies(df)
    comparators = {
        "isolation_forest": scores,
        "lof": scores * 0.9,
        "ocsvm": scores * 0.8,
    }
    seed_scores = {42: scores, 43: scores * 1.01, 44: scores * 0.99}
    return esb.build_scoreboard(
        scores,
        proxies,
        FRAME_V1_FEATURE_NAMES,
        comparator_scores=comparators,
        seed_scores=seed_scores,
        n_bootstrap=50,
    )


# ---------------------------------------------------------------------------
# Cálculo de EF@k / P@k
# ---------------------------------------------------------------------------


def test_ef_at_1pct_matches_known_signal(synthetic_df):
    df, scores = synthetic_df
    y = esb.build_proxies(df)["card_testing"]
    # base_rate = 20/1000 = 0.02. top-1% = 10 filas, todas card_testing -> P@1%=1.0.
    # EF@1% = 1.0 / 0.02 = 50.
    ef1 = esb._ef_at(y, scores, 0.01)
    assert ef1 == pytest.approx(50.0, rel=1e-6)


def test_precision_at_1pct(synthetic_df):
    df, scores = synthetic_df
    block = esb.evaluate_proxy(scores, esb.build_proxies(df)["card_testing"], n_bootstrap=20)
    assert block["precision_at_1pct"] == pytest.approx(1.0)


def test_build_proxies_uses_rules_thresholds(synthetic_df):
    df, _ = synthetic_df
    proxies = esb.build_proxies(df)
    assert proxies["card_testing"].sum() == 20
    assert proxies["velocity_extreme"].sum() == 10
    assert proxies["refund_negative_control"].sum() == 60
    # typed_union = OR (card_testing y velocity_extreme son disjuntos aquí).
    assert proxies["typed_union"].sum() == 30


# ---------------------------------------------------------------------------
# Esquema del scoreboard
# ---------------------------------------------------------------------------


def test_scoreboard_top_level_schema(scoreboard):
    for key in ("proxies", "global", "multiseed", "comparators", "disjointness", "meta"):
        assert key in scoreboard


def test_proxy_block_has_all_required_keys(scoreboard):
    required = {
        "base_rate",
        "auc",
        "ap",
        "ef_at_1pct",
        "ef_at_1pct_ci_low",
        "ef_at_1pct_ci_high",
        "ef_at_5pct",
        "ef_at_5pct_ci_low",
        "ef_at_5pct_ci_high",
        "precision_at_1pct",
    }
    for t in esb.TYPED_PROXIES + [esb.UNION_KEY, esb.REFUND_KEY]:
        assert t in scoreboard["proxies"], f"falta proxy {t}"
        assert required.issubset(scoreboard["proxies"][t]), f"claves faltantes en {t}"


def test_global_block_schema(scoreboard):
    g = scoreboard["global"]
    assert set(g) >= {"mwu_u", "mwu_p", "r_rb"}
    # Señal fuerte -> r_rb positivo y p pequeño.
    assert g["r_rb"] > 0
    assert g["mwu_p"] < 0.05


def test_comparators_schema(scoreboard):
    comp = scoreboard["comparators"]
    assert set(comp) == {"isolation_forest", "lof", "ocsvm"}
    for m in comp.values():
        assert set(m) == {"ef_at_1pct", "ef_at_5pct", "ap", "precision_at_1pct"}


def test_multiseed_schema_and_gating(scoreboard):
    ms = scoreboard["multiseed"]
    assert "card_testing" in ms
    assert set(ms["card_testing"]) >= {"ef1_relative_range", "gated"}
    assert ms["card_testing"]["gated"] is True
    # new_user_burst está fuera del gate de estabilidad.
    if "new_user_burst" in ms:
        assert ms["new_user_burst"]["gated"] is False


def test_disjointness_recorded(scoreboard):
    dj = scoreboard["disjointness"]
    assert dj["disjoint"] is True
    assert dj["overlap"] == []


def test_bootstrap_ci_brackets_point_estimate(scoreboard):
    ct = scoreboard["proxies"]["card_testing"]
    assert ct["ef_at_1pct_ci_low"] <= ct["ef_at_1pct"] <= ct["ef_at_1pct_ci_high"]


# ---------------------------------------------------------------------------
# Disyunción — falla ruidoso si un campo de regla es feature del modelo
# ---------------------------------------------------------------------------


def test_build_scoreboard_raises_on_circular_features(synthetic_df):
    df, scores = synthetic_df
    proxies = esb.build_proxies(df)
    poisoned = list(FRAME_V1_FEATURE_NAMES) + ["user_txn_count_24h"]
    with pytest.raises(ValueError, match="Disyunción"):
        esb.build_scoreboard(scores, proxies, poisoned, n_bootstrap=10)


# ---------------------------------------------------------------------------
# Import cruzado: generate_v2_tables consume el scoreboard sin error
# ---------------------------------------------------------------------------


def test_generate_v2_tables_consumes_scoreboard(scoreboard, tmp_path):
    written = tbl.generate_v2_tables(scoreboard, tmp_path)
    assert len(written) == len(V2_TABLE_FILENAMES)
    for name in V2_TABLE_FILENAMES:
        path = tmp_path / name
        assert path.exists(), f"falta {name}"
        content = path.read_text(encoding="utf-8")
        assert r"\begin{tabular}" in content
        assert r"\toprule" in content


def test_he2_type_passes_helper_on_scoreboard(scoreboard):
    # El helper de latex_tables debe poder leer los bloques que produjimos.
    ct = scoreboard["proxies"]["card_testing"]
    verdict = tbl.he2_type_passes(ct)
    assert verdict is True  # EF@1%=50, CI low >1 con señal fuerte


def test_empty_proxy_yields_none_metrics():
    # Un proxy sin positivos no rompe: métricas None (celdas --- en la tabla).
    scores = np.linspace(1, 0, 100)
    y = np.zeros(100, dtype=np.int8)
    block = esb.evaluate_proxy(scores, y, n_bootstrap=10)
    assert block["base_rate"] == 0.0
    assert block["ef_at_1pct"] is None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_end_to_end_with_parquet(synthetic_df, tmp_path):
    df, scores = synthetic_df
    df = df.copy()
    df["score"] = scores
    df["score_lof"] = scores * 0.9
    df["score_ocsvm"] = scores * 0.8
    df["score_seed42"] = scores
    df["score_seed43"] = scores * 1.01
    df["score_seed44"] = scores * 0.99
    parquet = tmp_path / "scores.parquet"
    df.to_parquet(parquet)
    out = tmp_path / "sb.json"

    rc = esb.main(
        [
            "--scores",
            str(parquet),
            "--features",
            "frame-v1",
            "--proxy-set",
            "typed_v2",
            "--negative-control",
            "refund",
            "--split",
            "test",
            "--seeds",
            "42,43,44",
            "--bootstrap",
            "20",
            "--output",
            str(out),
        ]
    )
    assert rc == 0
    assert out.exists()
    import json

    sb = json.loads(out.read_text())
    assert sb["comparators"]  # HE4 presente (los 3 modelos)
    assert sb["multiseed"]  # multiseed presente
    assert sb["meta"]["split"] == "test"

"""Tests for V2 confirmatory LaTeX tables (typed proxies + refund negative control).

TDD para Fase 3.4: generadores de tablas del scoreboard confirmatorio V2
(`output/revision/v2_confirmatory_scoreboard.json`, producido por el futuro
`scripts/eval_scoreboard.py`). Gates pre-registrados (RESUMEN-CADENA-
METODOLOGICA-V2-PIVOTE.md, seccion 7):

- HE1: Mann-Whitney U con p < 0,05 Y r_rb > 0,10 sobre la union tipificada.
- HE2: EF@1% >= 2 en >= 2 de 3 tipos no circulares, con IC95 cuyo limite
  inferior sea > 1 (estricto) en los tipos que satisfacen el gate.
- HE3 (control negativo): EF@1% refund en [0,8; 1,3] Y AUC en [0,45; 0,55].
- HE4: IF >= LOF y OC-SVM en >= 3 de 4 metricas (EF@1%, EF@5%, AP, P@1%).
- Multiseed: rango relativo EF@1% < 15% en tipos con gate.
"""

from __future__ import annotations

import copy
import re

import pytest

from fraud_detector.reporting import latex_tables as tbl

V2_TABLE_FILENAMES = [
    "table_v2_he1_mwu.tex",
    "table_v2_he2_ef_by_type.tex",
    "table_v2_he3_negative_control.tex",
    "table_v2_he4_comparison.tex",
    "table_v2_secondary_metrics.tex",
    "table_v2_multiseed_stability.tex",
]


@pytest.fixture
def scoreboard():
    """Scoreboard sintetico V2 donde HE1, HE2, HE3 y HE4 se respaldan."""

    def proxy(base_rate, auc, ap, ef1, ef1_lo, ef1_hi, ef5, ef5_lo, ef5_hi, p1):
        return {
            "base_rate": base_rate,
            "auc": auc,
            "ap": ap,
            "ef_at_1pct": ef1,
            "ef_at_1pct_ci_low": ef1_lo,
            "ef_at_1pct_ci_high": ef1_hi,
            "ef_at_5pct": ef5,
            "ef_at_5pct_ci_low": ef5_lo,
            "ef_at_5pct_ci_high": ef5_hi,
            "precision_at_1pct": p1,
        }

    return {
        "proxies": {
            "card_testing": proxy(0.0102, 0.621, 0.052, 5.08, 4.51, 5.72, 3.10, 2.80, 3.42, 0.052),
            "velocity_extreme": proxy(
                0.0034, 0.615, 0.031, 5.05, 4.42, 5.66, 3.05, 2.70, 3.40, 0.017
            ),
            "new_user_burst": proxy(
                0.0012, 0.608, 0.004, 1.36, 0.81, 2.10, 1.20, 0.90, 1.55, 0.002
            ),
            "typed_union": proxy(0.0140, 0.617, 0.068, 2.40, 2.10, 2.72, 2.00, 1.80, 2.22, 0.034),
            "refund_negative_control": proxy(
                0.0633, 0.498, 0.063, 1.19, 1.05, 1.32, 1.02, 0.95, 1.10, 0.075
            ),
        },
        "global": {"mwu_u": 1234567890.0, "mwu_p": 1.2e-25, "r_rb": 0.184},
        "multiseed": {
            "card_testing": {"ef1_relative_range": 0.020, "gated": True},
            "velocity_extreme": {"ef1_relative_range": 0.133, "gated": True},
            "new_user_burst": {"ef1_relative_range": 0.500, "gated": False},
            "typed_union": {"ef1_relative_range": 0.054, "gated": True},
        },
        "comparators": {
            "isolation_forest": {
                "ef_at_1pct": 2.40,
                "ef_at_5pct": 2.00,
                "ap": 0.068,
                "precision_at_1pct": 0.034,
            },
            "lof": {
                "ef_at_1pct": 1.80,
                "ef_at_5pct": 1.60,
                "ap": 0.050,
                "precision_at_1pct": 0.025,
            },
            "ocsvm": {
                "ef_at_1pct": 2.10,
                "ef_at_5pct": 1.90,
                "ap": 0.060,
                "precision_at_1pct": 0.029,
            },
        },
    }


def _assert_tabular(tex: str):
    assert r"\begin{table}" in tex
    assert r"\end{table}" in tex
    assert r"\begin{tabular}" in tex
    assert r"\end{tabular}" in tex
    assert r"\toprule" in tex
    assert r"\midrule" in tex
    assert r"\bottomrule" in tex


# --- HE1 ---


def test_he1_structure_and_verdict(scoreboard):
    tex = tbl.table_v2_he1_mwu(scoreboard)
    _assert_tabular(tex)
    assert "confirmatorio V2" in tex
    assert "0,184" in tex  # r_rb con coma decimal
    assert "Respaldada" in tex


def test_he1_rrb_at_gate_boundary_fails(scoreboard):
    sb = copy.deepcopy(scoreboard)
    sb["global"]["r_rb"] = 0.10  # gate exige r_rb > 0,10 estricto
    tex = tbl.table_v2_he1_mwu(sb)
    assert "No respaldada" in tex


def test_he1_p_above_alpha_fails(scoreboard):
    sb = copy.deepcopy(scoreboard)
    sb["global"]["mwu_p"] = 0.07
    tex = tbl.table_v2_he1_mwu(sb)
    assert "No respaldada" in tex


# --- HE2 ---


def test_he2_structure_and_pass(scoreboard):
    tex = tbl.table_v2_he2_ef_by_type(scoreboard)
    _assert_tabular(tex)
    assert "confirmatorio V2" in tex
    assert "5,080" in tex  # EF card_testing con coma decimal
    assert "Respaldada" in tex


def test_he2_ef_exactly_2_passes(scoreboard):
    """EF@1% = 2,0 exacto satisface el gate (>= 2)."""
    sb = copy.deepcopy(scoreboard)
    for t in ["card_testing", "velocity_extreme"]:
        sb["proxies"][t]["ef_at_1pct"] = 2.0
        sb["proxies"][t]["ef_at_1pct_ci_low"] = 1.5
        sb["proxies"][t]["ef_at_1pct_ci_high"] = 2.5
    tex = tbl.table_v2_he2_ef_by_type(sb)
    assert "No respaldada" not in tex
    assert "Respaldada" in tex


def test_he2_ci_low_exactly_1_fails(scoreboard):
    """LI = 1,0 exacto NO pasa (gate exige LI > 1 estricto)."""
    sb = copy.deepcopy(scoreboard)
    sb["proxies"]["velocity_extreme"]["ef_at_1pct_ci_low"] = 1.0
    # Solo card_testing queda cumpliendo ambas condiciones -> 1 de 3 < 2
    tex = tbl.table_v2_he2_ef_by_type(sb)
    assert "No respaldada" in tex


def test_he2_only_one_type_above_2_fails(scoreboard):
    sb = copy.deepcopy(scoreboard)
    sb["proxies"]["velocity_extreme"]["ef_at_1pct"] = 1.5
    tex = tbl.table_v2_he2_ef_by_type(sb)
    assert "No respaldada" in tex


# --- HE3 (control negativo) ---


def test_he3_within_bands_supported(scoreboard):
    tex = tbl.table_v2_he3_negative_control(scoreboard)
    _assert_tabular(tex)
    assert "confirmatorio V2" in tex
    assert "1,190" in tex  # EF refund con coma decimal
    assert "Respaldada" in tex


def test_he3_ef_out_of_band_fails(scoreboard):
    sb = copy.deepcopy(scoreboard)
    sb["proxies"]["refund_negative_control"]["ef_at_1pct"] = 1.50  # > 1,3
    tex = tbl.table_v2_he3_negative_control(sb)
    assert "No respaldada" in tex


def test_he3_auc_out_of_band_fails(scoreboard):
    sb = copy.deepcopy(scoreboard)
    sb["proxies"]["refund_negative_control"]["auc"] = 0.60  # fuera de [0,45; 0,55]
    tex = tbl.table_v2_he3_negative_control(sb)
    assert "No respaldada" in tex


def test_he3_band_edges_inclusive(scoreboard):
    sb = copy.deepcopy(scoreboard)
    sb["proxies"]["refund_negative_control"]["ef_at_1pct"] = 1.3
    sb["proxies"]["refund_negative_control"]["auc"] = 0.55
    tex = tbl.table_v2_he3_negative_control(sb)
    assert "No respaldada" not in tex


# --- HE4 ---


def test_he4_structure_and_pass(scoreboard):
    tex = tbl.table_v2_he4_comparison(scoreboard)
    _assert_tabular(tex)
    assert "confirmatorio V2" in tex
    assert "LOF" in tex and "OC-SVM" in tex
    assert r"\textbf" in tex  # metricas donde IF gana en negrita
    assert "4/4" in tex
    assert "Respaldada" in tex


def test_he4_if_loses_two_metrics_fails(scoreboard):
    sb = copy.deepcopy(scoreboard)
    sb["comparators"]["lof"]["ap"] = 0.90
    sb["comparators"]["lof"]["ef_at_1pct"] = 9.0
    tex = tbl.table_v2_he4_comparison(sb)
    assert "2/4" in tex
    assert "No respaldada" in tex


def test_he4_tie_counts_as_win(scoreboard):
    """HE4 es 'comparable o superior': empate exacto cuenta para IF."""
    sb = copy.deepcopy(scoreboard)
    sb["comparators"]["lof"]["ap"] = sb["comparators"]["isolation_forest"]["ap"]
    tex = tbl.table_v2_he4_comparison(sb)
    assert "4/4" in tex


# --- Secundarias ---


def test_secondary_metrics_table(scoreboard):
    tex = tbl.table_v2_secondary_metrics(scoreboard)
    _assert_tabular(tex)
    assert "confirmatorio V2" in tex
    assert "sin gate" in tex.lower()
    assert "0,621" in tex  # AUC card_testing con coma decimal
    # No emite veredicto de hipotesis
    assert "Respaldada" not in tex


# --- Multiseed ---


def test_multiseed_stability_table(scoreboard):
    tex = tbl.table_v2_multiseed_stability(scoreboard)
    _assert_tabular(tex)
    assert "confirmatorio V2" in tex
    assert "15" in tex  # umbral 15%
    assert "2,0" in tex  # card_testing 2,0%
    assert "Sí" in tex


def test_multiseed_above_threshold_fails(scoreboard):
    sb = copy.deepcopy(scoreboard)
    sb["multiseed"]["card_testing"]["ef1_relative_range"] = 0.20
    tex = tbl.table_v2_multiseed_stability(sb)
    assert "No" in tex


def test_multiseed_ungated_type_shows_no_verdict(scoreboard):
    tex = tbl.table_v2_multiseed_stability(scoreboard)
    # new_user_burst no tiene gate (varianza estructural): se reporta sin veredicto
    assert "---" in tex or "sin gate" in tex.lower()


# --- Formato ---


def test_comma_decimal_everywhere(scoreboard):
    """Ninguna celda numerica debe usar punto decimal."""
    for fn in [
        tbl.table_v2_he1_mwu,
        tbl.table_v2_he2_ef_by_type,
        tbl.table_v2_he3_negative_control,
        tbl.table_v2_he4_comparison,
        tbl.table_v2_secondary_metrics,
        tbl.table_v2_multiseed_stability,
    ]:
        tex = fn(scoreboard)
        body = tex.split(r"\midrule")[1].split(r"\bottomrule")[0]
        assert not re.search(r"\d\.\d", body), f"punto decimal en {fn.__name__}: {body}"


def test_labels_are_unique(scoreboard):
    labels = set()
    for fn in [
        tbl.table_v2_he1_mwu,
        tbl.table_v2_he2_ef_by_type,
        tbl.table_v2_he3_negative_control,
        tbl.table_v2_he4_comparison,
        tbl.table_v2_secondary_metrics,
        tbl.table_v2_multiseed_stability,
    ]:
        m = re.search(r"\\label\{([^}]+)\}", fn(scoreboard))
        assert m, f"sin label en {fn.__name__}"
        labels.add(m.group(1))
    assert len(labels) == 6


# --- Orquestador ---


def test_generate_v2_tables_writes_files(scoreboard, tmp_path):
    written = tbl.generate_v2_tables(scoreboard, tmp_path)
    for name in V2_TABLE_FILENAMES:
        assert (tmp_path / name).exists(), f"falta {name}"
        content = (tmp_path / name).read_text(encoding="utf-8")
        assert r"\begin{tabular}" in content
    assert len(written) == len(V2_TABLE_FILENAMES)


def test_generate_v2_tables_missing_sections_graceful(tmp_path):
    """Un scoreboard incompleto no debe lanzar excepcion (celdas ---)."""
    written = tbl.generate_v2_tables({"proxies": {}}, tmp_path)
    assert len(written) == len(V2_TABLE_FILENAMES)
    for name in V2_TABLE_FILENAMES:
        assert (tmp_path / name).exists()

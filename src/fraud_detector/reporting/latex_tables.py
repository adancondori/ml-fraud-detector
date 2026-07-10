"""LaTeX table generator for thesis Cap 3. Booktabs format, APA 7 style.

Contiene dos familias de generadores:

1. **V1 (diagnostico)**: funciones ``table_*`` originales contra el proxy de
   reembolso. Se conservan intactas como material de la fase diagnostica.

2. **V2 (confirmatorio)**: funciones ``table_v2_*`` contra el scoreboard del
   confirmatorio V2 (variable criterio = union tipificada no circular;
   refund = control negativo). Salida en espanol con coma decimal, archivos
   ``table_v2_*.tex``. Orquestador: :func:`generate_v2_tables`.

Esquema de entrada V2 (``output/revision/v2_confirmatory_scoreboard.json``,
producido por ``scripts/eval_scoreboard.py`` — Fase 5A)::

    {
      "proxies": {
        # claves: card_testing | velocity_extreme | new_user_burst |
        #         typed_union | refund_negative_control
        "<proxy_type>": {
          "base_rate": float,            # tasa base del tipo en test
          "auc": float,                  # AUC-ROC (secundaria, sin gate)
          "ap": float,                   # Average Precision (secundaria)
          "ef_at_1pct": float,           # EF@1% (metrica titular)
          "ef_at_1pct_ci_low": float,    # IC bootstrap 95% (limite inferior)
          "ef_at_1pct_ci_high": float,   # IC bootstrap 95% (limite superior)
          "ef_at_5pct": float,
          "ef_at_5pct_ci_low": float,
          "ef_at_5pct_ci_high": float,
          "precision_at_1pct": float
        }, ...
      },
      "global": {                        # sobre la union tipificada
        "mwu_u": float,                  # estadistico U de Mann-Whitney
        "mwu_p": float,                  # p-value
        "r_rb": float                    # rank-biserial
      },
      "multiseed": {                     # semillas 42/43/44
        "<proxy_type>": {
          "ef1_relative_range": float,   # rango/mediana de EF@1%
          "gated": bool                  # si el tipo participa del gate <15%
        }, ...
      },
      "comparators": {                   # HE4, sobre la union tipificada
        # claves: isolation_forest | lof | ocsvm
        "<model>": {
          "ef_at_1pct": float, "ef_at_5pct": float,
          "ap": float, "precision_at_1pct": float
        }, ...
      }
    }

Gates pre-registrados (RESUMEN-CADENA-METODOLOGICA-V2-PIVOTE.md, seccion 7),
congelados ANTES de la corrida confirmatoria:

- HE1: p < 0,05 Y r_rb > 0,10 (estricto) sobre la union tipificada.
- HE2: EF@1% >= 2 en al menos 2 de los 3 tipos no circulares, con IC95 cuyo
  limite inferior sea > 1 (estricto) en los tipos que satisfacen el gate.
- HE3 (control negativo): EF@1% refund en [0,8; 1,3] Y AUC en [0,45; 0,55]
  (bandas de equivalencia inclusivas).
- HE4: IF >= LOF y OC-SVM en al menos 3 de 4 metricas (EF@1%, EF@5%, AP,
  P@1%); el empate exacto cuenta a favor de IF ("comparable o superior").
- Multiseed: rango relativo EF@1% < 15% (estricto) en tipos con gate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from fraud_detector.features.engineering import FEATURE_NAMES
from fraud_detector.utils.logger import logger


def _esc(text: str) -> str:
    """Escape LaTeX special characters."""
    for c, r in [("%", r"\%"), ("&", r"\&"), ("_", r"\_"), ("#", r"\#"), ("$", r"\$")]:
        text = str(text).replace(c, r)
    return text


def _fmt(v, decimals=4) -> str:
    if v is None:
        return "---"
    if isinstance(v, bool):
        return "Sí" if v else "No"
    if isinstance(v, float):
        return f"{v:.{decimals}f}"
    return str(v)


def _wrap_table(content: str, caption: str, label: str) -> str:
    return (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        "\\small\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        f"{content}"
        "\\end{table}\n"
    )


def _save(tex: str, path: Optional[Path]) -> str:
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(tex, encoding="utf-8")
        logger.info(f"Table saved: {path}")
    return tex


def table_model_comparison(results: Dict, path: Optional[Path] = None) -> str:
    models = ["isolation_forest", "lof", "ocsvm"]
    names = {"isolation_forest": "Isolation Forest", "lof": "LOF", "ocsvm": "OC-SVM"}
    metrics = [
        ("AUC-ROC", "auc_roc"),
        ("AP", "average_precision"),
        ("P@5\\%", "precision_at_5pct"),
        ("EF@5\\%", "ef_at_5pct"),
    ]
    rows = []
    for label, key in metrics:
        vals = []
        for m in models:
            he2 = results.get(m, {}).get("he2", {})
            he3 = results.get(m, {}).get("he3", {})
            v = he2.get(key) or he3.get(key)
            vals.append(_fmt(v))
        rows.append(f"  {label} & {' & '.join(vals)} \\\\")
    header = " & ".join(names[m] for m in models)
    body = (
        "\\begin{tabular}{l" + "r" * len(models) + "}\n"
        "\\toprule\n"
        f"Métrica & {header} \\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )
    tex = _wrap_table(
        body, "Comparación de modelos en el conjunto de prueba", "tab:model-comparison"
    )
    return _save(tex, path)


def table_he1_results(results: Dict, path: Optional[Path] = None) -> str:
    models = ["isolation_forest", "lof", "ocsvm"]
    names = {"isolation_forest": "IF", "lof": "LOF", "ocsvm": "OC-SVM"}
    rows = []
    for m in models:
        he1 = results.get(m, {}).get("he1", {})
        pass_str = "Sí" if he1.get("he1_pass") else "No"
        rows.append(
            f"  {names[m]} & {_fmt(he1.get('U_statistic'), 0)} & "
            f"{he1.get('p_value', 0):.2e} & {_fmt(he1.get('rank_biserial_r'))} & "
            f"{_fmt(he1.get('cles'))} & {pass_str} \\\\"
        )
    body = (
        "\\begin{tabular}{lrrrrc}\n\\toprule\n"
        "Modelo & U & p-value & r & CLES & Pasa \\\\\n\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n"
    )
    tex = _wrap_table(body, "HE1: Separación estadística de scores", "tab:he1-results")
    return _save(tex, path)


def table_he2_results(results: Dict, path: Optional[Path] = None) -> str:
    models = ["isolation_forest", "lof", "ocsvm"]
    names = {"isolation_forest": "IF", "lof": "LOF", "ocsvm": "OC-SVM"}
    rows = []
    for m in models:
        he2 = results.get(m, {}).get("he2", {})
        ci = results.get(m, {}).get("bootstrap_ci_auc", {})
        pass_str = "Sí" if he2.get("he2_pass") else "No"
        rows.append(
            f"  {names[m]} & {_fmt(he2.get('auc_roc'))} & "
            f"[{_fmt(ci.get('lower'))}, {_fmt(ci.get('upper'))}] & "
            f"{_fmt(he2.get('average_precision'))} & "
            f"{pass_str} \\\\"
        )
    body = (
        "\\begin{tabular}{lcccc}\n\\toprule\n"
        "Modelo & AUC-ROC & CI 95\\% & AP & Pasa \\\\\n\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n"
    )
    tex = _wrap_table(body, "HE2: Capacidad discriminativa", "tab:he2-results")
    return _save(tex, path)


def table_he3_results(results: Dict, path: Optional[Path] = None) -> str:
    models = ["isolation_forest", "lof", "ocsvm"]
    names = {"isolation_forest": "IF", "lof": "LOF", "ocsvm": "OC-SVM"}
    rows = []
    for m in models:
        he3 = results.get(m, {}).get("he3", {})
        vals = [_fmt(he3.get(f"ef_at_{k}pct")) for k in [1, 2, 5, 10]]
        rows.append(f"  {names[m]} & " + " & ".join(vals) + " \\\\")
    body = (
        "\\begin{tabular}{lrrrr}\n\\toprule\n"
        "Modelo & EF@1\\% & EF@2\\% & EF@5\\% & EF@10\\% \\\\\n\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n"
    )
    tex = _wrap_table(body, "HE3: Factor de enriquecimiento por percentil", "tab:he3-results")
    return _save(tex, path)


def table_he4_comparison(results: Dict, path: Optional[Path] = None) -> str:
    he4 = results.get("he4", {})
    mc = he4.get("metrics_comparison", {})
    wins = set(he4.get("if_wins_on", []))
    metrics = [
        ("AUC-ROC", "auc_roc"),
        ("AP", "ap"),
        ("P@5\\%", "precision_at_5pct"),
        ("EF@5\\%", "ef_at_5pct"),
    ]
    rows = []
    for label, key in metrics:
        vals = []
        for m in ["isolation_forest", "lof", "ocsvm"]:
            v = _fmt(mc.get(m, {}).get(key))
            if m == "isolation_forest" and key in wins:
                v = f"\\textbf{{{v}}}"
            vals.append(v)
        rows.append(f"  {label} & " + " & ".join(vals) + " \\\\")
    body = (
        "\\begin{tabular}{lrrr}\n\\toprule\n"
        "Métrica & IF & LOF & OC-SVM \\\\\n\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n"
    )
    tex = _wrap_table(
        body, f"HE4: IF gana {he4.get('if_wins', 0)}/4 métricas", "tab:he4-comparison"
    )
    return _save(tex, path)


def table_bootstrap_ci(results: Dict, path: Optional[Path] = None) -> str:
    models = ["isolation_forest", "lof", "ocsvm"]
    names = {"isolation_forest": "IF", "lof": "LOF", "ocsvm": "OC-SVM"}
    rows = []
    for m in models:
        ci_auc = results.get(m, {}).get("bootstrap_ci_auc", {})
        ci_ap = results.get(m, {}).get("bootstrap_ci_ap", {})
        rows.append(
            f"  {names[m]} & {_fmt(ci_auc.get('mean'))} & [{_fmt(ci_auc.get('lower'))}, {_fmt(ci_auc.get('upper'))}] & "
            f"{_fmt(ci_ap.get('mean'))} & [{_fmt(ci_ap.get('lower'))}, {_fmt(ci_ap.get('upper'))}] \\\\"
        )
    body = (
        "\\begin{tabular}{lcccc}\n\\toprule\n"
        "Modelo & AUC media & CI 95\\% AUC & AP media & CI 95\\% AP \\\\\n\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n"
    )
    tex = _wrap_table(
        body, "Intervalos de confianza bootstrap (1000 iteraciones)", "tab:bootstrap-ci"
    )
    return _save(tex, path)


def table_sensitivity_proxy(sens: Dict, path: Optional[Path] = None) -> str:
    ps = sens.get("proxy_sensitivity", {})
    rows = []
    for proxy in ["unified", "tipo_a", "wide"]:
        d = ps.get(proxy, {})
        rows.append(
            f"  {_esc(proxy)} & {_fmt(d.get('auc_roc'))} & {_fmt(d.get('ap'))} & {_fmt(d.get('base_rate'), 4)} \\\\"
        )
    rows.append("\\midrule")
    rows.append(
        f"  $\\Delta$ AUC (tipo\\_a) & \\multicolumn{{3}}{{c}}{{{_fmt(ps.get('delta_auc_tipo_a'))}}} \\\\"
    )
    rows.append(
        f"  Robusto ($\\Delta < 0.05$) & \\multicolumn{{3}}{{c}}{{{_fmt(ps.get('robust'))}}} \\\\"
    )
    body = (
        "\\begin{tabular}{lrrr}\n\\toprule\n"
        "Proxy & AUC-ROC & AP & Tasa base \\\\\n\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n"
    )
    tex = _wrap_table(
        body, "Sensibilidad del proxy: unificado vs. Tipo A vs. amplio", "tab:sensitivity-proxy"
    )
    return _save(tex, path)


def table_sensitivity_per_type(sens: Dict, path: Optional[Path] = None) -> str:
    pt = sens.get("per_type_metrics", {})
    rows = []
    for tipo in ["tipo_a", "tipo_b", "tipo_c", "tipo_d", "tipo_e"]:
        d = pt.get(tipo, {})
        auc = _fmt(d.get("auc_roc")) if d.get("auc_roc") is not None else "---"
        ap = _fmt(d.get("ap")) if d.get("ap") is not None else "---"
        ef = _fmt(d.get("ef_at_5pct")) if d.get("ef_at_5pct") is not None else "---"
        rows.append(
            f"  {_esc(tipo)} & {d.get('count', 0):,} & {_fmt(d.get('rate'), 4)} & {auc} & {ap} & {ef} \\\\"
        )
    body = (
        "\\begin{tabular}{lrrrrr}\n\\toprule\n"
        "Tipo & N & Tasa & AUC & AP & EF@5\\% \\\\\n\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n"
    )
    tex = _wrap_table(body, "Métricas desagregadas por tipo de proxy", "tab:sensitivity-per-type")
    return _save(tex, path)


def table_sensitivity_feature18(sens: Dict, path: Optional[Path] = None) -> str:
    f18 = sens.get("feature18_sensitivity", {})
    rows = [
        f"  AUC (31 features) & {_fmt(f18.get('auc_31_features'))} \\\\",
        f"  AUC (30 features, sin F18) & {_fmt(f18.get('auc_30_features'))} \\\\",
        f"  $\\Delta$ AUC & {_fmt(f18.get('delta_auc'))} \\\\",
        f"  Baja sensibilidad ($\\Delta < 0.02$) & {_fmt(f18.get('low_sensitivity'))} \\\\",
        f"  Jaccard top-5\\% & {_fmt(f18.get('jaccard_top5pct'))} \\\\",
        f"  Spearman $\\rho$ & {_fmt(f18.get('spearman_r'))} \\\\",
    ]
    body = (
        "\\begin{tabular}{lr}\n\\toprule\n"
        "Métrica & Valor \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
    )
    tex = _wrap_table(
        body, "Sensibilidad de Feature \\#18 (user\\_reversal\\_ratio\\_30d)", "tab:sensitivity-f18"
    )
    return _save(tex, path)


def table_ablation_31vs21(sens: Dict, path: Optional[Path] = None) -> str:
    abl = sens.get("ablation_31_vs_21", {})
    m31 = abl.get("model_31", {})
    m21 = abl.get("model_21", {})
    delta = abl.get("delta", {})
    metrics = [
        ("AUC-ROC", "auc_roc"),
        ("AP", "ap"),
        ("P@5\\%", "precision_at_5pct"),
        ("EF@5\\%", "enrichment_factor"),
    ]
    rows = []
    for label, key in metrics:
        rows.append(
            f"  {label} & {_fmt(m31.get(key))} & {_fmt(m21.get(key))} & {_fmt(delta.get(key))} \\\\"
        )
    body = (
        "\\begin{tabular}{lrrr}\n\\toprule\n"
        "Métrica & IF-31 & IF-21 & $\\Delta$ \\\\\n\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n"
    )
    tex = _wrap_table(body, "Ablación: IF-31 vs. IF-21 (grupos F, G, H)", "tab:ablation-31vs21")
    return _save(tex, path)


def table_temporal_stability(results: Dict, path: Optional[Path] = None) -> str:
    rows = []
    for m in ["isolation_forest", "lof", "ocsvm"]:
        names = {"isolation_forest": "IF", "lof": "LOF", "ocsvm": "OC-SVM"}
        ts = results.get(m, {}).get("temporal_stability", {}).get("monthly_auc", {})
        vals = [
            _fmt(ts.get(month, {}).get("auc_roc"))
            for month in ["2025-09", "2025-10", "2025-11", "2025-12"]
        ]
        rows.append(f"  {names[m]} & " + " & ".join(vals) + " \\\\")
    body = (
        "\\begin{tabular}{lrrrr}\n\\toprule\n"
        "Modelo & Sep & Oct & Nov & Dic \\\\\n\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n"
    )
    tex = _wrap_table(
        body, "Estabilidad temporal: AUC-ROC mensual (test set)", "tab:temporal-stability"
    )
    return _save(tex, path)


def table_hypothesis_summary(results: Dict, path: Optional[Path] = None) -> str:
    he1 = results.get("isolation_forest", {}).get("he1", {})
    he2 = results.get("isolation_forest", {}).get("he2", {})
    he3 = results.get("isolation_forest", {}).get("he3", {})
    he4 = results.get("he4", {})
    hypotheses = [
        (
            "HE1",
            "Separación estadística",
            f"r = {_fmt(he1.get('rank_biserial_r'))}",
            he1.get("he1_pass"),
        ),
        (
            "HE2",
            "Capacidad discriminativa",
            f"AUC = {_fmt(he2.get('auc_roc'))}",
            he2.get("he2_pass"),
        ),
        (
            "HE3",
            "Concentración top-K",
            f"EF@5\\% = {_fmt(he3.get('ef_at_5pct'))}",
            he3.get("he3_pass"),
        ),
        (
            "HE4",
            "IF $\\geq$ competidores",
            f"{he4.get('if_wins', 0)}/4 métricas",
            he4.get("he4_pass"),
        ),
    ]
    rows = []
    for name, desc, evidence, passed in hypotheses:
        verdict = "Respaldada" if passed else "No respaldada"
        row = f"  {name} & {desc} & {evidence} & {verdict}" + " \\\\"
        rows.append(row)
    body = (
        "\\begin{tabular}{llll}\n\\toprule\n"
        "Hipótesis & Descripción & Evidencia & Veredicto \\\\\n\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n"
    )
    tex = _wrap_table(body, "Resumen de validación de hipótesis", "tab:hypothesis-summary")
    return _save(tex, path)


def table_metrics_by_segment(
    sens: Dict, segment_key: str, caption: str, label: str, path: Optional[Path] = None
) -> str:
    segs = sens.get("segment_metrics", {}).get(segment_key, {})
    metrics = [
        ("AUC", "auc_roc"),
        ("AP", "ap"),
        ("P@5\\%", "precision_at_5pct"),
        ("EF@5\\%", "enrichment_factor"),
    ]
    rows = []
    for seg_name, d in sorted(segs.items()):
        vals = [_fmt(d.get(k)) for _, k in metrics]
        n = f"{d.get('n_transactions', 0):,}"
        rows.append(f"  {_esc(seg_name)} & {n} & " + " & ".join(vals) + " \\\\")
    body = (
        "\\begin{tabular}{lrrrrr}\n\\toprule\n"
        "Segmento & N & "
        + " & ".join(l for l, _ in metrics)
        + " \\\\\n\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n"
    )
    tex = _wrap_table(body, caption, label)
    return _save(tex, path)


def table_anomaly_types(sens: Dict, path: Optional[Path] = None) -> str:
    typo = sens.get("anomaly_typology", {}).get("type_distribution", {})
    rows = []
    for t in [
        "amount",
        "velocity",
        "discount",
        "temporal",
        "credit_flow",
        "role_deviation",
        "diversity",
        "reversal",
        "mixed",
    ]:
        d = typo.get(t, {})
        rows.append(f"  {_esc(t)} & {d.get('count', 0):,} & {_fmt(d.get('pct'), 2)}\\% \\\\")
    body = (
        "\\begin{tabular}{lrr}\n\\toprule\n"
        "Tipo & N & \\% \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
    )
    tex = _wrap_table(body, "Tipología de anomalías (top-5\\%, SHAP)", "tab:anomaly-types")
    return _save(tex, path)


def table_user_risk_profile(sens: Dict, path: Optional[Path] = None) -> str:
    up = sens.get("user_risk_profiles", {})
    rows = [
        f"  Usuarios totales & {up.get('n_users_total', 0):,} \\\\",
        f"  Usuarios flagged ($c > 0.10$) & {up.get('n_users_flagged', 0):,} \\\\",
        f"  \\% flagged & {_fmt(up.get('pct_users_flagged'), 2)}\\% \\\\",
    ]
    summary = up.get("flagged_users_summary", {})
    if summary:
        rows.append(
            f"  Concentración media (flagged) & {_fmt(summary.get('mean_concentration'))} \\\\"
        )
        rows.append(f"  Concentración máxima & {_fmt(summary.get('max_concentration'))} \\\\")
    body = (
        "\\begin{tabular}{lr}\n\\toprule\n"
        "Métrica & Valor \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
    )
    tex = _wrap_table(body, "Perfil de riesgo agregado por usuario", "tab:user-risk-profile")
    return _save(tex, path)


def table_posthoc_facility(posthoc: Dict, path: Optional[Path] = None) -> str:
    ph = posthoc.get("posthoc_analysis", {})
    facs = ph.get("facility_concentration", {}).get("top_10_facilities", [])
    rows = []
    for f in facs[:10]:
        rows.append(
            f"  {f.get('facility_id', '')} & {f.get('n_transactions', 0):,} & "
            f"{_fmt(f.get('anomaly_rate'), 4)} & {_fmt(f.get('anomaly_enrichment'), 2)}x \\\\"
        )
    body = (
        "\\begin{tabular}{lrrr}\n\\toprule\n"
        "Facility & N & Tasa anom. & Enriquecimiento \\\\\n\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n"
    )
    tex = _wrap_table(
        body, "Top 10 centros con mayor concentración de anomalías", "tab:posthoc-facility"
    )
    return _save(tex, path)


def table_posthoc_currency(posthoc: Dict, path: Optional[Path] = None) -> str:
    ph = posthoc.get("posthoc_analysis", {})
    currs = ph.get("currency_concentration", {}).get("currencies_affected", [])
    rows = []
    for c in currs[:15]:
        rows.append(
            f"  {_esc(c.get('currency', ''))} & {c.get('n_transactions', 0):,} & "
            f"{_fmt(c.get('anomaly_rate'), 4)} & {_fmt(c.get('anomaly_enrichment'), 2)}x \\\\"
        )
    body = (
        "\\begin{tabular}{lrrr}\n\\toprule\n"
        "Moneda & N & Tasa anom. & Enriquecimiento \\\\\n\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n"
    )
    tex = _wrap_table(body, "Distribución de anomalías por moneda", "tab:posthoc-currency")
    return _save(tex, path)


def table_posthoc_manager(posthoc: Dict, path: Optional[Path] = None) -> str:
    ph = posthoc.get("posthoc_analysis", {})
    mgr = ph.get("manager_concentration", {})
    agg = mgr.get("aggregate_only", {})
    rows = [
        f"  Txns con intervención manager & {agg.get('n_transactions_with_manager_intervention', 0):,} \\\\",
        f"  Anomalías con manager & {agg.get('n_anomalies_with_manager_intervention', 0):,} \\\\",
        f"  Tasa anomalía (manager) & {_fmt(agg.get('anomaly_rate_with_manager_intervention'), 4)} \\\\",
    ]
    body = (
        "\\begin{tabular}{lr}\n\\toprule\n"
        "Métrica & Valor \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
    )
    tex = _wrap_table(
        body,
        "Concentración de anomalías en pagos con intervención de manager",
        "tab:posthoc-manager",
    )
    return _save(tex, path)


# ---------------------------------------------------------------------------
# V2 — Confirmatorio (union tipificada + refund como control negativo)
# ---------------------------------------------------------------------------
# Esquema de entrada y gates pre-registrados: ver docstring del modulo.

# Tipos no circulares que participan del gate HE2.
V2_TYPED_PROXIES = ["card_testing", "velocity_extreme", "new_user_burst"]
V2_UNION_KEY = "typed_union"
V2_REFUND_KEY = "refund_negative_control"

V2_PROXY_NAMES = {
    "card_testing": "Card testing",
    "velocity_extreme": "Velocidad extrema",
    "new_user_burst": "Usuario nuevo (ráfaga)",
    "typed_union": "Unión tipificada",
    "refund_negative_control": "Reembolso (control negativo)",
}

# Gates pre-registrados (congelados antes del confirmatorio V2).
V2_GATE_HE1_ALPHA = 0.05
V2_GATE_HE1_R_RB = 0.10  # r_rb > 0,10 (estricto)
V2_GATE_HE2_EF = 2.0  # EF@1% >= 2
V2_GATE_HE2_CI_LOW = 1.0  # limite inferior IC95 > 1 (estricto)
V2_GATE_HE2_MIN_TYPES = 2  # en al menos 2 de 3 tipos
V2_BAND_HE3_EF = (0.8, 1.3)  # banda de equivalencia EF@1% refund (inclusiva)
V2_BAND_HE3_AUC = (0.45, 0.55)  # banda de equivalencia AUC refund (inclusiva)
V2_GATE_HE4_MIN_WINS = 3  # IF >= comparadores en >= 3 de 4 metricas
V2_GATE_MULTISEED_MAX = 0.15  # rango relativo EF@1% < 15% (estricto)

V2_HE4_METRICS = [
    ("EF@1\\%", "ef_at_1pct"),
    ("EF@5\\%", "ef_at_5pct"),
    ("AP", "ap"),
    ("P@1\\%", "precision_at_1pct"),
]


def _fmt_es(v, decimals=3) -> str:
    """Formato numérico en español: coma decimal, ``---`` para faltantes."""
    if v is None:
        return "---"
    if isinstance(v, bool):
        return "Sí" if v else "No"
    if isinstance(v, float):
        return f"{v:.{decimals}f}".replace(".", ",")
    return str(v)


def _fmt_es_int(v) -> str:
    """Entero con separador de miles como espacio fino LaTeX (``\\,``).

    Se evita el punto como separador de miles para que nunca se confunda
    con un decimal en las tablas con coma decimal.
    """
    if v is None:
        return "---"
    return f"{float(v):,.0f}".replace(",", "\\,")


def _fmt_es_sci(v) -> str:
    """Notación científica con coma decimal (p-values)."""
    if v is None:
        return "---"
    return f"{float(v):.2e}".replace(".", ",")


def _fmt_es_pct(v, decimals=1) -> str:
    """Proporción [0,1] como porcentaje en español."""
    if v is None:
        return "---"
    return f"{float(v) * 100:.{decimals}f}".replace(".", ",") + "\\%"


def _verdict(passed: Optional[bool]) -> str:
    if passed is None:
        return "---"
    return "Respaldada" if passed else "No respaldada"


def he2_type_passes(d: Dict) -> Optional[bool]:
    """Un tipo satisface el gate HE2 si EF@1% >= 2 Y LI del IC95 > 1."""
    ef = d.get("ef_at_1pct")
    ci_low = d.get("ef_at_1pct_ci_low")
    if ef is None or ci_low is None:
        return None
    return ef >= V2_GATE_HE2_EF and ci_low > V2_GATE_HE2_CI_LOW


def table_v2_he1_mwu(scoreboard: Dict, path: Optional[Path] = None) -> str:
    """HE1 confirmatorio V2: Mann-Whitney U sobre la unión tipificada."""
    g = scoreboard.get("global", {})
    p, r_rb = g.get("mwu_p"), g.get("r_rb")
    p_pass = None if p is None else p < V2_GATE_HE1_ALPHA
    r_pass = None if r_rb is None else r_rb > V2_GATE_HE1_R_RB
    he1_pass = None if (p_pass is None or r_pass is None) else (p_pass and r_pass)
    rows = [
        f"  U de Mann-Whitney & {_fmt_es_int(g.get('mwu_u'))} & --- & --- \\\\",
        f"  p-value & {_fmt_es_sci(p)} & $< 0,05$ & {_fmt_es(p_pass)} \\\\",
        f"  $r_{{rb}}$ (rank-biserial) & {_fmt_es(r_rb)} & $> 0,10$ & {_fmt_es(r_pass)} \\\\",
        "\\midrule",
        f"  Veredicto HE1 & \\multicolumn{{3}}{{c}}{{{_verdict(he1_pass)}}} \\\\",
    ]
    body = (
        "\\begin{tabular}{lrcc}\n\\toprule\n"
        "Estadístico & Valor & Gate & Cumple \\\\\n\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n"
    )
    tex = _wrap_table(
        body,
        "HE1 (confirmatorio V2): separación de scores respecto de la unión tipificada",
        "tab:v2-he1-mwu",
    )
    return _save(tex, path)


def table_v2_he2_ef_by_type(scoreboard: Dict, path: Optional[Path] = None) -> str:
    """HE2 confirmatorio V2: EF@1% por tipo con IC95 y veredicto del gate."""
    proxies = scoreboard.get("proxies", {})
    rows = []
    n_pass = 0
    n_known = 0
    for t in V2_TYPED_PROXIES:
        d = proxies.get(t, {})
        tp = he2_type_passes(d)
        if tp is not None:
            n_known += 1
            n_pass += int(tp)
        ci = (
            f"[{_fmt_es(d.get('ef_at_1pct_ci_low'))}; {_fmt_es(d.get('ef_at_1pct_ci_high'))}]"
            if d.get("ef_at_1pct_ci_low") is not None
            else "---"
        )
        rows.append(
            f"  {V2_PROXY_NAMES[t]} & {_fmt_es(d.get('ef_at_1pct'))} & {ci} & "
            f"{_fmt_es(tp)} \\\\"
        )
    he2_pass = None if n_known == 0 else n_pass >= V2_GATE_HE2_MIN_TYPES
    rows.append("\\midrule")
    rows.append(
        f"  Tipos que cumplen ($\\geq {V2_GATE_HE2_MIN_TYPES}$ de {len(V2_TYPED_PROXIES)}) & "
        f"\\multicolumn{{3}}{{c}}{{{n_pass}/{len(V2_TYPED_PROXIES)}}} \\\\"
    )
    rows.append(f"  Veredicto HE2 & \\multicolumn{{3}}{{c}}{{{_verdict(he2_pass)}}} \\\\")
    body = (
        "\\begin{tabular}{lrcc}\n\\toprule\n"
        "Tipo & EF@1\\% & IC 95\\% & Cumple gate \\\\\n\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n"
    )
    tex = _wrap_table(
        body,
        "HE2 (confirmatorio V2): EF@1\\% por tipo no circular "
        "(gate: EF $\\geq$ 2 en $\\geq$ 2 de 3 tipos, con LI del IC95 $>$ 1)",
        "tab:v2-he2-ef-by-type",
    )
    return _save(tex, path)


def table_v2_he3_negative_control(scoreboard: Dict, path: Optional[Path] = None) -> str:
    """HE3 confirmatorio V2: control negativo (refund) vs. bandas de equivalencia."""
    d = scoreboard.get("proxies", {}).get(V2_REFUND_KEY, {})
    ef, auc = d.get("ef_at_1pct"), d.get("auc")
    ef_in = None if ef is None else (V2_BAND_HE3_EF[0] <= ef <= V2_BAND_HE3_EF[1])
    auc_in = None if auc is None else (V2_BAND_HE3_AUC[0] <= auc <= V2_BAND_HE3_AUC[1])
    he3_pass = None if (ef_in is None or auc_in is None) else (ef_in and auc_in)
    rows = [
        f"  EF@1\\% (reembolso) & {_fmt_es(ef)} & [0,8; 1,3] & {_fmt_es(ef_in)} \\\\",
        f"  AUC-ROC (reembolso) & {_fmt_es(auc)} & [0,45; 0,55] & {_fmt_es(auc_in)} \\\\",
        "\\midrule",
        f"  Veredicto HE3 & \\multicolumn{{3}}{{c}}{{{_verdict(he3_pass)}}} \\\\",
    ]
    body = (
        "\\begin{tabular}{lrcc}\n\\toprule\n"
        "Métrica & Valor & Banda de equivalencia & Dentro de banda \\\\\n\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n"
    )
    tex = _wrap_table(
        body,
        "HE3 (confirmatorio V2): validez discriminante — control negativo de reembolso",
        "tab:v2-he3-negative-control",
    )
    return _save(tex, path)


def table_v2_he4_comparison(scoreboard: Dict, path: Optional[Path] = None) -> str:
    """HE4 confirmatorio V2: IF vs. LOF vs. OC-SVM sobre la unión tipificada."""
    comp = scoreboard.get("comparators", {})
    if_m = comp.get("isolation_forest", {})
    lof_m = comp.get("lof", {})
    ocsvm_m = comp.get("ocsvm", {})
    rows = []
    wins = 0
    n_known = 0
    for label, key in V2_HE4_METRICS:
        iv, lv, ov = if_m.get(key), lof_m.get(key), ocsvm_m.get(key)
        win = None
        if iv is not None and lv is not None and ov is not None:
            n_known += 1
            win = iv >= lv and iv >= ov  # empate cuenta a favor de IF
            wins += int(win)
        if_cell = _fmt_es(iv)
        if win:
            if_cell = f"\\textbf{{{if_cell}}}"
        rows.append(f"  {label} & {if_cell} & {_fmt_es(lv)} & {_fmt_es(ov)} \\\\")
    he4_pass = None if n_known < len(V2_HE4_METRICS) else wins >= V2_GATE_HE4_MIN_WINS
    rows.append("\\midrule")
    rows.append(
        f"  IF $\\geq$ comparadores & \\multicolumn{{3}}{{c}}{{{wins}/{len(V2_HE4_METRICS)} "
        f"métricas (gate: $\\geq {V2_GATE_HE4_MIN_WINS}$/4)}} \\\\"
    )
    rows.append(f"  Veredicto HE4 & \\multicolumn{{3}}{{c}}{{{_verdict(he4_pass)}}} \\\\")
    body = (
        "\\begin{tabular}{lrrr}\n\\toprule\n"
        "Métrica & IF & LOF & OC-SVM \\\\\n\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n"
    )
    tex = _wrap_table(
        body,
        "HE4 (confirmatorio V2): IF vs. LOF vs. OC-SVM sobre la unión tipificada",
        "tab:v2-he4-comparison",
    )
    return _save(tex, path)


def table_v2_secondary_metrics(scoreboard: Dict, path: Optional[Path] = None) -> str:
    """Métricas secundarias V2 (AUC/AP por tipo y unión), sin gate."""
    proxies = scoreboard.get("proxies", {})
    rows = []
    for t in V2_TYPED_PROXIES + [V2_UNION_KEY]:
        d = proxies.get(t, {})
        rows.append(
            f"  {V2_PROXY_NAMES[t]} & {_fmt_es_pct(d.get('base_rate'), 2)} & "
            f"{_fmt_es(d.get('auc'))} & {_fmt_es(d.get('ap'))} & "
            f"{_fmt_es(d.get('precision_at_1pct'))} \\\\"
        )
    body = (
        "\\begin{tabular}{lrrrr}\n\\toprule\n"
        "Tipo & Tasa base & AUC-ROC & AP & P@1\\% \\\\\n\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n"
    )
    tex = _wrap_table(
        body,
        "Métricas secundarias (confirmatorio V2, sin gate): AUC-ROC y AP "
        "por tipo y unión tipificada",
        "tab:v2-secondary-metrics",
    )
    return _save(tex, path)


def table_v2_multiseed_stability(scoreboard: Dict, path: Optional[Path] = None) -> str:
    """Estabilidad multi-semilla V2: rango relativo de EF@1% vs. umbral 15\\%."""
    ms = scoreboard.get("multiseed", {})
    rows = []
    for t in V2_TYPED_PROXIES + [V2_UNION_KEY]:
        d = ms.get(t)
        if d is None:
            continue
        rel = d.get("ef1_relative_range")
        gated = d.get("gated", True)
        if not gated:
            verdict = "--- (sin gate)"
        elif rel is None:
            verdict = "---"
        else:
            verdict = "Sí" if rel < V2_GATE_MULTISEED_MAX else "No"
        rows.append(f"  {V2_PROXY_NAMES[t]} & {_fmt_es_pct(rel)} & {verdict} \\\\")
    body = (
        "\\begin{tabular}{lrc}\n\\toprule\n"
        "Tipo & Rango relativo EF@1\\% & Cumple ($< 15\\%$) \\\\\n\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n"
    )
    tex = _wrap_table(
        body,
        "Estabilidad multi-semilla (confirmatorio V2): rango relativo de "
        "EF@1\\% entre semillas 42/43/44",
        "tab:v2-multiseed-stability",
    )
    return _save(tex, path)


def generate_v2_tables(scoreboard: Dict, output_dir: Path) -> Dict[str, Path]:
    """Genera todas las tablas del confirmatorio V2 desde el scoreboard JSON.

    Args:
        scoreboard: dict con el esquema documentado en el docstring del módulo
            (contenido de ``output/revision/v2_confirmatory_scoreboard.json``).
        output_dir: directorio destino (p. ej. ``output/tables``).

    Returns:
        Mapa nombre de archivo -> ruta escrita.
    """
    output_dir = Path(output_dir)
    generators = {
        "table_v2_he1_mwu.tex": table_v2_he1_mwu,
        "table_v2_he2_ef_by_type.tex": table_v2_he2_ef_by_type,
        "table_v2_he3_negative_control.tex": table_v2_he3_negative_control,
        "table_v2_he4_comparison.tex": table_v2_he4_comparison,
        "table_v2_secondary_metrics.tex": table_v2_secondary_metrics,
        "table_v2_multiseed_stability.tex": table_v2_multiseed_stability,
    }
    written: Dict[str, Path] = {}
    for name, fn in generators.items():
        path = output_dir / name
        fn(scoreboard, path=path)
        written[name] = path
    logger.info(f"V2 tables generated: {len(written)} files in {output_dir}")
    return written

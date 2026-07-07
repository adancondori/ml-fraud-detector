"""LaTeX table generator for thesis Cap 3. Booktabs format, APA 7 style."""

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

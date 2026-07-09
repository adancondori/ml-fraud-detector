"""Genera el reporte comparativo old vs new de facility stats con evidencia.

frame-normalization-v1 (spec artefactos-stats — Requirements: Reporte
comparativo obligatorio + Evidencia reproducible del refresh).

Produce JSON + resumen MD con: deltas por facility, veredicto material_change
con umbral declarado, y bloque de evidencia (snapshot, query de conteo,
conteos, rutas + SHA-256 de artefactos old/new y comando ejecutado).

Uso:
  ./venv/bin/python scripts/compare_facility_stats.py \
      --old output/models/facility_stats_v1.json \
      --new output/models/candidates/facility_stats_v1_candidate.json \
      --old-thresholds output/models/thresholds_segmented_v1.json \
      --new-thresholds output/models/candidates/thresholds_segmented_v1_candidate.json \
      --snapshot "2026-07-09T00:00:00Z" \
      --count-query "SELECT count() FROM payments FINAL WHERE ..." \
      --universe-rows 3137086 \
      --out-dir output/reports/stats_refresh_20260709
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    from fraud_detector.stats.compare import compare_stats, render_summary_md

    parser = argparse.ArgumentParser(description="Reporte comparativo de facility stats")
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--old-thresholds", type=Path, default=None)
    parser.add_argument("--new-thresholds", type=Path, default=None)
    parser.add_argument("--snapshot", default=None, help="Fecha/hora del snapshot ClickHouse")
    parser.add_argument("--count-query", default=None, help="Query de conteo del universo")
    parser.add_argument("--universe-rows", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    with open(args.old) as f:
        old_stats = json.load(f)
    with open(args.new) as f:
        new_stats = json.load(f)

    report = compare_stats(old_stats, new_stats)

    evidence = {
        "snapshot": args.snapshot,
        "count_query": args.count_query,
        "universe_rows": args.universe_rows,
        "train_rows_old": old_stats.get("train_rows"),
        "train_rows_new": new_stats.get("train_rows"),
        "old_stats_path": str(args.old),
        "old_stats_sha256": sha256_of(args.old),
        "new_stats_path": str(args.new),
        "new_stats_sha256": sha256_of(args.new),
        "command": " ".join(shlex.quote(a) for a in sys.argv),
    }
    if args.old_thresholds and args.old_thresholds.exists():
        evidence["old_thresholds_path"] = str(args.old_thresholds)
        evidence["old_thresholds_sha256"] = sha256_of(args.old_thresholds)
    if args.new_thresholds and args.new_thresholds.exists():
        evidence["new_thresholds_path"] = str(args.new_thresholds)
        evidence["new_thresholds_sha256"] = sha256_of(args.new_thresholds)

    report["evidence"] = evidence

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "stats_compare_report.json"
    md_path = args.out_dir / "stats_compare_summary.md"

    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    md_path.write_text(render_summary_md(report, evidence=evidence))

    print(f"Reporte JSON: {json_path}")
    print(f"Resumen MD:   {md_path}")
    print(f"material_change: {report['material_change']}")
    for reason in report["material_reasons"][:20]:
        print(f"  - {reason}")
    if len(report["material_reasons"]) > 20:
        print(f"  ... y {len(report['material_reasons']) - 20} más")


if __name__ == "__main__":
    main()

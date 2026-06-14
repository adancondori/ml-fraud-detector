#!/bin/bash
# Cleanup script for this session — removes:
#   - val_features.parquet.bak (47MB, identical to current val_features.parquet)
#   - scripts/fix_warm_history_revaluate.py (negative result, fully documented in RESULTADO_FINAL.md)
#   - /tmp/*.log (my session's temp logs)
#
# Keeps all experiment scripts and JSON results as documentation.
# Run with: bash scripts/cleanup_session.sh

set -e
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
echo "Cleanup running in $PROJECT_ROOT"

# 1. .bak files
if [ -f "$PROJECT_ROOT/data/processed/val_features.parquet.bak" ]; then
    rm "$PROJECT_ROOT/data/processed/val_features.parquet.bak"
    echo "  ✓ removed val_features.parquet.bak"
fi
if [ -f "$PROJECT_ROOT/data/processed/test_features.parquet.bak" ]; then
    rm "$PROJECT_ROOT/data/processed/test_features.parquet.bak"
    echo "  ✓ removed test_features.parquet.bak"
fi

# 2. The negative-result script
if [ -f "$PROJECT_ROOT/scripts/fix_warm_history_revaluate.py" ]; then
    rm "$PROJECT_ROOT/scripts/fix_warm_history_revaluate.py"
    echo "  ✓ removed scripts/fix_warm_history_revaluate.py (negative result, see RESULTADO_FINAL.md)"
fi

# 3. /tmp logs from this session
for f in /tmp/honest_auc.log /tmp/clean_honest.log /tmp/ensemble.log /tmp/grid_unified.log \
         /tmp/engineered.log /tmp/raw_features.log /tmp/raw2.log /tmp/he4.log \
         /tmp/token_extract.log /tmp/token_extract2.log /tmp/tokens.log /tmp/validate.log; do
    if [ -f "$f" ]; then
        rm "$f"
        echo "  ✓ removed $f"
    fi
done

echo "Cleanup done."

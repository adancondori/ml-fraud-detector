"""validate_universe_filter: verify the facility stats artifact metadata.

Checks that:
  1. universe_filter string matches the scorer's exact filter expression.
  2. train_rows in the artifact matches len(sample_df) within 0.1% tolerance.
  3. schema_version == 'facility-stats-v1'.
  4. Facility coverage: len(facilities) == tz_df['facility_id'].nunique()
     (i.e. every facility with a known timezone has an entry).
  5. n_facilities == facility coverage count.
  6. All facility entries have a non-empty iana_tz.

Raises AssertionError with a descriptive message on any violation.
Returns True if all checks pass (convenient for scripting and testing).
"""

from __future__ import annotations

import pandas as pd

EXPECTED_UNIVERSE_FILTER = (
    "_peerdb_is_deleted=0 AND payment_method NOT IN ('reversal','free') AND FINAL"
)
EXPECTED_SCHEMA_VERSION = "facility-stats-v1"
ROW_COUNT_TOLERANCE = 0.001  # 0.1%


def validate_universe_filter(
    stats: dict,
    sample_df: pd.DataFrame,
    tz_df: pd.DataFrame,
) -> bool:
    """Verify the stats artifact universe filter, row count, and facility coverage.

    Args:
        stats: The dict loaded from facility_stats_v1.json.
        sample_df: The training DataFrame (or a representative sample) used to
            check that train_rows in the artifact is consistent with the actual
            parquet. Must have at least one row.
        tz_df: The facility timezone DataFrame (output/revision/facility_tz.parquet).
            Used to compute the expected facility count (1876 in production).

    Returns:
        True if all assertions pass.

    Raises:
        AssertionError: if any check fails, with a descriptive message.
    """
    # 1. Schema version
    assert stats.get("schema_version") == EXPECTED_SCHEMA_VERSION, (
        f"schema_version mismatch: got '{stats.get('schema_version')}', "
        f"expected '{EXPECTED_SCHEMA_VERSION}'"
    )

    # 2. Universe filter string (exact match)
    actual_filter = stats.get("universe_filter", "")
    assert actual_filter == EXPECTED_UNIVERSE_FILTER, (
        f"universe_filter mismatch:\n"
        f"  got:      '{actual_filter}'\n"
        f"  expected: '{EXPECTED_UNIVERSE_FILTER}'"
    )

    # 3. Row count within tolerance
    artifact_rows = stats.get("train_rows", 0)
    parquet_rows = len(sample_df)
    if artifact_rows > 0:
        relative_diff = abs(artifact_rows - parquet_rows) / artifact_rows
        assert relative_diff < ROW_COUNT_TOLERANCE, (
            f"train_rows mismatch: artifact={artifact_rows}, "
            f"sample_df={parquet_rows}, "
            f"relative_diff={relative_diff:.4%} > tolerance {ROW_COUNT_TOLERANCE:.4%}"
        )

    # 4. Facility coverage: artifact must cover ALL facilities in tz_df
    expected_n = int(tz_df["facility_id"].nunique())
    actual_n_facilities = len(stats.get("facilities", {}))
    assert actual_n_facilities == expected_n, (
        f"facility coverage incomplete: len(facilities)={actual_n_facilities} "
        f"!= tz_df.facility_id.nunique()={expected_n}. "
        f"Builder likely iterated only train_df.groupby() (~689) instead of tz_map (1876)."
    )

    # 5. n_facilities field must match facility coverage
    artifact_n_facilities = stats.get("n_facilities", -1)
    assert (
        artifact_n_facilities == expected_n
    ), f"n_facilities={artifact_n_facilities} != expected {expected_n}"

    # 6. Every facility entry must have a non-empty iana_tz
    missing_tz = [
        fid for fid, entry in stats.get("facilities", {}).items() if not entry.get("iana_tz")
    ]
    assert not missing_tz, (
        f"{len(missing_tz)} facilities have empty/missing iana_tz: "
        f"{missing_tz[:10]}{'...' if len(missing_tz) > 10 else ''}"
    )

    return True

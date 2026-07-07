"""FacilityStatsBuilder: compute per-facility reference stats from training data.

Produces a dict suitable for JSON serialization with the following top-level keys:
  schema_version, built_at, universe_filter, train_rows, n_facilities,
  min_n_threshold, global_fallback, currency_fallbacks, facilities.

The 'facilities' dict maps str(facility_id) -> entry for EVERY facility_id
present in tz_map (1876 in production), regardless of whether that facility
appeared in train_df. Facilities absent from train_df receive fallback_level
'currency' or 'global' with magnitude stats (median/mean/iqr) set to None.

IQR guard rule (production spec): iqr_guarded = max(iqr, 1.0).
Do NOT use iqr + 1e-6 (amplifies noise for near-uniform distributions).
"""

from __future__ import annotations

import datetime
from typing import Optional

import numpy as np
import pandas as pd

from fraud_detector.stats.tz_mapping import resolve_iana

# Minimum number of train rows required for a currency to get dedicated fallback stats.
# Currencies below this threshold fall back to global (not per-currency) stats.
_MIN_CURRENCY_N = 1_000  # monedas con menos rows en train caen a global


class FacilityStatsBuilder:
    """Build the per-facility stats artifact from the training universe.

    Usage::

        stats = FacilityStatsBuilder().build(train_df, tz_map, fid_currency)
    """

    MIN_N: int = 30  # facilities with n < MIN_N use currency/global fallback

    def build(
        self,
        train_df: pd.DataFrame,
        tz_map: dict[int, str],
        fid_currency: dict[int, str],
    ) -> dict:
        """Compute per-facility stats.

        Args:
            train_df: DataFrame filtered to the scorer universe
                (_peerdb_is_deleted=0 AND payment_method NOT IN ('reversal','free')).
                Required columns: amount (numeric), facility_id (int), currency (str).
            tz_map: Mapping {facility_id -> rails_tz_name} for ALL facilities
                (including those absent from train_df). Typically derived from
                output/revision/facility_tz.parquet.
            fid_currency: Mapping {facility_id -> dominant_currency} for facilities
                that appear in train_df. Used to choose the currency fallback.

        Returns:
            A dict ready for json.dump() with schema_version='facility-stats-v1'.
        """
        train_df = train_df.copy()
        train_df["amount"] = pd.to_numeric(train_df["amount"], errors="coerce")

        # --- Global fallback (computed over all train rows) ---
        global_fallback = self._compute_global_fallback(train_df)

        # --- Currency fallbacks for all currencies with n >= _MIN_CURRENCY_N in train ---
        eligible_currencies = (
            train_df[train_df["currency"] != "EMPTY"]
            .groupby("currency")
            .size()
            .pipe(lambda s: s[s >= _MIN_CURRENCY_N])
            .index.tolist()
        )
        # Always include USD even if below threshold
        if "USD" not in eligible_currencies:
            eligible_currencies.append("USD")
        currency_fallbacks = self._compute_currency_fallbacks(train_df, eligible_currencies)

        # --- Per-facility stats (base loop is tz_map, not train_df.groupby) ---
        # This ensures ALL 1876 facilities have an entry even with no train history.
        per_fid_stats = self._compute_per_facility(train_df)

        facilities: dict[str, dict] = {}
        for fid, rails_tz in tz_map.items():
            iana_tz = resolve_iana(rails_tz)
            fid_key = str(fid)
            dominant_currency = fid_currency.get(fid, "USD")

            if fid in per_fid_stats:
                raw = per_fid_stats[fid]
                n = raw["n"]
                iqr = raw["iqr"]
                iqr_guarded = max(iqr, 1.0)

                if n >= self.MIN_N:
                    # Facility has enough history for its own stats.
                    facilities[fid_key] = {
                        "n": n,
                        "median": _to_float(raw["median"]),
                        "mean": _to_float(raw["mean"]),
                        "iqr": _to_float(iqr),
                        "iqr_guarded": _to_float(iqr_guarded),
                        "iana_tz": iana_tz,
                        "fallback_level": "facility",
                    }
                else:
                    # Facility exists in train but n < MIN_N: use currency fallback stats
                    # for magnitude, but retain its own n and iana_tz for auditing.
                    fallback_stats, fallback_level = self._resolve_fallback(
                        dominant_currency, currency_fallbacks, global_fallback
                    )
                    facilities[fid_key] = {
                        "n": n,
                        "median": fallback_stats["median"],
                        "mean": fallback_stats["mean"],
                        "iqr": _to_float(iqr),
                        "iqr_guarded": _to_float(iqr_guarded),
                        "iana_tz": iana_tz,
                        "fallback_level": fallback_level,
                    }
            else:
                # Facility is in tz_map but has NO rows in train_df.
                # Provide iana_tz + currency fallback stats so real-time path
                # never hits a KeyError for new/cold-start facilities.
                fallback_stats, fallback_level = self._resolve_fallback(
                    dominant_currency, currency_fallbacks, global_fallback
                )
                facilities[fid_key] = {
                    "n": 0,
                    "median": fallback_stats["median"],
                    "mean": fallback_stats["mean"],
                    "iqr": None,
                    "iqr_guarded": fallback_stats["iqr_guarded"],
                    "iana_tz": iana_tz,
                    "fallback_level": fallback_level,
                }

        # Hard invariant: every facility in tz_map must be represented.
        assert len(facilities) == len(
            tz_map
        ), f"facilities coverage {len(facilities)} != tz_map size {len(tz_map)}"

        return {
            "schema_version": "facility-stats-v1",
            "built_at": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "universe_filter": (
                "_peerdb_is_deleted=0 AND payment_method NOT IN ('reversal','free') AND FINAL"
            ),
            "train_rows": int(len(train_df)),
            "n_facilities": int(len(facilities)),
            "min_n_threshold": int(self.MIN_N),
            "min_currency_n_threshold": int(_MIN_CURRENCY_N),
            "global_fallback": global_fallback,
            "currency_fallbacks": currency_fallbacks,
            "facilities": facilities,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_global_fallback(self, train_df: pd.DataFrame) -> dict:
        """Compute magnitude stats across all rows in train_df."""
        amounts = train_df["amount"].dropna()
        q1 = float(amounts.quantile(0.25))
        q3 = float(amounts.quantile(0.75))
        iqr = q3 - q1
        iqr_guarded = max(iqr, 1.0)
        return {
            "median": _to_float(float(amounts.median())),
            "mean": _to_float(float(amounts.mean())),
            "iqr": _to_float(iqr),
            "iqr_guarded": _to_float(iqr_guarded),
            "n": int(len(amounts)),
            "fallback_level": "global",
        }

    def _compute_currency_fallbacks(
        self,
        train_df: pd.DataFrame,
        currencies: list[str],
    ) -> dict[str, dict]:
        """Compute magnitude stats per currency for the given currency list."""
        fallbacks: dict[str, dict] = {}
        for currency in currencies:
            mask = train_df["currency"] == currency
            amounts = train_df.loc[mask, "amount"].dropna()
            if len(amounts) == 0:
                continue
            q1 = float(amounts.quantile(0.25))
            q3 = float(amounts.quantile(0.75))
            iqr = q3 - q1
            iqr_guarded = max(iqr, 1.0)
            fallbacks[currency] = {
                "median": _to_float(float(amounts.median())),
                "mean": _to_float(float(amounts.mean())),
                "iqr": _to_float(iqr),
                "iqr_guarded": _to_float(iqr_guarded),
                "n": int(len(amounts)),
                "fallback_level": "currency",
            }
        return fallbacks

    def _compute_per_facility(self, train_df: pd.DataFrame) -> dict[int, dict]:
        """Group by facility_id and compute raw stats per facility.

        Returns a dict {facility_id (int) -> {n, median, mean, iqr}} for
        facilities that actually appear in train_df.
        """
        grp = train_df.groupby("facility_id")["amount"]
        counts = grp.size()
        medians = grp.median()
        means = grp.mean()
        q1s = grp.quantile(0.25)
        q3s = grp.quantile(0.75)

        result: dict[int, dict] = {}
        for fid in counts.index:
            iqr = float(q3s[fid] - q1s[fid])
            result[int(fid)] = {
                "n": int(counts[fid]),
                "median": float(medians[fid]),
                "mean": float(means[fid]),
                "iqr": iqr,
            }
        return result

    @staticmethod
    def _resolve_fallback(
        currency: str,
        currency_fallbacks: dict[str, dict],
        global_fallback: dict,
    ) -> tuple[dict, str]:
        """Return (fallback_stats_dict, fallback_level) for a given currency."""
        if currency in currency_fallbacks:
            return currency_fallbacks[currency], "currency"
        return global_fallback, "global"


def _to_float(value: Optional[float]) -> Optional[float]:
    """Convert numpy scalar to native Python float, preserving None."""
    if value is None:
        return None
    if isinstance(value, float) and (value != value):  # NaN
        return None
    return float(value)

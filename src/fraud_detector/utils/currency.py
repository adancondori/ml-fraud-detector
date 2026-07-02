"""
Currency normalization for the anomaly-detection pipeline.

The code accepts two exchange-rate layouts:

1. Manual monthly file with direct USD rate:
   `year_month,currency,rate_to_usd`
2. ClickHouse snapshot exported from `default.exchange_rates`:
   `base_currency,target_currency,conversion_rate,timestamp`

Internally both formats are converted into the same lookup:
`(year_month, currency) -> rate_to_usd`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from fraud_detector.utils.logger import logger

_FALLBACK_RATES: Dict[str, float] = {
    "USD": 1.000000,
    "CAD": 0.720000,
    "MYR": 0.225000,
    "HNL": 0.039800,
    "AUD": 0.645000,
    "NIO": 0.027000,
    "ILS": 0.270000,
    "GTQ": 0.130000,
    "PKR": 0.003560,
    "HKD": 0.128400,
    "SGD": 0.750000,
    "COP": 0.000240,
    "BWP": 0.074000,
    "AED": 0.272300,
    "EUR": 1.080000,
    "RWF": 0.000730,
    "JPY": 0.006800,
    "MXN": 0.050000,
    "INR": 0.012000,
    "NZD": 0.580000,
}


def fallback_rate(currency: str | None) -> float:
    """Return the static fallback USD conversion rate for a currency."""
    currency = (currency or "USD").upper()
    return _FALLBACK_RATES.get(currency, 1.0)


def normalize_amount_value(amount: float | int | None, currency: str | None) -> float:
    """Normalize a single local-currency amount to USD with fallback rates."""
    return float(amount or 0.0) * fallback_rate(currency)


def clickhouse_rate_case(column: str = "currency") -> str:
    """Build a ClickHouse multiIf expression matching the fallback rate table."""
    parts: list[str] = []
    for currency, rate in sorted(_FALLBACK_RATES.items()):
        parts.extend([f"upper({column}) = '{currency}'", f"{rate:.12f}"])
    parts.append("1.0")
    return f"multiIf({', '.join(parts)})"


class CurrencyNormalizer:
    """Normalize local-currency amounts into USD."""

    MONETARY_COLS = ("amount", "discount", "tax", "tip")

    def __init__(self, rate_lookup: Dict[Tuple[str, str], float]) -> None:
        self._rate_lookup = rate_lookup
        self._fallback_by_currency = self._build_currency_fallbacks()
        logger.info(
            "CurrencyNormalizer initialized with "
            f"{len(rate_lookup)} monthly entries and "
            f"{len(self._fallback_by_currency)} currencies"
        )

    def _build_currency_fallbacks(self) -> Dict[str, float]:
        by_currency: Dict[str, list[float]] = {}
        for (_, currency), rate in self._rate_lookup.items():
            by_currency.setdefault(currency, []).append(float(rate))
        return {currency: float(np.median(rates)) for currency, rates in by_currency.items()}

    @classmethod
    def from_csv(cls, path: str | Path) -> "CurrencyNormalizer":
        """Load a normalizer from a CSV export."""
        csv_path = Path(path)
        if not csv_path.exists():
            logger.warning(f"Exchange-rate file not found at {csv_path}; using fallback rates.")
            return cls.from_fallback()

        df = pd.read_csv(csv_path)
        rate_lookup = cls._parse_rate_lookup(df)
        logger.info(f"Loaded {len(rate_lookup)} exchange-rate entries from {csv_path}")
        return cls(rate_lookup)

    @classmethod
    def from_fallback(cls) -> "CurrencyNormalizer":
        rate_lookup = {("fallback", currency): rate for currency, rate in _FALLBACK_RATES.items()}
        return cls(rate_lookup)

    @staticmethod
    def _parse_rate_lookup(df: pd.DataFrame) -> Dict[Tuple[str, str], float]:
        columns = set(df.columns)

        if {"year_month", "currency", "rate_to_usd"}.issubset(columns):
            return {
                (str(row["year_month"]), str(row["currency"]).upper()): float(row["rate_to_usd"])
                for _, row in df.iterrows()
            }

        if {"base_currency", "target_currency", "conversion_rate", "timestamp"}.issubset(columns):
            subset = df.copy()
            subset["base_currency"] = subset["base_currency"].astype(str).str.upper()
            subset["target_currency"] = subset["target_currency"].astype(str).str.upper()
            subset["timestamp"] = pd.to_datetime(subset["timestamp"], errors="coerce")
            subset = subset[subset["base_currency"] == "USD"].copy()
            subset["year_month"] = subset["timestamp"].dt.to_period("M").astype(str)

            rate_lookup: Dict[Tuple[str, str], float] = {}
            for _, row in subset.iterrows():
                target = row["target_currency"]
                conversion_rate = float(row["conversion_rate"])
                rate_lookup[(row["year_month"], target)] = 1.0 / conversion_rate
            rate_lookup.setdefault(("fallback", "USD"), 1.0)
            return rate_lookup

        raise ValueError(
            "Exchange-rate CSV must contain either "
            "`year_month,currency,rate_to_usd` or "
            "`base_currency,target_currency,conversion_rate,timestamp`."
        )

    def get_rate(self, year_month: str, currency: str) -> float:
        currency = (currency or "USD").upper()

        exact = self._rate_lookup.get((year_month, currency))
        if exact is not None:
            return exact

        fallback = self._fallback_by_currency.get(currency)
        if fallback is not None:
            return fallback

        static = _FALLBACK_RATES.get(currency)
        if static is not None:
            return static

        logger.warning(
            f"Missing exchange rate for currency={currency}, year_month={year_month}; "
            "defaulting to 1.0."
        )
        return 1.0

    def normalize(
        self,
        df: pd.DataFrame,
        amount_col: str = "amount",
        currency_col: str = "currency",
        timestamp_col: str = "created_at",
    ) -> pd.DataFrame:
        """Normalize amount-like columns to USD and preserve local values."""
        out = df.copy()

        timestamps = pd.to_datetime(out[timestamp_col], errors="coerce")
        if timestamps.isna().any():
            raise ValueError(f"Column '{timestamp_col}' contains invalid timestamps")

        year_month = timestamps.dt.to_period("M").astype(str)
        currency = out[currency_col].fillna("USD").astype(str).str.upper().replace("", "USD")

        rate_series = pd.Series(
            [self.get_rate(ym, cur) for ym, cur in zip(year_month, currency)],
            index=out.index,
            dtype=np.float64,
        )

        out["exchange_rate_applied"] = rate_series.astype(np.float32)
        out["exchange_rate_source"] = np.where(
            currency.eq("USD"),
            "identity",
            "lookup_or_fallback",
        )

        for col in self.MONETARY_COLS:
            if col not in out.columns:
                continue
            out[f"{col}_local"] = out[col].astype(np.float32)
            out[col] = (out[col].astype(np.float64) * rate_series).astype(np.float32)

        non_usd = int((currency != "USD").sum())
        logger.info(
            "Currency normalization applied: "
            f"{len(out):,} rows, {currency.nunique()} currencies, "
            f"{non_usd:,} non-USD rows"
        )
        return out

    def summary(self) -> pd.DataFrame:
        rows = [
            {"year_month": year_month, "currency": currency, "rate_to_usd": rate}
            for (year_month, currency), rate in sorted(self._rate_lookup.items())
        ]
        return pd.DataFrame(rows)

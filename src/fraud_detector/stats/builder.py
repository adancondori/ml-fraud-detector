"""FacilityStatsBuilder: compute per-facility reference stats from training data.

Produces a dict suitable for JSON serialization with the following top-level keys:
  schema_version, built_at, universe_filter, stats_window_start, stats_window_end,
  shadow_period_start, observed_min_created_at, observed_max_created_at,
  amount_source, train_rows, n_facilities, min_n_threshold,
  min_currency_n_threshold, global_fallback, currency_fallbacks, facilities.

The 'facilities' dict maps str(facility_id) -> entry for EVERY facility_id
present in iana_map (all live facilities in production), regardless of whether
that facility appeared in train_df. Facilities absent from train_df receive
fallback_level 'currency' or 'global' with magnitude stats set to None.

iana_tz source (frame-normalization-v1, design D6): the IANA timezone comes
straight from the replicated column ``facilities.tzinfo_identifier`` (same
source Rails sends in the real-time payload). There is NO Rails->IANA mapping
dictionary anymore. An empty/None identifier yields ``iana_tz: null`` and the
scoring path degrades through the normal fallback chain (payload -> artifact
-> Etc/UTC) instead of breaking the build.

IQR guard rule (production spec): iqr_guarded = max(iqr, 1.0).
Do NOT use iqr + 1e-6 (amplifies noise for near-uniform distributions).
"""

from __future__ import annotations

import datetime
from typing import Optional

import pandas as pd

# Universo canónico (decisión humana 3, frame-normalization-v1):
# los 4 predicados literales que el batch scorer y el loader aplican.
CANONICAL_UNIVERSE_FILTER = (
    "FINAL AND _peerdb_is_deleted=0 AND "
    "payment_method NOT IN ('reversal','free') AND user_id != 0"
)

# Mapping de monto fuente (design D9): la columna interna 'amount' de los
# pipelines offline es un alias de payments.reservation_paid_out. No existe
# una columna física payments.amount.
CANONICAL_AMOUNT_SOURCE = "payments.reservation_paid_out AS amount"

# Minimum number of train rows required for a currency to get dedicated fallback stats.
# Currencies below this threshold fall back to global (not per-currency) stats.
_MIN_CURRENCY_N = 1_000  # monedas con menos rows en train caen a global


class FacilityStatsBuilder:
    """Build the per-facility stats artifact from the training universe.

    Usage::

        stats = FacilityStatsBuilder().build(
            train_df,
            iana_map,
            fid_currency,
            stats_window_start="2025-01-01",
            stats_window_end="2025-06-30",
            shadow_period_start="2025-07-01",
        )
    """

    MIN_N: int = 30  # facilities with n < MIN_N use currency/global fallback

    def build(
        self,
        train_df: pd.DataFrame,
        iana_map: dict[int, Optional[str]],
        fid_currency: dict[int, str],
        *,
        stats_window_start: str,
        stats_window_end: str,
        shadow_period_start: str,
        amount_source: str = CANONICAL_AMOUNT_SOURCE,
    ) -> dict:
        """Compute per-facility stats.

        Args:
            train_df: DataFrame filtered to the canonical scorer universe
                (FINAL AND _peerdb_is_deleted=0 AND payment_method NOT IN
                ('reversal','free') AND user_id != 0).
                Required columns: amount (numeric), facility_id (int),
                currency (str), created_at (datetime, no-nulo, parseable).
                created_at es columna REQUERIDA: el gate anti-fuga temporal
                valida sobre las mismas filas que luego se agregan.
            iana_map: Mapping {facility_id -> tzinfo_identifier} for ALL live
                facilities (including those absent from train_df), read from the
                replicated ClickHouse column ``facilities.tzinfo_identifier``.
                Empty/None values produce ``iana_tz: null`` without aborting.
            fid_currency: Mapping {facility_id -> dominant_currency} for facilities
                that appear in train_df. Used to choose the currency fallback.
            stats_window_start: ISO date (YYYY-MM-DD) — inicio de la ventana de
                pagos usada para las stats.
            stats_window_end: ISO date — fin (inclusivo) de la ventana de stats.
            shadow_period_start: ISO date — inicio del período de shadow/
                evaluación. La generación FALLA si la ventana lo solapa
                (se exige stats_window_end < shadow_period_start).
            amount_source: Declaración del mapping de monto fuente.

        Returns:
            A dict ready for json.dump() with schema_version='facility-stats-v1'.
            Incluye observed_min_created_at / observed_max_created_at (ISO date
            YYYY-MM-DD) con la procedencia temporal real de train_df.

        Raises:
            ValueError: si la ventana de stats solapa el período de shadow o
                está invertida (metadata declarada), o si los datos de
                train_df["created_at"] no pertenecen a la ventana declarada /
                solapan el shadow (anti-fuga temporal EJECUTADO sobre datos,
                spec anti-fuga-stats).
        """
        self._validate_window(stats_window_start, stats_window_end, shadow_period_start)

        # Anti-fuga temporal EJECUTADO: validar pertenencia real de las filas
        # ANTES de copiar/coercer amount (fallar barato sin mutar un df que se
        # rechazará). Devuelve las fechas observadas como datetime.date.
        observed_min, observed_max = self._validate_temporal_provenance(
            train_df, stats_window_start, stats_window_end, shadow_period_start
        )

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

        # --- Per-facility stats (base loop is iana_map, not train_df.groupby) ---
        # This ensures ALL live facilities have an entry even with no train history.
        per_fid_stats = self._compute_per_facility(train_df)

        facilities: dict[str, dict] = {}
        for fid, tzinfo_identifier in iana_map.items():
            # Columna replicada tal cual; vacío/None -> null (sin diccionario propio).
            iana_tz = tzinfo_identifier or None
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
                # Facility is in iana_map but has NO rows in train_df.
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

        # Hard invariant: every facility in iana_map must be represented.
        assert len(facilities) == len(
            iana_map
        ), f"facilities coverage {len(facilities)} != iana_map size {len(iana_map)}"

        return {
            "schema_version": "facility-stats-v1",
            "built_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "universe_filter": CANONICAL_UNIVERSE_FILTER,
            "stats_window_start": stats_window_start,
            "stats_window_end": stats_window_end,
            "shadow_period_start": shadow_period_start,
            "observed_min_created_at": observed_min.isoformat(),
            "observed_max_created_at": observed_max.isoformat(),
            "amount_source": amount_source,
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

    @staticmethod
    def _validate_window(
        stats_window_start: str,
        stats_window_end: str,
        shadow_period_start: str,
    ) -> None:
        """Anti-fuga temporal: la ventana de stats debe terminar antes del shadow."""
        start = datetime.date.fromisoformat(stats_window_start)
        end = datetime.date.fromisoformat(stats_window_end)
        shadow_start = datetime.date.fromisoformat(shadow_period_start)

        if start >= end:
            raise ValueError(
                f"ventana de stats invertida o vacía: "
                f"stats_window_start={stats_window_start} >= stats_window_end={stats_window_end}"
            )
        if end >= shadow_start:
            raise ValueError(
                f"la ventana de stats solapa el período de shadow/evaluación: "
                f"stats_window_end={stats_window_end} >= "
                f"shadow_period_start={shadow_period_start} (se exige end < T)"
            )

    @staticmethod
    def _validate_temporal_provenance(
        train_df: pd.DataFrame,
        stats_window_start: str,
        stats_window_end: str,
        shadow_period_start: str,
    ) -> tuple[datetime.date, datetime.date]:
        """Anti-fuga temporal EJECUTADO sobre train_df["created_at"].

        Verifica pertenencia REAL de las filas a la ventana declarada y
        ausencia de solape con el shadow. Comparación date-vs-date en la
        referencia naive del parquet (el corte train/shadow del extractor):
        reduce created_at a .date() y compara contra date.fromisoformat(...),
        sin tz_localize (no asume una zona no registrada). El resultado es
        determinista e independiente de la TZ del proceso.

        Returns:
            (observed_min, observed_max) como datetime.date.

        Raises:
            ValueError: train_df vacío (0 filas), columna faltante, nulos/NaT,
                no parseable, observed_min < stats_window_start,
                observed_max > stats_window_end, o
                observed_max >= shadow_period_start (fuga).
        """
        if len(train_df) == 0:
            # Guarda de vaciedad: con 0 filas, s.min()/s.max() dan NaT y
            # NaT.date() vs date.fromisoformat levantaría un TypeError opaco.
            # El contrato D1/D2 exige fallar barato con ValueError accionable
            # ANTES de agregar estadísticas.
            raise ValueError(
                "train_df vacío: no hay filas para validar procedencia "
                "temporal (el gate anti-fuga requiere al menos una fila con created_at)"
            )

        if "created_at" not in train_df.columns:
            raise ValueError(
                "train_df no contiene la columna 'created_at': no puedo "
                "garantizar anti-fuga temporal (columna requerida)"
            )

        try:
            s = pd.to_datetime(train_df["created_at"], errors="raise")
        except (ValueError, TypeError) as exc:
            raise ValueError(
                "created_at no es datetime ni parseable: "
                f"dtype={train_df['created_at'].dtype}, no puedo comparar "
                f"contra la ventana ({exc})"
            ) from exc

        n_null = int(s.isna().sum())
        if n_null > 0:
            raise ValueError(
                f"created_at contiene {n_null} nulos/NaT: no puedo validar "
                "pertenencia temporal de esas filas"
            )

        start = datetime.date.fromisoformat(stats_window_start)
        end = datetime.date.fromisoformat(stats_window_end)
        shadow_start = datetime.date.fromisoformat(shadow_period_start)

        observed_min = s.min().date()
        observed_max = s.max().date()

        if observed_min < start:
            raise ValueError(
                "created_at fuera de ventana: "
                f"observed_min={observed_min.isoformat()} < "
                f"stats_window_start={stats_window_start}"
            )
        # El solape con shadow (fuga) es la condición más severa y específica:
        # se comprueba antes que el fin de ventana genérico. Como _validate_window
        # ya garantiza end < shadow, un observed_max >= shadow es también > end,
        # pero el mensaje de fuga es el accionable correcto (spec: borde estricto).
        if observed_max >= shadow_start:
            raise ValueError(
                "FUGA TEMPORAL: "
                f"observed_max={observed_max.isoformat()} >= "
                f"shadow_period_start={shadow_period_start} (train solapa shadow)"
            )
        if observed_max > end:
            raise ValueError(
                "created_at fuera de ventana: "
                f"observed_max={observed_max.isoformat()} > "
                f"stats_window_end={stats_window_end}"
            )

        return observed_min, observed_max

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

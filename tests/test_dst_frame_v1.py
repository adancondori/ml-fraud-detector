"""Test DST-correctness for FrameV1FeatureCalculator._local_hour_dow.

Verifica que la conversión UTC→hora local sea correcta para ≥2 zonas latinoamericanas
(Buenos Aires sin DST, La Paz sin DST) y para spring-forward en América/New York.

Criterio de aceptación FRAME-03.
"""
from __future__ import annotations

import pandas as pd
import pytest

from fraud_detector.scoring.features_frame_v1 import FrameV1FeatureCalculator

OFF_HOURS = frozenset({23, 0, 1, 2, 3, 4, 5, 6})


class TestDstLocalHour:
    """Verifica la conversión UTC → hora local con DST correcto."""

    def test_new_york_spring_forward_after(self):
        """2025-03-09T07:00:00Z en America/New_York = 03:00 EDT (tras spring-forward).

        Spring-forward ocurre a las 02:00 EST → 03:00 EDT el 2025-03-09.
        Antes: UTC-05; después: UTC-04.
        07:00 UTC = 03:00 EDT. 3 está en OFF_HOURS → is_off_hours = True.
        """
        ts = pd.Timestamp("2025-03-09T07:00:00")  # naive UTC
        hour, dow = FrameV1FeatureCalculator._local_hour_dow(ts, "America/New_York")
        assert hour == 3, f"Esperado 3 AM EDT, obtenido {hour}"
        assert dow == 6, f"2025-03-09 es domingo (dow=6), obtenido {dow}"
        assert hour in OFF_HOURS, "3 AM debe estar en OFF_HOURS"

    def test_new_york_spring_forward_before(self):
        """2025-03-09T06:00:00Z en America/New_York = 01:00 EST (antes de spring-forward)."""
        ts = pd.Timestamp("2025-03-09T06:00:00")  # naive UTC
        hour, dow = FrameV1FeatureCalculator._local_hour_dow(ts, "America/New_York")
        assert hour == 1, f"Esperado 1 AM EST, obtenido {hour}"
        assert hour in OFF_HOURS, "1 AM debe estar en OFF_HOURS"

    def test_buenos_aires_no_dst(self):
        """Argentina no tiene DST — UTC-3 todo el año.

        2025-10-05T03:00:00Z = 00:00 ART. Primera zona latinoamericana.
        """
        ts = pd.Timestamp("2025-10-05T03:00:00")  # naive UTC
        hour, dow = FrameV1FeatureCalculator._local_hour_dow(
            ts, "America/Argentina/Buenos_Aires"
        )
        assert hour == 0, f"Esperado medianoche ART (-3h), obtenido {hour}"
        assert dow == 6, f"2025-10-05 es domingo (dow=6), obtenido {dow}"
        assert hour in OFF_HOURS, "medianoche debe estar en OFF_HOURS"

    def test_la_paz_no_dst(self):
        """Bolivia UTC-4, sin DST todo el año.

        2025-11-15T12:00:00Z = 08:00 BOT. Segunda zona latinoamericana requerida.
        """
        ts = pd.Timestamp("2025-11-15T12:00:00")  # naive UTC
        hour, dow = FrameV1FeatureCalculator._local_hour_dow(ts, "America/La_Paz")
        assert hour == 8, f"Esperado 08:00 BOT (-4h), obtenido {hour}"
        assert dow == 5, f"2025-11-15 es sábado (dow=5), obtenido {dow}"
        assert hour not in OFF_HOURS, "8 AM no está en OFF_HOURS"

    def test_naive_timestamp_is_localized_not_converted(self):
        """Un timestamp naive debe tz_localize('UTC') — nunca tz_convert directo sobre naive.

        Con tz_convert directo sobre un timestamp naive, pandas levanta TypeError.
        Este test verifica que el método maneja correctamente el timestamp naive.
        """
        ts = pd.Timestamp("2025-06-01T12:00:00")  # naive, sin tzinfo
        assert ts.tzinfo is None, "El timestamp debe ser naive para el test"
        # No debe levantar excepción
        hour, dow = FrameV1FeatureCalculator._local_hour_dow(ts, "America/New_York")
        # 12:00 UTC en junio (EDT = UTC-4) → 08:00 EDT
        assert hour == 8, f"Esperado 8 AM EDT en verano, obtenido {hour}"

    def test_dow_is_0_on_monday(self):
        """Verificar convención dow: 0=lunes."""
        # 2025-06-02 es lunes
        ts = pd.Timestamp("2025-06-02T12:00:00")  # naive UTC
        _, dow = FrameV1FeatureCalculator._local_hour_dow(ts, "Etc/UTC")
        assert dow == 0, f"2025-06-02 es lunes (dow=0), obtenido {dow}"

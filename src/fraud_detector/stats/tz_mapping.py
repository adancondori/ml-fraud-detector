"""Rails timezone name -> IANA timezone name mapping.

Covers the 64 unique Rails timezone names present in
output/revision/facility_tz.parquet (1876 facilities, 0 nulls).

The base 51 entries are sourced from scripts/exp_frames_improvement.py
(RAILS_TZ_TO_IANA). The remaining 13 entries are the zones present in
facility_tz.parquet that were absent from the prototype dict.
"""
from __future__ import annotations

from zoneinfo import ZoneInfo

# fmt: off
RAILS_TO_IANA: dict[str, str] = {
    # ---- From prototype (exp_frames_improvement.py RAILS_TZ_TO_IANA) ----
    "Eastern Time (US & Canada)": "America/New_York",
    "Central Time (US & Canada)": "America/Chicago",
    "Mountain Time (US & Canada)": "America/Denver",
    "Pacific Time (US & Canada)": "America/Los_Angeles",
    "Arizona": "America/Phoenix",
    "Hawaii": "Pacific/Honolulu",
    "Atlantic Time (Canada)": "America/Halifax",
    "Central America": "America/Guatemala",
    "Caracas": "America/Caracas",
    "Puerto Rico": "America/Puerto_Rico",
    "Kuala Lumpur": "Asia/Kuala_Lumpur",
    "Melbourne": "Australia/Melbourne",
    "Sydney": "Australia/Sydney",
    "Hong Kong": "Asia/Hong_Kong",
    "Singapore": "Asia/Singapore",
    "Monterrey": "America/Monterrey",
    "Quito": "America/Guayaquil",
    "Abu Dhabi": "Asia/Dubai",
    "Bogota": "America/Bogota",
    "Karachi": "Asia/Karachi",
    "Islamabad": "Asia/Karachi",
    "Brisbane": "Australia/Brisbane",
    "Mexico City": "America/Mexico_City",
    "Guadalajara": "America/Mexico_City",
    "Tijuana": "America/Tijuana",
    "Cairo": "Africa/Cairo",
    "Jerusalem": "Asia/Jerusalem",
    "Tokyo": "Asia/Tokyo",
    "Sapporo": "Asia/Tokyo",
    "London": "Europe/London",
    "Auckland": "Pacific/Auckland",
    "Wellington": "Pacific/Auckland",
    "Mumbai": "Asia/Kolkata",
    "New Delhi": "Asia/Kolkata",
    "Berlin": "Europe/Berlin",
    "Istanbul": "Europe/Istanbul",
    "Dublin": "Europe/Dublin",
    "La Paz": "America/La_Paz",
    "Athens": "Europe/Athens",
    "Perth": "Australia/Perth",
    "Stockholm": "Europe/Stockholm",
    "Bangkok": "Asia/Bangkok",
    "Hanoi": "Asia/Bangkok",
    "Pretoria": "Africa/Johannesburg",
    "Zurich": "Europe/Zurich",
    "Kyiv": "Europe/Kyiv",
    "Brasilia": "America/Sao_Paulo",
    "Sri Jayawardenepura": "Asia/Colombo",
    "Buenos Aires": "America/Argentina/Buenos_Aires",
    "New Caledonia": "Pacific/Noumea",
    "Paris": "Europe/Paris",
    # ---- 13 zones present in facility_tz.parquet but absent from prototype ----
    "Baku": "Asia/Baku",
    "Brussels": "Europe/Brussels",
    "Darwin": "Australia/Darwin",
    "Fiji": "Pacific/Fiji",
    "Madrid": "Europe/Madrid",
    "Magadan": "Asia/Magadan",
    "Mazatlan": "America/Mazatlan",
    "Montevideo": "America/Montevideo",
    "Nairobi": "Africa/Nairobi",
    "Santiago": "America/Santiago",
    "Sarajevo": "Europe/Sarajevo",
    "Sofia": "Europe/Sofia",
    "UTC": "Etc/UTC",
}
# fmt: on

assert len(RAILS_TO_IANA) == 64, (
    "RAILS_TO_IANA debe cubrir las 64 zonas Rails presentes en facility_tz.parquet"
    f" — got {len(RAILS_TO_IANA)}"
)

# Validate all IANA names are loadable by zoneinfo at import time.
# This prevents typos from silently propagating to the real-time path.
for _rails_name, _iana_name in RAILS_TO_IANA.items():
    try:
        ZoneInfo(_iana_name)
    except Exception as exc:  # pragma: no cover
        raise ValueError(
            f"IANA name '{_iana_name}' (for Rails '{_rails_name}') is not loadable "
            f"by zoneinfo: {exc}"
        ) from exc


def resolve_iana(rails_name: str) -> str:
    """Return the IANA timezone name for a given Rails timezone name.

    Falls back to 'Etc/UTC' for unknown or empty Rails names so that
    the caller never receives a KeyError — unknown facilities are treated
    as UTC rather than crashing the real-time path.

    Args:
        rails_name: Rails timezone string (e.g. 'Eastern Time (US & Canada)').

    Returns:
        IANA timezone string (e.g. 'America/New_York'), or 'Etc/UTC' if unknown.
    """
    return RAILS_TO_IANA.get(rails_name, "Etc/UTC")

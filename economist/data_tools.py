"""Structured external data tools for the Economist (ported from the reference
tools agent): FRED, BLS, NOAA daily weather and US public holidays.

Rules:
- Every value returned comes from the API response; nothing is estimated.
- Missing API keys or API failures return {"error": ...} so the Economist
  can say the source was unavailable instead of inventing numbers.
- Keys are read lazily from the environment, then Secret Manager, so a
  missing key never breaks startup.
- These tools never touch client data.
"""

from __future__ import annotations

import logging
import os
import re
from functools import lru_cache
from typing import Any

import requests

logger = logging.getLogger("economist-data-tools")

TIMEOUT_SECONDS = 15
MAX_POINTS = 60

# Useful FRED series for marketing-demand context (any valid ID also works).
FRED_SERIES = {
    "UMCSENT": "University of Michigan consumer sentiment",
    "FEDFUNDS": "Effective federal funds rate (%)",
    "CPIAUCSL": "CPI, all items (index)",
    "CUSR0000SETG01": "CPI, airline fares (index)",
    "CUSR0000SEFV": "CPI, food away from home (index)",
    "UNRATE": "Unemployment rate (%)",
    "RSXFS": "Retail sales excl. food services ($M)",
    "PCE": "Personal consumption expenditures ($B)",
    "AIRRPMTSID11": "US air carrier revenue passenger miles",
    "DSPIC96": "Real disposable personal income ($B)",
}

_SECRETS = {
    "FRED_API_KEY": "ASTROBOT_FRED_API_KEY",
    "NOAA_API_TOKEN": "ASTROBOT_NOAA_API_TOKEN",
    "BLS_API_KEY": "ASTROBOT_BLS_API_KEY",
}

_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")
_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@lru_cache(maxsize=None)
def _secret(name: str) -> str:
    """API key from env (ASTROBOT_<NAME>) or Secret Manager secret <NAME>."""
    value = os.getenv(_SECRETS[name], "").strip()
    if value:
        return value
    project = os.getenv("GOOGLE_CLOUD_PROJECT", "").strip()
    if not project:
        return ""
    try:
        from google.cloud import secretmanager

        client = secretmanager.SecretManagerServiceClient()
        response = client.access_secret_version(
            request={"name": f"projects/{project}/secrets/{name}/versions/latest"}
        )
        return response.payload.data.decode("utf-8").strip()
    except Exception as exc:  # NotFound, PermissionDenied, auth errors
        logger.warning("Secret %s unavailable: %s", name, type(exc).__name__)
        return ""


def _missing_key(source: str, name: str) -> dict[str, Any]:
    return {
        "source": source,
        "error": (
            f"{source} is not configured: set {_SECRETS[name]} or create "
            f"Secret Manager secret '{name}'."
        ),
    }


def _summarise(points: list[dict[str, Any]]) -> dict[str, Any]:
    values = [p["value"] for p in points if isinstance(p.get("value"), (int, float))]
    if not values:
        return {}
    first, last = values[0], values[-1]
    return {
        "first": first,
        "last": last,
        "min": min(values),
        "max": max(values),
        "change_pct": round((last - first) / first * 100, 2) if first else None,
        "points": len(values),
    }


def _as_float(raw: Any) -> float | None:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _http_error(source: str, exc: Exception) -> dict[str, Any]:
    logger.warning("%s request failed: %s", source, exc)
    return {"source": source, "error": f"{source} request failed: {type(exc).__name__}"}


# ============================================================
# FRED
# ============================================================

def get_fred_series(
    series_id: str,
    observation_start: str,
    observation_end: str,
) -> dict[str, Any]:
    """Fetch a FRED economic time series (St. Louis Fed).

    Useful series: UMCSENT (consumer sentiment), FEDFUNDS (Fed funds rate),
    CPIAUCSL (CPI), CUSR0000SETG01 (airline fares CPI), CUSR0000SEFV (food
    away from home CPI), UNRATE (unemployment), RSXFS (retail sales),
    PCE (consumer spending), AIRRPMTSID11 (air revenue passenger miles),
    DSPIC96 (real disposable income).

    Args:
        series_id: FRED series ID, e.g. "UMCSENT".
        observation_start: Start date YYYY-MM-DD.
        observation_end: End date YYYY-MM-DD.
    """
    if not _ID_PATTERN.match(series_id or ""):
        return {"source": "FRED", "error": "Invalid series_id."}
    if not (_DATE_PATTERN.match(observation_start or "") and _DATE_PATTERN.match(observation_end or "")):
        return {"source": "FRED", "error": "Dates must be YYYY-MM-DD."}
    key = _secret("FRED_API_KEY")
    if not key:
        return _missing_key("FRED", "FRED_API_KEY")

    try:
        response = requests.get(
            "https://api.stlouisfed.org/fred/series/observations",
            params={
                "series_id": series_id.upper(),
                "api_key": key,
                "file_type": "json",
                "observation_start": observation_start,
                "observation_end": observation_end,
            },
            timeout=TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        observations = response.json().get("observations", [])
    except Exception as exc:
        return _http_error("FRED", exc)

    points = [
        {"date": o.get("date"), "value": _as_float(o.get("value"))}
        for o in observations
        if _as_float(o.get("value")) is not None
    ]
    if not points:
        return {"source": "FRED", "series_id": series_id, "error": "No data for this series and date range."}
    return {
        "source": "FRED (Federal Reserve Bank of St. Louis)",
        "series_id": series_id.upper(),
        "description": FRED_SERIES.get(series_id.upper(), ""),
        "summary": _summarise(points),
        "observations": points[-MAX_POINTS:],
    }


# ============================================================
# BLS
# ============================================================

def get_bls_series(
    series_id: str,
    start_year: str,
    end_year: str,
) -> dict[str, Any]:
    """Fetch a US Bureau of Labor Statistics time series.

    Examples: LNS14000000 (unemployment rate), CES0000000001 (total nonfarm
    employment), CUUR0000SA0 (CPI-U), CES7000000001 (leisure & hospitality
    employment).

    Args:
        series_id: BLS series ID.
        start_year: First year, e.g. "2025".
        end_year: Last year, e.g. "2026".
    """
    if not _ID_PATTERN.match(series_id or ""):
        return {"source": "BLS", "error": "Invalid series_id."}
    if not (str(start_year).isdigit() and str(end_year).isdigit()):
        return {"source": "BLS", "error": "Years must be numbers like 2026."}

    payload: dict[str, Any] = {
        "seriesid": [series_id],
        "startyear": str(start_year),
        "endyear": str(end_year),
    }
    key = _secret("BLS_API_KEY")
    if key:  # optional; without it BLS applies lower rate limits
        payload["registrationkey"] = key

    try:
        response = requests.post(
            "https://api.bls.gov/publicAPI/v2/timeseries/data/",
            json=payload,
            timeout=TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        body = response.json()
    except Exception as exc:
        return _http_error("BLS", exc)

    if body.get("status") != "REQUEST_SUCCEEDED":
        return {"source": "BLS", "series_id": series_id,
                "error": "; ".join(body.get("message", [])) or "BLS request failed."}

    series = (body.get("Results", {}).get("series") or [{}])[0]
    points = [
        {"date": f"{d.get('year')}-{d.get('periodName')}", "value": _as_float(d.get("value"))}
        for d in reversed(series.get("data", []))
        if _as_float(d.get("value")) is not None
    ]
    if not points:
        return {"source": "BLS", "series_id": series_id, "error": "No data for this series and range."}
    return {
        "source": "US Bureau of Labor Statistics",
        "series_id": series_id,
        "summary": _summarise(points),
        "observations": points[-MAX_POINTS:],
    }


# ============================================================
# NOAA
# ============================================================

def get_noaa_daily_weather(
    station_id: str,
    start_date: str,
    end_date: str,
) -> dict[str, Any]:
    """Fetch daily weather (max/min temperature in F, precipitation in inches)
    from NOAA's Climate Data Online GHCND dataset.

    Example stations: GHCND:USW00023169 (Las Vegas airport),
    GHCND:USW00012839 (Miami airport), GHCND:USW00013874 (Atlanta airport).

    Args:
        station_id: NOAA station ID, e.g. "GHCND:USW00023169".
        start_date: YYYY-MM-DD.
        end_date: YYYY-MM-DD (at most one year after start_date).
    """
    if not _ID_PATTERN.match(station_id or ""):
        return {"source": "NOAA", "error": "Invalid station_id."}
    if not (_DATE_PATTERN.match(start_date or "") and _DATE_PATTERN.match(end_date or "")):
        return {"source": "NOAA", "error": "Dates must be YYYY-MM-DD."}
    token = _secret("NOAA_API_TOKEN")
    if not token:
        return _missing_key("NOAA", "NOAA_API_TOKEN")

    try:
        response = requests.get(
            "https://www.ncei.noaa.gov/cdo-web/api/v2/data",
            headers={"token": token},
            params={
                "datasetid": "GHCND",
                "stationid": station_id,
                "startdate": start_date,
                "enddate": end_date,
                "datatypeid": ["TMAX", "TMIN", "PRCP"],
                "units": "standard",
                "limit": 1000,
            },
            timeout=TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        results = response.json().get("results", [])
    except Exception as exc:
        return _http_error("NOAA", exc)

    by_day: dict[str, dict[str, Any]] = {}
    for row in results:
        day = str(row.get("date", ""))[:10]
        by_day.setdefault(day, {"date": day})[row.get("datatype", "").lower()] = row.get("value")
    days = [by_day[d] for d in sorted(by_day)]
    if not days:
        return {"source": "NOAA", "station_id": station_id, "error": "No weather data for this station and range."}
    return {
        "source": "NOAA Climate Data Online (GHCND)",
        "station_id": station_id,
        "units": {"tmax": "F", "tmin": "F", "prcp": "in"},
        "days": days[-MAX_POINTS:],
    }


# ============================================================
# HOLIDAYS
# ============================================================

def get_us_holidays(year: str, state_code: str = "") -> dict[str, Any]:
    """US public holidays for a year (source: Nager.Date): federal holidays,
    plus holidays specific to one state when state_code is given (e.g. "NV").
    State coverage in Nager.Date is incomplete.

    Args:
        year: Four-digit year, e.g. "2026".
        state_code: Optional two-letter US state code.
    """
    if not (str(year).isdigit() and len(str(year)) == 4):
        return {"source": "Nager.Date", "error": "Year must be four digits."}
    state = str(state_code or "").strip().upper()
    if state and not re.fullmatch(r"[A-Z]{2}", state):
        return {"source": "Nager.Date", "error": "state_code must be two letters."}

    try:
        response = requests.get(
            f"https://date.nager.at/api/v3/PublicHolidays/{year}/US",
            timeout=TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        holidays = response.json()
    except Exception as exc:
        return _http_error("Nager.Date", exc)

    federal = [
        {"date": h.get("date"), "name": h.get("name")}
        for h in holidays
        if h.get("global")
    ]
    state_specific = [
        {"date": h.get("date"), "name": h.get("name")}
        for h in holidays
        if state and not h.get("global") and f"US-{state}" in (h.get("counties") or [])
    ]
    result = {
        "source": "Nager.Date public holiday API",
        "year": str(year),
        "federal": federal,
    }
    if state:
        result["state"] = state
        result["state_specific"] = state_specific
        result["note"] = (
            "Nager.Date covers federal holidays fully but only some state "
            "holidays; state-specific days may be missing. Check the state's "
            "official calendar for completeness."
        )
    return result


ECONOMIST_DATA_TOOLS = [
    get_fred_series,
    get_bls_series,
    get_noaa_daily_weather,
    get_us_holidays,
]

__all__ = ["ECONOMIST_DATA_TOOLS", "FRED_SERIES"]

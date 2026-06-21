"""
Fetches real-time context (time, location, weather) for outfit recommendations.

Two API calls:
  1. ip-api.com    — IP geolocation → city, country, lat/lon
  2. open-meteo.com — weather from lat/lon → temp, conditions, wind

fetch_cached() adds a 5-minute TTL so repeated tool calls in one session
don't re-hit the network. fetch() is always live.
"""
import json
import ssl
import time as _time
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import certifi

# macOS Python doesn't trust the system keychain; certifi ships its own CA bundle.
_SSL_CTX = ssl.create_default_context(cafile=certifi.where())

_WMO: dict[int, str] = {
    0: "clear sky",
    1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "icy fog",
    51: "light drizzle", 53: "drizzle", 55: "heavy drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains",
    80: "light showers", 81: "showers", 82: "heavy showers",
    85: "light snow showers", 86: "heavy snow showers",
    95: "thunderstorm", 96: "thunderstorm with hail", 99: "severe thunderstorm",
}

_TTL_SECONDS = 300  # 5 minutes
_cache: tuple["Context", float] | None = None  # (result, monotonic timestamp)


@dataclass
class Context:
    now: str
    city: str
    country: str
    temp_f: float
    feels_like_f: float
    conditions: str
    wind_mph: float


def _get_json(url: str, timeout: int = 5) -> dict:
    with urllib.request.urlopen(url, timeout=timeout, context=_SSL_CTX) as resp:
        return json.loads(resp.read())


def fetch() -> Optional[Context]:
    """Fetch live weather — always makes network calls."""
    try:
        loc = _get_json("http://ip-api.com/json/")
        lat, lon = loc["lat"], loc["lon"]
        weather = _get_json(
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            f"&current=temperature_2m,apparent_temperature,weather_code,wind_speed_10m"
            f"&temperature_unit=fahrenheit&wind_speed_unit=mph"
        )["current"]
        now = datetime.now()
        hour = now.strftime("%I").lstrip("0") or "12"
        return Context(
            now=now.strftime(f"%A {hour}:%M %p"),
            city=loc.get("city", ""),
            country=loc.get("country", ""),
            temp_f=weather["temperature_2m"],
            feels_like_f=weather["apparent_temperature"],
            conditions=_WMO.get(weather["weather_code"], "unknown"),
            wind_mph=weather["wind_speed_10m"],
        )
    except Exception:
        return None


def fetch_cached() -> Optional[Context]:
    """Return cached weather if <5 min old, otherwise fetch live and cache it."""
    global _cache
    if _cache is not None:
        ctx, ts = _cache
        if _time.monotonic() - ts < _TTL_SECONDS:
            return ctx
    ctx = fetch()
    if ctx is not None:
        _cache = (ctx, _time.monotonic())
    return ctx


def to_string(ctx: Context) -> str:
    """Format a Context as a short readable string for model tool responses."""
    return (
        f"Date/time: {ctx.now}\n"
        f"Location: {ctx.city}, {ctx.country}\n"
        f"Temperature: {ctx.temp_f:.0f}°F (feels like {ctx.feels_like_f:.0f}°F)\n"
        f"Conditions: {ctx.conditions}\n"
        f"Wind: {ctx.wind_mph:.0f} mph"
    )

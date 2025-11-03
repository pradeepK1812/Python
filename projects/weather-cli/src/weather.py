import requests
import json
import os
import time

CACHE_FILE = "weather_cache.json"
CACHE_EXPIRY = 3600  # 1 hour

GEO_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
REQUEST_TIMEOUT = 8  # faster timeout


# Weather code mapping
WEATHER_CODE_MAP = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    61: "Light rain",
    63: "Moderate rain",
    65: "Heavy rain",
    71: "Light snow",
    73: "Moderate snow",
    75: "Heavy snow",
    95: "Thunderstorm",
    99: "Thunderstorm w/ hail",
}


def load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    return {}


def save_cache(cache):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f)


def _normalize_city(city: str):
    return city.strip().title()


def _try_variations(city: str):
    """Try different formats for better matching"""
    parts = city.split()
    variations = [city]
    if len(parts) == 2:
        variations.append(", ".join(parts))  # New Delhi → New, Delhi

    # Prioritize India fallback
    if not city.lower().endswith(", india"):
        variations.append(f"{city}, India")

    return variations


def _geocode_first_result(city: str):
    try:
        resp = requests.get(
            GEO_URL, params={"name": city, "count": 1}, timeout=REQUEST_TIMEOUT
        )
        resp.raise_for_status()
        results = resp.json().get("results")
        return results[0] if results else None
    except requests.RequestException:
        return None


def get_weather(city: str) -> dict:
    if not city or not city.strip():
        raise ValueError("City name is empty.")

    normalized_city = _normalize_city(city)
    cache = load_cache()

    # Cache hit
    if normalized_city in cache:
        entry = cache[normalized_city]
        if time.time() - entry["timestamp"] < CACHE_EXPIRY:
            return entry["weather"]

    # Try variations
    for query in _try_variations(normalized_city):
        geo = _geocode_first_result(query)
        if not geo:
            continue

        lat, lon = geo["latitude"], geo["longitude"]
        resolved_name = geo["name"]

        try:
            wresp = requests.get(
                WEATHER_URL,
                params={"latitude": lat, "longitude": lon, "current_weather": True},
                timeout=REQUEST_TIMEOUT,
            )
            wresp.raise_for_status()
            wdata = wresp.json()
        except requests.RequestException as e:
            raise RuntimeError(f"Network error: {e}")

        current = wdata.get("current_weather")
        if not current:
            continue

        code = int(current.get("weathercode", -1))
        weather_text = WEATHER_CODE_MAP.get(code, f"Weather code {code}")

        weather_info = {
            "resolved_city": resolved_name,
            "latitude": lat,
            "longitude": lon,
            "temperature": current.get("temperature"),
            "windspeed": current.get("windspeed"),
            "weathercode": code,
            "weather": weather_text,
        }

        # Save to cache
        cache[normalized_city] = {"timestamp": time.time(), "weather": weather_info}
        save_cache(cache)

        return weather_info

    raise ValueError(f"Could not fetch weather for '{city}'. Try a different name.")

import requests

GEO_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
REQUEST_TIMEOUT = 10

# Map of Open-Meteo weather codes to text
WEATHER_CODE_MAP = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    71: "Slight snow fall",
    73: "Moderate snow fall",
    75: "Heavy snow fall",
    95: "Thunderstorm",
    99: "Thunderstorm with hail",
}


def _normalize_city(city: str) -> str:
    """Normalize city input (trim, capitalize words)."""
    return " ".join(word.capitalize() for word in city.strip().split())


def _try_variations(city: str):
    """Generate query variations for better matching."""
    parts = city.split()
    variations = [city]
    if len(parts) == 2:
        variations.append(", ".join(parts))  # e.g., "New Delhi" → "New, Delhi"
    if " " not in city and not city.lower().endswith(", india"):
        variations.append(f"{city}, India")
    return variations


def _geocode_first_result(city: str):
    """Fetch first valid geocode result from Open-Meteo."""
    try:
        resp = requests.get(
            GEO_URL,
            params={"name": city, "count": 1, "language": "en"},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results")
        if not results:
            return None
        return results[0]
    except requests.RequestException:
        return None


def get_weather(city: str) -> dict:
    """
    Fetch current weather for a city using Open-Meteo API
    with fallback and normalization.
    """
    if not city or not city.strip():
        raise ValueError("City name is empty.")

    normalized_city = _normalize_city(city)
    possible_queries = _try_variations(normalized_city)

    # Fallback for common Indian cities
    if normalized_city.lower() in {"mumbai", "bombay"}:
        possible_queries.insert(0, "Mumbai, India")
    elif normalized_city.lower() in {"new delhi", "delhi"}:
        possible_queries.insert(0, "New Delhi, India")

    for query in possible_queries:
        geo = _geocode_first_result(query)
        if not geo:
            continue

        latitude = geo.get("latitude")
        longitude = geo.get("longitude")
        resolved_name = geo.get("name")

        try:
            wresp = requests.get(
                WEATHER_URL,
                params={
                    "latitude": latitude,
                    "longitude": longitude,
                    "current_weather": True,
                },
                timeout=REQUEST_TIMEOUT,
            )
            wresp.raise_for_status()
            wdata = wresp.json()
        except requests.RequestException as e:
            raise RuntimeError(f"Network error while fetching weather: {e}")

        current = wdata.get("current_weather")
        if not current:
            continue

        code = int(current.get("weathercode", -1))
        weather_text = WEATHER_CODE_MAP.get(code, f"Weather code {code}")

        return {
            "resolved_city": resolved_name,
            "latitude": latitude,
            "longitude": longitude,
            "temperature": current.get("temperature"),
            "windspeed": current.get("windspeed"),
            "weathercode": code,
            "weather": weather_text,
        }

    raise ValueError(f"Could not find location for '{city}'. Try a different spelling.")

import pytest
from weather import get_weather

def test_weather_returns_valid_data():
    """Basic smoke test for get_weather function."""
    data = get_weather("Mumbai")
    assert "temperature" in data
    assert "windspeed" in data
    assert "weather" in data
    assert data["resolved_city"].lower() in ["mumbai", "bombay"]

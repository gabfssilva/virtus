"""Functions become tools; the model calls them until it can answer. Sync tools run in a thread."""

import asyncio
import json
from urllib.parse import urlencode
from urllib.request import urlopen

from pydantic import BaseModel

from virtus import agent, observe, tool
from virtus.providers import openrouter
from virtus.tools import http

CURRENT = [
    "temperature_2m",
    "apparent_temperature",
    "relative_humidity_2m",
    "precipitation",
    "weather_code",
    "cloud_cover",
    "pressure_msl",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
]
DAILY = [
    "weather_code",
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "precipitation_probability_max",
    "wind_speed_10m_max",
    "uv_index_max",
    "sunrise",
    "sunset",
]


class Place(BaseModel):
    name: str
    latitude: float
    longitude: float
    admin1: str | None = None
    country: str | None = None


class Geocoding(BaseModel):
    results: list[Place] = []


@tool
def get_weather(city: str) -> str:
    """Current conditions and the 7-day forecast for a city, from Open-Meteo. Weather codes are WMO codes."""
    found = Geocoding.model_validate_json(_get("https://geocoding-api.open-meteo.com/v1/search", name=city, count=1))
    if not found.results:
        return f"No city named {city}."
    place = found.results[0]
    forecast = _get(
        "https://api.open-meteo.com/v1/forecast",
        latitude=place.latitude,
        longitude=place.longitude,
        current=",".join(CURRENT),
        daily=",".join(DAILY),
        timezone="auto",
    )
    location = ", ".join(part for part in (place.name, place.admin1, place.country) if part)
    return json.dumps({"location": location, "forecast": json.loads(forecast)})


def _get(url: str, **params: str | int | float) -> bytes:
    with urlopen(f"{url}?{urlencode(params)}", timeout=10) as response:
        return response.read()


@agent(provider=openrouter("mercury-3"), capabilities=[observe(), http()])
async def weather_agent(city: str) -> str:
    """Get the weather for the city {city} via OpenMeteo API."""
    ...


async def main() -> None:
    print(await weather_agent("São Paulo"))


asyncio.run(main())

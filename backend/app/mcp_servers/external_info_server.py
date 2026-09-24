import json
from datetime import (
    datetime,
)
from zoneinfo import (
    ZoneInfo,
)

import httpx
from fastmcp import FastMCP


mcp = FastMCP(
    "NodAgent External Info Server"
)


@mcp.tool()
def get_current_time(
    timezone: str,
) -> str:
    """
    Get the current date and time for an
    IANA timezone.

    Example:
    Asia/Shanghai
    Asia/Tokyo
    America/New_York
    """

    try:
        now = datetime.now(
            ZoneInfo(timezone)
        )

    except Exception:
        return json.dumps(
            {
                "error": (
                    "Invalid timezone"
                ),
                "timezone": timezone,
            },
            ensure_ascii=False,
        )

    return json.dumps(
        {
            "timezone": timezone,
            "datetime": (
                now.isoformat()
            ),
        },
        ensure_ascii=False,
    )


@mcp.tool()
async def get_weather(
    city: str,
) -> str:
    """
    Get current weather for a city.

    This tool uses the public Open-Meteo
    geocoding and weather APIs.
    """

    city = city.strip()

    if not city:
        return json.dumps(
            {
                "error": (
                    "City cannot be empty"
                )
            },
            ensure_ascii=False,
        )

    timeout = httpx.Timeout(
        10.0
    )

    async with httpx.AsyncClient(
        timeout=timeout
    ) as client:
        geo_response = (
            await client.get(
                (
                    "https://geocoding-api."
                    "open-meteo.com/v1/search"
                ),
                params={
                    "name": city,
                    "count": 1,
                    "language": "en",
                    "format": "json",
                },
            )
        )

        geo_response.raise_for_status()

        geo_data = (
            geo_response.json()
        )

        results = (
            geo_data.get(
                "results",
                []
            )
        )

        if not results:
            return json.dumps(
                {
                    "error": (
                        "City not found"
                    ),
                    "city": city,
                },
                ensure_ascii=False,
            )

        location = results[0]

        latitude = location[
            "latitude"
        ]

        longitude = location[
            "longitude"
        ]

        weather_response = (
            await client.get(
                (
                    "https://api."
                    "open-meteo.com/"
                    "v1/forecast"
                ),
                params={
                    "latitude": (
                        latitude
                    ),
                    "longitude": (
                        longitude
                    ),
                    "current": (
                        "temperature_2m,"
                        "apparent_temperature,"
                        "weather_code,"
                        "wind_speed_10m"
                    ),
                    "timezone": "auto",
                },
            )
        )

        weather_response.raise_for_status()

        weather_data = (
            weather_response.json()
        )

    return json.dumps(
        {
            "city": location.get(
                "name"
            ),
            "country": location.get(
                "country"
            ),
            "latitude": latitude,
            "longitude": longitude,
            "timezone": (
                weather_data.get(
                    "timezone"
                )
            ),
            "current": (
                weather_data.get(
                    "current"
                )
            ),
        },
        ensure_ascii=False,
    )


if __name__ == "__main__":
    mcp.run(
        transport="stdio"
    )

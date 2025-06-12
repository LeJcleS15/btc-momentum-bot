import requests
import json
import time
import logging
from datetime import datetime, timedelta


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)


def fetch_drift_ohlcv(
    market_index: int = 1,
    market_type: str = "perp",
    resolution: int = 1,
    days: int = 60,
    output_path: str = "btc_1m.json",
) -> int:
    """
    Fetch OHLCV data from Drift for given market over a lookback window.
    """
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)
    start_ms = int(start_time.timestamp() * 1000)
    end_ms = int(end_time.timestamp() * 1000)

    logger.info(
        f"Fetching {market_type.upper()} data: {start_time.date()} to {end_time.date()}"
    )

    url = "https://mainnet-beta.api.drift.trade/tv/history"
    headers = {
        "origin": "https://app.drift.trade",
        "referer": "https://app.drift.trade",
    }

    interval_ms = resolution * 60 * 1000
    step = 2000 * interval_ms
    all_candles = []

    for t in range(start_ms, end_ms, step):
        fr, to = t, min(t + step, end_ms)
        params = {
            "marketIndex": market_index,
            "marketType": market_type,
            "resolution": resolution,
            "from": fr,
            "to": to,
        }

        logger.info(
            f"Fetching {datetime.fromtimestamp(fr / 1000)} → {datetime.fromtimestamp(to / 1000)}"
        )
        r = requests.get(url, headers=headers, params=params)
        r.raise_for_status()
        candles = r.json().get("candles", [])
        all_candles.extend(candles)
        time.sleep(0.2)

    logger.info(f"Total candles: {len(all_candles)}")

    with open(output_path, "w") as f:
        json.dump({"candles": all_candles}, f)

    logger.info(f"Saved to {output_path}")
    return len(all_candles)


if __name__ == "__main__":
    fetch_drift_ohlcv(
        market_index=1,
        market_type="perp",
        resolution=1,
        days=185,
        output_path="./data/btc_1m_185days.json",
    )

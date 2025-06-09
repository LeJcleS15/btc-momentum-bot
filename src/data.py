import requests
import json
import time
from datetime import datetime, timedelta


def log(msg):
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {msg}")


def fetch_btc_1m_2months():
    """
    Fetch exactly 3 months of BTC-PERP 1m data from Drift
    """

    end_time = datetime.now()
    start_time = end_time - timedelta(days=60)

    start_ms = int(start_time.timestamp() * 1000)
    end_ms = int(end_time.timestamp() * 1000)

    log(f"Fetching BTC-PERP 1m: {start_time.date()} to {end_time.date()}")

    url = "https://mainnet-beta.api.drift.trade/tv/history"
    headers = {
        "origin": "https://app.drift.trade",
        "referer": "https://app.drift.trade",
    }

    all_candles = []
    interval_ms = 60 * 1000  # 1 minute
    step = 2000 * interval_ms  # 2000 bars per request

    for t in range(start_ms, end_ms, step):
        fr, to = t, min(t + step, end_ms)

        log(
            f"Fetching {datetime.fromtimestamp(fr / 1000)} → {datetime.fromtimestamp(to / 1000)}"
        )

        params = {
            # https://github.com/drift-labs/protocol-v2/blob/acf51a0f57a467334948ad54b3799ac3014fdb9e/sdk/src/constants/perpMarkets.ts#L402
            "marketIndex": 1,
            "marketType": "perp",
            "resolution": 1,
            "from": fr,
            "to": to,
        }

        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        candles = response.json().get("candles", [])

        log(f"Fetched {len(candles)} candles")
        all_candles.extend(candles)

        time.sleep(0.2)

    log(f"Total candles: {len(all_candles)}")

    with open("btc_1m_2months.json", "w") as f:
        json.dump({"candles": all_candles}, f)

    log(f"Saved to btc_1m_2months.json")
    return len(all_candles)


if __name__ == "__main__":
    fetch_btc_1m_2months()

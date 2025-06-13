import json
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_data(file_path: str = "./data/btc_1m_185days.json") -> pd.DataFrame:
    with open(file_path) as f:
        data = json.load(f)

    df = pd.DataFrame(data["candles"])
    df["timestamp"] = pd.to_datetime(df["start"].astype(int), unit="ms")
    df = df.set_index("timestamp").sort_index()

    return df[["oracleClose", "fillHigh", "fillLow", "fillClose", "baseVolume"]].astype(
        float
    )


def calculate_spreads(df: pd.DataFrame) -> pd.DataFrame:
    df["spread_bps"] = ((df["fillHigh"] - df["fillLow"]) / df["oracleClose"]) * 10000
    df["buy_cost_bps"] = (
        (df["fillHigh"] - df["oracleClose"]) / df["oracleClose"]
    ) * 10000
    df["sell_cost_bps"] = (
        (df["oracleClose"] - df["fillLow"]) / df["oracleClose"]
    ) * 10000
    df["round_trip_bps"] = df["buy_cost_bps"] + df["sell_cost_bps"]
    return df


def analyze_periods(df: pd.DataFrame) -> dict:
    end_ts = df.index[-1]
    return {
        "last_30_days": df[df.index >= end_ts - pd.Timedelta(days=30)],
        "last_60_days": df[df.index >= end_ts - pd.Timedelta(days=60)],
        "last_90_days": df[df.index >= end_ts - pd.Timedelta(days=90)],
        "full_period": df,
    }


def get_stats(df: pd.DataFrame) -> dict:
    return {
        "candles": len(df),
        "avg_spread": df["spread_bps"].mean(),
        "med_spread": df["spread_bps"].median(),
        "p90_spread": df["spread_bps"].quantile(0.9),
        "avg_round_trip": df["round_trip_bps"].mean(),
        "med_round_trip": df["round_trip_bps"].median(),
        "avg_buy_cost": df["buy_cost_bps"].mean(),
        "avg_sell_cost": df["sell_cost_bps"].mean(),
    }


def main():
    logger.info("BTC spread cost analysis")

    df = load_data()
    df = calculate_spreads(df)
    periods = analyze_periods(df)

    for name, data in periods.items():
        stats = get_stats(data)

        logger.info(f"{name.upper().replace('_', ' ')}")
        logger.info(f"Candles: {stats['candles']:,}")
        logger.info(f"Avg spread: {stats['avg_spread']:.2f} bps")
        logger.info(f"Median spread: {stats['med_spread']:.2f} bps")
        logger.info(f"90th %ile: {stats['p90_spread']:.2f} bps")
        logger.info(f"Round trip (avg): {stats['avg_round_trip']:.2f} bps")
        logger.info(f"Round trip (med): {stats['med_round_trip']:.2f} bps")
        logger.info(f"Buy cost: {stats['avg_buy_cost']:.2f} bps")
        logger.info(f"Sell cost: {stats['avg_sell_cost']:.2f} bps\n")

    last_month = periods["last_30_days"]
    daily_cost = last_month["round_trip_bps"].mean() * 15  # 15 round trips/day
    monthly_cost = daily_cost * 30

    logger.info("IMPACT (30 trades/day)")
    logger.info(f"Daily cost: {daily_cost:.1f} bps")
    logger.info(f"Monthly cost: {monthly_cost:.1f} bps ({monthly_cost / 100:.2f}%)")


if __name__ == "__main__":
    main()

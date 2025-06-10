import json
import pandas as pd
import numpy as np
import logging
import random
from typing import List, Optional, NamedTuple
import multiprocessing as mp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)


class Position(NamedTuple):
    signal: int
    entry_price: float
    quantity: float
    capital: float  # USD


class MarkToMarket(NamedTuple):
    pnl: float  # USD
    equity: float  # USD


def load_price_data(file_path: str = "./data/btc_1m_3months.json") -> pd.DataFrame:
    logger.info(f"Loading price data from {file_path}")

    with open(file_path) as f:
        data = json.load(f)

    df = pd.DataFrame(data["candles"])
    df["timestamp"] = pd.to_datetime(df["start"].astype(int), unit="ms")
    df = df.set_index("timestamp")
    df["price"] = df["oracleClose"].astype(float)
    df["volume"] = df["baseVolume"].astype(float)  # BTC volume

    initial_rows = len(df)
    df = df[~df.index.duplicated(keep="first")]
    df = df.sort_index()
    final_rows = len(df)

    logger.info(
        f"Loaded {final_rows} rows (removed {initial_rows - final_rows} duplicates)"
    )
    return df[["price", "volume"]]


def precompute_all_emas(price_data: pd.DataFrame, max_window: int = 250) -> dict:
    logger.info(f"Pre-computing EMAs for windows 2 to {max_window}")

    ema_cache = {}
    for window in range(2, max_window + 1):
        ema_cache[window] = price_data["price"].ewm(span=window).mean()

    logger.info(f"Cached {len(ema_cache)} EMA series")
    return ema_cache


def generate_momentum_signal(
    fast_ema: pd.Series,
    slow_ema: pd.Series,
    volume: pd.Series = None,
    vol_threshold: float = 0.01,
) -> pd.Series:
    momentum = fast_ema - slow_ema
    price_signal = np.sign(momentum).shift(2)

    if volume is not None:
        # only trade when volume > recent average
        vol_ma = volume.rolling(20).mean()  # 20-period volume average
        volume_filter = (volume > vol_ma * vol_threshold).shift(2)
        price_signal = price_signal * volume_filter

    return price_signal.dropna()


def create_ensemble_signal(
    price_data: pd.DataFrame, strategy_configs: List[dict], ema_cache: dict = None
) -> pd.DataFrame:
    logger.debug(f"Creating ensemble from {len(strategy_configs)} strategies")

    if len(strategy_configs) == 0:
        result = price_data.copy()
        result["ensemble_signal"] = 0
        return result

    individual_signals = []
    for config in strategy_configs:
        fast, slow = config["fast"], config["slow"]

        if ema_cache and fast in ema_cache and slow in ema_cache:
            ema_fast = ema_cache[fast]
            ema_slow = ema_cache[slow]
        else:
            ema_fast = price_data["price"].ewm(span=fast).mean()
            ema_slow = price_data["price"].ewm(span=slow).mean()

        signal = generate_momentum_signal(ema_fast, ema_slow, price_data["volume"])
        individual_signals.append(signal)

    df = price_data.copy()
    signals_matrix = pd.concat(individual_signals, axis=1)
    df["ensemble_signal"] = signals_matrix.median(axis=1)

    return df.dropna()


def open_new_position(
    signal: int, price: float, btc_qty: float, capital: float
) -> Position:
    return Position(signal, price, btc_qty, capital)


def calculate_mtm(position: Position, current_price: float) -> MarkToMarket:
    pnl = position.signal * position.quantity * (current_price - position.entry_price)
    equity = position.capital + pnl
    return MarkToMarket(pnl, equity)


def backtest_ensemble(
    df: pd.DataFrame,
    strategies: List[dict],
    btc_qty: float = 1,
    ema_cache: dict = None,
) -> pd.DataFrame:
    if len(strategies) == 0:
        return pd.DataFrame()

    df_signals = create_ensemble_signal(df, strategies, ema_cache)

    if len(df_signals) == 0:
        return pd.DataFrame()

    timeline = []
    position: Optional[Position] = None
    initial_capital_usd = 0

    for _, row in df_signals.iterrows():
        ts = row.name
        price = row.price
        signal = int(row.ensemble_signal)

        snap = {
            "ts": ts,
            "price": price,
            "signal": signal,
            "is_entry": False,
            "is_exit": False,
            "qty": 0.0,
            "pnl": 0.0,
            "equity": 0.0,
        }

        if position is not None:
            mtm = calculate_mtm(position, price)
            snap["qty"] = position.quantity
            snap["pnl"] = mtm.pnl
            snap["equity"] = mtm.equity

            if signal != position.signal:
                snap["is_exit"] = True
                current_equity = mtm.equity

                if signal != 0:
                    position = open_new_position(signal, price, btc_qty, current_equity)
                    snap["is_entry"] = True
                    snap["qty"] = position.quantity
                    snap["equity"] = current_equity
                else:
                    position = None

        elif signal != 0:
            initial_capital_usd = price * btc_qty
            position = open_new_position(signal, price, btc_qty, initial_capital_usd)
            snap["is_entry"] = True
            snap["qty"] = position.quantity
            snap["equity"] = initial_capital_usd

        else:
            # When no position and no signal, carry forward last equity
            if len(timeline) > 0:
                snap["equity"] = timeline[-1]["equity"]

        timeline.append(snap)

    if (
        position is not None
        and len(timeline) > 0
        and timeline[-1]["ts"] != df_signals.index[-1]
    ):
        last_row = df_signals.iloc[-1]
        mtm = calculate_mtm(position, last_row.price)
        timeline.append(
            {
                "ts": last_row.name,
                "price": last_row.price,
                "signal": position.signal,
                "is_entry": False,
                "is_exit": True,
                "qty": position.quantity,
                "pnl": mtm.pnl,
                "equity": mtm.equity,
            }
        )

    return pd.DataFrame(timeline).set_index("ts")


def calculate_performance_metrics(timeline: pd.DataFrame, capital_0: float) -> dict:
    if capital_0 == 0 or timeline.empty:
        return {
            k: 0
            for k in [
                "net_pnl",
                "total_return_pct",
                "sharpe_ratio",
                "max_drawdown_pct",
                "win_rate_pct",
                "trades_per_year",
                "avg_hold_hours",
            ]
        }

    equity = timeline["equity"].ffill().dropna()
    duration = (timeline.index[-1] - timeline.index[0]).total_seconds() / 86400

    trades = timeline[timeline.is_exit]
    trade_returns = trades["equity"].diff().dropna()
    hold_durations = trades.index.to_series().diff().dt.total_seconds().dropna() / 3600

    rf_ann = 0.0406
    avg_hold = hold_durations.mean() if not hold_durations.empty else 1
    rf_trade = rf_ann * (avg_hold / 8760)
    excess = trade_returns / capital_0 - rf_trade
    std = excess.std()

    return {
        "net_pnl": equity.iloc[-1] - capital_0,
        "total_return_pct": (equity.iloc[-1] / capital_0 - 1) * 100,
        "sharpe_ratio": (excess.mean() / std * np.sqrt(8760 / avg_hold))
        if std > 1e-8
        else 0,
        "max_drawdown_pct": 100 * (equity / equity.cummax() - 1).min(),
        "win_rate_pct": 100 * (trade_returns > 0).mean()
        if len(trade_returns) > 0
        else 0,
        "trades_per_year": len(trade_returns) / duration * 365 if duration > 0 else 0,
        "avg_hold_hours": avg_hold,
    }


def test_single_strategy(
    price_data: pd.DataFrame, fast_window: int, slow_window: int, ema_cache: dict = None
) -> dict:
    if fast_window >= slow_window:
        return None

    try:
        config = [{"fast": fast_window, "slow": slow_window}]
        timeline = backtest_ensemble(price_data, config, ema_cache=ema_cache)

        if len(timeline) < 10:
            return None

        first_capital = timeline[timeline.is_entry]["equity"].iloc[0]
        metrics = calculate_performance_metrics(timeline, first_capital)

        return {
            "fast": fast_window,
            "slow": slow_window,
            "sharpe_ratio": metrics["sharpe_ratio"],
            "total_return_pct": metrics["total_return_pct"],
            "trades_per_year": metrics["trades_per_year"],
        }

    except Exception as e:
        logger.debug(f"Strategy test failed for {fast_window}/{slow_window}: {e}")
        return None


def test_strategy_worker(args):
    fast, slow, train_data, ema_cache = args
    return test_single_strategy(train_data, fast, slow, ema_cache)


def search_optimal_strategies_parallel(
    price_data: pd.DataFrame,
    max_strategies: int = 5,
    num_tests: int = 500,
    n_cores: int = None,
) -> tuple:
    if n_cores is None:
        n_cores = mp.cpu_count() - 1

    logger.info(
        f"Parallel search testing {num_tests} EMA combinations using {n_cores} cores"
    )
    logger.info(f"Using {len(price_data)} rows for strategy search")

    ema_cache = precompute_all_emas(price_data, 120)

    random.seed(42)
    ema_pairs = list(
        set((random.randint(2, 30), random.randint(32, 120)) for _ in range(num_tests))
    )

    worker_args = [(f, s, price_data, ema_cache) for f, s in ema_pairs]

    with mp.Pool(processes=n_cores) as pool:
        results = pool.map(test_strategy_worker, worker_args)

    valid_results = [r for r in results if r is not None]
    valid_results.sort(key=lambda x: x["sharpe_ratio"], reverse=True)

    selected = valid_results[:max_strategies]
    logger.info("Top strategies selected:")
    for i, s in enumerate(selected):
        logger.info(
            f"  {i + 1}. EMA({s['fast']},{s['slow']}) - Sharpe={s['sharpe_ratio']:.3f}, Return={s['total_return_pct']:.2f}%"
        )

    return [{"fast": s["fast"], "slow": s["slow"]} for s in selected]


def main():
    logger.info("BTC Momentum Strategy Optimization")

    df = load_price_data()

    # time-based boundaries
    end_ts = df.index[-1]
    last_month_start = end_ts - pd.Timedelta(days=30)
    two_months_prior = last_month_start - pd.Timedelta(days=60)

    # isolate last 3 months
    df = df[df.index >= two_months_prior]

    # extract prior 2 months and split 70/30
    prior_2m = df[(df.index >= two_months_prior) & (df.index < last_month_start)]
    split_idx = int(len(prior_2m) * 0.7)
    train_data = prior_2m.iloc[:split_idx]
    val30_data = prior_2m.iloc[split_idx:]

    # final validation on last month
    test_data = df[df.index >= last_month_start]

    logger.info(
        f"Data split - Train: {len(train_data)} rows, Validation: {len(val30_data)} rows, Test: {len(test_data)} rows"
    )

    strategies = search_optimal_strategies_parallel(train_data)

    if not strategies:
        logger.error("No viable strategies found!")
        return

    def evaluate(stage, data):
        ema_cache = precompute_all_emas(data, 250)
        result = backtest_ensemble(data, strategies, ema_cache=ema_cache)
        if result.empty:
            logger.error(f"No trades on {stage} set!")
            return

        capital_0 = (
            result[result.is_entry]["equity"].iloc[0]
            if len(result[result.is_entry]) > 0
            else 1000.0
        )
        metrics = calculate_performance_metrics(result, capital_0)

        logger.info(f"{stage.title()} Results:")
        logger.info(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
        logger.info(f"  Total Return: {metrics['total_return_pct']:.2f}%")
        logger.info(f"  Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
        logger.info(
            f"  Win Rate: {metrics['win_rate_pct']:.1f}% | Trades/Year: {metrics['trades_per_year']:.0f}"
        )

    evaluate("train", train_data)
    evaluate("validation", val30_data)
    evaluate("test", test_data)


def test():
    logger.info("Ensemble Backtest on Multiple Time Periods")

    df = load_price_data()
    logger.info(f"Full dataset: {len(df)} rows")

    strategies = [
        {"fast": 15, "slow": 108},
        {"fast": 18, "slow": 99},
        {"fast": 15, "slow": 116},
        {"fast": 19, "slow": 99},
        {"fast": 16, "slow": 102},
    ]

    # Time-based boundaries
    end_ts = df.index[-1]
    last_month_start = end_ts - pd.Timedelta(days=30)
    two_months_prior = last_month_start - pd.Timedelta(days=60)

    # Create three datasets
    last_month_data = df[df.index >= last_month_start]
    full_3mo_data = df[df.index >= two_months_prior]
    prior_2mo_data = df[(df.index >= two_months_prior) & (df.index < last_month_start)]

    def run_backtest(data, name):
        logger.info(f"{name} ({len(data)} rows):")

        ema_cache = precompute_all_emas(data, 250)
        timeline = backtest_ensemble(data, strategies, ema_cache=ema_cache)

        if timeline.empty:
            logger.info(f"  No trades generated!")
            return

        first_capital = (
            timeline[timeline.is_entry]["equity"].iloc[0]
            if len(timeline[timeline.is_entry]) > 0
            else 1000.0
        )
        metrics = calculate_performance_metrics(timeline, first_capital)

        logger.info(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
        logger.info(f"  Total Return: {metrics['total_return_pct']:.2f}%")
        logger.info(f"  Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
        logger.info(
            f"  Win Rate: {metrics['win_rate_pct']:.1f}% | Trades/Year: {metrics['trades_per_year']:.0f}"
        )
        logger.info(f"  Average Hold Time: {metrics['avg_hold_hours']:.2f} hours")

    # Run backtests on all three periods
    run_backtest(last_month_data, "Last Month Only")
    run_backtest(full_3mo_data, "Full 3 Months")
    run_backtest(prior_2mo_data, "Prior 2 Months Only")


if __name__ == "__main__":
    # main()
    test()

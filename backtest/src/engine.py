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


def load_price_data(file_path: str = "./data/btc_1m_185days.json") -> pd.DataFrame:
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
    vol_threshold: float = 1.5,
) -> pd.Series:
    momentum = fast_ema - slow_ema
    price_signal = -np.sign(momentum).shift(2)

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


def calculate_unrealized_pnl(position: Position, current_price: float) -> float:
    return position.signal * position.quantity * (current_price - position.entry_price)


def backtest_ensemble(
    df: pd.DataFrame,
    strategies: List[dict],
    btc_qty: float = 0.03,
    ema_cache: dict = None,
) -> pd.DataFrame:
    if len(strategies) == 0:
        return pd.DataFrame()

    df_signals = create_ensemble_signal(df, strategies, ema_cache)

    if len(df_signals) == 0:
        return pd.DataFrame()

    timeline = []
    position: Optional[Position] = None

    # Fixed initial capital - never changes
    initial_capital = btc_qty * df_signals.iloc[0]["price"]
    cumulative_realized_pnl = 0.0

    for _, row in df_signals.iterrows():
        ts = row.name
        price = row.price
        signal = int(row.ensemble_signal)

        # Calculate current unrealized PnL
        unrealized_pnl = 0.0
        if position is not None:
            unrealized_pnl = calculate_unrealized_pnl(position, price)

        # Total equity = initial capital + all realized PnL + current unrealized PnL
        total_equity = initial_capital + cumulative_realized_pnl + unrealized_pnl

        snap = {
            "ts": ts,
            "price": price,
            "signal": signal,
            "is_entry": False,
            "is_exit": False,
            "qty": position.quantity if position else 0.0,
            "pnl": unrealized_pnl,
            "equity": total_equity,
            "volume_usd": 0.0,
        }

        # Handle position changes
        if position is not None:
            if signal != position.signal:
                # Exit current position - realize PnL
                snap["is_exit"] = True
                snap["volume_usd"] = abs(position.quantity * price)
                cumulative_realized_pnl += unrealized_pnl
                position = None

                # Update after realizing PnL
                snap["equity"] = initial_capital + cumulative_realized_pnl
                snap["pnl"] = 0.0
                snap["qty"] = 0.0

                # Enter new position if signal != 0
                if signal != 0:
                    position = open_new_position(
                        signal, price, btc_qty, initial_capital
                    )
                    snap["is_entry"] = True
                    snap["qty"] = btc_qty
                    snap["volume_usd"] += abs(btc_qty * price)

        elif signal != 0:
            # No current position, but signal says enter
            position = open_new_position(signal, price, btc_qty, initial_capital)
            snap["is_entry"] = True
            snap["qty"] = btc_qty
            snap["volume_usd"] = abs(btc_qty * price)

        timeline.append(snap)

    # Force exit at end if position exists - avoid double counting
    if position is not None:
        # Only add final exit if we haven't already processed the last timestamp
        if len(timeline) == 0 or timeline[-1]["ts"] != df_signals.index[-1]:
            last_row = df_signals.iloc[-1]
            final_unrealized_pnl = calculate_unrealized_pnl(position, last_row.price)
            cumulative_realized_pnl += final_unrealized_pnl

            timeline.append(
                {
                    "ts": last_row.name,
                    "price": last_row.price,
                    "signal": 0,
                    "is_entry": False,
                    "is_exit": True,
                    "qty": 0.0,
                    "pnl": 0.0,
                    "equity": initial_capital + cumulative_realized_pnl,
                    "volume_usd": abs(position.quantity * last_row.price),
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
                "total_volume_usd",
                "avg_daily_volume",
                "avg_hourly_volume",
                "actual_trades",
                "signal_flips",
            ]
        }

    equity = timeline["equity"].ffill().dropna()
    duration_days = (timeline.index[-1] - timeline.index[0]).total_seconds() / 86400
    duration_hours = duration_days * 24

    # Trade analysis
    exit_trades = timeline[timeline.is_exit]
    trade_returns = exit_trades["equity"].diff().dropna()
    hold_durations = (
        exit_trades.index.to_series().diff().dt.total_seconds().dropna() / 3600
    )

    # Count signal flips (trades with both entry and exit in same row)
    signal_flips = timeline[timeline.is_entry & timeline.is_exit]

    # Volume calculation
    total_volume_usd = timeline["volume_usd"].sum()

    # Risk-free rate calculation
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
        "trades_per_year": len(trade_returns) / duration_days * 365
        if duration_days > 0
        else 0,
        "avg_hold_hours": avg_hold,
        "total_volume_usd": total_volume_usd,
        "avg_daily_volume": total_volume_usd / duration_days
        if duration_days > 0
        else 0,
        "avg_hourly_volume": total_volume_usd / duration_hours
        if duration_hours > 0
        else 0,
        "actual_trades": len(trade_returns),
        "signal_flips": len(signal_flips),
    }


def test_single_strategy(
    price_data: pd.DataFrame, fast_window: int, slow_window: int, ema_cache: dict = None
) -> dict:
    if fast_window >= slow_window:
        return None

    try:
        config = [{"fast": fast_window, "slow": slow_window}]
        timeline = backtest_ensemble(
            price_data, config, btc_qty=0.03, ema_cache=ema_cache
        )

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
    # ema_pairs = list(
    #     set((random.randint(2, 30), random.randint(32, 120)) for _ in range(num_tests))
    # )
    ema_pairs = [(f, s) for f in range(4, 16, 2) for s in range(f + 10, f + 25, 3)]
    random.shuffle(ema_pairs)
    ema_pairs = ema_pairs[:num_tests]

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


def search():
    logger.info("BTC Momentum Strategy Optimization")

    df = load_price_data()

    # Time-based boundaries for 6-month data
    end_ts = df.index[-1]

    test_start = end_ts - pd.Timedelta(days=30)

    # Validation: 1 month before test (days 31-60 from end)
    validation_start = test_start - pd.Timedelta(days=30)

    # Train on last 2 months instead of first 4 months
    train_start = test_start - pd.Timedelta(days=90)  # Last 3 months total

    # Create datasets
    train_data = df[
        (df.index >= train_start) & (df.index < validation_start)
    ]  # Days 61-90 from end (1 month)
    validation_data = df[
        (df.index >= validation_start) & (df.index < test_start)
    ]  # Days 31-60 from end (1 month)
    test_data = df[df.index >= test_start]  # Last 1 month

    logger.info(
        f"Data split - Train: {len(train_data)} rows, Validation: {len(validation_data)} rows, Test: {len(test_data)} rows"
    )

    candidates = search_optimal_strategies_parallel(train_data, max_strategies=20)

    # Step 2: Rank using validation Sharpe
    val_emas = precompute_all_emas(validation_data, 250)
    results = []
    for strat in candidates:
        timeline = backtest_ensemble(validation_data, [strat], ema_cache=val_emas)
        if timeline.empty:
            continue
        cap0 = timeline[timeline.is_entry]["equity"].iloc[0]
        metrics = calculate_performance_metrics(timeline, cap0)
        results.append((metrics["sharpe_ratio"], strat))

    # Step 3: Select top-N by validation performance
    results.sort(reverse=True, key=lambda x: x[0])
    strategies = [s for _, s in results[:5]]

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
    evaluate("validation", validation_data)
    evaluate("test", test_data)


def main():
    logger.info("Ensemble Backtest on Multiple Time Periods")

    df = load_price_data()
    logger.info(f"Full dataset: {len(df)} rows")

    strategies = [
        {"fast": 6, "slow": 19},
        {"fast": 6, "slow": 22},
        {"fast": 8, "slow": 21},
        {"fast": 4, "slow": 26},
        {"fast": 4, "slow": 23},
    ]

    # strategies = [
    #     {"fast": 14, "slow": 32},
    #     {"fast": 15, "slow": 30},
    #     {"fast": 12, "slow": 35},
    #     {"fast": 14, "slow": 31},
    #     {"fast": 13, "slow": 31},
    # ]

    # Time-based boundaries
    end_ts = df.index[-1]
    last_month_start = end_ts - pd.Timedelta(days=30)
    two_months_prior = last_month_start - pd.Timedelta(days=90)

    # Create three datasets
    datasets = {
        "last_month": df[df.index >= last_month_start],
        "full_3mo": df[df.index >= two_months_prior],
        "prior_2mo": df[(df.index >= two_months_prior) & (df.index < last_month_start)],
    }

    all_metrics = []

    def run_backtest(data, period_name, period_key):
        logger.info(f"{period_name} ({len(data)} rows):")

        ema_cache = precompute_all_emas(data, 250)
        timeline = backtest_ensemble(
            data, strategies, btc_qty=0.03, ema_cache=ema_cache
        )

        if timeline.empty:
            logger.info("  No trades generated!")
            return None, None

        first_capital = (
            timeline[timeline.is_entry]["equity"].iloc[0]
            if len(timeline[timeline.is_entry]) > 0
            else 1000.0
        )
        metrics = calculate_performance_metrics(timeline, first_capital)

        # Add period info to metrics
        metrics_row = {
            "period": period_key,
            "period_name": period_name,
            "start_date": data.index[0],
            "end_date": data.index[-1],
            "rows": len(data),
            "initial_capital": first_capital,
            **metrics,
        }
        all_metrics.append(metrics_row)

        # Save timeline for this period
        # timeline_file = f"./results/timeline_{period_key}.csv"
        # timeline.to_csv(timeline_file)
        # logger.info(f"  Saved timeline to {timeline_file}")
        logger.info(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
        logger.info(f"  Total Return: {metrics['total_return_pct']:.2f}%")
        logger.info(f"  Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
        logger.info(
            f"  Win Rate: {metrics['win_rate_pct']:.1f}% | Trades/Year: {metrics['trades_per_year']:.0f}"
        )
        logger.info(f"  Total Volume: ${metrics['total_volume_usd']:,.0f}")
        logger.info(
            f"  Daily Volume: ${metrics['avg_daily_volume']:,.0f} | Hourly Volume: ${metrics['avg_hourly_volume']:,.0f}"
        )
        logger.info(
            f"  Actual Trades: {metrics['actual_trades']} | Signal Flips: {metrics['signal_flips']} | Trades/Year: {metrics['trades_per_year']:.0f}"
        )

        return timeline, metrics

    # Create results directory
    # import os

    # os.makedirs("./results", exist_ok=True)

    # Run backtests on all three periods
    for period_key, data in datasets.items():
        period_names = {
            "last_month": "Last Month Only",
            "full_3mo": "Full 3 Months",
            "prior_2mo": "Prior 2 Months Only",
        }
        run_backtest(data, period_names[period_key], period_key)

    # Save summary metrics
    # if all_metrics:
    #     metrics_df = pd.DataFrame(all_metrics)
    #     metrics_df.to_csv("./results/performance_summary.csv", index=False)
    #     logger.info("Saved performance summary to ./results/performance_summary.csv")

    # logger.info("All results saved to ./results/ directory")


if __name__ == "__main__":
    main()
    # search()

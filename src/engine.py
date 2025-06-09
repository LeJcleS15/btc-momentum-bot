import json
import pandas as pd
import numpy as np
import logging
import random
from typing import List, Optional, NamedTuple
import multiprocessing as mp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class Position(NamedTuple):
    signal: int
    entry_price: float
    quantity: float
    capital: float


class MarkToMarket(NamedTuple):
    pnl: float
    equity: float


def load_price_data(file_path: str = "./data/btc_1m_3months.json") -> pd.DataFrame:
    """Load and clean BTC price data from JSON file."""
    logger.info(f"Loading price data from {file_path}")

    with open(file_path) as f:
        data = json.load(f)

    df = pd.DataFrame(data["candles"])
    df["timestamp"] = pd.to_datetime(df["start"].astype(int), unit="ms")
    df = df.set_index("timestamp")
    df["price"] = df["oracleClose"].astype(float)

    # Clean data
    initial_rows = len(df)
    df = df[~df.index.duplicated(keep="first")]
    df = df.sort_index()
    final_rows = len(df)

    logger.info(
        f"Loaded {final_rows} rows (removed {initial_rows - final_rows} duplicates)"
    )
    return df[["price"]]


def precompute_all_emas(price_data: pd.DataFrame, max_window: int = 250) -> dict:
    """Pre-compute EMAs for all windows to avoid recalculation."""
    logger.info(f"Pre-computing EMAs for windows 2 to {max_window}")

    ema_cache = {}
    for window in range(2, max_window + 1):
        ema_cache[window] = price_data["price"].ewm(span=window).mean()

    logger.info(f"Cached {len(ema_cache)} EMA series")
    return ema_cache


def generate_momentum_signal(fast_ema: pd.Series, slow_ema: pd.Series) -> pd.Series:
    """Generate momentum signals using pre-computed EMAs."""
    momentum = fast_ema - slow_ema
    return np.sign(momentum).shift(2).dropna()


def create_ensemble_signal(
    price_data: pd.DataFrame, strategy_configs: List[dict], ema_cache: dict = None
) -> pd.DataFrame:
    """Combine multiple momentum strategies using median voting."""
    logger.debug(f"Creating ensemble from {len(strategy_configs)} strategies")

    if len(strategy_configs) == 0:
        result = price_data.copy()
        result["ensemble_signal"] = 0
        return result

    individual_signals = []
    for config in strategy_configs:
        fast, slow = config["fast"], config["slow"]

        # Use cache if available, otherwise compute on demand
        if ema_cache and fast in ema_cache and slow in ema_cache:
            ema_fast = ema_cache[fast]
            ema_slow = ema_cache[slow]
        else:
            ema_fast = price_data["price"].ewm(span=fast).mean()
            ema_slow = price_data["price"].ewm(span=slow).mean()

        signal = generate_momentum_signal(ema_fast, ema_slow)
        individual_signals.append(signal)

    # Combine signals
    df = price_data.copy()
    signals_matrix = pd.concat(individual_signals, axis=1)
    df["ensemble_signal"] = signals_matrix.median(axis=1)

    return df.dropna()


def open_new_position(signal: int, price: float, available_capital: float) -> Position:
    """Open a new trading position."""
    quantity = available_capital / price
    logger.debug(
        f"Opening position: signal={signal}, price={price:.2f}, qty={quantity:.6f}"
    )
    return Position(signal, price, quantity, available_capital)


def calculate_mtm(position: Position, current_price: float) -> MarkToMarket:
    """Calculate mark-to-market PnL and equity."""
    pnl = position.signal * position.quantity * (current_price - position.entry_price)
    equity = position.capital + pnl
    return MarkToMarket(pnl, equity)


def backtest_ensemble(
    df: pd.DataFrame,
    strategies: List[dict],
    initial_capital: float = 1.0,
    ema_cache: dict = None,
) -> pd.DataFrame:
    """Backtest ensemble strategy with every-minute rebalancing"""
    if len(strategies) == 0:
        return pd.DataFrame()

    # Create ensemble signals
    df_signals = create_ensemble_signal(df, strategies, ema_cache)

    if len(df_signals) == 0:
        return pd.DataFrame()

    timeline = []
    position: Optional[Position] = None
    capital_0 = initial_capital

    # Rebalance every minute
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
            "equity": capital_0,
        }

        # Mark existing position
        if position is not None:
            mtm = calculate_mtm(position, price)
            snap["qty"] = position.quantity
            snap["pnl"] = mtm.pnl
            snap["equity"] = mtm.equity

            # Check if signal changed
            if signal != position.signal:
                snap["is_exit"] = True
                cap_1 = mtm.equity

                if signal != 0:  # Open new position
                    position = open_new_position(signal, price, cap_1)
                    snap["is_entry"] = True
                    snap["qty"] = position.quantity
                else:
                    position = None

        # No position, maybe open
        elif signal != 0:
            position = open_new_position(signal, price, capital_0)
            snap["is_entry"] = True
            snap["qty"] = position.quantity
            snap["equity"] = position.capital

        timeline.append(snap)

    # Final close
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
    """Test a single momentum strategy using real backtest and compute_performance."""
    if fast_window >= slow_window:
        return None

    try:
        config = [{"fast": fast_window, "slow": slow_window}]
        timeline = backtest_ensemble(
            price_data, config, initial_capital=1.0, ema_cache=ema_cache
        )

        if len(timeline) < 10:  # Need some trades
            return None

        metrics = calculate_performance_metrics(timeline, 1.0)

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
    """Worker function for parallel strategy testing."""
    fast, slow, train_data, ema_cache = args
    return test_single_strategy(train_data, fast, slow, ema_cache)


def search_optimal_strategies_parallel(
    max_strategies: int = 5, num_tests: int = 500, n_cores: int = None
) -> tuple:
    """Search for optimal momentum strategies using parallel random EMA pairs."""

    if n_cores is None:
        n_cores = mp.cpu_count() - 1  # Leave one core free

    logger.info(
        f"Parallel search testing {num_tests} EMA combinations using {n_cores} cores"
    )

    # Load and split data once
    price_data = load_price_data()
    train_split = int(len(price_data) * 0.8)
    train_data = price_data.iloc[:train_split]
    logger.info(f"Using {len(train_data)} rows for strategy search")

    # Pre-compute EMAs for efficiency
    ema_cache = precompute_all_emas(train_data, 120)  # Reduced from 250

    # Generate random EMA pairs
    random.seed(42)  # For reproducibility
    ema_pairs = []

    for _ in range(num_tests):
        fast = random.randint(2, 30)  # 2-30 minutes
        slow = random.randint(fast + 2, 120)  # slow must be > fast, up to 2 hours
        ema_pairs.append((fast, slow))

    # Remove duplicates
    ema_pairs = list(set(ema_pairs))
    logger.info(f"Testing {len(ema_pairs)} unique random EMA combinations")

    # Prepare arguments for parallel processing
    worker_args = [(fast, slow, train_data, ema_cache) for fast, slow in ema_pairs]

    # Run parallel processing
    logger.info(f"Starting parallel processing with {n_cores} workers")

    with mp.Pool(processes=n_cores) as pool:
        results = pool.map(test_strategy_worker, worker_args)

    # Filter out None results
    valid_results = [r for r in results if r is not None]

    logger.info(
        f"Found {len(valid_results)} viable strategies from {len(ema_pairs)} tested"
    )

    # Sort by Sharpe ratio and take the best
    valid_results.sort(key=lambda x: x["sharpe_ratio"], reverse=True)
    selected = valid_results[:max_strategies]

    logger.info(f"Selected {len(selected)} best strategies:")
    for i, s in enumerate(selected):
        logger.info(
            f"  {i + 1}: EMA({s['fast']},{s['slow']}) - Sharpe={s['sharpe_ratio']:.3f}, Return={s['total_return_pct']:.2f}%"
        )

    return [{"fast": s["fast"], "slow": s["slow"]} for s in selected], price_data


def main():
    """Main execution function."""
    logger.info("=== BTC Momentum Strategy Optimization ===")

    # Find optimal strategies and get data
    optimal_strategies, price_data = search_optimal_strategies_parallel()

    if len(optimal_strategies) == 0:
        logger.error("No viable strategies found!")
        return

    # Split data
    train_split = int(len(price_data) * 0.8)
    training_data = price_data.iloc[:train_split]
    validation_data = price_data.iloc[train_split:]

    logger.info(f"Testing ensemble on {len(training_data)} training rows")

    # Pre-compute EMAs for training data
    training_ema_cache = precompute_all_emas(training_data, 250)

    # Run ensemble backtest on training data
    training_results = backtest_ensemble(
        training_data, optimal_strategies, ema_cache=training_ema_cache
    )

    if len(training_results) == 0:
        logger.error("No trades generated on training data!")
        return

    # Calculate training metrics
    training_metrics = calculate_performance_metrics(training_results, 1.0)

    logger.info("=== TRAINING ENSEMBLE RESULTS ===")
    logger.info(f"  Sharpe Ratio: {training_metrics['sharpe_ratio']:.3f}")
    logger.info(f"  Total Return: {training_metrics['total_return_pct']:.2f}%")
    logger.info(f"  Trades/Year: {training_metrics['trades_per_year']:.0f}")
    logger.info(f"  Win Rate: {training_metrics['win_rate_pct']:.1f}%")
    logger.info(f"  Max Drawdown: {training_metrics['max_drawdown_pct']:.2f}%")

    logger.info(f"Testing ensemble on {len(validation_data)} validation rows")

    # Pre-compute EMAs for validation data
    validation_ema_cache = precompute_all_emas(validation_data, 250)

    # Run ensemble backtest on validation data
    ensemble_results = backtest_ensemble(
        validation_data, optimal_strategies, ema_cache=validation_ema_cache
    )

    if len(ensemble_results) == 0:
        logger.error("No trades generated on validation data!")
        return

    # Calculate validation metrics
    metrics = calculate_performance_metrics(ensemble_results, 1.0)

    logger.info("=== VALIDATION ENSEMBLE RESULTS ===")
    logger.info(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
    logger.info(f"  Total Return: {metrics['total_return_pct']:.2f}%")
    logger.info(f"  Trades/Year: {metrics['trades_per_year']:.0f}")
    logger.info(f"  Win Rate: {metrics['win_rate_pct']:.1f}%")
    logger.info(f"  Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")


if __name__ == "__main__":
    main()

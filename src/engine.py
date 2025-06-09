import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, NamedTuple, Tuple


def log(msg):
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {msg}")


class Position(NamedTuple):
    signal: int
    px_entry: float
    qty: float
    capital_net: float


class MTM(NamedTuple):
    pnl: float
    equity: float


def load_btc_data(filepath="./data/btc_1m_3months.json") -> pd.DataFrame:
    with open(filepath) as f:
        data = json.load(f)
    df = pd.DataFrame(data["candles"])
    df["start"] = pd.to_datetime(df["start"].astype(int), unit="ms")
    df = df.set_index("start")
    df["close"] = df["oracleClose"].astype(float)
    return df[["close"]]


def add_ema_momentum_signal(
    df: pd.DataFrame, fast: int, slow: int, threshold: float, lag: int
) -> pd.DataFrame:
    df = df.copy()
    df["ema_fast"] = df["close"].ewm(span=fast).mean()
    df["ema_slow"] = df["close"].ewm(span=slow).mean()
    df["momentum"] = df["ema_fast"] - df["ema_slow"]
    df["signal_raw"] = np.select(
        [df["momentum"] > threshold, df["momentum"] < -threshold], [1, -1], default=0
    )
    df["signal_yest"] = df["signal_raw"].shift(lag)
    return df.dropna(subset=["signal_yest"])


def open_position(sig: int, price: float, capital: float) -> Position:
    qty = capital / price
    return Position(sig, price, qty, capital)


def mark_to_market(pos: Position, current_price: float) -> MTM:
    pnl = pos.signal * pos.qty * (current_price - pos.px_entry)
    equity = pos.capital_net + pnl
    return MTM(pnl, equity)


def make_snap(ts, signal, price):
    return {
        "ts": ts,
        "price": price,
        "signal": signal,
        "is_entry": False,
        "is_exit": False,
        "qty": 0.0,
        "pnl": 0.0,
        "equity": np.nan,
    }


def backtest(
    df: pd.DataFrame, initial_capital: float = 1.0, rebalance_freq: int = 60
) -> Tuple[pd.DataFrame, float]:
    timeline = []
    position: Optional[Position] = None
    capital_0 = initial_capital

    rebalance_mask = pd.Series(False, index=df.index)
    rebalance_mask.iloc[::rebalance_freq] = True

    for i, row in enumerate(df.itertuples(index=True)):
        ts = row.Index
        price = row.close
        rebalance_now = rebalance_mask.iloc[i]
        signal = int(row.signal_yest)
        snap = make_snap(ts, signal, price)

        # mark existing position
        if position is not None:
            mtm = mark_to_market(position, price)
            equity_now = mtm.equity

            snap["qty"] = position.qty
            snap["pnl"] = mtm.pnl
            snap["equity"] = equity_now

            if rebalance_now and signal != position.signal:  # close old position
                cap_1 = equity_now
                snap["is_exit"] = True
                snap["equity"] = cap_1

                if signal:  # open new position immediately
                    position = open_position(signal, price, cap_1)
                    snap["is_entry"] = True
                    snap["qty"] = position.qty
                else:
                    position = None

        # no position, maybe open
        elif signal and rebalance_now:
            position = open_position(signal, price, capital_0)
            snap["is_entry"] = True
            snap["qty"] = position.qty
            snap["equity"] = position.capital_net

        timeline.append(snap)

    # final close
    if position:
        row = df.iloc[-1]
        ts = row.name
        price = row.close
        mtm = mark_to_market(position, price)
        cap_1 = mtm.equity

        timeline.append(
            {
                "ts": ts,
                "price": price,
                "signal": position.signal,
                "is_entry": False,
                "is_exit": True,
                "qty": position.qty,
                "pnl": mtm.pnl,
                "equity": cap_1,
            }
        )

    return pd.DataFrame(timeline).set_index("ts"), capital_0


def compute_performance(timeline: pd.DataFrame, initial_capital: float) -> dict:
    if initial_capital == 0 or timeline.empty:
        return {
            k: 0
            for k in [
                "net_usd",
                "final_pct",
                "sharpe",
                "min_drawdown",
                "win_rate",
                "trades_per_year",
                "avg_hold_hrs",
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
    excess = trade_returns / initial_capital - rf_trade
    std = excess.std()

    return {
        "net_usd": equity.iloc[-1] - initial_capital,
        "final_pct": (equity.iloc[-1] / initial_capital - 1) * 100,
        "sharpe": (excess.mean() / std * np.sqrt(8760 / avg_hold)) if std > 1e-8 else 0,
        "min_drawdown": 100 * (equity / equity.cummax() - 1).min(),
        "win_rate": 100 * (trade_returns > 0).mean() if len(trade_returns) > 0 else 0,
        "trades_per_year": len(trade_returns) / duration * 365,
        "avg_hold_hrs": avg_hold,
    }


def main():
    df_full = load_btc_data()

    # Split last 30 days for OOS
    split_point = df_full.index[-1] - pd.Timedelta(days=30)
    df_train = df_full[df_full.index < split_point]
    df_test = df_full[df_full.index >= split_point]

    # Fixed strategy params
    fast, slow = 9, 21
    threshold = 0.05
    lag = 15
    initial_capital = 1.0

    # Compute signals once
    df_signals_train = add_ema_momentum_signal(df_train, fast, slow, threshold, lag)
    df_signals_test = add_ema_momentum_signal(df_test, fast, slow, threshold, lag)

    update_freq_list = [5, 10, 15, 20, 30, 45, 60, 90]

    for freq in update_freq_list:
        timeline, _ = backtest(df_signals_train, initial_capital, freq)
        stats = compute_performance(timeline, initial_capital)

        timeline_oos, _ = backtest(df_signals_test, initial_capital, freq)
        stats_oos = compute_performance(timeline_oos, initial_capital)

        if stats["sharpe"] > 1:
            print(
                f"freq={freq:3d} → "
                f"[TRAIN] Sharpe={stats['sharpe']:.2f} R={stats['final_pct']:5.1f}% | "
                f"[OOS] Sharpe={stats_oos['sharpe']:.2f} R={stats_oos['final_pct']:5.1f}%"
            )


main()

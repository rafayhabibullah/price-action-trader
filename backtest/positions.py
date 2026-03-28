import pandas as pd
import numpy as np


def simulate_trades(
    df: pd.DataFrame,
    starting_capital: float,
    risk_per_trade: float,
    max_bars: int = 100,
) -> pd.DataFrame:
    """
    Simulate trades from a signal DataFrame.

    df must have columns: open, high, low, close, signal, sl, tp.
    signal: 1=long, -1=short, 0=no trade.
    sl/tp: stop loss and take profit prices.

    Returns DataFrame of completed trades.
    """
    capital = starting_capital
    trades = []
    in_trade = False
    entry_price = sl = tp = direction = entry_bar = 0
    size = 0.0

    opens = df["open"].values
    highs = df["high"].values
    lows = df["low"].values
    signals = df["signal"].values
    sls = df["sl"].values
    tps = df["tp"].values

    for i in range(len(df)):
        ts = df.index[i]

        if in_trade:
            # Check exit conditions on this bar
            exit_price = None
            exit_reason = None

            if direction == 1:  # long
                if lows[i] <= sl:
                    exit_price = sl
                    exit_reason = "sl"
                elif highs[i] >= tp:
                    exit_price = tp
                    exit_reason = "tp"
                elif (i - entry_bar) >= max_bars:
                    exit_price = opens[i]
                    exit_reason = "timeout"
            else:  # short
                if highs[i] >= sl:
                    exit_price = sl
                    exit_reason = "sl"
                elif lows[i] <= tp:
                    exit_price = tp
                    exit_reason = "tp"
                elif (i - entry_bar) >= max_bars:
                    exit_price = opens[i]
                    exit_reason = "timeout"

            if exit_price is not None:
                pnl = (exit_price - entry_price) * direction * size
                capital += pnl
                r_risk = abs(entry_price - sl) * size
                r_multiple = pnl / r_risk if r_risk > 0 else 0.0
                trades.append({
                    "entry_time": df.index[entry_bar],
                    "exit_time": ts,
                    "direction": direction,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "sl": sl,
                    "tp": tp,
                    "size": size,
                    "pnl": pnl,
                    "r_multiple": r_multiple,
                    "exit_reason": exit_reason,
                    "equity": capital,
                })
                in_trade = False

        if not in_trade and signals[i] != 0 and not np.isnan(sls[i]):
            # Enter trade on next bar open — but we enter at current bar open
            # to keep simple vectorized model (signal on close, fill on same bar open)
            direction = int(signals[i])
            entry_price = opens[i]
            sl = sls[i]
            tp = tps[i] if not np.isnan(tps[i]) else (
                entry_price + 2 * abs(entry_price - sl) * direction
            )
            risk_amount = capital * risk_per_trade
            price_risk = abs(entry_price - sl)
            size = risk_amount / price_risk if price_risk > 0 else 0.0
            entry_bar = i
            in_trade = True

    return pd.DataFrame(trades)

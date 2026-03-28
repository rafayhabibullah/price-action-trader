from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass

from strategies.base import BaseStrategy
from trading.symbols import to_alpaca, is_crypto
from trading import config


@dataclass
class Signal:
    asset: str        # Internal format: "BTC/USDT" or "AAPL"
    timeframe: str    # "1h", "4h", "1d"
    strategy: str     # Strategy class name
    direction: int    # 1=long, -1=short
    entry_price: float
    sl: float
    tp: float


def scan_all(
    client,
    strategies: list[BaseStrategy],
    assets_crypto: list[str],
    assets_stocks: list[str],
    timeframes: list[str],
) -> list[Signal]:
    """Scan all asset/timeframe combos and return active signals."""
    market_open = client.is_market_open()
    signals: list[Signal] = []

    for asset in assets_crypto + assets_stocks:
        if not is_crypto(asset) and not market_open:
            continue  # skip stocks when market is closed

        alpaca_sym = to_alpaca(asset)

        for tf in timeframes:
            df = client.get_bars(alpaca_sym, tf, limit=config.CANDLE_HISTORY)
            if df is None or len(df) < 50:
                continue

            for strat in strategies:
                try:
                    sig_df = strat.generate_signals(df)
                except Exception:
                    continue

                # Check second-to-last bar (1-bar shifted signal fires on latest open)
                if len(sig_df) < 2:
                    continue

                row = sig_df.iloc[-2]
                signal_val = int(row["signal"])
                sl_val = float(row["sl"]) if not pd.isna(row["sl"]) else float("nan")
                tp_val = float(row["tp"]) if not pd.isna(row["tp"]) else float("nan")

                if signal_val == 0 or np.isnan(sl_val) or np.isnan(tp_val):
                    continue

                entry_price = float(df["close"].iloc[-1])

                # Filter: SL must be at least 0.1% from entry
                sl_dist_pct = abs(entry_price - sl_val) / entry_price
                if sl_dist_pct < config.MIN_SL_DISTANCE_PCT:
                    continue

                signals.append(
                    Signal(
                        asset=asset,
                        timeframe=tf,
                        strategy=strat.__class__.__name__,
                        direction=signal_val,
                        entry_price=entry_price,
                        sl=sl_val,
                        tp=tp_val,
                    )
                )

    return signals

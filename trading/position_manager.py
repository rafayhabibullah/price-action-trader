from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone

from trading.symbols import to_alpaca, to_internal, is_crypto
from trading import config as _cfg


@dataclass
class CycleResult:
    timestamp: str
    account_equity: float
    open_positions: int
    signals_found: int
    orders_placed: int
    positions_closed: int
    errors: list[str] = field(default_factory=list)
    actions: list[dict] = field(default_factory=list)


def run_cycle(
    client,
    scan_fn,
    strategies: list,
    config_override: dict | None = None,
) -> CycleResult:
    """Execute one full trading cycle. Returns a CycleResult summary."""
    cfg = {
        "MAX_CONCURRENT": _cfg.MAX_CONCURRENT,
        "RISK_PER_TRADE": _cfg.RISK_PER_TRADE,
        "MAX_BARS": _cfg.MAX_BARS,
        "ASSETS_CRYPTO": _cfg.ASSETS_CRYPTO,
        "ASSETS_STOCKS": _cfg.ASSETS_STOCKS,
        "TIMEFRAMES": _cfg.TIMEFRAMES,
    }
    if config_override:
        cfg.update(config_override)

    result = CycleResult(
        timestamp=datetime.now(timezone.utc).isoformat(),
        account_equity=0.0,
        open_positions=0,
        signals_found=0,
        orders_placed=0,
        positions_closed=0,
    )

    # 1. Query account state
    account = client.get_account()
    result.account_equity = float(account.get("equity", 0))
    positions = client.get_positions()
    result.open_positions = len(positions)

    # 2. Check timeouts
    now = datetime.now(timezone.utc)
    # 100 bars × 1h/bar (most conservative timeframe) = 100 hours timeout
    timeout_hours = float(cfg["MAX_BARS"])  # MAX_BARS hours at 1h timeframe
    for pos in positions:
        try:
            created_at_str = pos.get("created_at", "")
            created_at = datetime.fromisoformat(
                created_at_str.replace("Z", "+00:00")
            )
            hours_held = (now - created_at).total_seconds() / 3600
            if hours_held >= timeout_hours:
                symbol = pos["symbol"]
                client.close_position(symbol)
                result.positions_closed += 1
                result.actions.append({
                    "action": "close_timeout",
                    "symbol": symbol,
                    "hours_held": round(hours_held, 1),
                })
        except Exception as e:
            result.errors.append(f"timeout check error: {e}")

    # 3. Scan for signals
    signals = scan_fn(
        client,
        strategies,
        assets_crypto=cfg["ASSETS_CRYPTO"],
        assets_stocks=cfg["ASSETS_STOCKS"],
        timeframes=cfg["TIMEFRAMES"],
    )
    result.signals_found = len(signals)

    # 4. Filter signals
    open_assets = {to_internal(p["symbol"]) for p in positions}
    # Also exclude assets we just timed out
    for action in result.actions:
        if action["action"] == "close_timeout":
            open_assets.add(to_internal(action["symbol"]))

    slots_available = cfg["MAX_CONCURRENT"] - result.open_positions
    if slots_available <= 0:
        return result

    # Deduplicate: one signal per asset
    seen_assets: set[str] = set()
    filtered: list = []
    for sig in signals:
        if sig.asset in open_assets:
            continue
        if sig.asset in seen_assets:
            continue
        seen_assets.add(sig.asset)
        filtered.append(sig)
        if len(filtered) >= slots_available:
            break

    # 5. Place bracket orders
    for sig in filtered:
        try:
            alpaca_sym = to_alpaca(sig.asset)
            side = "buy" if sig.direction == 1 else "sell"

            risk_amount = result.account_equity * cfg["RISK_PER_TRADE"]
            price_risk = abs(sig.entry_price - sig.sl)
            raw_qty = risk_amount / price_risk

            if is_crypto(sig.asset):
                qty = round(raw_qty, 8)
            else:
                qty = math.floor(raw_qty)
                if qty < 1:
                    result.actions.append({
                        "action": "skip_insufficient_qty",
                        "symbol": alpaca_sym,
                        "raw_qty": round(raw_qty, 4),
                    })
                    continue

            order = client.place_bracket_order(
                symbol=alpaca_sym,
                side=side,
                qty=qty,
                take_profit=sig.tp,
                stop_loss=sig.sl,
            )
            result.orders_placed += 1
            result.actions.append({
                "action": "place_order",
                "symbol": alpaca_sym,
                "side": side,
                "qty": qty,
                "sl": sig.sl,
                "tp": sig.tp,
                "strategy": sig.strategy,
                "timeframe": sig.timeframe,
                "order_id": order.get("id"),
            })
        except Exception as e:
            result.errors.append(f"order error {sig.asset}: {e}")

    return result

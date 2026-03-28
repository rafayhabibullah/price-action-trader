"""
Entry point for Cloud Run HTTP handler and local CLI.

Cloud Run invocation:
    python -m trading.runner
    POST / → runs one cycle, returns JSON

Local testing:
    python -m trading.runner --once
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict

from flask import Flask, jsonify

from trading import config
from trading.alpaca_client import AlpacaClient
from trading.scanner import scan_all
from trading.position_manager import run_cycle
from strategies.support_resistance import SupportResistanceStrategy
from strategies.fair_value_gaps import FairValueGapStrategy
from strategies.candlestick_patterns import CandlestickPatternStrategy
from strategies.market_structure import MarketStructureStrategy


app = Flask(__name__)

STRATEGIES = [
    SupportResistanceStrategy(),
    FairValueGapStrategy(),
    CandlestickPatternStrategy(),
    MarketStructureStrategy(),
]


def _build_client() -> AlpacaClient:
    return AlpacaClient(
        api_key=config.ALPACA_API_KEY,
        secret_key=config.ALPACA_SECRET_KEY,
        base_url=config.ALPACA_BASE_URL,
    )


@app.route("/", methods=["POST"])
def handle():
    try:
        client = _build_client()
        result = run_cycle(client, scan_fn=scan_all, strategies=STRATEGIES)
        output = asdict(result)
        print(json.dumps(output))  # captured by Cloud Logging
        return jsonify(output), 200
    except Exception as e:
        print(f"[runner] handler error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    if "--once" in sys.argv:
        # Run one cycle locally and print result
        client = _build_client()
        result = run_cycle(client, scan_fn=scan_all, strategies=STRATEGIES)
        print(json.dumps(asdict(result), indent=2))
    else:
        app.run(host="0.0.0.0", port=8080)

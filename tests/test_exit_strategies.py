import pytest
from optimization.exit_strategies import FixedRR, TrailingStop, PartialTP


def test_fixed_rr_long_tp():
    exit_cfg = FixedRR(rr_ratio=2.0)
    tp = exit_cfg.take_profit(entry=100.0, sl=98.0, direction=1)
    assert abs(tp - 104.0) < 0.01  # entry + 2 * risk


def test_fixed_rr_short_tp():
    exit_cfg = FixedRR(rr_ratio=2.0)
    tp = exit_cfg.take_profit(entry=100.0, sl=102.0, direction=-1)
    assert abs(tp - 96.0) < 0.01   # entry - 2 * risk


def test_trailing_stop_initial_sl_unchanged():
    exit_cfg = TrailingStop(trail_pct=0.02)
    new_sl = exit_cfg.update_sl(current_sl=98.0, current_price=100.0, direction=1, entry=100.0)
    assert new_sl >= 98.0


def test_trailing_stop_moves_up_with_price():
    exit_cfg = TrailingStop(trail_pct=0.02)
    new_sl = exit_cfg.update_sl(current_sl=98.0, current_price=110.0, direction=1, entry=100.0)
    expected = 110.0 * (1 - 0.02)
    assert abs(new_sl - expected) < 0.01


def test_partial_tp_first_target():
    exit_cfg = PartialTP(first_tp_r=1.0, first_tp_pct=0.5)
    tp1 = exit_cfg.first_target(entry=100.0, sl=98.0, direction=1)
    assert abs(tp1 - 102.0) < 0.01  # entry + 1R


def test_get_params_all_exits():
    for cls in [FixedRR(2.0), TrailingStop(0.02), PartialTP(1.0, 0.5)]:
        assert isinstance(cls.get_params(), dict)

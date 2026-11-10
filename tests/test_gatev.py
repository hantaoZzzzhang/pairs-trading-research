import numpy as np
import pandas as pd

from src.backtest import run_backtest
from src.gatev import (
    generate_gatev_signals,
    normalize_total_return_prices,
    pair_distance,
    rank_distance_pairs,
    select_non_overlapping_pairs,
)
from run_gatev_comparison import build_windows


def test_normalization_and_distance_use_formation_observations():
    formation = pd.DataFrame(
        {"A": [10.0, 11.0, 12.0], "B": [20.0, 22.0, 21.0]},
        index=pd.date_range("2020-01-01", periods=3),
    )
    normalized = normalize_total_return_prices(formation)
    assert normalized.iloc[0].tolist() == [1.0, 1.0]
    expected = np.mean(np.square(np.array([0.0, 0.0, 0.15])))
    assert np.isclose(pair_distance(normalized["A"], normalized["B"]), expected)


def test_distance_pair_ranking():
    formation = pd.DataFrame(
        {
            "A": [10.0, 11.0, 12.0, 13.0],
            "B": [20.0, 22.0, 24.0, 26.0],
            "C": [10.0, 15.0, 8.0, 20.0],
        },
        index=pd.date_range("2020-01-01", periods=4),
    )
    ranked = rank_distance_pairs(formation, max_pairs=2)
    assert not ranked.empty
    assert {ranked.iloc[0]["x"], ranked.iloc[0]["y"]} == {"A", "B"}


def test_non_overlapping_pair_selection_does_not_reuse_tickers():
    ranked = pd.DataFrame(
        {
            "x": ["A", "A", "C"],
            "y": ["B", "C", "D"],
            "distance": [0.01, 0.02, 0.03],
            "formation_mean": [0.0, 0.0, 0.0],
            "formation_std": [0.1, 0.1, 0.1],
        }
    )
    selected = select_non_overlapping_pairs(ranked, max_pairs=2)
    assert len(selected) == 2
    selected_tickers = list(selected["x"]) + list(selected["y"])
    assert len(selected_tickers) == len(set(selected_tickers))


def test_gatev_entry_cross_exit_and_end_flattening():
    idx = pd.date_range("2020-01-01", periods=7)
    spread = np.array([0.0, 0.30, 0.40, 0.10, -0.10, -0.30, -0.40])
    normalized = pd.DataFrame({"A": 1.0 + spread, "B": 1.0}, index=idx)
    signals = generate_gatev_signals(
        normalized,
        pair=("A", "B"),
        formation_std=0.10,
        trading_index=idx,
        entry_sds=2.0,
    )
    assert signals.loc[idx[1], "entry_flag"] == 1
    assert signals.loc[idx[4], "exit_flag"] == 1
    assert signals.loc[idx[-1], "side"] == 0


def test_gatev_signals_ignore_prices_after_trading_window():
    formation_idx = pd.date_range("2020-01-01", periods=3)
    trading_idx = pd.date_range("2020-01-04", periods=3)
    future_idx = pd.date_range("2020-01-07", periods=2)
    idx = formation_idx.append(trading_idx).append(future_idx)
    spread = np.array([0.0, 0.05, 0.0, 0.30, 0.10, -0.10, 99.0, -99.0])
    normalized = pd.DataFrame({"A": 1.0 + spread, "B": 1.0}, index=idx)
    with_future = generate_gatev_signals(
        normalized, ("A", "B"), formation_std=0.10,
        trading_index=trading_idx, entry_sds=2.0,
    )
    without_future = generate_gatev_signals(
        normalized.loc[:trading_idx[-1]], ("A", "B"), formation_std=0.10,
        trading_index=trading_idx, entry_sds=2.0,
    )
    pd.testing.assert_frame_equal(with_future, without_future)


def test_windows_have_non_overlapping_trading_periods():
    windows = build_windows("2018-01-01", "2022-01-01")
    assert len(windows) == 2
    assert windows[0].trading_end <= windows[1].formation_start


def test_next_bar_execution_and_transaction_costs():
    idx = pd.date_range("2020-01-01", periods=6)
    prices = pd.DataFrame({"X": 100.0, "Y": 100.0}, index=idx)
    signals = pd.DataFrame(0.0, index=idx, columns=["w_x", "w_y"])
    signals.loc[idx[1], ["w_x", "w_y"]] = [1.0, -1.0]
    # Flat at idx[3] is executed at idx[4] by the existing next-bar convention.
    result = run_backtest(
        prices=prices,
        signals=signals,
        pair=("X", "Y"),
        fee_bps=1.0,
        slippage_bps=1.0,
        capital=100_000.0,
        dollar_per_leg=50_000.0,
    )
    trades = result["trades"]
    assert trades.index.min() == idx[2]
    assert result["portfolio"]["equity"].iloc[-1] < 100_000.0

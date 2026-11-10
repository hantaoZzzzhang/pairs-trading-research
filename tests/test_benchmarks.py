import numpy as np
import pandas as pd

from src.benchmarks import equal_weight_buy_and_hold


def test_equal_weight_buy_and_hold_tracks_common_growth():
    prices = pd.DataFrame(
        {"A": [100.0, 110.0, 120.0], "B": [100.0, 100.0, 100.0]},
        index=pd.date_range("2024-01-01", periods=3),
    )
    result = equal_weight_buy_and_hold(
        prices,
        capital=100_000.0,
        fee_bps=0.0,
        slippage_bps=0.0,
    )

    assert np.isclose(result["summary"]["gross_final_equity"], 110_000.0)
    assert np.isclose(result["summary"]["final_equity"], 110_000.0)

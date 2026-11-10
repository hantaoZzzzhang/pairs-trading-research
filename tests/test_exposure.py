import pandas as pd

from src.exposure import aggregate_stock_exposures


def test_aggregate_stock_exposures_keeps_offsetting_long_and_short():
    idx = pd.date_range("2024-01-01", periods=1)
    frames = [
        pd.DataFrame({"HPQ": [2_500.0], "AAPL": [-2_500.0]}, index=idx),
        pd.DataFrame({"HPQ": [-2_500.0], "CSCO": [2_500.0]}, index=idx),
    ]

    result = aggregate_stock_exposures(frames)
    hpq = result[result["ticker"] == "HPQ"].iloc[0]

    assert hpq["long_exposure"] == 2_500.0
    assert hpq["short_exposure"] == 2_500.0
    assert hpq["net_exposure"] == 0.0
    assert hpq["gross_exposure"] == 5_000.0

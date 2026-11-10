"""Stock-level exposure diagnostics for portfolios of pair books."""

from __future__ import annotations

import pandas as pd


EXPOSURE_COLUMNS = [
    "Date",
    "ticker",
    "long_exposure",
    "short_exposure",
    "net_exposure",
    "gross_exposure",
]


def aggregate_stock_exposures(position_frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Aggregate signed pair-leg values into daily stock-level exposures.

    Each input frame contains signed dollar position values, with one column
    per ticker. Positive values are long positions and negative values are
    short positions. Long and short amounts are aggregated separately so an
    offsetting long and short position is not hidden by a zero net exposure.
    """
    if not position_frames:
        return pd.DataFrame(columns=EXPOSURE_COLUMNS)

    combined = pd.concat(position_frames, axis=1).fillna(0.0)
    if combined.empty:
        return pd.DataFrame(columns=EXPOSURE_COLUMNS)

    long_by_stock = combined.clip(lower=0.0).T.groupby(level=0, sort=True).sum().T
    short_by_stock = (-combined.clip(upper=0.0)).T.groupby(level=0, sort=True).sum().T
    tickers = sorted(set(long_by_stock.columns) | set(short_by_stock.columns))

    rows: list[pd.DataFrame] = []
    for ticker in tickers:
        long_value = long_by_stock.get(ticker, pd.Series(0.0, index=combined.index))
        short_value = short_by_stock.get(ticker, pd.Series(0.0, index=combined.index))
        rows.append(
            pd.DataFrame(
                {
                    "Date": combined.index,
                    "ticker": ticker,
                    "long_exposure": long_value.to_numpy(dtype=float),
                    "short_exposure": short_value.to_numpy(dtype=float),
                    "net_exposure": (long_value - short_value).to_numpy(dtype=float),
                    "gross_exposure": (long_value + short_value).to_numpy(dtype=float),
                }
            )
        )

    return pd.concat(rows, ignore_index=True)[EXPOSURE_COLUMNS]

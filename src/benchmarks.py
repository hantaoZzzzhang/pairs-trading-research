"""Simple buy-and-hold benchmarks for Gatev category experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest import compute_metrics


def equal_weight_buy_and_hold(
    prices: pd.DataFrame,
    capital: float = 100_000.0,
    fee_bps: float = 1.0,
    slippage_bps: float = 1.0,
) -> dict:
    """Return an equal-weight buy-and-hold equity curve and summary.

    The benchmark uses the same cleaned, eligible universe as the Gatev run.
    It buys equal dollar amounts at the beginning and liquidates at the end,
    charging the same one-way fee and slippage assumptions on both sides.
    """
    clean = prices.dropna().astype(float)
    if clean.empty or len(clean.columns) == 0:
        raise ValueError("Buy-and-hold benchmark requires non-empty prices.")

    normalized = clean.divide(clean.iloc[0], axis="columns")
    gross_equity = capital * normalized.mean(axis=1)
    rate = (fee_bps + slippage_bps) / 1e4
    net_equity = gross_equity.copy()
    net_equity.iloc[0] -= capital * rate
    net_equity.iloc[-1] -= gross_equity.iloc[-1] * rate
    daily_ret = net_equity.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    metrics = compute_metrics(
        equity=net_equity,
        daily_ret=daily_ret,
        daily_turnover=pd.Series(0.0, index=net_equity.index),
    )
    summary = {
        "method": "equal_weight_buy_hold",
        "usable_tickers": len(clean.columns),
        "start_equity": capital,
        "gross_final_equity": float(gross_equity.iloc[-1]),
        "final_equity": float(net_equity.iloc[-1]),
        "net_profit": float(net_equity.iloc[-1] - capital),
        **metrics,
    }
    return {"equity": net_equity, "summary": summary}

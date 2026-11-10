"""Gatev distance-based pairs-trading experiment using yfinance data.

The runner intentionally contains only the Gatev method. It uses 12 months of
formation followed by 6 non-overlapping months of trading, with all selection,
normalization, and signal thresholds confined to each formation window.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.backtest import compute_metrics, run_backtest
from src.data import align_prices, download_prices, drop_sparse_tickers, validate_prices
from src.exposure import aggregate_stock_exposures
from src.gatev import (
    generate_gatev_signals,
    normalize_total_return_prices,
    rank_distance_pairs,
    select_non_overlapping_pairs,
)
from src.viz import plot_equity_and_drawdown


@dataclass(frozen=True)
class Window:
    formation_start: pd.Timestamp
    formation_end: pd.Timestamp
    trading_end: pd.Timestamp


def build_windows(
    start: str,
    end: str,
    formation_months: int = 12,
    trading_months: int = 6,
) -> list[Window]:
    """Build chronological, non-overlapping formation/trading windows."""
    cursor = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    windows: list[Window] = []
    while True:
        formation_end = cursor + pd.DateOffset(months=formation_months)
        trading_end = formation_end + pd.DateOffset(months=trading_months)
        if trading_end > end_ts:
            break
        windows.append(Window(cursor, formation_end, trading_end))
        cursor = trading_end
    return windows


def _slice_prices(prices: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return prices.loc[(prices.index >= start) & (prices.index < end)]


def _run_window(
    prices: pd.DataFrame,
    window: Window,
    max_pairs: int,
    entry_threshold: float,
    fee_bps: float,
    slippage_bps: float,
    capital: float,
    dollar_per_leg: float,
    non_overlapping: bool = True,
) -> dict | None:
    formation = _slice_prices(prices, window.formation_start, window.formation_end)
    trading = _slice_prices(prices, window.formation_end, window.trading_end)
    if formation.empty or trading.empty:
        return None

    ranked = rank_distance_pairs(formation, max_pairs=None)
    if non_overlapping:
        ranked = select_non_overlapping_pairs(ranked, max_pairs=max_pairs)
    else:
        ranked = ranked.head(max_pairs)
    if ranked.empty:
        return None

    combined = _slice_prices(prices, window.formation_start, window.trading_end)
    normalized = normalize_total_return_prices(combined)
    pair_capital = capital / len(ranked)
    pair_dollar_per_leg = dollar_per_leg / len(ranked)
    pair_portfolios = []
    pair_positions = []
    pair_signals = []
    selected_pairs = []
    for _, selected in ranked.iterrows():
        x = str(selected["x"])
        y = str(selected["y"])
        signals = generate_gatev_signals(
            normalized,
            pair=(x, y),
            formation_std=float(selected["formation_std"]),
            trading_index=trading.index,
            entry_sds=entry_threshold,
        )
        bt = run_backtest(
            prices=trading,
            signals=signals,
            pair=(x, y),
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            capital=pair_capital,
            dollar_per_leg=pair_dollar_per_leg,
        )
        pair_portfolios.append(bt["portfolio"])
        pair_positions.append(bt["positions"])
        pair_signals.append(signals)
        selected_pairs.append(f"{x}-{y}")

    stock_exposure = aggregate_stock_exposures(pair_positions)
    stock_exposure.insert(0, "formation_start", window.formation_start.date().isoformat())
    stock_exposure.insert(1, "formation_end", window.formation_end.date().isoformat())
    stock_exposure.insert(2, "trading_end", window.trading_end.date().isoformat())

    portfolio = pd.DataFrame(index=trading.index)
    portfolio["equity"] = sum(p["equity"] for p in pair_portfolios)
    portfolio["daily_ret"] = portfolio["equity"].pct_change().fillna(0.0)
    portfolio["daily_turnover"] = pd.concat(
        [p["daily_turnover"] for p in pair_portfolios], axis=1
    ).mean(axis=1)
    portfolio["drawdown"] = (
        portfolio["equity"] - portfolio["equity"].cummax()
    ) / portfolio["equity"].cummax()
    metrics = compute_metrics(
        equity=portfolio["equity"],
        daily_ret=portfolio["daily_ret"],
        daily_turnover=portfolio["daily_turnover"],
    )
    row = {
        "method": "gatev",
        "pair_selection": "non_overlapping" if non_overlapping else "top_distance_allow_overlap",
        "formation_start": window.formation_start.date().isoformat(),
        "formation_end": window.formation_end.date().isoformat(),
        "trading_end": window.trading_end.date().isoformat(),
        "pair_count": len(selected_pairs),
        "selected_pairs": ";".join(selected_pairs),
        "distance_best_pair": float(ranked.loc[0, "distance"]),
        "trade_count": int(sum(s["entry_flag"].sum() for s in pair_signals)),
        "start_equity": float(portfolio["equity"].iloc[0]),
        "final_equity": float(portfolio["equity"].iloc[-1]),
        "net_profit": float(portfolio["equity"].iloc[-1] - portfolio["equity"].iloc[0]),
        **metrics,
    }
    return {
        "row": row,
        "portfolio": portfolio,
        "signals": pair_signals,
        "stock_exposure": stock_exposure,
    }


def _run_all_windows(
    prices: pd.DataFrame,
    windows: Iterable[Window],
    max_pairs: int,
    entry_threshold: float,
    fee_bps: float,
    slippage_bps: float,
    capital: float,
    dollar_per_leg: float,
    non_overlapping: bool = True,
) -> dict:
    results = []
    for window in windows:
        result = _run_window(
            prices, window, max_pairs, entry_threshold,
            fee_bps, slippage_bps, capital, dollar_per_leg,
            non_overlapping,
        )
        if result is not None:
            results.append(result)

    if not results:
        return {
            "rows": [],
            "summary": {"method": "gatev", "windows_traded": 0},
            "stock_exposure": pd.DataFrame(),
        }

    returns = pd.concat([r["portfolio"]["daily_ret"] for r in results]).sort_index()
    turnover = pd.concat([r["portfolio"]["daily_turnover"] for r in results]).reindex(returns.index)
    equity = capital * (1.0 + returns).cumprod()
    metrics = compute_metrics(equity=equity, daily_ret=returns, daily_turnover=turnover)
    summary = {
        "method": "gatev",
        "pair_selection": "non_overlapping" if non_overlapping else "top_distance_allow_overlap",
        "windows_traded": len(results),
        "trade_count": int(sum(r["row"]["trade_count"] for r in results)),
        "start_equity": capital,
        "final_equity": float(equity.iloc[-1]),
        "net_profit": float(equity.iloc[-1] - capital),
        **metrics,
    }
    stock_exposure = pd.concat(
        [r["stock_exposure"] for r in results], ignore_index=True
    )
    return {
        "rows": [r["row"] for r in results],
        "summary": summary,
        "equity": equity,
        "stock_exposure": stock_exposure,
    }


def _write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tickers",
        default=(
            "AAPL,ADI,AMAT,ANET,APH,AVGO,CSCO,DELL,GLW,HPE,HPQ,IBM,"
            "INTC,KLAC,LRCX,MU,NTAP,QCOM,STX,TXN,WDC,ZBRA"
        ),
        help="Comma-separated eligible technology-hardware universe",
    )
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2025-01-01")
    parser.add_argument("--max-pairs", type=int, default=20)
    parser.add_argument(
        "--allow-overlap",
        action="store_true",
        help="Use raw top-distance pairs even when a ticker appears in multiple pairs",
    )
    parser.add_argument("--fee-bps", type=float, default=1.0)
    parser.add_argument("--slippage-bps", type=float, default=1.0)
    parser.add_argument("--capital", type=float, default=100_000.0)
    parser.add_argument("--dollar-per-leg", type=float, default=50_000.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    raw = download_prices(tickers, start=args.start, end=args.end)
    prices = align_prices(raw)
    prices = validate_prices(prices, max_ffill_days=5)
    prices = drop_sparse_tickers(prices, max_nan_frac=0.02)
    if prices.empty or len(prices.columns) < 2:
        raise ValueError("Insufficient clean price data for Gatev experiment.")

    windows = build_windows(args.start, args.end)
    if not windows:
        raise ValueError("Date range must contain at least 12 formation months and 6 trading months.")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("results") / "gatev_runs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    base = _run_all_windows(
        prices, windows, args.max_pairs, 2.0,
        args.fee_bps, args.slippage_bps, args.capital, args.dollar_per_leg,
        non_overlapping=not args.allow_overlap,
    )
    _write_rows(out_dir / "windows.csv", base["rows"])
    _write_rows(out_dir / "summary.csv", [base["summary"]])
    if not base["stock_exposure"].empty:
        base["stock_exposure"].to_csv(
            out_dir / "stock_exposure.csv",
            index=False,
            encoding="utf-8-sig",
        )

    if "equity" in base:
        equity = base["equity"]
        drawdown = (equity - equity.cummax()) / equity.cummax()
        plot_equity_and_drawdown(
            pd.DataFrame({"equity": equity, "drawdown": drawdown}),
            str(out_dir / "equity.png"),
            title="Gatev walk-forward equity",
        )

    sensitivity = []
    cost_scenarios = [
        ("low_cost", 0.5, 0.5),
        ("base_cost", args.fee_bps, args.slippage_bps),
        ("high_cost", 2.0, 2.0),
    ]
    for scenario, fee_bps, slippage_bps in cost_scenarios:
        result = _run_all_windows(
            prices, windows, args.max_pairs, 2.0,
            fee_bps, slippage_bps, args.capital, args.dollar_per_leg,
            non_overlapping=not args.allow_overlap,
        )
        sensitivity.append({
            "sensitivity_type": "cost",
            "scenario": scenario,
            "entry_threshold": 2.0,
            "fee_bps": fee_bps,
            "slippage_bps": slippage_bps,
            **result["summary"],
        })
    for entry_threshold in (1.5, 2.0, 2.5):
        result = _run_all_windows(
            prices, windows, args.max_pairs, entry_threshold,
            args.fee_bps, args.slippage_bps, args.capital, args.dollar_per_leg,
            non_overlapping=not args.allow_overlap,
        )
        sensitivity.append({
            "sensitivity_type": "entry_threshold",
            "scenario": "base_cost",
            "entry_threshold": entry_threshold,
            "fee_bps": args.fee_bps,
            "slippage_bps": args.slippage_bps,
            **result["summary"],
        })
    _write_rows(out_dir / "sensitivity.csv", sensitivity)

    print(f"Saved Gatev outputs to {out_dir}")
    print(pd.DataFrame([base["summary"]]).to_string(index=False))


if __name__ == "__main__":
    main()

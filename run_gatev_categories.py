"""Run the same Gatev experiment across several pre-specified universes."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from run_gatev_comparison import _run_all_windows, _write_rows, build_windows
from src.benchmarks import equal_weight_buy_and_hold
from src.data import align_prices, download_prices, drop_sparse_tickers, validate_prices
from src.viz import plot_equity_and_drawdown


CATEGORY_UNIVERSES: dict[str, list[str]] = {
    "technology_hardware": [
        "AAPL", "ADI", "AMAT", "ANET", "APH", "AVGO", "CSCO", "DELL",
        "GLW", "HPE", "HPQ", "IBM", "INTC", "KLAC", "LRCX", "MU",
        "NTAP", "QCOM", "STX", "TXN", "WDC", "ZBRA",
    ],
    "semiconductors": [
        "ADI", "AMD", "AMAT", "ASML", "AVGO", "INTC", "KLAC", "LRCX",
        "MCHP", "MRVL", "MU", "NVDA", "NXPI", "ON", "QCOM", "SWKS",
        "TER", "TXN",
    ],
    "airlines": [
        "AAL", "ALGT", "ALK", "DAL", "JBLU", "LUV", "SAVE", "SKYW", "UAL",
    ],
    "retail": [
        "BBY", "COST", "DG", "DLTR", "HD", "KSS", "KR", "LOW", "M",
        "ROST", "TGT", "TJX", "WMT",
    ],
}


def _download_available_prices(
    tickers: list[str],
    start: str,
    end: str,
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Download a category while isolating ticker-specific data failures."""
    raw: dict[str, pd.DataFrame] = {}
    failed: list[str] = []
    for ticker in tickers:
        try:
            raw.update(download_prices([ticker], start=start, end=end))
        except Exception as exc:  # yfinance can fail for one delisted/timed-out ticker
            failed.append(ticker)
            print(f"Skipping {ticker}: {exc}")
    return raw, failed


def _run_category(
    category: str,
    tickers: list[str],
    start: str,
    end: str,
    max_pairs: int,
    fee_bps: float,
    slippage_bps: float,
    capital: float,
    dollar_per_leg: float,
    out_dir: Path,
) -> dict:
    raw, failed_tickers = _download_available_prices(tickers, start, end)
    prices = drop_sparse_tickers(
        validate_prices(align_prices(raw), max_ffill_days=5),
        max_nan_frac=0.02,
    )
    windows = build_windows(start, end)
    if prices.empty or len(prices.columns) < 2 or not windows:
        raise ValueError(f"Insufficient data for category: {category}")

    result = _run_all_windows(
        prices,
        windows,
        max_pairs=max_pairs,
        entry_threshold=2.0,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        capital=capital,
        dollar_per_leg=dollar_per_leg,
        non_overlapping=True,
    )
    category_dir = out_dir / category
    category_dir.mkdir(parents=True, exist_ok=True)
    _write_rows(category_dir / "windows.csv", result["rows"])
    _write_rows(category_dir / "summary.csv", [result["summary"]])
    if not result["stock_exposure"].empty:
        result["stock_exposure"].to_csv(
            category_dir / "stock_exposure.csv",
            index=False,
            encoding="utf-8-sig",
        )

    if "equity" in result:
        equity = result["equity"]
        drawdown = (equity - equity.cummax()) / equity.cummax()
        plot_equity_and_drawdown(
            pd.DataFrame({"equity": equity, "drawdown": drawdown}),
            str(category_dir / "equity.png"),
            title=f"Gatev walk-forward equity: {category}",
        )

    benchmark = equal_weight_buy_and_hold(
        prices,
        capital=capital,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
    )
    benchmark_summary = {
        "category": category,
        "requested_tickers": len(tickers),
        "usable_tickers": len(prices.columns),
        "failed_tickers": ",".join(failed_tickers),
        **benchmark["summary"],
    }
    pd.DataFrame([benchmark_summary]).to_csv(
        category_dir / "buy_hold_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    gatev_summary = {
        "category": category,
        "requested_tickers": len(tickers),
        "usable_tickers": len(prices.columns),
        "failed_tickers": ",".join(failed_tickers),
        **result["summary"],
    }
    return [gatev_summary, benchmark_summary]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2025-01-01")
    parser.add_argument("--max-pairs", type=int, default=20)
    parser.add_argument("--fee-bps", type=float, default=1.0)
    parser.add_argument("--slippage-bps", type=float, default=1.0)
    parser.add_argument("--capital", type=float, default=100_000.0)
    parser.add_argument("--dollar-per-leg", type=float, default=50_000.0)
    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("results") / "gatev_categories" / run_id
    summaries = []
    for category, tickers in CATEGORY_UNIVERSES.items():
        print(f"Running {category}: {len(tickers)} tickers")
        summaries.extend(
            _run_category(
                category,
                tickers,
                args.start,
                args.end,
                args.max_pairs,
                args.fee_bps,
                args.slippage_bps,
                args.capital,
                args.dollar_per_leg,
                out_dir,
            )
        )

    pd.DataFrame(summaries).to_csv(
        out_dir / "category_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(f"Saved category outputs to {out_dir}")
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()

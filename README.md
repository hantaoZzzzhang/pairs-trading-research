# Pairs Trading Research Framework

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

An end-to-end research framework for Gatev-style distance-based pairs trading.
The project demonstrates **data collection, formation-period pair selection,
walk-forward backtesting, and performance analysis**.

---

## Features
- **Data handling**
  - Pulls daily stock prices with `yfinance`
  - Cleans and aligns data into a single table for analysis
- **Pair selection**
  - Normalizes adjusted-close prices at the start of each formation period
  - Ranks eligible pairs by mean squared normalized-price distance
- **Strategy**
  - Uses a fixed formation-period spread standard deviation
  - Enters beyond the formation threshold and exits at a zero crossing
- **Backtesting**
  - Uses chronological 12-month formation and 6-month trading windows
  - Executes signals at the next trading-day close
  - Tracks profit/loss, portfolio value, and drawdowns
  - Reports core metrics: growth, volatility, Sharpe ratio, hit rate
- **Visualization**
  - Plot of the spread with entry/exit signals
  - Equity curve with drawdown shading

---

## Quickstart

Clone the repository:

```bash
git clone https://github.com/<your-username>/pairs-trading.git
cd pairs-trading
```

Create and activate a virtual environment:

```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Gatev workflow:

```bash
python run_gatev_comparison.py --help
```

See [Gatev Configuration](#gatev-configuration) for a reproducible experiment
command and the available options.

---

## Example Results

Running `run_gatev_comparison.py` produces:

- **Console summary**:
  - Selected pairs and formation-period distances
  - Window-level and aggregate performance metrics
  - Starting capital vs. ending equity
  - CAGR, Sharpe ratio, maximum drawdown, turnover and trade count
- **Files** saved to a timestamped directory under `results/gatev_runs/`:
  - Window-level and aggregate CSV results
  - Fee, slippage and entry-threshold sensitivity CSV
  - Equity and drawdown figure

---

## Project Structure

```
pairs-trading/
  README.md
  LICENSE
  requirements.txt
  run_gatev_comparison.py  # primary Gatev experiment runner
  run_gatev_categories.py  # same experiment across fixed universes
  src/                     # source code
    data.py                # download, clean, validate prices
    gatev.py               # normalization, distance ranking, signals
    backtest.py            # backtesting + metrics
    exposure.py            # stock-level long/short exposure diagnostics
    viz.py                 # plotting utilities
  data/
    raw/                   # raw price data (ignored)
    processed/             # cleaned data (ignored)
  results/
    gatev_runs/<run-id>/   # CSV results, sensitivity tables, and figures
  tests/                   # pytest unit tests, including exposure checks
```

The earlier `main.py`, `pairs.py`, and `strategy.py` files are retained as
legacy code and are not the documented Gatev experiment path.

---

## Gatev Configuration

The documented research experiment uses the Gatev distance-based strategy and
is configured through command-line options in `run_gatev_comparison.py`:

```bash
python run_gatev_comparison.py \
  --tickers AAPL,ADI,AMAT,ANET,APH,AVGO,CSCO,DELL,GLW,HPE,HPQ,IBM,INTC,KLAC,LRCX,MU,NTAP,QCOM,STX,TXN,WDC,ZBRA \
  --start 2018-01-01 \
  --end 2025-01-01 \
  --max-pairs 20 \
  --fee-bps 1.0 \
  --slippage-bps 1.0 \
  --capital 100000
```

The default methodology uses a 12-month formation period followed by a
non-overlapping 6-month trading period. The base entry threshold is two
formation-period standard deviations; the runner also records sensitivity to
alternative entry thresholds and transaction costs. `--dollar-per-leg` controls
the dollar exposure assigned to each pair leg. By default, pair selection is
greedy and non-overlapping: once a ticker is selected in one pair, it cannot be
selected in another pair in the same formation window. Use `--allow-overlap`
only to reproduce the raw top-distance benchmark.

To run the same configuration across the pre-specified technology-hardware,
semiconductor, airline, and retail universes:

```bash
python run_gatev_categories.py --start 2018-01-01 --end 2025-01-01 --max-pairs 20
```

This creates a timestamped directory under `results/gatev_categories/`, with a
`category_comparison.csv` file and one subdirectory per category. The comparison
includes the Gatev strategy and an equal-weight buy-and-hold benchmark using
the same eligible tickers. A ticker that
cannot be downloaded is skipped and recorded in `failed_tickers`; the number of
usable tickers is reported so that the result is not mistaken for a complete
universe.

One run using the default non-overlapping selector and the same eligible
universe for an equal-weight buy-and-hold benchmark produced:

| Category | Method | Usable tickers | CAGR | Sharpe | Max drawdown | Net profit |
|---|---|---:|---:|---:|---:|---:|
| Technology hardware | Gatev | 22 | 1.00% | 0.21 | -5.95% | $1,996.78 |
| Technology hardware | Buy and hold | 22 | 19.64% | 0.76 | -38.51% | $250,018.99 |
| Semiconductors | Gatev | 18 | 1.69% | 0.39 | -3.93% | $3,387.61 |
| Semiconductors | Buy and hold | 18 | 26.52% | 1.17 | -49.68% | $417,327.35 |
| Airlines | Gatev | 8 | -2.47% | -0.36 | -13.96% | -$4,857.70 |
| Airlines | Buy and hold | 8 | -1.32% | 0.18 | -65.56% | -$8,901.65 |
| Retail | Gatev | 13 | -1.72% | -0.21 | -11.78% | -$3,392.18 |
| Retail | Buy and hold | 13 | 12.04% | 0.65 | -30.06% | $121,242.95 |

The direct-investment benchmark uses the same cleaned eligible tickers, equal
weights, date range, and fee/slippage assumptions. For semiconductors, the
large gap between the buy-and-hold and Gatev results suggests that sector-wide
appreciation, rather than the pair signal, explains most of the raw investment
return in this sample. The benchmark has substantially larger drawdowns, so
the comparison is about return attribution, not a claim that either strategy
dominates on every risk dimension.

These are one historical run, not evidence that one category will remain
profitable. The airline result excluded `SAVE` because Yahoo Finance returned no
usable historical data for that ticker.

---

## Requirements
- Python 3.10+  
- Libraries: pandas, numpy, statsmodels, matplotlib, yfinance, pyarrow  
  (see `requirements.txt` for full list)

---

## Notes & Disclaimer
This project is for **educational purposes only** and should not be used as financial advice.

## Gatev Distance-Based Experiment

The repository now includes a separate Gatev-style distance strategy and an
experiment runner:

```text
src/gatev.py                 # normalization, distance ranking, Gatev signals
run_gatev_comparison.py      # Gatev-only walk-forward experiment
tests/test_gatev.py          # distance, signals, timing, costs, leakage tests
```

Run the Gatev experiment with the expanded technology-hardware universe and up to 20 selected pairs:

```bash
python run_gatev_comparison.py --tickers AAPL,ADI,AMAT,ANET,APH,AVGO,CSCO,DELL,GLW,HPE,HPQ,IBM,INTC,KLAC,LRCX,MU,NTAP,QCOM,STX,TXN,WDC,ZBRA --max-pairs 20 --start 2018-01-01 --end 2025-01-01
```

Each run creates a timestamped directory under `results/gatev_runs/`
containing window-level CSVs, summary CSVs, cost/entry-threshold sensitivity,
stock-level exposure diagnostics, and separate equity figures. Existing result
figures are not overwritten.

### Methodology

The experiment uses cleaned prices, chronological windows, fixed capital,
next-bar execution, fees, slippage, and the default 12-month formation period
followed by 6 non-overlapping months of trading.

The Gatev method:

1. Normalizes each stock's adjusted-close series to 1.0 at the beginning of the
   formation period. Adjusted close is used as a practical total-return proxy;
   it is not a CRSP point-in-time total-return series.
2. Ranks pairs by `D_ij = mean((P_i - P_j)^2)` using formation observations only.
3. Uses the formation-period standard deviation of the normalized spread only.
4. At each trading-day close, enters when the absolute normalized spread exceeds
   two formation-period standard deviations. The existing backtester executes
   that signal at the next trading-day close; therefore, the signal day's close
   is not used as the execution price.
5. Closes after a zero/sign crossing or at the end of the trading period. The
   final signal is flattened one bar before the end so next-bar execution closes
   the position on the final trading close. No rolling z-score, stop, or time
   stop is silently added to the Gatev method.

The current experiment intentionally excludes the cointegration baseline. The
legacy cointegration modules and tests remain in the repository for historical
experiments, but they are not imported or executed by this Gatev runner.

### Limitations

- The small KO/PEP universe belongs to the legacy demo and is only an
  implementation smoke test. The documented Gatev experiment uses a larger,
  pre-specified universe, but it is still not a survivorship-bias-free study.
- Yahoo Finance adjusted prices are an approximation to total-return prices;
  survivorship bias, delistings, ticker changes, corporate actions, and data
  availability can affect results.
- Pair selection, normalization, and thresholds are isolated to each formation
  window. Do not fit them on the complete backtest period.
- The optional raw top-distance benchmark (`--allow-overlap`) may reuse the
  same stock across pairs. The default non-overlapping selector avoids this
  particular concentration within each formation window, but does not remove
  all correlation or common-factor risk.
- Sensitivity results are diagnostic; parameters must not be selected using the
  final test period.
- Reported hit rate is the fraction of positive-return days, not the fraction
  of profitable round-trip trades. The runner also reports pair trade count.

### References

- [Gatev, Goetzmann, and Rouwenhorst (2006), “Pairs Trading: Performance of a Relative-Value Arbitrage Rule”](https://doi.org/10.1093/rfs/hhj020)
- [Zhu (2024), “Examining Pairs Trading Profitability”](https://economics.yale.edu/sites/default/files/2024-05/Zhu_Pairs_Trading.pdf)
- [Upstream pairs-trading project](https://github.com/alitariq1/pairs-trading)

---

## License
[MIT](LICENSE)

## Acknowledgments
- [yfinance](https://github.com/ranaroussi/yfinance) for market data
- [statsmodels](https://www.statsmodels.org/) for econometric tools
- [pandas](https://pandas.pydata.org/), [numpy](https://numpy.org/), [matplotlib](https://matplotlib.org/) for the core stack

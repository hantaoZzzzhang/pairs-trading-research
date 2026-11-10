"""Gatev-style distance pairs trading.

The implementation follows the formation/trading convention described in
Gatev, Goetzmann, and Rouwenhorst (2006) and Zhu (2024): prices are normalized
at the start of the formation period, pairs are ranked by mean squared distance
on formation observations only, and trading uses the formation-period spread
standard deviation. Signals are generated at a close and executed by the
existing backtester on the next close.
"""

from __future__ import annotations

import itertools
from typing import Sequence

import numpy as np
import pandas as pd


DISTANCE_COLUMNS = ["x", "y", "distance", "formation_mean", "formation_std"]


def normalize_total_return_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Normalize each price series to 1.0 at the first formation observation."""
    if prices.empty:
        raise ValueError("Cannot normalize an empty price matrix.")
    clean = prices.astype(float).copy()
    first = clean.iloc[0]
    if first.isna().any() or (first <= 0).any():
        raise ValueError("Formation prices must have positive values on the first date.")
    return clean.divide(first, axis="columns")


def normalized_spread(x: pd.Series, y: pd.Series) -> pd.Series:
    """Return x-y after aligning and removing missing observations."""
    xy = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    return (xy["x"] - xy["y"]).rename("normalized_spread")


def pair_distance(x: pd.Series, y: pd.Series) -> float:
    """Compute D_ij = mean_t[(P_i,t - P_j,t)^2]."""
    spread = normalized_spread(x, y)
    if spread.empty:
        raise ValueError("Pair has no overlapping observations.")
    return float(np.mean(np.square(spread.to_numpy(dtype=float))))


def formation_spread_stats(x: pd.Series, y: pd.Series) -> dict[str, float]:
    """Return formation distance, mean spread, and population std deviation."""
    spread = normalized_spread(x, y)
    if spread.empty:
        raise ValueError("Pair has no overlapping formation observations.")
    return {
        "distance": float(np.mean(np.square(spread.to_numpy(dtype=float)))),
        "formation_mean": float(spread.mean()),
        # Zhu's displayed variance uses 1/T; ddof=0 matches that convention.
        "formation_std": float(spread.std(ddof=0)),
    }


def rank_distance_pairs(
    formation_prices: pd.DataFrame,
    max_pairs: int | None = 20,
    sector_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Rank eligible pairs by formation-period normalized-price distance."""
    normalized = normalize_total_return_prices(formation_prices)
    rows: list[dict[str, float | str]] = []
    for x, y in itertools.combinations(normalized.columns, 2):
        if sector_map is not None and sector_map.get(x) != sector_map.get(y):
            continue
        try:
            stats = formation_spread_stats(normalized[x], normalized[y])
        except (TypeError, ValueError):
            continue
        rows.append({"x": x, "y": y, **stats})

    if not rows:
        return pd.DataFrame(columns=DISTANCE_COLUMNS)
    ranked = (
        pd.DataFrame(rows, columns=DISTANCE_COLUMNS)
        .sort_values("distance", ascending=True)
        .reset_index(drop=True)
    )
    return ranked if max_pairs is None else ranked.head(max_pairs)


def select_non_overlapping_pairs(
    ranked_pairs: pd.DataFrame,
    max_pairs: int = 20,
) -> pd.DataFrame:
    """Greedily select low-distance pairs without reusing a ticker.

    ``ranked_pairs`` must already be sorted by formation distance. This is a
    portfolio-construction extension to the original top-M rule: it reduces
    stock-level concentration and prevents a ticker from being long in one
    selected pair while short in another selected pair.
    """
    if ranked_pairs.empty or max_pairs <= 0:
        return ranked_pairs.iloc[0:0].copy()

    used: set[str] = set()
    selected_indices: list[int] = []
    for index, row in ranked_pairs.iterrows():
        x, y = str(row["x"]), str(row["y"])
        if x in used or y in used:
            continue
        selected_indices.append(index)
        used.update((x, y))
        if len(selected_indices) >= max_pairs:
            break

    return ranked_pairs.loc[selected_indices].reset_index(drop=True)


def generate_gatev_signals(
    normalized_prices: pd.DataFrame,
    pair: tuple[str, str],
    formation_std: float,
    trading_index: pd.Index,
    entry_sds: float = 2.0,
) -> pd.DataFrame:
    """Generate fixed-threshold Gatev signals for one trading period.

    Positive x-y spread means x is the leader, so the signal is short x/long y;
    negative spread means long x/short y. A position closes when the sign of
    the spread changes (including a zero crossing). The final position is
    forced flat one bar before the end so the existing next-bar backtester
    closes it on the final trading close.
    """
    x, y = pair
    if formation_std <= 0 or not np.isfinite(formation_std):
        raise ValueError("formation_std must be positive and finite.")
    if len(trading_index) == 0:
        return pd.DataFrame(
            columns=["spread", "formation_z", "z", "side", "entry_flag", "exit_flag", "w_x", "w_y"],
            index=trading_index,
        )

    spread = normalized_spread(normalized_prices[x], normalized_prices[y]).reindex(trading_index)
    threshold = float(entry_sds * formation_std)
    side = np.zeros(len(trading_index), dtype=int)
    entry_flag = np.zeros(len(trading_index), dtype=int)
    exit_flag = np.zeros(len(trading_index), dtype=int)
    current = 0

    for i, value in enumerate(spread.to_numpy(dtype=float)):
        if not np.isfinite(value):
            if current != 0:
                current = 0
                exit_flag[i] = 1
        elif current != 0 and np.sign(value) != current:
            current = 0
            exit_flag[i] = 1
        elif current == 0 and abs(value) > threshold:
            current = int(np.sign(value))
            entry_flag[i] = 1
        side[i] = current

    # End-of-period liquidation is known in advance. Suppress an entry on the
    # final signal date and close any prior position at the final trading close.
    if len(side) >= 2:
        entry_flag[-1] = 0
        side[-1] = 0
        if side[-2] != 0:
            side[-2] = 0
            exit_flag[-2] = 1
    else:
        entry_flag[-1] = 0
        side[-1] = 0

    w_y = side.astype(float)
    w_x = -w_y
    out = pd.DataFrame(
        {
            "spread": spread.to_numpy(dtype=float),
            "formation_z": spread.to_numpy(dtype=float) / formation_std,
            "z": spread.to_numpy(dtype=float) / formation_std,
            "side": side,
            "entry_flag": entry_flag,
            "exit_flag": exit_flag,
            "w_x": w_x,
            "w_y": w_y,
        },
        index=trading_index,
    )
    out.index.name = "Date"
    return out

"""Anomaly detection on a monthly business metric.

Seasonality is the classic trap (a Q4 peak is not an anomaly), so with >= 15 months of history we
work on the *year-over-year growth rate* of the per-day value (cancels seasonality and leap-year /
month-length effects); otherwise on month-over-month growth.

Detection is a scan statistic: for every contiguous window of 1-6 months we compute a two-sample
t-statistic of "growth inside the window" vs "growth everywhere else" and keep the most extreme
window (up to two windows). That recovers multi-month incidents that single-month z-scores miss.
An Isolation Forest on the same features gives an independent second opinion.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


@dataclass
class AnomalyWindow:
    start: pd.Timestamp
    end: pd.Timestamp  # exclusive
    months: int
    severity: float  # t-statistic of the window vs the rest (negative = drop)
    corroborated: float  # share of months the Isolation Forest also flagged


@dataclass
class AnomalyResult:
    method: str
    table: pd.DataFrame
    windows: list[AnomalyWindow]


def _robust_z(x: pd.Series) -> pd.Series:
    med = x.median()
    scale = max((x - med).abs().median() * 1.4826, 0.005)
    return (x - med) / scale


def _scan(sig: pd.Series, direction: str, t_thresh: float, max_len: int = 6, max_windows: int = 2):
    vals, n = sig.to_numpy(), len(sig)
    taken = np.zeros(n, dtype=bool)
    found = []
    for _ in range(max_windows):
        best = None
        for L in range(1, min(max_len, n // 2) + 1):
            for i in range(0, n - L + 1):
                if taken[i:i + L].any():
                    continue
                mask = np.ones(n, dtype=bool)
                mask[i:i + L] = False
                mask &= ~taken
                outside = vals[mask]
                if len(outside) < 5:
                    continue
                sd = max(outside.std(ddof=1), 0.01)
                t = (vals[i:i + L].mean() - outside.mean()) / (sd * np.sqrt(1 / L + 1 / len(outside)))
                score = -t if direction == "down" else t if direction == "up" else abs(t)
                if best is None or score > best[0]:
                    best = (score, i, L, t)
        if best is None or best[0] < t_thresh:
            break
        _, i, L, t = best
        taken[i:i + L] = True
        found.append((i, L, float(t)))
    return found


def detect_anomalies(series: pd.Series, direction: str = "down", t_thresh: float = 3.5) -> AnomalyResult:
    s = series.sort_index().astype(float)
    s.index = pd.DatetimeIndex(s.index)
    per_day = s / s.index.days_in_month
    mom = per_day.pct_change()
    if len(s) >= 15:
        sig, method = per_day.pct_change(12), "yoy_growth_scan"
    else:
        sig, method = mom, "mom_growth_scan"
    z = _robust_z(sig.dropna()).reindex(s.index)

    feats = pd.DataFrame({"sig": sig, "mom": mom}).dropna()
    iso = pd.Series(False, index=s.index)
    if len(feats) >= 8:
        model = IsolationForest(n_estimators=200, contamination=0.2, random_state=0).fit(feats)
        iso.loc[feats.index] = model.predict(feats) == -1

    clean = sig.dropna()
    windows: list[AnomalyWindow] = []
    flagged = pd.Series(False, index=s.index)
    for i, L, t in sorted(_scan(clean, direction, t_thresh)):
        seg = clean.index[i:i + L]
        flagged.loc[seg] = True
        windows.append(AnomalyWindow(
            start=pd.Timestamp(seg[0]), end=pd.Timestamp(seg[-1]) + pd.DateOffset(months=1),
            months=L, severity=t, corroborated=float(iso.loc[seg].mean()),
        ))
    table = pd.DataFrame({"value": s, "mom": s.pct_change(), "signal": sig, "z": z, "flagged": flagged, "iforest": iso})
    return AnomalyResult(method, table, windows)

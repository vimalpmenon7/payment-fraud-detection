"""Leakage-safe feature engineering.

Every customer-history feature for a transaction uses ONLY transactions strictly earlier in the
sorted order, so a feature value never changes when later data arrives (tested in tests/).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import CATEGORIES

BASE_FEATURES = ["amount", "log_amount", "hour", "is_night", "device_new", "country_mismatch",
                 "txn_count_24h", "hours_since_last", "first_txn", "amount_to_cust_mean"]
FEATURES = BASE_FEATURES + [f"cat_{c}" for c in CATEGORIES]
NO_HISTORY_HOURS = 720.0  # used when a customer has no earlier transaction


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a feature frame aligned to ``df`` (which must be sorted by ``ts``)."""
    if not df["ts"].is_monotonic_increasing:
        raise ValueError("df must be sorted by ts")
    n = len(df)
    ts = df["ts"].to_numpy(dtype=np.int64)
    amt = df["amount"].to_numpy(dtype=float)
    count_24h = np.zeros(n)
    since_last = np.full(n, NO_HISTORY_HOURS)
    first = np.ones(n)
    ratio = np.ones(n)

    for idx in df.groupby("customer_id", sort=False).indices.values():
        t, a = ts[idx], amt[idx]
        k = np.arange(len(idx))
        start = np.searchsorted(t, t - 86_400, side="left")
        count_24h[idx] = k - start  # earlier txns within the last 24h
        if len(idx) > 1:
            since_last[idx[1:]] = (t[1:] - t[:-1]) / 3600.0
            first[idx[1:]] = 0.0
            prev_mean = (np.cumsum(a) - a)[1:] / k[1:]
            ratio[idx[1:]] = a[1:] / np.maximum(prev_mean, 1e-6)

    hour = (ts // 3600) % 24
    out = pd.DataFrame({
        "amount": amt, "log_amount": np.log1p(amt), "hour": hour, "is_night": (hour < 5).astype(int),
        "device_new": df["device_new"].to_numpy(), "country_mismatch": df["country_mismatch"].to_numpy(),
        "txn_count_24h": count_24h, "hours_since_last": np.minimum(since_last, NO_HISTORY_HOURS),
        "first_txn": first, "amount_to_cust_mean": np.clip(ratio, 0, 50),
    }, index=df.index)
    for c in CATEGORIES:
        out[f"cat_{c}"] = (df["merchant_category"].to_numpy() == c).astype(int)
    return out[FEATURES]

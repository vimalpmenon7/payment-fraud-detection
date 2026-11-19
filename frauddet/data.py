"""Synthetic card-transaction generator with known fraud patterns (no real data is used)."""

from __future__ import annotations

import numpy as np
import pandas as pd

CATEGORIES = ["grocery", "fuel", "restaurant", "travel", "electronics", "gift_cards", "crypto"]
CATEGORY_RISK = {"grocery": 0.0, "fuel": 0.0, "restaurant": 0.0, "travel": 0.3,
                 "electronics": 1.0, "gift_cards": 1.6, "crypto": 1.8}
CATEGORY_P = [0.32, 0.18, 0.2, 0.1, 0.1, 0.06, 0.04]


def generate_transactions(n: int = 20_000, n_customers: int = 600, days: int = 60, seed: int = 7) -> pd.DataFrame:
    """Return a DataFrame sorted by time with columns:
    txn_id, customer_id, ts (epoch seconds), amount, merchant_category,
    device_new, country_mismatch, is_fraud.

    Fraud is drawn from a logistic risk model (night, new device, foreign country, risky merchant,
    unusually large amount) and often comes in bursts of follow-up transactions, so velocity features help.
    """
    rng = np.random.default_rng(seed)
    cust_mean = rng.lognormal(mean=np.log(40), sigma=0.5, size=n_customers)
    t0 = 1_790_000_000
    span = days * 86_400

    cust = rng.integers(0, n_customers, size=n)
    ts = t0 + np.sort(rng.integers(0, span, size=n))
    # more activity in the daytime: thin out some night transactions
    hour = (ts // 3600) % 24
    keep = (hour >= 7) | (rng.random(n) < 0.35)
    cust, ts = cust[keep], ts[keep]
    n = len(ts)
    hour = (ts // 3600) % 24

    amount = cust_mean[cust] * rng.lognormal(0, 0.6, size=n)
    cat = rng.choice(len(CATEGORIES), size=n, p=CATEGORY_P)
    risk = np.array([CATEGORY_RISK[CATEGORIES[c]] for c in cat])
    device_new = (rng.random(n) < 0.04).astype(int)
    mismatch = (rng.random(n) < 0.03).astype(int)
    night = ((hour < 5)).astype(int)

    logit = (-6.2 + 1.6 * night + 2.4 * device_new + 2.4 * mismatch + 1.1 * risk
             + 0.9 * np.log(np.maximum(amount / cust_mean[cust], 0.1)))
    p = 1 / (1 + np.exp(-logit))
    fraud = (rng.random(n) < p).astype(int)

    df = pd.DataFrame({"customer_id": cust, "ts": ts, "amount": amount.round(2),
                       "merchant_category": [CATEGORIES[c] for c in cat],
                       "device_new": device_new, "country_mismatch": mismatch, "is_fraud": fraud})

    # fraud bursts: follow-up fraudulent transactions minutes later on the same customer
    extra = []
    for _, row in df[df.is_fraud == 1].iterrows():
        if rng.random() < 0.5:
            for k in range(int(rng.integers(1, 4))):
                extra.append({"customer_id": row.customer_id,
                              "ts": int(row.ts + rng.integers(60, 1800) * (k + 1)),
                              "amount": round(float(row.amount * rng.lognormal(0.2, 0.3)), 2),
                              "merchant_category": rng.choice(["electronics", "gift_cards", "crypto"]),
                              "device_new": 1, "country_mismatch": int(rng.random() < 0.5), "is_fraud": 1})
    if extra:
        df = pd.concat([df, pd.DataFrame(extra)], ignore_index=True)
    df = df.sort_values(["ts", "customer_id"], kind="mergesort").reset_index(drop=True)
    df.insert(0, "txn_id", [f"T{i:06d}" for i in range(len(df))])
    return df

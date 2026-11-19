import tempfile
import unittest

import numpy as np
import pandas as pd

from frauddet.data import generate_transactions
from frauddet.features import FEATURES, build_features
from frauddet.model import best_threshold, expected_cost, load, save, score, time_split, train

DF = generate_transactions(n=12_000, seed=3)


class DataTests(unittest.TestCase):
    def test_reproducible_and_sorted(self):
        a, b = generate_transactions(2000, seed=1), generate_transactions(2000, seed=1)
        pd.testing.assert_frame_equal(a, b)
        self.assertTrue(a["ts"].is_monotonic_increasing)
        self.assertEqual(a["txn_id"].is_unique, True)

    def test_fraud_rate_is_realistically_low(self):
        rate = DF["is_fraud"].mean()
        self.assertTrue(0.005 < rate < 0.08, rate)


class FeatureTests(unittest.TestCase):
    def test_columns_and_no_nans(self):
        f = build_features(DF)
        self.assertEqual(list(f.columns), FEATURES)
        self.assertFalse(f.isna().any().any())

    def test_no_future_leakage(self):
        """Features of early rows must be identical whether or not later rows exist."""
        k = 6000
        early_only = build_features(DF.iloc[:k])
        with_future = build_features(DF).iloc[:k]
        pd.testing.assert_frame_equal(early_only, with_future)

    def test_velocity_counts_only_earlier_transactions(self):
        df = pd.DataFrame({"ts": [1000, 1100, 1200, 1300 + 90_000], "customer_id": [1, 1, 1, 1],
                           "amount": [10.0, 10.0, 10.0, 10.0], "merchant_category": ["fuel"] * 4,
                           "device_new": [0] * 4, "country_mismatch": [0] * 4})
        f = build_features(df)
        self.assertEqual(f["txn_count_24h"].tolist(), [0, 1, 2, 0])
        self.assertEqual(f["first_txn"].tolist(), [1, 0, 0, 0])

    def test_unsorted_input_rejected(self):
        with self.assertRaises(ValueError):
            build_features(DF.iloc[::-1])


class ModelTests(unittest.TestCase):
    def test_time_split_is_chronological_and_complete(self):
        tr, va, te = time_split(DF)
        self.assertLessEqual(tr["ts"].max(), va["ts"].min())
        self.assertLessEqual(va["ts"].max(), te["ts"].min())
        self.assertEqual(len(tr) + len(va) + len(te), len(DF))
        with self.assertRaises(ValueError):
            time_split(DF, 0.9, 0.2)

    def test_expected_cost_and_threshold_on_toy_data(self):
        y = np.array([1, 0, 0, 1, 0])
        amt = np.array([500.0, 20.0, 30.0, 400.0, 10.0])
        p = np.array([0.9, 0.2, 0.1, 0.8, 0.05])
        self.assertEqual(expected_cost(y, amt, p >= 0.5, review_cost=5.0), 10.0)  # 2 flagged, none missed
        self.assertEqual(expected_cost(y, amt, np.zeros(5, bool), review_cost=5.0), 900.0)
        self.assertTrue(0.2 < best_threshold(y, amt, p) <= 0.9)

    def test_train_beats_base_rate_and_saves(self):
        model, res = train(DF)
        self.assertGreater(res.test["pr_auc"], 3 * res.base_rate_test)
        self.assertGreater(res.test["roc_auc"], 0.8)
        self.assertLess(res.test["expected_cost"], res.test["cost_if_no_model"])
        self.assertEqual(len(res.top_features), 5)
        with tempfile.TemporaryDirectory() as d:
            save(model, res, d)
            out = score(DF.iloc[-500:].reset_index(drop=True), load(d))
        self.assertEqual(list(out.columns), ["txn_id", "fraud_probability", "flagged"])
        self.assertTrue(out["fraud_probability"].between(0, 1).all())

    def test_too_little_data_raises(self):
        with self.assertRaises(ValueError):
            train(generate_transactions(n=300, seed=1))


if __name__ == "__main__":
    unittest.main()

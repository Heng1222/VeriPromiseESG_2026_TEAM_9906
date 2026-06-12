import unittest

import pandas as pd

from app.model.compare_result import (
    ALL_COLUMNS,
    align_results,
    evaluate_results,
)


class CompareResultTests(unittest.TestCase):
    def make_frame(self, rows):
        return pd.DataFrame(rows, columns=ALL_COLUMNS)

    def test_missing_prediction_is_scored_as_error(self):
        truth = self.make_frame(
            [
                [1, "Yes", "already", "Yes", "Clear"],
                [2, "Yes", "within_2_years", "Yes", "Not Clear"],
            ]
        )
        pred = self.make_frame(
            [
                [1, "Yes", "already", "Yes", "Clear"],
                [2, "Yes", "N/A", "Yes", "Not Clear"],
            ]
        )

        results = evaluate_results(truth, pred)
        timeline = results["tasks"]["verification_timeline"]

        self.assertEqual(timeline["evaluated_count"], 2)
        self.assertEqual(timeline["invalid_count"], 1)
        self.assertLess(timeline["macro_f1"], 1.0)

    def test_timeline_alias_is_canonicalized(self):
        truth = self.make_frame(
            [[1, "Yes", "more_than_5_years", "Yes", "Clear"]]
        )
        pred = self.make_frame(
            [[1, "Yes", "longer_than_5_years", "Yes", "Clear"]]
        )

        results = evaluate_results(truth, pred)
        self.assertEqual(
            results["tasks"]["verification_timeline"]["macro_f1"],
            0.25,
        )

    def test_id_mismatch_raises(self):
        truth = self.make_frame([[1, "No", "N/A", "N/A", "N/A"]])
        pred = self.make_frame([[2, "No", "N/A", "N/A", "N/A"]])

        with self.assertRaises(ValueError):
            align_results(truth, pred)


if __name__ == "__main__":
    unittest.main()

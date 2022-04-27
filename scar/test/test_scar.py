import pandas as pd
from scar import model
import unittest


class ScarIntegration(unittest.TestCase):
    """
    Test activation_functions.py functions.
    """

    def test_scar(self):
        raw_count = pd.read_pickle("scar/test/raw_counts.pickle")
        ambient_profile = pd.read_pickle("scar/test/ambient_profile.pickle")
        expected_output = pd.read_pickle("scar/test/output_assignment.pickle")

        scarObj = model(
            raw_count=raw_count.values,
            ambient_profile=ambient_profile,
            feature_type="sgRNA",
        )

        scarObj.train(epochs=40, batch_size=64)

        scarObj.inference()

        self.assertTrue(scarObj.feature_assignment.equals(expected_output))

from logging import root
import pandas as pd
from pathlib import Path
from scAR import model
import unittest

class ScarIntegration(unittest.TestCase):
    """
    Test activation_functions.py functions.
    """    

    def test_scar(self):
        raw_count = pd.read_pickle("scAR/test/raw_counts.pickle")
        empty_profile=pd.read_pickle("scAR/test/ambient_profile.pickle")
        expected_output=pd.read_pickle("scAR/test/output_assignment.pickle")

        scarObj = model(
            raw_count=raw_count.values, empty_profile=empty_profile, scRNAseq_tech="CROPseq"
        )

        scarObj.train(epochs=40, batch_size=64)

        scarObj.inference()

        self.assertTrue(scarObj.feature_assignment.equals(expected_output))

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

from scar import model, data_generator
import unittest


class ScarIntegration(unittest.TestCase):
    """
    Functional testing
    """

    def test_scar(self):
        raw_count = pd.read_pickle("scar/test/raw_counts.pickle")
        ambient_profile = pd.read_pickle("scar/test/ambient_profile.pickle")
        expected_output = pd.read_pickle("scar/test/output_assignment.pickle")

        scarObj = model(
            raw_count=raw_count.values,
            ambient_profile=ambient_profile,
            feature_type="sgRNAs",
        )

        scarObj.train(epochs=40, batch_size=64)

        scarObj.inference()

        self.assertTrue(scarObj.feature_assignment.equals(expected_output))

    def test_scar_data_generator(self):
        """
        Functional testing of data_generator module
        """
        np.random.seed(8)
        citeseq = data_generator.citeseq(6000, 6, 50)
        citeseq.generate()

        citeseq_raw_counts = pd.read_pickle("scar/test/citeseq_raw_counts.pickle")

        self.assertTrue(np.equal(citeseq.obs_count, citeseq_raw_counts.values))

    def test_scar_citeseq(self):
        """
        Functional testing of scAR
        """
        citeseq_raw_counts = pd.read_pickle("scar/test/citeseq_raw_counts.pickle")
        citeseq_ambient_profile = pd.read_pickle(
            "scar/test/citeseq_ambient_profile.pickle"
        )
        citeseq_native_signals = pd.read_pickle(
            "scar/test/citeseq_native_counts.pickle"
        )

        citeseq_scar = model(
            raw_count=citeseq_raw_counts.values,
            ambient_profile=citeseq_ambient_profile.values,
            feature_type="ADTs",
        )

        citeseq_scar.train(epochs=50, batch_size=64, verbose=False)
        citeseq_scar.inference()

        dist = euclidean_distances(
            citeseq_native_signals.values, citeseq_scar.native_counts
        )
        mean_dist = (np.eye(dist.shape[0]) * dist).sum() / dist.shape[0]

        self.assertTrue(mean_dist < 50)

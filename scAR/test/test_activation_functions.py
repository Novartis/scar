""" This tests the activation functions. """
import decimal
import unittest
import numpy
import torch
from decimal import Decimal
from scAR.main._activation_functions import mytanh, hnormalization, mySoftplus


class ActivationFunctionsTest(unittest.TestCase):
    """
    Test activation_functions.py functions.
    """

    def test_mytanh(self):
        """
        Test mytanh().
        """
        self.assertEqual(mytanh(Decimal(1)).quantize(decimal.Decimal('.01'), rounding=decimal.ROUND_DOWN),
                         Decimal(0.88).quantize(decimal.Decimal('.01'),
                                                rounding=decimal.ROUND_DOWN))

    def test_hnormalization(self):
        """
        Test hnormalization().
        """
        self.assertTrue(torch.allclose(hnormalization(torch.tensor(numpy.full((20, 8), 1))).double(),
                                       torch.tensor(numpy.full((20, 8), 0.1250))))

    def test_mySoftplus(self):
        """
        Test mySoftplus().
        """
        self.assertTrue(torch.allclose(mySoftplus(torch.tensor(numpy.full((20, 8), 0.1))).double(),
                                       torch.tensor(numpy.full((20, 8), 0.7444))))

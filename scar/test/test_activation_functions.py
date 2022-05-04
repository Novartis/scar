""" This tests the activation functions. """
import decimal
import unittest
import numpy
import torch
from decimal import Decimal
from scar.main._activation_functions import mytanh, hnormalization, mysoftplus


class ActivationFunctionsTest(unittest.TestCase):
    """
    Test activation_functions.py functions.
    """

    def test_mytanh(self):
        """
        Test mytanh().
        """
        self.assertEqual(
            Decimal(mytanh()(torch.tensor(1.0, dtype=torch.float32)).item()).quantize(
                decimal.Decimal(".01"), rounding=decimal.ROUND_DOWN
            ),
            Decimal(0.88).quantize(decimal.Decimal(".01"), rounding=decimal.ROUND_DOWN),
        )

    def test_hnormalization(self):
        """
        Test hnormalization().
        """
        self.assertTrue(
            torch.allclose(
                hnormalization()(torch.tensor(numpy.full((20, 8), 1))).double(),
                torch.tensor(numpy.full((20, 8), 0.1250)),
            )
        )

    def test_mysoftplus(self):
        """
        Test mysoftplus().
        """
        self.assertTrue(
            torch.allclose(
                mysoftplus()(
                    torch.tensor(numpy.full((20, 8), 0.01), dtype=torch.float32)
                ).double(),
                torch.tensor(numpy.full((20, 8), 0.3849)),
            )
        )

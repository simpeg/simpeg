from discretize.tests import OrderTest
import numpy as np
import matplotlib.pyplot as plt
from SimPEG.regularization import SmoothnessFullGradient


class RegOrderTest(OrderTest):
    meshTypes = ["uniformTensorMesh", "uniformTree"]
    meshSizes = [4, 8, 16, 32]
    meshDimension = 2

    def getError(self):
        true_val = 59.2176264065362 / 2
        x = self.M.cell_centers[:, 0]
        y = self.M.cell_centers[:, 1]
        # a function that is zero at edge with zero derivative
        f_cc = (1 - np.cos(2 * x * np.pi)) * (1 - np.cos(2 * y * np.pi))

        reg = SmoothnessFullGradient(self.M, alphas=[1, 1])
        return reg(f_cc) - true_val

    def test_orderWeakCellGradIntegral(self):
        self.orderTest()

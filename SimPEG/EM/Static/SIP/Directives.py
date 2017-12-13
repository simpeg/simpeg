from SimPEG.Directives import InversionDirective
import numpy as np


class UpdateSensWeighting(InversionDirective):
    """
    Directive to take care of re-weighting
    the non-linear spectral IP problems.
    """
    mapping = None
    ComboRegFun = True
    ComboMisfitFun = True
    JtJdiag = None

    def initialize(self):

        # if getattr(self, 'JtJdiag', None) is None:
        #     # Get sum square of columns of J
        print ("Intial")
        self.JtJdiag = self.prob.getJtJdiag()

        # Compute normalized weights
        self.wr = self.getWr(self.JtJdiag)

        # Update the regularization
        self.updateReg()

        # Send a copy of JtJdiag for the preconditioner
        # self.updateOpt()
        # Rescale weights
        # self.regScale()

    def endIter(self):
        print ("update")
        # Get sum square of columns of J
        JtJdiag = self.prob.getJtJdiag()

        # Compute normalized weights
        self.wr = self.getWr(JtJdiag)

        # Update the regularization
        self.updateReg()

        # Send a copy of JtJdiag for the preconditioner
        # self.updateOpt()
        # Rescale weights
        # self.regScale()

    # Not sure this is really important ...
    # GaussNewton should handle scale issue in theory
    def regScale(self):
        """
            Update the scales used by regularization
        """

        # Currently implemented specifically for MVI-S
        # Need to be generalized if used by others

    def getWr(self, JtJdiag):
        """
            Take the diagonal of JtJ and return
            a normalized sensitivty weighting vector
        """

        # fm = self.reg.objfcts[0].objfcts[0]
        # max_a = max(abs(fm.W * (self.reg.objfcts[0].mapping*self.invProb.model)))
        # fm = self.reg.objfcts[1].objfcts[0]
        # max_b = max(abs(fm.W * (self.reg.objfcts[0].mapping*self.invProb.model)))
        # fm = self.reg.objfcts[2].objfcts[0]
        # max_c = max(abs(fm.W * (self.reg.objfcts[0].mapping*self.invProb.model)))

        # r0 = max_a/max_a
        # r1 = max_b/max_a
        # r2 = max_c/max_a

        wr = np.sqrt(JtJdiag)
        wr = wr / wr.max()
        n = int(len(wr)/2)
        # ratio = np.r_[r0, r1, r2]

        wmax = wr.reshape((n, 2), order="F").max(axis=0)
        wr = np.r_[np.ones(n), np.ones(n)*wmax[1]/wmax[0]]
        # wr = np.r_[r0*np.sqrt(wr), r1*np.sqrt(wr), r2*np.sqrt(wr)]
        return wr

    def updateReg(self):
        """
            Update the cell weights
        """
        # print (" >> Update weights")
        for reg in self.reg.objfcts:
            reg.cell_weights = reg.mapping * self.wr
            reg.model = self.invProb.model

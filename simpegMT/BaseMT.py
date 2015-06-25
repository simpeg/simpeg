from simpegEM.FDEM import BaseFDEMProblem
from SurveyMT import SurveyMT
from DataMT import DataMT
from FieldsMT import FieldsMT
from SimPEG import SolverLU as SimpegSolver, mkvc
import numpy as np

class BaseMTProblem(BaseFDEMProblem):

    def __init__(self, mesh, **kwargs):
        BaseFDEMProblem.__init__(self, mesh, **kwargs)

    # Set the default pairs of the problem
    surveyPair = SurveyMT
    dataPair = DataMT
    fieldsPair = FieldsMT

    # Set the solver
    Solver = SimpegSolver
    solverOpts = {}

    verbose = False
    # Notes:
    # Use the forward and devs from BaseFDEMProblem
    # Might need to add more stuff here.

    def Jvec(self, m, v, f=None):
        """
        Function to calculate the data sensitivities dD/dm times a vector.

            :param numpy.ndarray (nC, 1) - conductive model
            :param numpy.ndarray (nC, 1) - random vector
            :param MTfields object (optional) - MT fields object, if not given it is calculated
            :rtype: MTdata object
            :return: Data sensitivities wrt m
        """

        # Calculate the fields
        if f is None:
           f = self.fields(m)
        # Set current model
        self.curModel = m
        # Initiate the Jv object
        Jv = self.dataPair(self.survey)

        # Loop all the frequenies
        for freq in self.survey.freqs:
            dA_du = self.getA(freq) #
            dA_duI = self.Solver(dA_du, **self.solverOpts)

            for src in self.survey.getSrcByFreq(freq):
                # We need fDeriv_m = df/du*du/dm + df/dm
                # Construct du/dm, it requires a solve
                ftype = self._fieldType + 'Solution'
                u_src = f[src, ftype]
                dA_dm = self.getADeriv_m(freq, u_src, v)
                dRHS_dm = self.getRHSDeriv_m(freq, v)
                if dRHS_dm is None:
                    du_dm = dA_duI * ( - dA_dm )
                else:
                    du_dm = dA_duI * ( - dA_dm + dRHS_dm )
                # Calculate the projection derivatives
                for rx in src.rxList:
                    # Get the stacked derivative
                    # df_duFun = getattr(f, '_fDeriv_u', None)
                    # df_dmFun = getattr(f, '_fDeriv_m', None)
                    # df_dm = df_dmFun(src,v,adjoint=False)
                    # if df_dm is None:
                    #     fDeriv_m = df_duFun(src, du_dm, adjoint=False)
                    # else:
                    #     fDeriv_m = df_duFun(src, du_dm, adjoint=False) + df_dm
                    # Not needed for now. Since PDeriv does this currently.

                    # Get the projection derivative
                    PDeriv = lambda v: rx.projectFieldsDeriv(src, self.mesh, f, v) # wrt u, also have wrt m
                    Jv[src, rx] = PDeriv(du_dm)
        # Return the vectorized sensitivities
        return mkvc(Jv)

    def Jtvec(self, m, v, f=None):
        if f is None:
            f = self.fields(m)

        self.curModel = m

        # Ensure v is a data object.
        if not isinstance(v, self.dataPair):
            v = self.dataPair(self.survey, v)

        Jtv = np.zeros(m.size)

        for freq in self.survey.freqs:
            AT = self.getA(freq).T
            ATinv = self.Solver(AT, **self.solverOpts)

            for src in self.survey.getSrcByFreq(freq):
                ftype = self._fieldType + 'Solution'
                u_src = f[src, ftype]

                for rx in src.rxList:
                    # Get the adjoint projectFieldsDeriv
                    PTv = rx.projectFieldsDeriv(src, self.mesh, f, v[src, rx], adjoint=True) # wrt u, need possibility wrt m
                    # Get the
                    dA_duIT = ATinv * PTv
                    dA_dmT = self.getADeriv_m(freq, u_src, dA_duIT, adjoint=True)
                    dRHS_dmT = self.getRHSDeriv_m(freq, dA_duIT, adjoint=True)
                    # Make du_dmT
                    if dRHS_dmT is None:
                        du_dmT = -dA_dmT
                    else:
                        du_dmT = -dA_dmT + dRHS_dmT
                    # Select the correct component
                    real_or_imag = rx.projComp
                    if real_or_imag == 'real':
                        Jtv +=  du_dmT.real
                    elif real_or_imag == 'imag':
                        Jtv +=  -du_dmT.real
                    else:
                        raise Exception('Must be real or imag')

        return Jtv
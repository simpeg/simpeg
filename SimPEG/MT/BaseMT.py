from SimPEG import SolverLU as SimpegSolver, PropMaps, Utils, mkvc, sp, np
from SimPEG.EM.FDEM.ProblemFDEM import BaseFDEMProblem
from .SurveyMT import Survey, Data
from .FieldsMT import BaseMTFields


class BaseMTProblem(BaseFDEMProblem):
    """
        Base class for all Natural source problems.
    """

    def __init__(self, mesh, **kwargs):
        BaseFDEMProblem.__init__(self, mesh, **kwargs)
        Utils.setKwargs(self, **kwargs)
    # Set the default pairs of the problem
    surveyPair = Survey
    dataPair = Data
    fieldsPair = BaseMTFields

    # Set the solver
    Solver = SimpegSolver
    solverOpts = {}

    verbose = False
    # Notes:
    # Use the forward and devs from BaseFDEMProblem
    # Might need to add more stuff here.

    ## NEED to clean up the Jvec and Jtvec to use Zero and Identities for None components.
    def Jvec(self, m, v, f=None):
        """
        Function to calculate the data sensitivities dD/dm times a vector.

            :param numpy.ndarray m (nC, 1) - conductive model
            :param numpy.ndarray v (nC, 1) - random vector
            :param MTfields object (optional) - MT fields object, if not given it is calculated
            :rtype: MTdata object
            :return: Data sensitivities wrt m
        """

        # Calculate the fields
        if f is None:
           f= self.fields(m)
        # Set current model
        self.curModel = m
        # Initiate the Jv object
        Jv = []

        # Loop all the frequenies
        for freq in self.survey.freqs:
            dA_du = self.getA(freq) #

            dA_duI = self.Solver(dA_du, **self.solverOpts)

            for src in self.survey.getSrcByFreq(freq):
                # We need fDeriv_m = df/du*du/dm + df/dm
                # Construct du/dm, it requires a solve
                # NOTE: need to account for the 2 polarizations in the derivatives.
                f_src = f[src,:]
                # dA_dm and dRHS_dm should be of size nE,2, so that we can multiply by dA_duI. The 2 columns are each of the polarizations.
                dA_dm = self.getADeriv_m(freq, f_src, v) # Size: nE,2 (u_px,u_py) in the columns.
                dRHS_dm = self.getRHSDeriv_m(freq, v) # Size: nE,2 (u_px,u_py) in the columns.
                if dRHS_dm is None:
                    du_dm = dA_duI * ( -dA_dm )
                else:
                    du_dm = dA_duI * ( -dA_dm + dRHS_dm )
                # Calculate the projection derivatives
                for rx in src.rxList:
                    # Get the projection derivative
                    # v should be of size 2*nE (for 2 polarizations)
                    PDeriv_u = lambda t: rx.evalDeriv(src, self.mesh, f, t) # wrt u, we don't have have PDeriv wrt m
                    # Jv[src, rx] = PDeriv_u(mkvc(du_dm))
                    Jv.append(PDeriv_u(mkvc(du_dm)))
            dA_duI.clean()
        # Return the vectorized sensitivities
        return np.hstack(Jv)

    def Jtvec(self, m, v, f=None):
        """
        Function to calculate the transpose of the data sensitivities (dD/dm)^T times a vector.

            :param numpy.ndarray m (nC, 1) - conductive model
            :param numpy.ndarray v (nD, 1) - vector
            :param MTfields object u (optional) - MT fields object, if not given it is calculated
            :rtype: MTdata object
            :return: Data sensitivities wrt m
        """

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
                f_src = f[src, :]

                for rx in src.rxList:
                    # Get the adjoint evalDeriv
                    # PTv needs to be nE,
                    PTv = rx.evalDeriv(src, self.mesh, f, mkvc(v[src, rx],2), adjoint=True) # wrt u, need possibility wrt m
                    # Get the
                    dA_duIT = ATinv * PTv
                    dA_dmT = self.getADeriv_m(freq, f_src, mkvc(dA_duIT), adjoint=True)
                    dRHS_dmT = self.getRHSDeriv_m(freq, mkvc(dA_duIT), adjoint=True)
                    # Make du_dmT
                    if dRHS_dmT is None:
                        du_dmT = -dA_dmT
                    else:
                        du_dmT = -dA_dmT + dRHS_dmT
                    # Select the correct component
                    # du_dmT needs to be of size nC,
                    real_or_imag = rx.projComp
                    if real_or_imag == 'real':
                        Jtv +=  du_dmT.real
                    elif real_or_imag == 'imag':
                        Jtv +=  -du_dmT.real
                    else:
                        raise Exception('Must be real or imag')
            # Clean the factorization, clear memory.
            ATinv.clean()
        return Jtv

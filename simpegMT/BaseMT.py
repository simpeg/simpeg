from simpegEM.FDEM import BaseFDEMProblem
from SurveyMT import SurveyMT
from DataMT import DataMT
from FieldsMT import FieldsMT
from SimPEG import SolverLU as SimpegSolver, PropMaps, Utils, mkvc, sp, np

class BaseMTProblem(BaseFDEMProblem):

    def __init__(self, mesh, **kwargs):
        BaseFDEMProblem.__init__(self, mesh, **kwargs)
        Utils.setKwargs(self, **kwargs)
    # Set the default pairs of the problem
    surveyPair = SurveyMT
    dataPair = DataMT
    fieldsPair = FieldsMT

    # Pickleing support methods
    def __getstate__(self):
        '''
        Method that makes the dictionary of the object pickleble, removes non-pickleble elements of the object.

        Used when doing:
            pickle.dump(pickleFile,object)
        '''
        odict = self.__dict__.copy()
        # Remove fields that are not needed
        del odict['hook']
        del odict['setKwargs']
        # Return the dict
        return odict

    def __setstate__(self,odict):
        '''
        Function that sets a pickle dictionary in to an object.

        Used when doing:
            object = pickle.load(pickleFile)
        '''
        # Update the dict
        self.__dict__.update(odict)
        # Re-hook the methods to the object
        Utils.codeutils.hook(self,Utils.codeutils.hook)
        Utils.codeutils.hook(self,Utils.codeutils.setKwargs)

    # Set the solver
    Solver = SimpegSolver
    solverOpts = {}

    verbose = False
    # Notes:
    # Use the forward and devs from BaseFDEMProblem
    # Might need to add more stuff here.

    def Jvec(self, m, v, u=None):
        """
        Function to calculate the data sensitivities dD/dm times a vector.

            :param numpy.ndarray (nC, 1) - conductive model
            :param numpy.ndarray (nC, 1) - random vector
            :param MTfields object (optional) - MT fields object, if not given it is calculated
            :rtype: MTdata object
            :return: Data sensitivities wrt m
        """

        # Calculate the fields
        if u is None:
           u = self.fields(m)
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
                # NOTE: need to account for the 2 polarizations in the derivatives.
                u_src = u[src,:]
                # dA_dm and dRHS_dm should be of size nE,2, so that we can multiply by dA_duI. The 2 columns are each of the polarizations.
                dA_dm = self.getADeriv_m(freq, u_src, v) # Size: nE,2 (u_px,u_py) in the columns.
                dRHS_dm = self.getRHSDeriv_m(freq, v) # Size: nE,2 (u_px,u_py) in the columns.
                if dRHS_dm is None:
                    du_dm = dA_duI * ( -dA_dm )
                else:
                    du_dm = dA_duI * ( -dA_dm + dRHS_dm )
                # Calculate the projection derivatives
                for rx in src.rxList:
                    # Get the projection derivative
                    # v should be of size 2*nE (for 2 polarizations)
                    PDeriv_u = lambda t: rx.projectFieldsDeriv(src, self.mesh, u, t) # wrt u, we don't have have PDeriv wrt m
                    Jv[src, rx] = PDeriv_u(mkvc(du_dm))
        # Return the vectorized sensitivities
        return mkvc(Jv)

    def Jtvec(self, m, v, u=None):
        if u is None:
            u = self.fields(m)

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
                u_src = u[src, :]

                for rx in src.rxList:
                    # Get the adjoint projectFieldsDeriv
                    # PTv needs to be nE,
                    PTv = rx.projectFieldsDeriv(src, self.mesh, u, mkvc(v[src, rx],2), adjoint=True) # wrt u, need possibility wrt m
                    # Get the
                    dA_duIT = ATinv * PTv
                    dA_dmT = self.getADeriv_m(freq, u_src, mkvc(dA_duIT), adjoint=True)
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

        return Jtv
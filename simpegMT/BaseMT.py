from simpegEM.FDEM import BaseFDEMProblem
from SurveyMT import SurveyMT
from DataMT import DataMT
from FieldsMT import FieldsMT
from SimPEG import SolverLU as SimpegSolver

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
        if f is None:
           f = self.fields(m)

        self.curModel = m

        Jv = self.dataPair(self.survey)

        for freq in self.survey.freqs:
            dA_du = self.getA(freq) #
            dA_duI = self.Solver(dA_du, **self.solverOpts)

            for src in self.survey.getSrcByFreq(freq):
                ftype = self._fieldType + 'Solution'
                u_src = f[src, ftype]
                dA_dm = self.getADeriv_m(freq, u_src, v)
                dRHS_dm = self.getRHSDeriv_m(freq, v)
                if dRHS_dm is None:
                    du_dm = dA_duI * ( - dA_dm )
                else:
                    du_dm = dA_duI * ( - dA_dm + dRHS_dm )
                for rx in src.rxList:
                    # df_duFun = u.deriv_u(rx.fieldsUsed, m)
                    if 'e' in self._fieldType:
                        projField = 'b'
                    elif 'b' in self._fieldType:
                        projField = 'e'
                    df_duFun = getattr(f, '_%sDeriv_u'%projField, None)
                    df_du = df_duFun(src, du_dm, adjoint=False)
                    if df_du is not None:
                        du_dm = df_du

                    df_dmFun = getattr(f, '_%sDeriv_m'%projField, None)
                    df_dm = df_dmFun(src, v, adjoint=False)
                    if df_dm is not None:
                        du_dm += df_dm

                    P = lambda v: rx.projectFieldsDeriv(src, self.mesh, f, v) # wrt u, also have wrt m


                    Jv[src, rx] = P(du_dm)

        return Utils.mkvc(Jv)

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
                    PTv = rx.projectFieldsDeriv(src, self.mesh, f, v[src, rx], adjoint=True) # wrt u, need possibility wrt m

                    df_duTFun = getattr(f, '_%sDeriv_u'%rx.projField, None)
                    df_duT = df_duTFun(src, PTv, adjoint=True)
                    if df_duT is not None:
                        dA_duIT = ATinv * df_duT
                    else:
                        dA_duIT = ATinv * PTv

                    dA_dmT = self.getADeriv_m(freq, u_src, dA_duIT, adjoint=True)

                    dRHS_dmT = self.getRHSDeriv_m(src, dA_duIT, adjoint=True)

                    if dRHS_dmT is None:
                        du_dmT = - dA_dmT
                    else:
                        du_dmT = -dA_dmT + dRHS_dmT

                    df_dmFun = getattr(f, '_%sDeriv_m'%rx.projField, None)
                    dfT_dm = df_dmFun(src, PTv, adjoint=True)
                    if dfT_dm is not None:
                        du_dmT += dfT_dm

                    real_or_imag = rx.projComp
                    if real_or_imag == 'real':
                        Jtv +=   du_dmT.real
                    elif real_or_imag == 'imag':
                        Jtv += - du_dmT.real
                    else:
                        raise Exception('Must be real or imag')

        return Jtv
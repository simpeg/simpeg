import numpy as np

from ..induced_polarization import Simulation2DNodal as BaseSimulation2DNodal
from ..induced_polarization import (
    Simulation2DCellCentered as BaseSimulation2DCellCentered,
)
from .simulation import BaseSIPSimulation


class BaseSIPSimulation2D(BaseSIPSimulation):
    def __init__(self, mesh, **kwargs):
        self.storeJ = True
        super().__init__(mesh, **kwargs)

    def getJ(self, m, f=None):
        """
        Generate Full sensitivity matrix
        """

        if self.verbose:
            print(">> Compute Sensitivity matrix")

        if self._Jmatrix is not None:
            return self._Jmatrix
        else:
            if self._mini_survey is not None:
                survey = self._mini_survey
            else:
                survey = self.survey
            kys = self._quad_points
            weights = self._quad_weights

            if f is None:
                f = self.fields(m)

            Jt = np.zeros(
                (self.actMap.nP, int(self.survey.nD / self.survey.unique_times.size)),
                order="F",
            )
            for iky, ky in enumerate(kys):
                u_ky = f[:, self._solutionType, iky]
                istrt = 0
                for i_src, src in enumerate(survey.source_list):
                    u_src = u_ky[:, i_src]
                    for rx in src.receiver_list:
                        # wrt f, need possibility wrt m

                        if getattr(rx, 'projGLoc', None) is None:
                            if rx.orientation is not None:
                                rx.projGLoc = f._GLoc(rx.projField) + rx.orientation
                            else:
                                rx.projGLoc = f._GLoc(rx.projField)

                        P = rx.getP(self.mesh, rx.projGLoc).toarray()

                        ATinvdf_duT = self.Ainv[iky] * (P.T)

                        dA_dmT = self.getADeriv(ky, u_src, ATinvdf_duT, adjoint=True)
                        Jtv = -weights[iky] * dA_dmT  # RHS=0
                        iend = istrt + rx.nD
                        if rx.nD == 1:
                            Jt[:, istrt] += Jtv
                        else:
                            Jt[:, istrt:iend] += Jtv
                        istrt += rx.nD

            self._Jmatrix = self._mini_survey_data(Jt.T)
            # clean all factorization
            if self.Ainv[0] is not None:
                for i in range(self.nky):
                    self.Ainv[i].clean()
            return self._Jmatrix


class Simulation2DCellCentered(BaseSIPSimulation2D, BaseSimulation2DCellCentered):
    """
    2.5D cell centered Spectral IP problem
    """


class Simulation2DNodal(BaseSIPSimulation2D, BaseSimulation2DNodal):
    """
    2.5D nodal Spectral IP problem
    """


Simulation2DCellCentred = Simulation2DCellCentered

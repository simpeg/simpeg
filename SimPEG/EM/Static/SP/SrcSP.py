from SimPEG.EM.Static.DC import Src
from SimPEG import Props
from SimPEG.Utils import sdiag
import scipy.sparse as sp
import numpy as np

class StreamingCurrents(Src.BaseSrc):

    L = None
    mesh = None
    # "Hydraulic Head (m)"
    # "Streaming current source (A/m^3)"
    # "Streaming current density (A/m^2)"
    modelType = None

    def __init__(self, rxList, **kwargs):
        Src.BaseSrc.__init__(self, rxList, **kwargs)
        if self.L is None:
            raise Exception("SP source requires cross coupling coefficient L")
        if self.mesh is None:
            raise Exception("SP source requires mesh")

    def eval(self, prob):
        """

            Computing source term using:

            - Hydraulic head: h
            - Cross coupling coefficient: L

            .. math::

                -\nabla \cdot \vec{j}^s = \nabla \cdot L \nabla \phi \\

        """
        if prob._formulation == 'HJ':
            if self.modelType == "Head":
                q = -prob.Div*self.MfLiI*prob.Grad*prob.h
            elif self.modelType == "CurrentSource":
                q = prob.q
            elif self.modelType == "CurrentDensity":
                q = -prob.Div*prob.mesh.aveF2CCV.T*np.r_[prob.jsx, prob.jsy, prob.jsz]
            else:
                raise NotImplementedError()
        elif prob._formulation == 'EB':
            raise NotImplementedError()
        return q

    def evalDeriv(self, prob, v=None, adjoint=False):
        if prob._formulation == 'HJ':
            if adjoint:
                if self.modelType == "Head":
                    srcDeriv = - prob.hDeriv.T * prob.Grad.T * self.MfLiI.T * (prob.Div.T * v)
                elif self.modelType == "CurrentSource":
                    srcDeriv = prob.qDeriv.T * v
                elif self.modelType == "CurrentDensity":
                    jsDeriv = sp.vstack((prob.jsxDeriv, prob.jsyDeriv, prob.jszDeriv))
                    srcDeriv = - jsDeriv.T * prob.mesh.aveF2CCV * (prob.Div.T*v)
                else:
                    raise NotImplementedError()
            else:
                if self.modelType == "Head":
                    srcDeriv = -prob.Div*self.MfLiI*prob.Grad*(prob.hDeriv*v)
                elif self.modelType == "CurrentSource":
                    srcDeriv = prob.qDeriv * v
                elif self.modelType == "CurrentDensity":
                    jsDeriv = sp.vstack((prob.jsxDeriv, prob.jsyDeriv, prob.jszDeriv))
                    srcDeriv = -prob.Div*prob.mesh.aveF2CCV.T*(jsDeriv*v)
                else:
                    raise NotImplementedError()
        elif prob._formulation == 'EB':
            raise NotImplementedError()
        return srcDeriv

    @property
    def MfLi(self):
        """
            :code:`MfLi`
        """
        if getattr(self, '_MfLi', None) is None:
            self._MfLi = self.mesh.getFaceInnerProduct(1./self.L)
        return seself.lf._MfLi

    @property
    def MfLiI(self):
        """
            Inverse of :code:`_MfLiI`
        """
        if getattr(self, '_MfLiI', None) is None:
            self._MfLiI = self.mesh.getFaceInnerProduct(1./self.L, invMat=True)
        return self._MfLiI

if __name__ == '__main__':
    from SimPEG import Mesh, np
    mesh = Mesh.TensorMesh([10, 10])
    L = np.ones(mesh.nC)
    src = StreamingCurrents([], L=L, mesh=mesh)
    thing = src.MfLiI
    if thing is not None:
        pass

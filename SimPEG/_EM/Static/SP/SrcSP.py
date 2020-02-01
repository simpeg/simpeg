from SimPEG.EM.Static.DC import Src
from SimPEG import Props
from SimPEG.Utils import sdiag
from SimPEG import Utils
import scipy.sparse as sp
import numpy as np
from SimPEG.EM.Static.DC import Survey


class StreamingCurrents(Src.BaseSrc):

    L = None
    mesh = None
    modelType = None
    indActive = None

    def __init__(self, rxList, **kwargs):
        Src.BaseSrc.__init__(self, rxList, **kwargs)
        if self.modelType == "Head":
            if self.L is None:
                raise Exception(
                    "SP source requires cross coupling coefficient L"
                )
        elif self.modelType == "CurrentDensity":
            if self.indActive is None:
                self.indActive = np.ones(self.mesh.nC, dtype=bool)
            # This is for setting a Neuman condition on the topographic faces

            self.Grad = -sp.vstack(
                (
                    self.Pafx * self.mesh.faceDivx.T * self.V * self.Pac,
                    self.Pafy * self.mesh.faceDivy.T * self.V * self.Pac,
                    self.Pafz * self.mesh.faceDivz.T * self.V * self.Pac,
                )
            )

        if self.mesh is None:
            raise Exception("SP source requires mesh")

    def getq_from_j(self, j):
        q = self.Grad.T*self.mesh.aveCCV2F*j
        return q

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
                q = prob.Grad.T*self.MfLiI*prob.Grad*prob.h
            elif self.modelType == "CurrentSource":
                q = self.V * prob.q
            elif self.modelType == "CurrentDensity":
                q = self.Grad.T*self.mesh.aveCCV2F*np.r_[
                    prob.jsx, prob.jsy, prob.jsz
                ]
            else:
                raise NotImplementedError()
        elif prob._formulation == 'EB':
            raise NotImplementedError()
        return q

    def evalDeriv(self, prob, v=None, adjoint=False):
        if prob._formulation == 'HJ':
            if adjoint:
                if self.modelType == "Head":
                    srcDeriv = prob.hDeriv.T * prob.Grad.T * self.MfLiI.T * (
                        prob.Grad * v
                    )
                elif self.modelType == "CurrentSource":
                    srcDeriv = prob.qDeriv.T * (self.V * v)
                elif self.modelType == "CurrentDensity":
                    jsDeriv = sp.vstack(
                        (prob.jsxDeriv, prob.jsyDeriv, prob.jszDeriv)
                    )
                    srcDeriv = jsDeriv.T * self.mesh.aveCCV2F.T * (self.Grad*v)
                else:
                    raise NotImplementedError()
            else:
                if self.modelType == "Head":
                    srcDeriv = prob.Grad.T*self.MfLiI*prob.Grad*(prob.hDeriv*v)
                elif self.modelType == "CurrentSource":
                    srcDeriv = self.V * (prob.qDeriv * v)
                elif self.modelType == "CurrentDensity":
                    jsDeriv = sp.vstack(
                        (prob.jsxDeriv, prob.jsyDeriv, prob.jszDeriv)
                    )
                    srcDeriv = self.Grad.T * self.mesh.aveCCV2F*(jsDeriv*v)
                else:
                    raise NotImplementedError()
        elif prob._formulation == 'EB':
            raise NotImplementedError()
        return srcDeriv
    @property
    def V(self):
        """
            :code:`V`
        """
        if getattr(self, '_V', None) is None:
            self._V = Utils.sdiag(self.mesh.vol)
        return self._V

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

    @property
    def Pac(self):
        """
        diagonal matrix that nulls out inactive cells

        :rtype: scipy.sparse.csr_matrix
        :return: active cell diagonal matrix
        """
        if getattr(self, '_Pac', None) is None:
            if self.indActive is None:
                self._Pac = Utils.speye(self.mesh.nC)
            else:
                e = np.zeros(self.mesh.nC)
                e[self.indActive] = 1.
                # self._Pac = Utils.speye(self.mesh.nC)[:, self.indActive]
                self._Pac = Utils.sdiag(e)
        return self._Pac

    @property
    def Pafx(self):
        """
        diagonal matrix that nulls out inactive x-faces
        to full modelling space (ie. nFx x nindActive_Fx )

        :rtype: scipy.sparse.csr_matrix
        :return: active face-x diagonal matrix
        """
        if getattr(self, '_Pafx', None) is None:
            if self.indActive is None:
                self._Pafx = Utils.speye(self.mesh.nFx)
            else:
                indActive_Fx = self.mesh.aveFx2CC.T * self.indActive >= 1
                e = np.zeros(self.mesh.nFx)
                e[indActive_Fx] = 1.
                self._Pafx = Utils.sdiag(e)
        return self._Pafx

    @property
    def Pafy(self):
        """
        diagonal matrix that nulls out inactive y-faces
        to full modelling space (ie. nFy x nindActive_Fy )

        :rtype: scipy.sparse.csr_matrix
        :return: active face-y diagonal matrix
        """
        if getattr(self, '_Pafy', None) is None:
            if self.indActive is None:
                self._Pafy = Utils.speye(self.mesh.nFy)
            else:
                indActive_Fy = (self.mesh.aveFy2CC.T * self.indActive) >= 1
                e = np.zeros(self.mesh.nFy)
                e[indActive_Fy] = 1.
                self._Pafy = Utils.sdiag(e)
        return self._Pafy

    @property
    def Pafz(self):
        """
        diagonal matrix that nulls out inactive z-faces
        to full modelling space (ie. nFz x nindActive_Fz )

        :rtype: scipy.sparse.csr_matrix
        :return: active face-z diagonal matrix
        """
        if getattr(self, '_Pafz', None) is None:
            if self.indActive is None:
                self._Pafz = Utils.speye(self.mesh.nFz)
            else:
                indActive_Fz = (self.mesh.aveFz2CC.T * self.indActive) >= 1
                e = np.zeros(self.mesh.nFz)
                e[indActive_Fz] = 1.
                self._Pafz = Utils.sdiag(e)
        return self._Pafz


if __name__ == '__main__':
    from SimPEG import Mesh, np
    mesh = Mesh.TensorMesh([10, 10])
    L = np.ones(mesh.nC)
    src = StreamingCurrents([], L=L, mesh=mesh)
    thing = src.MfLiI
    if thing is not None:
        pass

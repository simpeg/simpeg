from __future__ import print_function
from . import Utils
from . import Maps
from . import Mesh
import numpy as np
import scipy.sparse as sp


class RegularizationMesh(object):
    """
    **Regularization Mesh**

    This contains the operators used in the regularization. Note that these
    are not necessarily true differential operators, but are constructed from
    a SimPEG Mesh.

    :param BaseMesh mesh: problem mesh
    :param numpy.array indActive: bool array, size nC, that is True where we have active cells. Used to reduce the operators so we regularize only on active cells
    """

    def __init__(self, mesh, indActive=None):
        self.mesh = mesh
        assert indActive is None or indActive.dtype == 'bool', 'indActive needs to be None or a bool'
        self.indActive = indActive

    @property
    def vol(self):
        """
        reduced volume vector
        :rtype: numpy.array
        :return: reduced cell volume
        """
        if getattr(self, '_vol', None) is None:
            self._vol = self._Pac.T * self.mesh.vol
        return self._vol

    @property
    def nC(self):
        """
        reduced number of cells
        :rtype: int
        :return: number of cells being regularized
        """
        if getattr(self, '_nC', None) is None:
            if self.indActive is None:
                self._nC = self.mesh.nC
            else:
                self._nC = sum(self.indActive)
        return self._nC

    @property
    def dim(self):
        """
        dimension of regularization mesh (1D, 2D, 3D)
        :rtype: int
        :return: dimension
        """
        if getattr(self, '_dim', None) is None:
            self._dim = self.mesh.dim
        return self._dim


    @property
    def _Pac(self):
        """
        projection matrix that takes from the reduced space of active cells to full modelling space (ie. nC x nindActive)
        :rtype: scipy.sparse.csr_matrix
        :return: active cell projection matrix
        """
        if getattr(self, '__Pac', None) is None:
            if self.indActive is None:
                self.__Pac = Utils.speye(self.mesh.nC)
            else:
                self.__Pac = Utils.speye(self.mesh.nC)[:,self.indActive]
        return self.__Pac

    @property
    def _Pafx(self):
        """
        projection matrix that takes from the reduced space of active x-faces to full modelling space (ie. nFx x nindActive_Fx )
        :rtype: scipy.sparse.csr_matrix
        :return: active face-x projection matrix
        """
        if getattr(self, '__Pafx', None) is None:
            if self.indActive is None:
                self.__Pafx = Utils.speye(self.mesh.nFx)
            else:
                indActive_Fx = (self.mesh.aveFx2CC.T * self.indActive) == 1
                self.__Pafx = Utils.speye(self.mesh.nFx)[:,indActive_Fx]
        return self.__Pafx

    @property
    def _Pafy(self):
        """
        projection matrix that takes from the reduced space of active y-faces to full modelling space (ie. nFy x nindActive_Fy )
        :rtype: scipy.sparse.csr_matrix
        :return: active face-y projection matrix
        """
        if getattr(self, '__Pafy', None) is None:
            if self.indActive is None:
                self.__Pafy = Utils.speye(self.mesh.nFy)
            else:
                indActive_Fy = (self.mesh.aveFy2CC.T * self.indActive) == 1
                self.__Pafy = Utils.speye(self.mesh.nFy)[:,indActive_Fy]
        return self.__Pafy

    @property
    def _Pafz(self):
        """
        projection matrix that takes from the reduced space of active z-faces to full modelling space (ie. nFz x nindActive_Fz )
        :rtype: scipy.sparse.csr_matrix
        :return: active face-z projection matrix
        """
        if getattr(self, '__Pafz', None) is None:
            if self.indActive is None:
                self.__Pafz = Utils.speye(self.mesh.nFz)
            else:
                indActive_Fz = (self.mesh.aveFz2CC.T * self.indActive) == 1
                self.__Pafz = Utils.speye(self.mesh.nFz)[:,indActive_Fz]
        return self.__Pafz

    @property
    def aveFx2CC(self):
        """
        averaging from active cell centers to active x-faces
        :rtype: scipy.sparse.csr_matrix
        :return: averaging from active cell centers to active x-faces
        """
        if getattr(self, '_aveFx2CC', None) is None:
            self._aveFx2CC =  self._Pac.T * self.mesh.aveFx2CC * self._Pafx
        return self._aveFx2CC

    @property
    def aveCC2Fx(self):
        """
        averaging from active x-faces to active cell centers
        :rtype: scipy.sparse.csr_matrix
        :return: averaging matrix from active x-faces to active cell centers
        """
        if getattr(self, '_aveCC2Fx', None) is None:
            self._aveCC2Fx =  Utils.sdiag(1./(self.aveFx2CC.T).sum(1)) * self.aveFx2CC.T
        return self._aveCC2Fx

    @property
    def aveFy2CC(self):
        """
        averaging from active cell centers to active y-faces
        :rtype: scipy.sparse.csr_matrix
        :return: averaging from active cell centers to active y-faces
        """
        if getattr(self, '_aveFy2CC', None) is None:
            self._aveFy2CC = self._Pac.T * self.mesh.aveFy2CC * self._Pafy
        return self._aveFy2CC

    @property
    def aveCC2Fy(self):
        """
        averaging from active y-faces to active cell centers
        :rtype: scipy.sparse.csr_matrix
        :return: averaging matrix from active y-faces to active cell centers
        """
        if getattr(self, '_aveCC2Fy', None) is None:
            self._aveCC2Fy =  Utils.sdiag(1./(self.aveFy2CC.T).sum(1)) * self.aveFy2CC.T
        return self._aveCC2Fy

    @property
    def aveFz2CC(self):
        """
        averaging from active cell centers to active z-faces
        :rtype: scipy.sparse.csr_matrix
        :return: averaging from active cell centers to active z-faces
        """
        if getattr(self, '_aveFz2CC', None) is None:
            self._aveFz2CC = self._Pac.T * self.mesh.aveFz2CC * self._Pafz
        return self._aveFz2CC

    @property
    def aveCC2Fz(self):
        """
        averaging from active z-faces to active cell centers
        :rtype: scipy.sparse.csr_matrix
        :return: averaging matrix from active z-faces to active cell centers
        """
        if getattr(self, '_aveCC2Fz', None) is None:
            self._aveCC2Fz =  Utils.sdiag(1./(self.aveFz2CC.T).sum(1)) * self.aveFz2CC.T
        return self._aveCC2Fz

    @property
    def cellDiffx(self):
        """
        cell centered difference in the x-direction
        :rtype: scipy.sparse.csr_matrix
        :return: differencing matrix for active cells in the x-direction
        """
        if getattr(self, '_cellDiffx', None) is None:
            self._cellDiffx = self._Pafx.T * self.mesh.cellGradx * self._Pac
        return self._cellDiffx

    @property
    def cellDiffy(self):
        """
        cell centered difference in the y-direction
        :rtype: scipy.sparse.csr_matrix
        :return: differencing matrix for active cells in the y-direction
        """
        if getattr(self, '_cellDiffy', None) is None:
            self._cellDiffy = self._Pafy.T * self.mesh.cellGrady * self._Pac
        return self._cellDiffy

    @property
    def cellDiffz(self):
        """
        cell centered difference in the z-direction
        :rtype: scipy.sparse.csr_matrix
        :return: differencing matrix for active cells in the z-direction
        """
        if getattr(self, '_cellDiffz', None) is None:
            self._cellDiffz = self._Pafz.T * self.mesh.cellGradz * self._Pac
        return self._cellDiffz

    @property
    def faceDiffx(self):
        """
        x-face differences
        :rtype: scipy.sparse.csr_matrix
        :return: differencing matrix for active faces in the x-direction
        """
        if getattr(self, '_faceDiffx', None) is None:
            self._faceDiffx = self._Pac.T * self.mesh.faceDivx * self._Pafx
        return self._faceDiffx

    @property
    def faceDiffy(self):
        """
        y-face differences
        :rtype: scipy.sparse.csr_matrix
        :return: differencing matrix for active faces in the y-direction
        """
        if getattr(self, '_faceDiffy', None) is None:
            self._faceDiffy = self._Pac.T * self.mesh.faceDivy * self._Pafy
        return self._faceDiffy

    @property
    def faceDiffz(self):
        """
        z-face differences
        :rtype: scipy.sparse.csr_matrix
        :return: differencing matrix for active faces in the z-direction
        """
        if getattr(self, '_faceDiffz', None) is None:
            self._faceDiffz = self._Pac.T * self.mesh.faceDivz * self._Pafz
        return self._faceDiffz

    @property
    def cellDiffxStencil(self):
        """
        cell centered difference stencil (no cell lengths include) in the x-direction
        :rtype: scipy.sparse.csr_matrix
        :return: differencing matrix for active cells in the x-direction
        """
        if getattr(self, '_cellDiffxStencil', None) is None:

            self._cellDiffxStencil = self._Pafx.T * self.mesh._cellGradxStencil() * self._Pac
        return self._cellDiffxStencil

    @property
    def cellDiffyStencil(self):
        """
        cell centered difference stencil (no cell lengths include) in the y-direction
        :rtype: scipy.sparse.csr_matrix
        :return: differencing matrix for active cells in the y-direction
        """
        if self.dim < 2: return None
        if getattr(self, '_cellDiffyStencil', None) is None:

            self._cellDiffyStencil = self._Pafy.T * self.mesh._cellGradyStencil() * self._Pac
        return self._cellDiffyStencil

    @property
    def cellDiffzStencil(self):
        """
        cell centered difference stencil (no cell lengths include) in the y-direction
        :rtype: scipy.sparse.csr_matrix
        :return: differencing matrix for active cells in the y-direction
        """
        if self.dim < 3: return None
        if getattr(self, '_cellDiffzStencil', None) is None:

            self._cellDiffzStencil = self._Pafz.T * self.mesh._cellGradzStencil() * self._Pac
        return self._cellDiffzStencil


class BaseRegularization(object):
    """
    **Base Regularization Class**

    This is used to regularize the model space::

        reg = Regularization(mesh)

    """

    counter = None

    mapPair = Maps.IdentityMap    #: A SimPEG.Map Class

    mapping = None    #: A SimPEG.Map instance.
    mesh    = None    #: A SimPEG.Mesh instance.
    mref    = None    #: Reference model.

    def __init__(self, mesh, mapping=None, indActive=None, **kwargs):
        Utils.setKwargs(self, **kwargs)
        assert isinstance(mesh, Mesh.BaseMesh), "mesh must be a SimPEG.Mesh object."
        if indActive is not None and indActive.dtype != 'bool':
            tmp = indActive
            indActive = np.zeros(mesh.nC, dtype=bool)
            indActive[tmp] = True
        if indActive is not None and mapping is None:
            mapping = Maps.IdentityMap(nP=indActive.nonzero()[0].size)

        self.regmesh = RegularizationMesh(mesh,indActive)
        self.mapping = mapping or self.mapPair(mesh)
        self.mapping._assertMatchesPair(self.mapPair)
        self.indActive = indActive

    @property
    def parent(self):
        """This is the parent of the regularization."""
        return getattr(self,'_parent',None)
    @parent.setter
    def parent(self, p):
        if getattr(self,'_parent',None) is not None:
            print('Regularization has switched to a new parent!')
        self._parent = p

    @property
    def inv(self): return self.parent.inv
    @property
    def invProb(self): return self.parent
    @property
    def reg(self): return self
    @property
    def opt(self): return self.parent.opt
    @property
    def prob(self): return self.parent.prob
    @property
    def survey(self): return self.parent.survey


    @property
    def W(self):
        """Full regularization weighting matrix W."""
        return sp.identity(self.regmesh.nC)

    @Utils.timeIt
    def eval(self, m):
        r = self.W * ( self.mapping * (m - self.mref) )
        return 0.5*r.dot(r)

    @Utils.timeIt
    def evalDeriv(self, m):
        """

        The regularization is:

        .. math::

            R(m) = \\frac{1}{2}\mathbf{(m-m_\\text{ref})^\\top W^\\top W(m-m_\\text{ref})}

        So the derivative is straight forward:

        .. math::

            R(m) = \mathbf{W^\\top W (m-m_\\text{ref})}

        """
        mD = self.mapping.deriv(m - self.mref)
        r = self.W * ( self.mapping * (m - self.mref) )
        return  mD.T * ( self.W.T * r )

    @Utils.timeIt
    def eval2Deriv(self, m, v=None):
        """
        Second derivative

        :param numpy.array m: geophysical model
        :param numpy.array v: vector to multiply
        :rtype: scipy.sparse.csr_matrix
        :return: WtW, or if v is supplied WtW*v (numpy.ndarray)

        The regularization is:

        .. math::

            R(m) = \\frac{1}{2}\mathbf{(m-m_\\text{ref})^\\top W^\\top W(m-m_\\text{ref})}

        So the second derivative is straight forward:

        .. math::

            R(m) = \mathbf{W^\\top W}

        """
        mD = self.mapping.deriv(m - self.mref)
        if v is None:
            return mD.T * self.W.T * self.W * mD

        return mD.T * ( self.W.T * ( self.W * ( mD * v) ) )

class Simple(BaseRegularization):
    """
    Simple regularization that does not include length scales in the derivatives.
    """

    mrefInSmooth = False  #: include mref in the smoothness?
    alpha_s      = Utils.dependentProperty('_alpha_s', 1.0, ['_W', '_Wsmall'], "Smallness weight")
    alpha_x      = Utils.dependentProperty('_alpha_x', 1.0, ['_W', '_Wx'],     "Weight for the first derivative in the x direction")
    alpha_y      = Utils.dependentProperty('_alpha_y', 1.0, ['_W', '_Wy'],     "Weight for the first derivative in the y direction")
    alpha_z      = Utils.dependentProperty('_alpha_z', 1.0, ['_W', '_Wz'],     "Weight for the first derivative in the z direction")
    cell_weights = 1.

    def __init__(self, mesh, mapping=None, indActive=None, **kwargs):
        BaseRegularization.__init__(self, mesh, mapping=mapping, indActive=indActive, **kwargs)

        if isinstance(self.cell_weights,float):
            self.cell_weights = np.ones(self.regmesh.nC) * self.cell_weights

    @property
    def Wsmall(self):
        """Regularization matrix Wsmall"""
        if getattr(self,'_Wsmall', None) is None:
            self._Wsmall = Utils.sdiag((self.alpha_s*self.cell_weights)**0.5)
        return self._Wsmall

    @property
    def Wx(self):
        """Regularization matrix Wx"""
        if getattr(self, '_Wx', None) is None:
            self._Wx = Utils.sdiag((self.alpha_x * (self.regmesh.aveCC2Fx*self.cell_weights))**0.5)*self.regmesh.cellDiffxStencil
        return self._Wx

    @property
    def Wy(self):
        """Regularization matrix Wy"""
        if getattr(self, '_Wy', None) is None:
            self._Wy = Utils.sdiag((self.alpha_y * (self.regmesh.aveCC2Fy*self.cell_weights))**0.5)*self.regmesh.cellDiffyStencil
        return self._Wy

    @property
    def Wz(self):
        """Regularization matrix Wz"""
        if getattr(self, '_Wz', None) is None:
            self._Wz = Utils.sdiag((self.alpha_z * (self.regmesh.aveCC2Fz*self.cell_weights))**0.5)*self.regmesh.cellDiffzStencil
        return self._Wz

#    @property
#    def Wsmooth(self):
#        """Full smoothness regularization matrix W"""
#        print('wtf why are we using Wsmooth')
#        raise NotImplementedError
#        if getattr(self, '_Wsmooth', None) is None:
#            wlist = (self.Wx,)
#            if self.regmesh.dim > 1:
#                wlist += (self.Wy,)
#            if self.regmesh.dim > 2:
#                wlist += (self.Wz,)
#            self._Wsmooth = sp.vstack(wlist)
#        return self._Wsmooth
#
#    @property
#    def W(self):
#        """Full regularization matrix W"""
#        print('wtf why are we using W')
#        if getattr(self, '_W', None) is None:
#            wlist = (self.Wsmall, self.Wx)
#            if self.regmesh.dim > 1:
#                wlist += (self.Wy,)
#            if self.regmesh.dim > 2:
#                wlist += (self.Wz,)
#            self._W = sp.vstack(wlist)
#        return self._W


    @Utils.timeIt
    def _evalSmall(self, m):
        r = self.Wsmall * ( self.mapping * (m - self.mref) )
        return 0.5 * r.dot(r)

    @Utils.timeIt
    def _evalSmallDeriv(self, m):
        r = self.Wsmall * ( self.mapping * (m - self.mref) )
        return r.T * ( self.Wsmall * self.mapping.deriv(m - self.mref) )

    @Utils.timeIt
    def _evalSmall2Deriv(self, m, v = None):
        rDeriv = self.Wsmall * ( self.mapping.deriv(m - self.mref) )
        if v is not None:
            return rDeriv.T * (rDeriv * v)
        return rDeriv.T * rDeriv

    @Utils.timeIt
    def _evalSmoothx(self, m):
        if self.mrefInSmooth == True:
            r = self.Wx * ( self.mapping * (m - self.mref) )
        elif self.mrefInSmooth == False:
            r = self.Wx * ( self.mapping * (m) )
        return 0.5 * r.dot(r)

    @Utils.timeIt
    def _evalSmoothy(self, m):
        if self.mrefInSmooth == True:
            r = self.Wy * ( self.mapping * (m - self.mref) )
        elif self.mrefInSmooth == False:
            r = self.Wy * ( self.mapping * (m) )
        return 0.5 * r.dot(r)

    @Utils.timeIt
    def _evalSmoothz(self, m):
        if self.mrefInSmooth == True:
            r = self.Wz * ( self.mapping * (m - self.mref) )
        elif self.mrefInSmooth == False:
            r = self.Wz * ( self.mapping * (m) )
        return 0.5 * r.dot(r)

    @Utils.timeIt
    def _evalSmooth(self, m):
        phiSmooth = self._evalSmoothx(m)
        if self.regmesh.dim > 1:
            phiSmooth += self._evalSmoothy(m)
        if self.regmesh.dim > 2:
            phiSmooth += self._evalSmoothz(m)
        return phiSmooth

    @Utils.timeIt
    def _evalSmoothxDeriv(self, m):
        if self.mrefInSmooth == True:
            r = self.Wx * ( self.mapping * ( m - self.mref ) )
            return r.T * ( self.Wx * self.mapping.deriv(m - self.mref) )
        elif self.mrefInSmooth == False:
            r = self.Wx * ( self.mapping * m )
            return r.T * ( self.Wx * self.mapping.deriv(m) )

    @Utils.timeIt
    def _evalSmoothx2Deriv(self, m, v=None):
        if self.mrefInSmooth == True:
            rDeriv = self.Wx * ( self.mapping.deriv( m - self.mref ) )
        elif self.mrefInSmooth == False:
            rDeriv = self.Wx * ( self.mapping.deriv(m) )

        if v is not None:
            return rDeriv.T * ( rDeriv * v )
        return rDeriv.T * rDeriv

    @Utils.timeIt
    def _evalSmoothyDeriv(self, m):
        if self.mrefInSmooth == True:
            r = self.Wy * ( self.mapping * ( m - self.mref ) )
            return r.T * ( self.Wy * self.mapping.deriv(m - self.mref) )
        elif self.mrefInSmooth == False:
            r = self.Wy * ( self.mapping * m )
            return r.T * ( self.Wy * self.mapping.deriv(m) )

    @Utils.timeIt
    def _evalSmoothy2Deriv(self, m, v=None):
        if self.mrefInSmooth == True:
            rDeriv = self.Wy * ( self.mapping.deriv( m - self.mref ) )
        elif self.mrefInSmooth == False:
            rDeriv = self.Wy * ( self.mapping.deriv(m) )

        if v is not None:
            return rDeriv.T * ( rDeriv * v )
        return rDeriv.T * rDeriv

    @Utils.timeIt
    def _evalSmoothzDeriv(self, m):
        if self.mrefInSmooth == True:
            r = self.Wz * ( self.mapping * ( m - self.mref ) )
            return r.T * ( self.Wz * self.mapping.deriv(m - self.mref) )
        elif self.mrefInSmooth == False:
            r = self.Wz * ( self.mapping * m )
            return r.T * ( self.Wz * self.mapping.deriv(m) )

    @Utils.timeIt
    def _evalSmoothz2Deriv(self, m, v=None):
        if self.mrefInSmooth == True:
            rDeriv = self.Wz * ( self.mapping.deriv( m - self.mref ) )
        elif self.mrefInSmooth == False:
            rDeriv = self.Wz * ( self.mapping.deriv(m) )

        if v is not None:
            return rDeriv.T * ( rDeriv * v )
        return rDeriv.T * rDeriv

    @Utils.timeIt
    def _evalSmoothDeriv(self, m):
        deriv = self._evalSmoothxDeriv(m)
        if self.regmesh.dim > 1:
            deriv += self._evalSmoothyDeriv(m)
        if self.regmesh.dim > 2:
            deriv += self._evalSmoothzDeriv(m)
        return deriv

    @Utils.timeIt
    def _evalSmooth2Deriv(self, m, v=None):
        deriv = self._evalSmoothx2Deriv(m, v)
        if self.regmesh.dim > 1:
            deriv += self._evalSmoothy2Deriv(m, v)
        if self.regmesh.dim > 2:
            deriv += self._evalSmoothz2Deriv(m, v)
        return deriv


    @Utils.timeIt
    def eval(self, m):
        return self._evalSmall(m) + self._evalSmooth(m)

    @Utils.timeIt
    def evalDeriv(self, m):
        """
        The regularization is:

        .. math::

            R(m) = \\frac{1}{2}\mathbf{(m-m_\\text{ref})^\\top W^\\top W(m-m_\\text{ref})}

        So the derivative is straight forward:

        .. math::

            R(m) = \mathbf{W^\\top W (m-m_\\text{ref})}

        """
        return self._evalSmallDeriv(m) + self._evalSmoothDeriv(m)

    @Utils.timeIt
    def eval2Deriv(self, m, v=None):
        return self._evalSmall2Deriv(m, v) + self._evalSmooth2Deriv(m, v)



class Tikhonov(Simple):
    """
    L2 Tikhonov regularization with both smallness and smoothness (first order
    derivative) contributions.

    .. math::
        \phi_m(\mathbf{m}) = \\alpha_s \| W_s (\mathbf{m} - \mathbf{m_{ref}} ) \|^2
        + \\alpha_x \| W_x \\frac{\partial}{\partial x} (\mathbf{m} - \mathbf{m_{ref}} ) \|^2
        + \\alpha_y \| W_y \\frac{\partial}{\partial y} (\mathbf{m} - \mathbf{m_{ref}} ) \|^2
        + \\alpha_z \| W_z \\frac{\partial}{\partial z} (\mathbf{m} - \mathbf{m_{ref}} ) \|^2

    Note if the key word argument `mrefInSmooth` is False, then mref is not
    included in the smoothness contribution.

    :param BaseMesh mesh: SimPEG mesh
    :param IdentityMap mapping: regularization mapping, takes the model from model space to the thing you want to regularize
    :param numpy.ndarray indActive: active cell indices for reducing the size of differential operators in the definition of a regularization mesh
    :param bool mrefInSmooth: (default = False) put mref in the smoothness component?
    :param float alpha_s: (default 1e-6) smallness weight
    :param float alpha_x: (default 1) smoothness weight for first derivative in the x-direction
    :param float alpha_y: (default 1) smoothness weight for first derivative in the y-direction
    :param float alpha_z: (default 1) smoothness weight for first derivative in the z-direction
    :param float alpha_xx: (default 1) smoothness weight for second derivative in the x-direction
    :param float alpha_yy: (default 1) smoothness weight for second derivative in the y-direction
    :param float alpha_zz: (default 1) smoothness weight for second derivative in the z-direction
    """
    mrefInSmooth = False  # put mref in the smoothness contribution
    alpha_s      = Utils.dependentProperty('_alpha_s', 1e-6, ['_W', '_Wsmall'], "Smallness weight")
    alpha_x      = Utils.dependentProperty('_alpha_x',  1.0, ['_W', '_Wx'],     "Weight for the first derivative in the x direction")
    alpha_y      = Utils.dependentProperty('_alpha_y',  1.0, ['_W', '_Wy'],     "Weight for the first derivative in the y direction")
    alpha_z      = Utils.dependentProperty('_alpha_z',  1.0, ['_W', '_Wz'],     "Weight for the first derivative in the z direction")
    alpha_xx     = Utils.dependentProperty('_alpha_xx', 0.0, ['_W', '_Wxx'],    "Weight for the second derivative in the x direction")
    alpha_yy     = Utils.dependentProperty('_alpha_yy', 0.0, ['_W', '_Wyy'],    "Weight for the second derivative in the y direction")
    alpha_zz     = Utils.dependentProperty('_alpha_zz', 0.0, ['_W', '_Wzz'],    "Weight for the second derivative in the z direction")

    def __init__(self, mesh, mapping=None, indActive=None, **kwargs):
        BaseRegularization.__init__(self, mesh, mapping=mapping, indActive=indActive, **kwargs)

    @property
    def Wsmall(self):
        """Regularization matrix Wsmall"""
        if getattr(self,'_Wsmall', None) is None:
            self._Wsmall = Utils.sdiag((self.regmesh.vol*self.alpha_s)**0.5)
        return self._Wsmall

    @property
    def Wx(self):
        """Regularization matrix Wx"""
        if getattr(self, '_Wx', None) is None:
            Ave_x_vol = self.regmesh.aveCC2Fx * self.regmesh.vol
            self._Wx = Utils.sdiag((Ave_x_vol*self.alpha_x)**0.5)*self.regmesh.cellDiffx
        return self._Wx

    @property
    def Wy(self):
        """Regularization matrix Wy"""
        if getattr(self, '_Wy', None) is None:
            Ave_y_vol = self.regmesh.aveCC2Fy * self.regmesh.vol
            self._Wy = Utils.sdiag((Ave_y_vol*self.alpha_y)**0.5)*self.regmesh.cellDiffy
        return self._Wy

    @property
    def Wz(self):
        """Regularization matrix Wz"""
        if getattr(self, '_Wz', None) is None:
            Ave_z_vol = self.regmesh.aveCC2Fz * self.regmesh.vol
            self._Wz = Utils.sdiag((Ave_z_vol*self.alpha_z)**0.5)*self.regmesh.cellDiffz
        return self._Wz

    @property
    def Wxx(self):
        """Regularization matrix Wxx"""
        if getattr(self, '_Wxx', None) is None:
            self._Wxx = Utils.sdiag((self.regmesh.vol*self.alpha_xx)**0.5)*self.regmesh.faceDiffx*self.regmesh.cellDiffx
        return self._Wxx

    @property
    def Wyy(self):
        """Regularization matrix Wyy"""
        if getattr(self, '_Wyy', None) is None:
            self._Wyy = Utils.sdiag((self.regmesh.vol*self.alpha_yy)**0.5)*self.regmesh.faceDiffy*self.regmesh.cellDiffy
        return self._Wyy

    @property
    def Wzz(self):
        """Regularization matrix Wzz"""
        if getattr(self, '_Wzz', None) is None:
            self._Wzz = Utils.sdiag((self.regmesh.vol*self.alpha_zz)**0.5)*self.regmesh.faceDiffz*self.regmesh.cellDiffz
        return self._Wzz


    @property
    def Wsmooth2(self):
        """Full smoothness regularization matrix W"""
        if getattr(self, '_Wsmooth', None) is None:
            wlist = (self.Wxx)
            if self.regmesh.dim > 1:
                wlist += (self.Wyy)
            if self.regmesh.dim > 2:
                wlist += (self.Wzz)
            self._Wsmooth = sp.vstack(wlist)
        return self._Wsmooth

    @Utils.timeIt
    def _evalSmoothxx(self, m):
        if self.mrefInSmooth == True:
            r = self.Wxx * ( self.mapping * (m - self.mref) )
        elif self.mrefInSmooth == False:
            r = self.Wxx * ( self.mapping * (m) )
        return 0.5 * r.dot(r)

    @Utils.timeIt
    def _evalSmoothyy(self, m):
        if self.mrefInSmooth == True:
            r = self.Wyy * ( self.mapping * (m - self.mref) )
        elif self.mrefInSmooth == False:
            r = self.Wyy * ( self.mapping * (m) )
        return 0.5 * r.dot(r)

    @Utils.timeIt
    def _evalSmoothzz(self, m):
        if self.mrefInSmooth == True:
            r = self.Wzz * ( self.mapping * (m - self.mref) )
        elif self.mrefInSmooth == False:
            r = self.Wzz * ( self.mapping * (m) )
        return 0.5 * r.dot(r)

    @Utils.timeIt
    def _evalSmooth2(self, m):
        phiSmooth2 = self._evalSmoothxx(m)
        if self.regmesh.dim > 1:
            phiSmooth2 += self._evalSmoothyy(m)
        if self.regmesh.dim > 2:
            phiSmooth2 += self._evalSmoothzz(m)
        return phiSmooth2

    @Utils.timeIt
    def _evalSmoothxxDeriv(self, m):
        if self.mrefInSmooth == True:
            r = self.Wxx * ( self.mapping * ( m - self.mref ) )
            return r.T * ( self.Wxx * self.mapping.deriv(m - self.mref) )
        elif self.mrefInSmooth == False:
            r = self.Wxx * ( self.mapping * m )
            return r.T * ( self.Wxx * self.mapping.deriv(m) )

    @Utils.timeIt
    def _evalSmoothyyDeriv(self, m):
        if self.mrefInSmooth == True:
            r = self.Wyy * ( self.mapping * ( m - self.mref ) )
            return r.T * ( self.Wyy * self.mapping.deriv(m - self.mref) )
        elif self.mrefInSmooth == False:
            r = self.Wyy * ( self.mapping * m )
            return r.T * ( self.Wyy * self.mapping.deriv(m) )

    @Utils.timeIt
    def _evalSmoothzzDeriv(self, m):
        if self.mrefInSmooth == True:
            r = self.Wzz * ( self.mapping * ( m - self.mref ) )
            return r.T * ( self.Wzz * self.mapping.deriv(m - self.mref) )
        elif self.mrefInSmooth == False:
            r = self.Wzz * ( self.mapping * m )
            return r.T * ( self.Wzz * self.mapping.deriv(m) )

    @Utils.timeIt
    def _evalSmoothxx2Deriv(self, m, v=None):
        if self.mrefInSmooth == True:
            rDeriv = self.Wxx * ( self.mapping.deriv( m - self.mref ) )
        elif self.mrefInSmooth == False:
            rDeriv = self.Wxx * self.mapping.deriv(m)
        if v is not None:
            return rDeriv.T * (rDeriv * v)
        return rDeriv.T * rDeriv

    @Utils.timeIt
    def _evalSmoothyy2Deriv(self, m, v=None):
        if self.mrefInSmooth == True:
            rDeriv = self.Wyy * ( self.mapping.deriv( m - self.mref ) )
        elif self.mrefInSmooth == False:
            rDeriv = self.Wyy * self.mapping.deriv(m)
        if v is not None:
            return rDeriv.T * (rDeriv * v)
        return rDeriv.T * rDeriv

    @Utils.timeIt
    def _evalSmoothzz2Deriv(self, m, v=None):
        if self.mrefInSmooth == True:
            rDeriv = self.Wzz * ( self.mapping.deriv( m - self.mref ) )
        elif self.mrefInSmooth == False:
            rDeriv = self.Wzz * self.mapping.deriv(m)
        if v is not None:
            return rDeriv.T * (rDeriv * v)
        return rDeriv.T * rDeriv

    @Utils.timeIt
    def _evalSmoothDeriv2(self, m):
        deriv = self._evalSmoothxxDeriv(m)
        if self.regmesh.dim > 1:
            deriv += self._evalSmoothyyDeriv(m)
        if self.regmesh.dim > 2:
            deriv += self._evalSmoothzzDeriv(m)
        return deriv

    @Utils.timeIt
    def _evalSmooth2Deriv2(self, m, v=None):
        deriv = self._evalSmoothxx2Deriv(m, v)
        if self.regmesh.dim > 1:
            deriv += self._evalSmoothyy2Deriv(m, v)
        if self.regmesh.dim > 2:
            deriv += self._evalSmoothzz2Deriv(m, v)
        return deriv


    @Utils.timeIt
    def eval(self, m):
        return self._evalSmall(m) + self._evalSmooth(m) + self._evalSmooth2(m)

    @Utils.timeIt
    def evalDeriv(self, m):
        """
        The regularization is:

        .. math::

            R(m) = \\frac{1}{2}\mathbf{(m-m_\\text{ref})^\\top W^\\top W(m-m_\\text{ref})}

        So the derivative is straight forward:

        .. math::

            R(m) = \mathbf{W^\\top W (m-m_\\text{ref})}

        """
        return self._evalSmallDeriv(m) + self._evalSmoothDeriv(m) + self._evalSmoothDeriv2(m)

    def eval2Deriv(self, m, v=None):
        """
        The regularization is:

        .. math::

            R(m) = \\frac{1}{2}\mathbf{(m-m_\\text{ref})^\\top W^\\top W(m-m_\\text{ref})}

        So the derivative is straight forward:

        .. math::

            R(m) = \mathbf{W^\\top W (m-m_\\text{ref})}

        """
        return self._evalSmall2Deriv(m, v) + self._evalSmooth2Deriv(m, v) + self._evalSmooth2Deriv2(m, v)



class Sparse(Simple):
    """
        The regularization is:

        .. math::

            R(m) = \\frac{1}{2}\mathbf{(m-m_\\text{ref})^\\top W^\\top R^\\top R W(m-m_\\text{ref})}

        where the IRLS weight

        .. math::

            R = \eta TO FINISH LATER!!!

        So the derivative is straight forward:

        .. math::

            R(m) = \mathbf{W^\\top R^\\top R W (m-m_\\text{ref})}

        The IRLS weights are recomputed after each beta solves.
        It is strongly recommended to do a few Gauss-Newton iterations
        before updating.
    """

    # set default values
    eps_p = 1e-1        # Threshold value for the model norm
    eps_q = 1e-1        # Threshold value for the model gradient norm
    curModel = None     # Requires model to compute the weights
    l2model = None
    gamma = 1.          # Model norm scaling to smooth out convergence
    norms = [0., 2., 2., 2.] # Values for norm on (m, dmdx, dmdy, dmdz)
    cell_weights = 1.        # Consider overwriting with sensitivity weights

    def __init__(self, mesh, mapping=None, indActive=None, **kwargs):
        Simple.__init__(self, mesh, mapping=mapping, indActive=indActive, **kwargs)

        if isinstance(self.cell_weights,float):
            self.cell_weights = np.ones(self.regmesh.nC) * self.cell_weights

    @property
    def Wsmall(self):
        """Regularization matrix Wsmall"""
        if getattr(self,'_Wsmall', None) is None:
            if getattr(self, 'curModel', None) is None:
                self.Rs = Utils.speye(self.regmesh.nC)

            else:
                f_m = self.mapping * (self.curModel - self.reg.mref)
                self.rs = self.R(f_m , self.eps_p, self.norms[0])
                self.Rs = Utils.sdiag( self.rs )

            self._Wsmall = Utils.sdiag((self.alpha_s*self.gamma*self.cell_weights)**0.5)*self.Rs

        return self._Wsmall

    @property
    def Wx(self):
        """Regularization matrix Wx"""
        if getattr(self,'_Wx', None) is None:
            if getattr(self, 'curModel', None) is None:
                self.Rx = Utils.speye(self.regmesh.cellDiffxStencil.shape[0])

            else:
                f_m = self.regmesh.cellDiffxStencil * (self.mapping * self.curModel)
                self.rx = self.R( f_m , self.eps_q, self.norms[1])
                self.Rx = Utils.sdiag( self.rx )

            self._Wx = Utils.sdiag(( self.alpha_x*self.gamma*(self.regmesh.aveCC2Fx*self.cell_weights))**0.5)*self.Rx*self.regmesh.cellDiffxStencil

        return self._Wx

    @property
    def Wy(self):
        """Regularization matrix Wy"""
        if getattr(self,'_Wy', None) is None:
            if getattr(self, 'curModel', None) is None:
                self.Ry = Utils.speye(self.regmesh.cellDiffyStencil.shape[0])

            else:
                f_m = self.regmesh.cellDiffyStencil * (self.mapping * self.curModel)
                self.ry = self.R( f_m , self.eps_q, self.norms[2])
                self.Ry = Utils.sdiag( self.ry )

            self._Wy = Utils.sdiag((self.alpha_y*self.gamma*(self.regmesh.aveCC2Fy*self.cell_weights))**0.5)*self.Ry*self.regmesh.cellDiffyStencil

        return self._Wy

    @property
    def Wz(self):
        """Regularization matrix Wz"""
        if getattr(self,'_Wz', None) is None:
            if getattr(self, 'curModel', None) is None:
                self.Rz = Utils.speye(self.regmesh.cellDiffzStencil.shape[0])

            else:
                f_m = self.regmesh.cellDiffzStencil * (self.mapping * self.curModel)
                self.rz = self.R( f_m , self.eps_q, self.norms[3])
                self.Rz = Utils.sdiag( self.rz )

            self._Wz = Utils.sdiag((self.alpha_z*self.gamma*(self.regmesh.aveCC2Fz*self.cell_weights))**0.5)*self.Rz*self.regmesh.cellDiffzStencil

        return self._Wz

    def R(self, f_m , eps, exponent):

        # Eta scaling is important for mix-norms...do not mess with it
        eta = (eps**(1.-exponent/2.))**0.5
        r = eta / (f_m**2.+ eps**2.)**((1.-exponent/2.)/2.)

        return r

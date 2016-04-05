import Utils, Maps, Mesh, numpy as np, scipy.sparse as sp

class BaseRegularization(object):
    """
    **Base Regularization Class**

    This is used to regularize the model space::

        reg = Regularization(mesh)

    """

    __metaclass__ = Utils.SimPEGMetaClass

    counter = None

    mapPair = Maps.IdentityMap    #: A SimPEG.Map Class

    mapping = None    #: A SimPEG.Map instance.
    mesh    = None    #: A SimPEG.Mesh instance.
    mref = None       #: Reference model.

    def __init__(self, mesh, mapping=None, indActive=None, **kwargs):
        Utils.setKwargs(self, **kwargs)
        self.mesh = mesh
        assert isinstance(mesh, Mesh.BaseMesh), "mesh must be a SimPEG.Mesh object."
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
            print 'Regularization has switched to a new parent!'
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
        return sp.identity(self.mapping.nP)


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

            :param numpy.array m: geophysical model
            :param numpy.array v: vector to multiply
            :rtype: scipy.sparse.csr_matrix or numpy.ndarray
            :return: WtW or WtW*v

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


class Tikhonov(BaseRegularization):
    """
    """
    smoothModel = True  #: SMOOTH and SMOOTH_MOD_DIF options
    alpha_s  = Utils.dependentProperty('_alpha_s', 1e-6, ['_W', '_Ws'], "Smallness weight")
    alpha_x  = Utils.dependentProperty('_alpha_x', 1.0, ['_W', '_Wx'], "Weight for the first derivative in the x direction")
    alpha_y  = Utils.dependentProperty('_alpha_y', 1.0, ['_W', '_Wy'], "Weight for the first derivative in the y direction")
    alpha_z  = Utils.dependentProperty('_alpha_z', 1.0, ['_W', '_Wz'], "Weight for the first derivative in the z direction")
    alpha_xx = Utils.dependentProperty('_alpha_xx', 0.0, ['_W', '_Wxx'], "Weight for the second derivative in the x direction")
    alpha_yy = Utils.dependentProperty('_alpha_yy', 0.0, ['_W', '_Wyy'], "Weight for the second derivative in the y direction")
    alpha_zz = Utils.dependentProperty('_alpha_zz', 0.0, ['_W', '_Wzz'], "Weight for the second derivative in the z direction")

    def __init__(self, mesh, mapping=None, indActive = None, **kwargs):
        BaseRegularization.__init__(self, mesh, mapping=mapping, **kwargs)
        self.indActive = indActive

    @property
    def Ws(self):
        """Regularization matrix Ws"""
        if getattr(self,'_Ws', None) is None:
            self._Ws = Utils.sdiag((self.mesh.vol*self.alpha_s)**0.5)
            if self.indActive is not None:
                Pac = Utils.speye(self.mesh.nC)[:,self.indActive]
                self._Ws = Pac.T * self._Ws * Pac
        return self._Ws

    @property
    def Wx(self):
        """Regularization matrix Wx"""
        if getattr(self, '_Wx', None) is None:
            Ave_x_vol = self.mesh.aveF2CC[:,:self.mesh.nFx].T*self.mesh.vol
            self._Wx = Utils.sdiag((Ave_x_vol*self.alpha_x)**0.5)*self.mesh.cellGradx

            if self.indActive is not None:
                indActive_Fx = (self.mesh.aveFx2CC.T * self.indActive) == 1
                Pac = Utils.speye(self.mesh.nC)[:,self.indActive]
                Pafx = Utils.speye(self.mesh.nFx)[:,indActive_Fx]
                self._Wx = Pafx.T*self._Wx*Pac

        return self._Wx

    @property
    def Wy(self):
        """Regularization matrix Wy"""
        if getattr(self, '_Wy', None) is None:
            Ave_y_vol = self.mesh.aveF2CC[:,self.mesh.nFx:np.sum(self.mesh.vnF[:2])].T*self.mesh.vol
            self._Wy = Utils.sdiag((Ave_y_vol*self.alpha_y)**0.5)*self.mesh.cellGrady

            if self.indActive is not None:
                indActive_Fy = (self.mesh.aveFy2CC.T * self.indActive) == 1
                Pac = Utils.speye(self.mesh.nC)[:,self.indActive]
                Pafy = Utils.speye(self.mesh.nFy)[:,indActive_Fy]
                self._Wy = Pafy.T*self._Wy*Pac

        return self._Wy

    @property
    def Wz(self):
        """Regularization matrix Wz"""
        if getattr(self, '_Wz', None) is None:
            Ave_z_vol = self.mesh.aveF2CC[:,np.sum(self.mesh.vnF[:2]):].T*self.mesh.vol
            self._Wz = Utils.sdiag((Ave_z_vol*self.alpha_z)**0.5)*self.mesh.cellGradz

            if self.indActive is not None:
                indActive_Fz = (self.mesh.aveFz2CC.T * self.indActive) == 1
                Pac = Utils.speye(self.mesh.nC)[:,self.indActive]
                Pafz = Utils.speye(self.mesh.nFz)[:,indActive_Fz]
                self._Wz = Pafz.T*self._Wz*Pac

        return self._Wz

    @property
    def Wxx(self):
        """Regularization matrix Wxx"""
        if getattr(self, '_Wxx', None) is None:
            self._Wxx = Utils.sdiag((self.mesh.vol*self.alpha_xx)**0.5)*self.mesh.faceDivx*self.mesh.cellGradx

            if self.indActive is not None:
                Pac = Utils.speye(self.mesh.nC)[:,self.indActive]
                self._Wxx = Pac.T*self._Wxx*Pac

        return self._Wxx

    @property
    def Wyy(self):
        """Regularization matrix Wyy"""
        if getattr(self, '_Wyy', None) is None:
            self._Wyy = Utils.sdiag((self.mesh.vol*self.alpha_yy)**0.5)*self.mesh.faceDivy*self.mesh.cellGrady

            if self.indActive is not None:
                Pac = Utils.speye(self.mesh.nC)[:,self.indActive]
                self._Wyy = Pac.T*self._Wyy*Pac

        return self._Wyy

    @property
    def Wzz(self):
        """Regularization matrix Wzz"""
        if getattr(self, '_Wzz', None) is None:
            self._Wzz = Utils.sdiag((self.mesh.vol*self.alpha_zz)**0.5)*self.mesh.faceDivz*self.mesh.cellGradz

            if self.indActive is not None:
                Pac = Utils.speye(self.mesh.nC)[:,self.indActive]
                self._Wzz = Pac.T*self._Wzz*Pac

        return self._Wzz

    @property
    def Wsmooth(self):
        """Full smoothness regularization matrix W"""
        if getattr(self, '_Wsmooth', None) is None:
            wlist = (self.Wx, self.Wxx)
            if self.mesh.dim > 1:
                wlist += (self.Wy, self.Wyy)
            if self.mesh.dim > 2:
                wlist += (self.Wz, self.Wzz)
            self._Wsmooth = sp.vstack(wlist)
        return self._Wsmooth

    @property
    def W(self):
        """Full regularization matrix W"""
        if getattr(self, '_W', None) is None:
            wlist = (self.Ws, self.Wsmooth)
            self._W = sp.vstack(wlist)
        return self._W

    @Utils.timeIt
    def eval(self, m):
        if self.smoothModel == True:
            r1 = self.Wsmooth * ( self.mapping * (m) )
            r2 = self.Ws * ( self.mapping * (m - self.mref) )
            return 0.5*(r1.dot(r1)+r2.dot(r2))
        elif self.smoothModel == False:
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
        if self.smoothModel == True:
            mD1 = self.mapping.deriv(m)
            mD2 = self.mapping.deriv(m - self.mref)
            r1 = self.Wsmooth * ( self.mapping * (m))
            r2 = self.Ws * ( self.mapping * (m - self.mref) )
            out1 = mD1.T * ( self.Wsmooth.T * r1 )
            out2 = mD2.T * ( self.Ws.T * r2 )
            out = out1+out2
        elif self.smoothModel == False:
            mD = self.mapping.deriv(m - self.mref)
            r = self.W * ( self.mapping * (m - self.mref) )
            out = mD.T * ( self.W.T * r )
        return out

# <<<<<<< HEAD

# class Simple(BaseRegularization):
#     """
#         Only for tensor mesh
#     """

#     smoothModel = True  #: SMOOTH and SMOOTH_MOD_DIF options
#     alpha_s  = Utils.dependentProperty('_alpha_s', 1.0, ['_W', '_Ws'], "Smallness weight")
#     alpha_x  = Utils.dependentProperty('_alpha_x', 1.0, ['_W', '_Wx'], "Weight for the first derivative in the x direction")
#     alpha_y  = Utils.dependentProperty('_alpha_y', 1.0, ['_W', '_Wy'], "Weight for the first derivative in the y direction")
#     alpha_z  = Utils.dependentProperty('_alpha_z', 1.0, ['_W', '_Wz'], "Weight for the first derivative in the z direction")
#     alpha_xx = Utils.dependentProperty('_alpha_xx', 0.0, ['_W', '_Wxx'], "Weight for the second derivative in the x direction")
#     alpha_yy = Utils.dependentProperty('_alpha_yy', 0.0, ['_W', '_Wyy'], "Weight for the second derivative in the y direction")
#     alpha_zz = Utils.dependentProperty('_alpha_zz', 0.0, ['_W', '_Wzz'], "Weight for the second derivative in the z direction")

#     def __init__(self, mesh, mapping=None, **kwargs):
#         BaseRegularization.__init__(self, mesh, mapping=mapping, **kwargs)



#     @property
#     def Ws(self):
#         """Regularization matrix Ws"""
#         if getattr(self,'_Ws', None) is None:
#                 self._Ws = Utils.sdiag((self.mesh.vol*self.alpha_s)**0.5)
#         return self._Ws

#     @property
#     def Wx(self):
#         """Regularization matrix Wx"""
#         if getattr(self, '_Wx', None) is None:
#             self._Wx = Utils.sdiag((self.mesh.vol*self.alpha_x)**0.5)*self.mesh.unitCellGradx
#         return self._Wx

#     @property
#     def Wy(self):
#         """Regularization matrix Wy"""
#         if getattr(self, '_Wy', None) is None:
#             self._Wy = Utils.sdiag((self.mesh.vol*self.alpha_y)**0.5)*self.mesh.unitCellGrady
#         return self._Wy

#     @property
#     def Wz(self):
#         """Regularization matrix Wz"""
#         if getattr(self, '_Wz', None) is None:
#             self._Wz = Utils.sdiag((self.mesh.vol*self.alpha_z)**0.5)*self.mesh.unitCellGradz
#         return self._Wz

#     @property
#     def Wxx(self):
#         """Regularization matrix Wxx"""
#         if getattr(self, '_Wxx', None) is None:
#             self._Wxx = Utils.sdiag((self.mesh.vol*self.alpha_xx)**0.5)*self.mesh.faceDivx*self.mesh.cellGradx
#         return self._Wxx

#     @property
#     def Wyy(self):
#         """Regularization matrix Wyy"""
#         if getattr(self, '_Wyy', None) is None:
#             self._Wyy = Utils.sdiag((self.mesh.vol*self.alpha_yy)**0.5)*self.mesh.faceDivy*self.mesh.cellGrady
#         return self._Wyy

#     @property
#     def Wzz(self):
#         """Regularization matrix Wzz"""
#         if getattr(self, '_Wzz', None) is None:
#             self._Wzz = Utils.sdiag((self.mesh.vol*self.alpha_zz)**0.5)*self.mesh.faceDivz*self.mesh.cellGradz
#         return self._Wzz

#     @property
#     def Wsmooth(self):
#         """Full smoothness regularization matrix W"""
#         if getattr(self, '_Wsmooth', None) is None:
#             wlist = (self.Wx, self.Wxx)
#             if self.mesh.dim > 1:
#                 wlist += (self.Wy, self.Wyy)
#             if self.mesh.dim > 2:
#                 wlist += (self.Wz, self.Wzz)
#             self._Wsmooth = sp.vstack(wlist)
#         return self._Wsmooth

#     @property
#     def W(self):
#         """Full regularization matrix W"""
#         if getattr(self, '_W', None) is None:
#             wlist = (self.Ws, self.Wsmooth)
#             self._W = sp.vstack(wlist)
#         return self._W

#     @Utils.timeIt
#     def eval(self, m):
#         if self.smoothModel == True:
#             r1 = self.Wsmooth * ( self.mapping * (m) )
#             r2 = self.Ws * ( self.mapping * (m - self.mref) )
#             return 0.5*(r1.dot(r1)+r2.dot(r2))
#         elif self.smoothModel == False:
#             r = self.W * ( self.mapping * (m - self.mref) )
#             return 0.5*r.dot(r)


#     @Utils.timeIt
#     def evalDeriv(self, m):
#         """

#         The regularization is:

#         .. math::

#             R(m) = \\frac{1}{2}\mathbf{(m-m_\\text{ref})^\\top W^\\top W(m-m_\\text{ref})}

#         So the derivative is straight forward:

#         .. math::

#             R(m) = \mathbf{W^\\top W (m-m_\\text{ref})}

#         """
#         if self.smoothModel == True:
#             mD1 = self.mapping.deriv(m)
#             mD2 = self.mapping.deriv(m - self.mref)
#             r1 = self.Wsmooth * ( self.mapping * (m))
#             r2 = self.Ws * ( self.mapping * (m - self.mref) )
#             out1 = mD1.T * ( self.Wsmooth.T * r1 )
#             out2 = mD2.T * ( self.Ws.T * r2 )
#             out = out1+out2
#         elif self.smoothModel == False:
#             mD = self.mapping.deriv(m - self.mref)
#             r = self.W * ( self.mapping * (m - self.mref) )
#             out = mD.T * ( self.W.T * r )
#         return out

# class SparseRegularization(Simple):

#     eps = 1e-1

#     m = None
#     gamma = 1.
#     p = 0.
#     qx = 2.
#     qy = 2.
#     qz = 2.

#     def __init__(self, mesh, mapping=None, **kwargs):
#         Simple.__init__(self, mesh, mapping=mapping, **kwargs)


#     @property
#     def Wsmooth(self):
#         """Full smoothness regularization matrix W"""
#         if getattr(self, '_Wsmooth', None) is None:
#             wlist = (self.Wx, self.Wxx)
#             if self.mesh.dim > 1:
#                 wlist += (self.Wy, self.Wyy)
#             if self.mesh.dim > 2:
#                 wlist += (self.Wz, self.Wzz)
#             self._Wsmooth = sp.vstack(wlist)
#         return self._Wsmooth

#     @property
#     def W(self):
#         """Full regularization matrix W"""
#         if getattr(self, '_W', None) is None:
#             wlist = (self.Ws, self.Wsmooth)
#             self._W = sp.vstack(wlist)
#         return self._W

#     @property
#     def Ws(self):
#         """Regularization matrix Ws"""
#         if getattr(self, 'm', None) is None:
#             self.Rs = Utils.speye(self.mesh.nC)

#         else:
#             f_m = self.m
#             self.rs = self.R(f_m , self.p, self.eps)
#             #print "Min rs: " + str(np.max(self.rs)) + "Max rs: " + str(np.min(self.rs))
#             self.Rs = Utils.sdiag( self.rs )

#         self._Ws = Utils.sdiag((self.mesh.vol*self.alpha_s*self.gamma)**0.5)*self.Rs

#         return self._Ws

#     @property
#     def Wx(self):
#         """Regularization matrix Wx"""

#         if getattr(self, 'm', None) is None:
#             self.Rx = Utils.speye(self.mesh.unitCellGradx.shape[0])

#         else:
#             f_m = self.mesh.unitCellGradx * self.m
#             self.rx = self.R( f_m , self.qx, self.eps)
#             self.Rx = Utils.sdiag( self.rx )

#         if getattr(self, '_Wx', None) is None:
#             self._Wx = Utils.sdiag((self.mesh.vol*self.alpha_x*self.gamma)**0.5)*self.Rx*self.mesh.unitCellGradx
#         return self._Wx

#     @property
#     def Wy(self):
#         """Regularization matrix Wy"""

#         if getattr(self, 'm', None) is None:
#             self.Ry = Utils.speye(self.mesh.unitCellGrady.shape[0])

#         else:
#             f_m = self.mesh.unitCellGrady * self.m
#             self.ry = self.R( f_m , self.qy, self.eps)
#             self.Ry = Utils.sdiag( self.ry )

#         if getattr(self, '_Wy', None) is None:
#             self._Wy = Utils.sdiag((self.mesh.vol*self.alpha_y*self.gamma)**0.5)*self.Ry*self.mesh.unitCellGrady
#         return self._Wy

#     @property
#     def Wz(self):
#         """Regularization matrix Wz"""

#         if getattr(self, 'm', None) is None:
#             self.Rz = Utils.speye(self.mesh.unitCellGradz.shape[0])

#         else:
#             f_m = self.mesh.unitCellGradz * self.m
#             self.rz = self.R( f_m , self.qz, self.eps)
#             self.Rz = Utils.sdiag( self.rz )

#         if getattr(self, '_Wz', None) is None:
#             self._Wz = Utils.sdiag((self.mesh.vol*self.alpha_z*self.gamma)**0.5)*self.Rz*self.mesh.unitCellGradz
#         return self._Wz


#     def R(self, f_m , p, dec):

#         eta = (self.eps**(1-p/2.))**0.5
#         r = eta / (f_m**2.+self.eps**2.)**((1-p/2.)/2.)

#         return r
# =======
# >>>>>>> 834de582844e8e1eac95819fbe03eed55dbeb001

from SimPEG import Utils, Maps, Mesh, sp, np
from SimPEG.Regularization import BaseRegularization, Simple

class MultiRegularization(Simple):
    """
    **MultiRegularization Class**

    This is used to regularize the model space
    having multiple models [m1, m2, m3, ...] ::

        reg = Regularization(mesh)

    """
    nModels = None # Number of models
    ratios = None  # Ratio for different models
    crossgrad = False # Use cross gradient or not
    betacross = 1.
    wx = []
    wy = []
    wz = []

    def __init__(self, mesh, mapping=None, indActive=None, **kwargs):
        BaseRegularization.__init__(self, mesh, mapping=mapping, indActive=indActive, **kwargs)
        if self.nModels == None:
            raise Exception("Put nModels as a initial input!")
        if self.ratios == None:
            self.ratios = [1. for imodel in range(self.nModels)]

    @property
    def Wsmall(self):
        """Regularization matrix Wsmall"""
        if getattr(self,'_Wsmall', None) is None:
            vecs = []
            for imodel in range(self.nModels):
                vecs.append((self.regmesh.vol*self.alpha_s*self.wght*self.ratios[imodel])**0.5)
            self._Wsmall = Utils.sdiag(np.hstack(vecs))
        return self._Wsmall

    @property
    def Wx(self):
        """Regularization matrix Wx"""
        if getattr(self, '_Wx', None) is None:
            mats = []
            for imodel in range(self.nModels):
                self.wx.append(Utils.sdiag((self.regmesh.aveCC2Fx * self.regmesh.vol*self.alpha_x*self.ratios[imodel]*(self.regmesh.aveCC2Fx*self.wght))**0.5))
                mats.append(self.wx[imodel]*self.regmesh.cellDiffxStencil)
            self._Wx = sp.block_diag(mats)
        return self._Wx

    @property
    def Wy(self):
        """Regularization matrix Wy"""
        if getattr(self, '_Wy', None) is None:
            mats = []
            for imodel in range(self.nModels):
                self.wy.append(Utils.sdiag((self.regmesh.aveCC2Fy * self.regmesh.vol*self.alpha_y*self.ratios[imodel]*(self.regmesh.aveCC2Fy*self.wght))**0.5))
                mats.append(self.wy[imodel]*self.regmesh.cellDiffyStencil)
            self._Wy = sp.block_diag(mats)
        return self._Wy

    @property
    def Wz(self):
        """Regularization matrix Wz"""
        if getattr(self, '_Wz', None) is None:
            mats = []
            for imodel in range(self.nModels):
                self.wz.append(Utils.sdiag((self.regmesh.aveCC2Fz * self.regmesh.vol*self.alpha_z*self.ratios[imodel]*(self.regmesh.aveCC2Fz*self.wght))**0.5))
                mats.append(self.wz[imodel]*self.regmesh.cellDiffzStencil)
            self._Wz = sp.block_diag(mats)
        return self._Wz

    @property
    def Wsmooth(self):
        """Full smoothness regularization matrix W"""
        if getattr(self, '_Wsmooth', None) is None:
            wlist = (self.Wx,)
            if self.regmesh.dim > 1:
                wlist += (self.Wy,)
            if self.regmesh.dim > 2:
                wlist += (self.Wz,)
            self._Wsmooth = sp.vstack(wlist)
        return self._Wsmooth

    @property
    def W(self):
        """Full regularization matrix W"""
        if getattr(self, '_W', None) is None:
            wlist = (self.Wsmall, self.Wsmooth)
            self._W = sp.vstack(wlist)
        return self._W


    @Utils.timeIt
    def eval(self, m):
        return self._evalSmall(m) + self._evalSmooth(m)

    @Utils.timeIt
    def _evalSmall(self, m):
        r = self.Wsmall * ( self.mapping * (m - self.mref) )
        return 0.5 * r.dot(r)

    @Utils.timeIt
    def _evalSmooth(self, m):
        if self.mrefInSmooth == True:
            r = self.Wsmooth * ( self.mapping * (m - self.mref) )
        elif self.mrefInSmooth == False:
            r = self.Wsmooth * ( self.mapping * m)
        return 0.5 * r.dot(r)

    def cross(a,b):
        ax, ay, az = a[0], a[1], a[2]
        bx, by, bz = b[0], b[1], b[2]
        cx = ay*bz - az*by
        cy = az*bx - ax*bz
        cz = ax*by - ay*bx
        return [cx, cy, cz]

     # TODO: Implement Cross Gradients..
    @Utils.timeIt
    def _evalCross(self, m):
        if self.crossgrad == False:
            return 0.
        elif self.crossgrad == True:
            M = (self.mapping * m).reshape((self.regmesh.nC, self.nModels), order="F")

            ax = self.regmesh.aveFx2CC*self.regmesh.wx[0]*M[:,0]
            ay = self.regmesh.aveFy2CC*self.regmesh.wy[0]*M[:,0]
            az = self.regmesh.aveFz2CC*self.regmesh.wz[0]*M[:,0]
            bx = self.regmesh.aveFx2CC*self.regmesh.wx[1]*M[:,1]
            by = self.regmesh.aveFy2CC*self.regmesh.wy[1]*M[:,1]
            bz = self.regmesh.aveFz2CC*self.regmesh.wz[1]*M[:,1]
            #ab
            out_ab = cross([ax, ay, az], [bx, by, bz])
            r = np.r_[out_ab[0], out_ab[1], out_ab[2]]*np.sqrt(self.betacross)

            if self.nModels == 3:
                cx = self.regmesh.aveFx2CC*self.regmesh.wx[1]*M[:,1]
                cy = self.regmesh.aveFy2CC*self.regmesh.wy[1]*M[:,1]
                cz = self.regmesh.aveFz2CC*self.regmesh.wz[1]*M[:,1]
                #ac
                out_ac = cross([ax, ay, az], [cx, cy, cz])
                #bc
                out_bc = cross([bx, by, bz], [cx, cy, cz])
                r = np.r_[r, np.hstack(out_ac)*np.sqrt(self.betacross), np.hstack(out_bc)*np.sqrt(self.betacross)]

            return 0.5 * r.dot(r)

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
        deriv = self._evalSmallDeriv(m) + self._evalSmoothDeriv(m)
        if self.crossgrad==True:
            deriv += self._evalCrossDeriv(m)
        return deriv

    @Utils.timeIt
    def _evalCrossDeriv(self,m):
        r = self.Wsmall * ( self.mapping * (m - self.mref) )
        return r.T * ( self.Wsmall * self.mapping.deriv(m - self.mref) )

    @Utils.timeIt
    def eval2Deriv(self, m, v=None):
        """
        Second derivative

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




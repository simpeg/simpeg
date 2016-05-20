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

    #  TODO: Implement Cross Gradients..
    # @Utils.timeIt
    # def _evalCross(self, m):
    #     if self.crossgrad == False:
    #         return 0.
    #     elif self.crossgrad == True:
    #         M = (self.mapping * m).reshape((self.regmesh.nC, self.nModels), order="F")

    #         for imodel in range(self.nModels):
    #             ux.append(self.regmesh.aveFx2CC*self.regmesh.wx[imodel]*M[:,imodel])
    #             uy.append(self.regmesh.aveFy2CC*self.regmesh.wy[imodel]*M[:,imodel])
    #             uz.append(self.regmesh.aveFz2CC*self.regmesh.wz[imodel]*M[:,imodel])

    #         ax, ay, az = ux[0], uy[0], uz[0]
    #         for imodel in range(1,self.nModels):
    #             bx, by, bz = ux[imodel], uy[imodel], uz[imodel]
    #             cx = ay*bz - az*by
    #             cy = az*bx - ax*bz
    #             cz = ax*by - ay*bx
    #             ax, ay, az = cx.copy(), cy.copy(), cz.copy()
    #         r = np.r_[ax, ay, az]*np.sqrt(self.betacross)

    #         return 0.5 * r.dot(r)


from SimPEG import Survey, Problem, Utils, np, sp 
from simpegEM.Utils.EMUtils import omega


class FieldsFDEM(Problem.Fields):
    """Fancy Field Storage for a FDEM survey."""
    knownFields = {}
    dtype = complex


class FieldsFDEM_e(FieldsFDEM):
    knownFields = {'e_sol':'E'}
    aliasFields = {
                    'e' : ['e_sol','E','_e'],
                    'ePrimary' : ['e_sol','E','_ePrimary'],
                    'eSecondary' : ['e_sol','E','_eSecondary'],
                    'b' : ['e_sol','F','_b'],
                    'bPrimary' : ['e_sol','F','_bPrimary'],
                    'bSecondary' : ['e_sol','F','_bSecondary']
                  }

    def __init__(self,mesh,survey,**kwargs):
        FieldsFDEM.__init__(self,mesh,survey,**kwargs)

    def startup(self):
        self._edgeCurl = self.survey.prob.mesh.edgeCurl

    def _ePrimary(self, e_sol, srcList):
        ePrimary = np.zeros_like(e_sol)
        for i, src in enumerate(srcList):
            ep = src.ePrimary(self.survey.prob)
            if ep is not None:
                ePrimary[:,i] = ep     
        return ePrimary

    def _eSecondary(self, e_sol, srcList):
        return e_sol 

    def _e(self, e_sol, srcList):
        return self._ePrimary(e_sol,srcList) + self._eSecondary(e_sol,srcList)

    def _bPrimary(self, e_sol, srcList):
        bPrimary = np.zeros([self._edgeCurl.shape[0],e_sol.shape[1]],dtype = complex)
        for i, src in enumerate(srcList):
            bp = src.bPrimary(self.survey.prob)
            if bp is not None:
                bPrimary[:,i] += bp
        return bPrimary

    def _bSecondary(self, e_sol, srcList): 
        C = self._edgeCurl
        b = (C * e_sol)
        for i, src in enumerate(srcList):
            b[:,i] *= - 1./(1j*omega(src.freq))
            S_m, _ = src.eval(self.survey.prob)
            if S_m is not None:
                b[:,i] += 1./(1j*omega(src.freq)) * S_m
        return b

    def _b(self, e_sol, srcList):
        return self._bPrimary(e_sol, srcList) + self._bSecondary(e_sol, srcList)

    def _bDeriv(self, e, srcList, v, adjoint=False):
        raise NotImplementedError('Fields Derivs Not Implemented Yet')
        # S_mDeriv,_ = src.getSourceDeriv(self.survey.prob, v, adjoint)
        # if S_mDeriv is None:
        #     return None
        # else:
        #     return 1./(1j*omega(src.freq)) * S_mDeriv


class FieldsFDEM_b(FieldsFDEM):
    knownFields = {'b_sol':'F'}
    aliasFields = {
                    'b' : ['b_sol','F','_b'],
                    'bPrimary' : ['b_sol','F','_bPrimary'],
                    'bSecondary' : ['b_sol','F','_bSecondary'],
                    'e' : ['b_sol','E','_e'],
                    'ePrimary' : ['b_sol','E','_ePrimary'],
                    'eSecondary' : ['b_sol','E','_eSecondary'],
                  }

    def __init__(self,mesh,survey,**kwargs):
        FieldsFDEM.__init__(self,mesh,survey,**kwargs)

    def startup(self):
        self._edgeCurl = self.survey.prob.mesh.edgeCurl
        self._MeSigmaI = self.survey.prob.MeSigmaI
        self._MfMui = self.survey.prob.MfMui

    def _bPrimary(self, b_sol, srcList):
        bPrimary = np.zeros_like(b_sol)
        for i, src in enumerate(srcList):
            bp = src.bPrimary(self.survey.prob)
            if bp is not None:
                bPrimary[:,i] = bp
        return bPrimary

    def _bSecondary(self, b_sol, srcList):
        return b_sol

    def _b(self, b_sol, srcList):
        return self._bPrimary(b_sol, srcList) + self._bSecondary(b_sol, srcList)  

    def _ePrimary(self, b_sol, srcList):
        ePrimary = np.zeros([self._edgeCurl.shape[1],b_sol.shape[1]],dtype = complex)
        for i,src in enumerate(srcList):
            ep = src.ePrimary(self.survey.prob)
            if ep is not None:
                ePrimary[:,i] = ep
        return ePrimary

    def _eSecondary(self, b_sol, srcList):
        e = self._MeSigmaI * ( self._edgeCurl.T * ( self._MfMui * b_sol))
        for i,src in enumerate(srcList): 
            _,S_e = src.eval(self.survey.prob)
            if S_e is not None:
                e += -self._MeSigmaI*S_e

        return e

    def _e(self, b_sol, srcList):
        return self._ePrimary(b_sol, srcList) + self._eSecondary(b_sol, srcList)

    def _eDeriv(self, b_sol, srcList, v, adjoint=False):
        raise NotImplementedError('Fields Derivs Not Implemented Yet')
        _,S_eDeriv = src.evalDeriv(self.survey.prob, v, adjoint)

        if S_eDeriv is None:
            return None
        else:
            return -S_eDeriv


class FieldsFDEM_j(FieldsFDEM):
    knownFields = {'j_sol':'F'}
    aliasFields = {
                    'j' : ['j_sol','F','_j'],
                    'jPrimary' : ['j_sol','F','_jPrimary'],
                    'jSecondary' : ['j_sol','F','_jSecondary'],
                    'h' : ['j_sol','E','_h'],
                    'hPrimary' : ['j_sol','E','_hPrimary'],
                    'hSecondary' : ['j_sol','E','_hSecondary'],
                  }

    def __init__(self,mesh,survey,**kwargs):
        FieldsFDEM.__init__(self,mesh,survey,**kwargs)

    def startup(self):
        self._edgeCurl = self.survey.prob.mesh.edgeCurl
        self._MeMuI = self.survey.prob.MeMuI
        self._MfRho = self.survey.prob.MfRho
        self._curModel = self.survey.prob.curModel

    def _jPrimary(self, j_sol, srcList):
        jPrimary = np.zeros_like(j_sol)
        for i, src in enumerate(srcList):
            jp = src.jPrimary(self.survey.prob) 
            if jp is not None:
                jPrimary[:,i] += jp
        return jPrimary

    def _jSecondary(self, j_sol, srcList):
        return j_sol

    def _j(self, j_sol, srcList):
        return self._jPrimary(j_sol, srcList) + self._jSecondary(j_sol, srcList)

    def _hPrimary(self, j_sol, srcList):
        hPrimary = np.zeros([self._edgeCurl.shape[1],j_sol.shape[1]],dtype = complex)
        for i, src in enumerate(srcList):
            hp = src.hPrimary(self.survey.prob)
            if hp is not None:
                hPrimary[:,i] = hp 
        return hPrimary

    def _hSecondary(self, j_sol, srcList):
        MeMuI = self._MeMuI
        C = self._edgeCurl
        MfRho = self._MfRho
        h =  MeMuI * (C.T * (MfRho * j_sol) ) 
        for i, src in enumerate(srcList):
            h[:,i] *= -1./(1j*omega(src.freq))
            S_m,_ = src.eval(self.survey.prob)
            if S_m is not None:
                h[:,i] += 1./(1j*omega(src.freq)) * MeMuI * S_m
        return h

    def _h(self, j_sol, srcList): 
        return self._hPrimary(j_sol, srcList) + self._hSecondary(j_sol, srcList)

    def _hDeriv(self, j_sol, srcList, v, adjoint=False):
        raise NotImplementedError('Fields Derivs Not Implemented Yet')
        sig = self._curModel.transform
        sigi = 1/sig
        dsig_dm = self._curModel.transformDeriv
        dsigi_dsig = -Utils.sdiag(sigi)**2
        dMf_dsigi = self.mesh.getFaceInnerProductDeriv(sigi)(j)
        sigi = self._MfRho

        S_mDeriv,_ = src.getSourceDeriv(self.survey.prob, v, adjoint)

        if not adjoint:
            h_Deriv= -(1./(1j*omega(freq))) * MeMuI * ( C.T * ( dMf_dsigi * ( dsigi_dsig * ( dsig_dm * v ) ) ) )
        else:
            h_Deriv= -(1./(1j*omega(freq))) * dsig_dm.T * ( dsigi_dsig.T * ( dMf_dsigi.T * ( C * ( MeMuI.T * v ) ) ) )

        if S_mDeriv is not None:
            return 1./(1j*omega(src.freq)) * S_mDeriv + h_Deriv


class FieldsFDEM_h(FieldsFDEM):
    knownFields = {'h_sol':'E'}
    aliasFields = {
                    'h' : ['h_sol','E','_h'],
                    'hPrimary' : ['h_sol','E','_hPrimary'],
                    'hSecondary' : ['h_sol','E','_hSecondary'],
                    'j' : ['h_sol','F','_j'],
                    'jPrimary' : ['h_sol','F','_jPrimary'],
                    'jSecondary' : ['h_sol','F','_jSecondary']
                  }

    def __init__(self,mesh,survey,**kwargs):
        FieldsFDEM.__init__(self,mesh,survey,**kwargs)

    def startup(self):
        self._edgeCurl = self.survey.prob.mesh.edgeCurl
        self._MeMuI = self.survey.prob.MeMuI
        self._MfRho = self.survey.prob.MfRho

    def _hPrimary(self, h_sol, srcList):
        hPrimary = np.zeros_like(h_sol)
        for i, src in enumerate(srcList):
            hp = src.hPrimary(self.survey.prob)
            if hp is not None:
                hPrimary[:,i] += hp
            return hPrimary

    def _hSecondary(self, h_sol, srcList):
        return h_sol

    def _h(self, h_sol, srcList):
        return self._hPrimary(h_sol, srcList) + self._hSecondary(h_sol, srcList)

    def _jPrimary(self, h_sol, srcList):
        jPrimary = np.zeros([self._edgeCurl.shape[0], h_sol.shape[1]])
        for i, src in enumerate(srcList):
            jp = src.jPrimary(self.survey.prob)
            if jp is not None:
                jPrimary[:,i] = jp 
        return jPrimary

    def _jSecondary(self, h_sol, srcList):
        j = self._edgeCurl*h_sol
        for i, src in enumerate(srcList):
            _,S_e = src.eval(self.survey.prob)
            if S_e is not None:
                j[:,i] += -S_e
        return j

    def _j(self, h_sol, srcList):
        return self._jPrimary(h_sol, srcList) + self._jSecondary(h_sol, srcList)

    def _jDeriv(self, h_sol, srcList, v, adjoint=False):
        raise NotImplementedError('Fields Derivs Not Implemented Yet')
        _,S_eDeriv = src.getSourceDeriv(self.survey.prob, v, adjoint)
        if S_eDeriv is None:
            return None
        else:
            return - S_eDeriv


    # def calcFields(self, sol, freq, fieldType, adjoint=False):
    #     j = sol
    #     if fieldType == 'j':
    #         return j
    #     elif fieldType == 'h':
    #         MeMuI = self._MeMuI
    #         C = self.mesh.edgeCurl
    #         MfRho = self._MfRho
    #         if not adjoint:
    #             h = -(1./(1j*omega(freq))) * MeMuI * ( C.T * ( MfRho * j ) )
    #         else:
    #             h = -(1./(1j*omega(freq))) * MfRho.T * ( C * ( MeMuI.T * j ) )
    #         return h
    #     raise NotImplementedError('fieldType "%s" is not implemented.' % fieldType)

    # def calcFieldsDeriv(self, sol, freq, fieldType, v, adjoint=False):
    #     j = sol
    #     if fieldType == 'j':
    #         return None
    #     elif fieldType == 'h':
    #         MeMuI = self._MeMuI
    #         C = self.mesh.edgeCurl
    #         sig = self._curModel.transform
    #         sigi = 1/sig
    #         dsig_dm = self._curModel.transformDeriv
    #         dsigi_dsig = -Utils.sdiag(sigi)**2
    #         dMf_dsigi = self.mesh.getFaceInnerProductDeriv(sigi)(j)
    #         sigi = self._MfRho
    #         if not adjoint:
    #             return -(1./(1j*omega(freq))) * MeMuI * ( C.T * ( dMf_dsigi * ( dsigi_dsig * ( dsig_dm * v ) ) ) )
    #         else:
    #             return -(1./(1j*omega(freq))) * dsig_dm.T * ( dsigi_dsig.T * ( dMf_dsigi.T * ( C * ( MeMuI.T * v ) ) ) )
    #     raise NotImplementedError('fieldType "%s" is not implemented.' % fieldType)


    # def calcFields(self, sol, freq, fieldType, adjoint=False):
    #     h = sol
    #     if fieldType == 'j':
    #         C = self.mesh.edgeCurl
    #         if adjoint:
    #             return C.T*h
    #         return C*h
    #     elif fieldType == 'h':
    #         return h
    #     raise NotImplementedError('fieldType "%s" is not implemented.' % fieldType)

    # def calcFieldsDeriv(self, sol, freq, fieldType, v, adjoint=False):
    #     return None
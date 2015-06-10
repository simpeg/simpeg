from SimPEG import Survey, Problem, Utils, np, sp 
from simpegEM.Utils.EMUtils import omega


class FieldsFDEM(Problem.Fields):
    """Fancy Field Storage for a FDEM survey."""
    knownFields = {}
    dtype = complex


class FieldsFDEM_e(FieldsFDEM):
    knownFields = {'eSolution':'E'}
    aliasFields = {
                    'e' : ['eSolution','E','_e'],
                    'ePrimary' : ['eSolution','E','_ePrimary'],
                    'eSecondary' : ['eSolution','E','_eSecondary'],
                    'b' : ['eSolution','F','_b'],
                    'bPrimary' : ['eSolution','F','_bPrimary'],
                    'bSecondary' : ['eSolution','F','_bSecondary']
                  }

    def __init__(self,mesh,survey,**kwargs):
        FieldsFDEM.__init__(self,mesh,survey,**kwargs)

    def startup(self):
        self._edgeCurl = self.survey.prob.mesh.edgeCurl

    # def getDeriv_u(self, fieldsList, src, v, adjoint=False):

    # def getDeriv_m(self, fieldsList, src, v, adjoint=False):

    def _ePrimary(self, eSolution, srcList):
        ePrimary = np.zeros_like(eSolution)
        for i, src in enumerate(srcList):
            ep = src.ePrimary(self.survey.prob)
            if ep is not None:
                ePrimary[:,i] = ep     
        return ePrimary

    def _eSecondary(self, eSolution, srcList):
        return eSolution 

    def _e(self, eSolution, srcList):
        return self._ePrimary(eSolution,srcList) + self._eSecondary(eSolution,srcList)

    def _eDeriv_u(self, src, v, adjoint = False):
        return None

    def _eDeriv_m(self, src, v, adjoint = False):
        # assuming primary does not depend on the model
        return None

    def _bPrimary(self, eSolution, srcList):
        bPrimary = np.zeros([self._edgeCurl.shape[0],eSolution.shape[1]],dtype = complex)
        for i, src in enumerate(srcList):
            bp = src.bPrimary(self.survey.prob)
            if bp is not None:
                bPrimary[:,i] += bp
        return bPrimary

    def _bSecondary(self, eSolution, srcList): 
        C = self._edgeCurl
        b = (C * eSolution)
        for i, src in enumerate(srcList):
            b[:,i] *= - 1./(1j*omega(src.freq))
            S_m, _ = src.eval(self.survey.prob)
            if S_m is not None:
                b[:,i] += 1./(1j*omega(src.freq)) * S_m
        return b

    def _bSecondaryDeriv_u(self, src, v, adjoint = False):
        C = self._edgeCurl
        if adjoint:
            return - 1./(1j*omega(src.freq)) * (C.T * v)
        return - 1./(1j*omega(src.freq)) * (C * v)

    def _bSecondaryDeriv_m(self, src, v, adjoint = False):
        S_mDeriv, _ = src.evalDeriv(self.survey.prob, adjoint)
        S_mDeriv = S_mDeriv(v)
        if S_mDeriv is not None:
            return 1./(1j * omega(src.freq)) * S_mDeriv
        return None

    def _b(self, eSolution, srcList):
        return self._bPrimary(eSolution, srcList) + self._bSecondary(eSolution, srcList)

    def _bDeriv_u(self, src, v, adjoint=False):
        # Primary does not depend on u
        return self._bSecondaryDeriv_u(src, v, adjoint)

    def _bDeriv_m(self, src, v, adjoint=False):
        # Assuming the primary does not depend on the model
        return self._bSecondaryDeriv_m(src, v, adjoint)


class FieldsFDEM_b(FieldsFDEM):
    knownFields = {'bSolution':'F'}
    aliasFields = {
                    'b' : ['bSolution','F','_b'],
                    'bPrimary' : ['bSolution','F','_bPrimary'],
                    'bSecondary' : ['bSolution','F','_bSecondary'],
                    'e' : ['bSolution','E','_e'],
                    'ePrimary' : ['bSolution','E','_ePrimary'],
                    'eSecondary' : ['bSolution','E','_eSecondary'],
                  }

    def __init__(self,mesh,survey,**kwargs):
        FieldsFDEM.__init__(self,mesh,survey,**kwargs)

    def startup(self):
        self._edgeCurl = self.survey.prob.mesh.edgeCurl
        self._MeSigmaI = self.survey.prob.MeSigmaI
        self._MfMui = self.survey.prob.MfMui
        self._MeSigmaIDeriv = self.survey.prob.MeSigmaIDeriv

    def _bPrimary(self, bSolution, srcList):
        bPrimary = np.zeros_like(bSolution)
        for i, src in enumerate(srcList):
            bp = src.bPrimary(self.survey.prob)
            if bp is not None:
                bPrimary[:,i] = bp
        return bPrimary

    def _bSecondary(self, bSolution, srcList):
        return bSolution

    def _b(self, bSolution, srcList):
        return self._bPrimary(bSolution, srcList) + self._bSecondary(bSolution, srcList)  

    def _bDeriv_u(self, src, v, adjoint=False):
        return None

    def _bDeriv_m(self, src, v, adjoint=False):
        # assuming primary does not depend on the model
        return None

    def _ePrimary(self, bSolution, srcList):
        ePrimary = np.zeros([self._edgeCurl.shape[1],bSolution.shape[1]],dtype = complex)
        for i,src in enumerate(srcList):
            ep = src.ePrimary(self.survey.prob)
            if ep is not None:
                ePrimary[:,i] = ep
        return ePrimary

    def _eSecondary(self, bSolution, srcList):
        e = self._MeSigmaI * ( self._edgeCurl.T * ( self._MfMui * bSolution))
        for i,src in enumerate(srcList): 
            _,S_e = src.eval(self.survey.prob)
            if S_e is not None:
                e += -self._MeSigmaI*S_e
        return e

    def _eSecondaryDeriv_u(self, src, v, adjoint=False):
        if not adjoint:
            return self._MeSigmaI * ( self._edgeCurl.T * ( self._MfMui * v) ) 
        else:
            return self._MfMui.T * (self._edgeCurl * (self._MeSigmaI.T * v))

    def _eSecondaryDeriv_m(self, src, v, adjoint=False):
        bsol = self[[src],'bSolution']
        _,S_e = src.eval(self.survey.prob)

        w = self._edgeCurl.T * (self._MfMui * bsol)
        if S_e is not None:
            w += -S_e

        if not adjoint:
            de_dm = self._MeSigmaIDeriv(w) * v
        elif adjoint: 
            de_dm = self._MeSigmaIDeriv(w).T * v

        _, S_eDeriv = src.evalDeriv(self.survey.prob, adjoint)
        Se_Deriv = S_eDeriv(v)

        if Se_Deriv is not None:
            de_dm += -self._MeSigmaI * Se_Deriv

        return de_dm

    def _e(self, bSolution, srcList):
        return self._ePrimary(bSolution, srcList) + self._eSecondary(bSolution, srcList)

    def _eDeriv_u(self, src, v, adjoint=False):
        return self._eSecondaryDeriv_u(src, v, adjoint)

    def _eDeriv_m(self, src, v, adjoint=False):
        # assuming primary doesn't depend on model
        return self._eSecondaryDeriv_m(src, v, adjoint)


class FieldsFDEM_j(FieldsFDEM):
    knownFields = {'jSolution':'F'}
    aliasFields = {
                    'j' : ['jSolution','F','_j'],
                    'jPrimary' : ['jSolution','F','_jPrimary'],
                    'jSecondary' : ['jSolution','F','_jSecondary'],
                    'h' : ['jSolution','E','_h'],
                    'hPrimary' : ['jSolution','E','_hPrimary'],
                    'hSecondary' : ['jSolution','E','_hSecondary'],
                  }

    def __init__(self,mesh,survey,**kwargs):
        FieldsFDEM.__init__(self,mesh,survey,**kwargs)

    def startup(self):
        self._edgeCurl = self.survey.prob.mesh.edgeCurl
        self._MeMuI = self.survey.prob.MeMuI
        self._MfRho = self.survey.prob.MfRho
        self._curModel = self.survey.prob.curModel

    def _jPrimary(self, jSolution, srcList):
        jPrimary = np.zeros_like(jSolution)
        for i, src in enumerate(srcList):
            jp = src.jPrimary(self.survey.prob) 
            if jp is not None:
                jPrimary[:,i] += jp
        return jPrimary

    def _jSecondary(self, jSolution, srcList):
        return jSolution

    def _j(self, jSolution, srcList):
        return self._jPrimary(jSolution, srcList) + self._jSecondary(jSolution, srcList)

    def _hPrimary(self, jSolution, srcList):
        hPrimary = np.zeros([self._edgeCurl.shape[1],jSolution.shape[1]],dtype = complex)
        for i, src in enumerate(srcList):
            hp = src.hPrimary(self.survey.prob)
            if hp is not None:
                hPrimary[:,i] = hp 
        return hPrimary

    def _hSecondary(self, jSolution, srcList):
        MeMuI = self._MeMuI
        C = self._edgeCurl
        MfRho = self._MfRho
        h =  MeMuI * (C.T * (MfRho * jSolution) ) 
        for i, src in enumerate(srcList):
            h[:,i] *= -1./(1j*omega(src.freq))
            S_m,_ = src.eval(self.survey.prob)
            if S_m is not None:
                h[:,i] += 1./(1j*omega(src.freq)) * MeMuI * S_m
        return h

    def _h(self, jSolution, srcList): 
        return self._hPrimary(jSolution, srcList) + self._hSecondary(jSolution, srcList)

    def _hDeriv(self, jSolution, srcList, v, adjoint=False):
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
    knownFields = {'hSolution':'E'}
    aliasFields = {
                    'h' : ['hSolution','E','_h'],
                    'hPrimary' : ['hSolution','E','_hPrimary'],
                    'hSecondary' : ['hSolution','E','_hSecondary'],
                    'j' : ['hSolution','F','_j'],
                    'jPrimary' : ['hSolution','F','_jPrimary'],
                    'jSecondary' : ['hSolution','F','_jSecondary']
                  }

    def __init__(self,mesh,survey,**kwargs):
        FieldsFDEM.__init__(self,mesh,survey,**kwargs)

    def startup(self):
        self._edgeCurl = self.survey.prob.mesh.edgeCurl
        self._MeMuI = self.survey.prob.MeMuI
        self._MfRho = self.survey.prob.MfRho

    def _hPrimary(self, hSolution, srcList):
        hPrimary = np.zeros_like(hSolution)
        for i, src in enumerate(srcList):
            hp = src.hPrimary(self.survey.prob)
            if hp is not None:
                hPrimary[:,i] += hp
            return hPrimary

    def _hSecondary(self, hSolution, srcList):
        return hSolution

    def _h(self, hSolution, srcList):
        return self._hPrimary(hSolution, srcList) + self._hSecondary(hSolution, srcList)

    def _jPrimary(self, hSolution, srcList):
        jPrimary = np.zeros([self._edgeCurl.shape[0], hSolution.shape[1]])
        for i, src in enumerate(srcList):
            jp = src.jPrimary(self.survey.prob)
            if jp is not None:
                jPrimary[:,i] = jp 
        return jPrimary

    def _jSecondary(self, hSolution, srcList):
        j = self._edgeCurl*hSolution
        for i, src in enumerate(srcList):
            _,S_e = src.eval(self.survey.prob)
            if S_e is not None:
                j[:,i] += -S_e
        return j

    def _j(self, hSolution, srcList):
        return self._jPrimary(hSolution, srcList) + self._jSecondary(hSolution, srcList)

    def _jDeriv(self, hSolution, srcList, v, adjoint=False):
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
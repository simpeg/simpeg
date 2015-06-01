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
                    'b' : ['e_sol','F','_b']
                  }

    def __init__(self,mesh,survey,**kwargs):
        FieldsFDEM.__init__(self,mesh,survey,**kwargs)

    def startup(self):
        self._edgeCurl = self.survey.prob.mesh.edgeCurl

    def _e(self, e_sol, srcList):
        e = e_sol
        for i, src in enumerate(srcList):
            e_p = src.e_p(self.survey.prob)
            if e_p is not None:
                e[:,i] += e_p     
        return e

    def _b(self, e_sol, srcList):
        C = self._edgeCurl
        b = (C * e_sol)
        for i, src in enumerate(srcList):
            b[:,i] *= - 1./(1j*omega(src.freq))
            S_m, _ = src.eval(self.survey.prob)
            if S_m is not None:
                b[:,i] += 1./(1j*omega(src.freq)) * S_m

            b_p = src.b_p(self.survey.prob)
            if b_p is not None:
                b[:,i] += b_p 

        return b

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
                    'e' : ['b_sol','E','_e']
                  }

    def __init__(self,mesh,survey,**kwargs):
        FieldsFDEM.__init__(self,mesh,survey,**kwargs)

    def startup(self):
        self._edgeCurl = self.survey.prob.mesh.edgeCurl
        self._MeSigmaI = self.survey.prob.MeSigmaI
        self._MfMui = self.survey.prob.MfMui

    def _b(self, b_sol, srcList):
        b = b_sol

        for i, src in enumerate(srcList):
            b_p = src.b_p(self.survey.prob)
            if b_p is not None:
                b[:,i] += b_p
        return b  

    def _e(self, b_sol, srcList):
        e = self._MeSigmaI * ( self._edgeCurl.T * ( self._MfMui * b_sol))

        for i,src in enumerate(srcList): 
            _,S_e = src.eval(self.survey.prob)
            if S_e is not None:
                e += -self._MeSigmaI*S_e

            e_p = src.e_p(self.survey.prob)
            if e_p is not None:
                e[:,i] += e_p

        return e

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
                    'h' : ['j_sol','E','_h']
                  }

    def __init__(self,mesh,survey,**kwargs):
        FieldsFDEM.__init__(self,mesh,survey,**kwargs)

    def startup(self):
        self._edgeCurl = self.survey.prob.mesh.edgeCurl
        self._MeMuI = self.survey.prob.MeMuI
        self._MfRho = self.survey.prob.MfRho
        self._curModel = self.survey.prob.curModel

    def _j(self, j_sol, srcList):
        j = j_sol
        for i, src in enumerate(srcList):
            j_p = src.j_p(self.survey.prob) 
            if j_p is not None:
                j[:,i] += j_p
        return j

    def _h(self, j_sol, srcList): 
        MeMuI = self._MeMuI
        C = self._edgeCurl
        MfRho = self._MfRho

        h =  MeMuI * (C.T * (MfRho * j_sol) ) 

        for i, src in enumerate(srcList):
            h[:,i] *= -1./(1j*omega(src.freq))
            S_m,_ = src.eval(self.survey.prob)
            if S_m is not None:
                h[:,i] += 1./(1j*omega(src.freq)) * MeMuI * S_m

            h_p = src.h_p(self.survey.prob)
            if h_p is not None:
                h[:,i] += h_p 

        return h

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
                    'j' : ['h_sol','F','_j']
                  }

    def __init__(self,mesh,survey,**kwargs):
        FieldsFDEM.__init__(self,mesh,survey,**kwargs)

    def startup(self):
        self._edgeCurl = self.survey.prob.mesh.edgeCurl
        self._MeMuI = self.survey.prob.MeMuI
        self._MfRho = self.survey.prob.MfRho

    def _h(self, h_sol, srcList):
        h = h_sol
        for i, src in enumerate(srcList):
            h_p = src.h_p(self.survey.prob)
            if h_p is not None:
                h[:,i] += h_p
            return h

    def _j(self, h_sol, srcList):
        j = self._edgeCurl*h_sol
        for i, src in enumerate(srcList):
            _,S_e = src.eval(self.survey.prob)
            if S_e is not None:
                j[:,i] += -S_e

            j_p = src.j_p(self.survey.prob)
            if j_p is not None:
                j[:,i] += j_p 
        return j

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
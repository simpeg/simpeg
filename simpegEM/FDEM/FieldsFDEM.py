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
                    'b_sec' : ['e_sol','F','_b_sec'],
                    'b' : ['e_sol','F','_b']
                  }

    def __init__(self,mesh,survey,**kwargs):
        FieldsFDEM.__init__(self,mesh,survey,**kwargs)

    def startup(self):
        self._edgeCurl = self.survey.prob.mesh.edgeCurl

    def _e(self, e_sol, src):
        e = e_sol
        e_p = src.e_p(self.survey.prob)
        if e_p is not None:
            e += e_p     
        return e

    def _b_sec(self, e_sol, src):
        C = self._edgeCurl
        b_sec =  - 1./(1j*omega(src.freq))*(C * e_sol)
        return b_sec

    def _b_secDeriv(self,e_sol, src, v, adjoint=False): 
        return None

    def _b(self, e_sol, src):
        b = self._b_sec(e_sol, src)
        S_m, _ = src.eval(self.survey.prob)
        if S_m is not None:
            b += 1./(1j*omega(src.freq)) * S_m

        b_p = src.b_p(self.survey.prob)
        if b_p is not None:
            b += b_p 

        return b

    def _bDeriv(self, e, src, v, adjoint=False):
        S_mDeriv,_ = src.getSourceDeriv(self.survey.prob, v, adjoint)
        b_secDeriv = self._b_secDeriv(e, src.freq, v, adjoint)
        if S_mDeriv is None & b_secDeriv is None:
            return None
        elif b_secDeriv is None:
            return 1./(1j*omega(src.freq)) * S_mDeriv
        elif S_mDeriv is None:
            return b_secDeriv
        else:
            return 1./(1j*omega(src.freq)) * S_mDeriv + b_secDeriv


class FieldsFDEM_b(FieldsFDEM):
    knownFields = {'b_sol':'F'}
    aliasFields = {
                    'b' : ['b_sol','F','_b'],
                    'e_sec' : ['b_sol','E','_e_sec'],
                    'e' : ['b_sol','E','_e']
                  }

    def __init__(self,mesh,survey,**kwargs):
        FieldsFDEM.__init__(self,mesh,survey,**kwargs)

    def startup(self):
        self._edgeCurl = self.survey.prob.mesh.edgeCurl
        self._MeSigmaI = self.survey.prob.MeSigmaI
        self._MfMui = self.survey.prob.MfMui
        # self._getSource = self.survey.prob.getSource
        # self._getSourceDeriv = self.survey.prob.getSourceDeriv 

    def _b(self, b_sol, src):
        b = b_sol
        b_p = src.b_p(self.survey.prob)
        if b_p is not None:
            b += b_p
        return b  

    def _e_sec(self, b_sol, src):
        return self._MeSigmaI * ( self._edgeCurl.T * ( self._MfMui * b_sol))

    def _e_secDeriv(self, b_sol, src, v, adjoint=False):
        return None

    def _e(self, b_sol, src):
        e = self._e_sec(b_sol,src)
        _,S_e = src.eval(self.survey.prob)
        if S_e is not None:
            e += -self._MeSigmaI*S_e

        e_p = src.e_p(self.survey.prob)
        if e_p is not None:
            e += e_p

        return e

    def _eDeriv(self, b_sol, src, v, adjoint=False):
        _,S_eDeriv = src.getSourceDeriv(self.survey.prob, v, adjoint)
        e_secDeriv = self._e_secDeriv(b_sol, src, v, adjoint)

        if S_eDeriv is None & e_secDeriv is None:
            return None
        elif e_secDeriv is None:
            return -S_eDeriv
        elif S_eDeriv is None:
            return e_secDeriv
        else:
            return e_secDeriv - S_eDeriv


class FieldsFDEM_j(FieldsFDEM):
    knownFields = {'j_sol':'F'}
    aliasFields = {
                    'j' : ['j_sol','F','_j'],
                    'h_sec' : ['j_sol','E','_h_sec'],
                    'h' : ['j_sol','E','_h']
                  }

    def __init__(self,mesh,survey,**kwargs):
        FieldsFDEM.__init__(self,mesh,survey,**kwargs)

    def startup(self):
        self._edgeCurl = self.survey.prob.mesh.edgeCurl
        self._MeMuI = self.survey.prob.MeMuI
        self._MfSigmai = self.survey.prob.MfSigmai
        # self._getSource = self.survey.prob.getSource
        # self._getSourceDeriv = self.survey.prob.getSourceDeriv 
        self._curModel = self.survey.prob.curModel

    def _j(self, j_sol, src):
        j = j_sol
        j_p = src.j_p(self.survey.prob) 
        if j_p is not None:
            j += j_p
        return j

    def _h_sec(self, j_sol, src): #v, adjoint=False
        return - 1./(1j*omega(src.freq)) * self._MeMuI * (self._edgeCurl.T * (self._MfSigmai * j_sol) ) 

    def _h_secDeriv(self, j_sol, src, v, adjoint=False): 
        MeMuI = self._MeMuI
        C = self._edgeCurl
        sig = self._curModel.transform
        sigi = 1/sig
        dsig_dm = self._curModel.transformDeriv
        dsigi_dsig = -Utils.sdiag(sigi)**2
        dMf_dsigi = self.mesh.getFaceInnerProductDeriv(sigi)(j)
        sigi = self._MfSigmai
        if not adjoint:
            return -(1./(1j*omega(freq))) * MeMuI * ( C.T * ( dMf_dsigi * ( dsigi_dsig * ( dsig_dm * v ) ) ) )
        else:
            return -(1./(1j*omega(freq))) * dsig_dm.T * ( dsigi_dsig.T * ( dMf_dsigi.T * ( C * ( MeMuI.T * v ) ) ) )

    def _h(self, j_sol, src): #v, adjoint=False
        h = self._h_sec(j_sol,src)
        S_m,_ = src.eval(self.survey.prob)
        if S_m is not None:
            h += 1./(1j*omega(src.freq)) * self._MeMuI * S_m
        h_p = src.h_p(self.survey.prob)
        if h_p is not None:
            h += h_p 
        return h

    def _hDeriv(self, j_sol, src, v, adjoint=False):
        S_mDeriv,_ = src.getSourceDeriv(self.survey.prob, v, adjoint)
        h_secDeriv = self._h_secDeriv(j_sol,src.freq, v, adjoint)
        if S_mDeriv is None & h_secDeriv is None:
            return None
        elif h_secDeriv is None:
            return 1./(1j*omega(src.freq)) * S_mDeriv
        elif S_mDeriv is None:
            return h_secDeriv
        else:
            return 1./(1j*omega(src.freq)) * S_mDeriv + h_secDeriv


class FieldsFDEM_h(FieldsFDEM):
    knownFields = {'h_sol':'E'}
    aliasFields = {
                    'h' : ['h_sol','E','_h'],
                    'j_sec' : ['h_sol','F','_j_sec'],
                    'j' : ['h_sol','F','_j']
                  }

    def __init__(self,mesh,survey,**kwargs):
        FieldsFDEM.__init__(self,mesh,survey,**kwargs)

    def startup(self):
        self._edgeCurl = self.survey.prob.mesh.edgeCurl
        self._MeMuI = self.survey.prob.MeMuI
        self._MfSigmai = self.survey.prob.MfSigmai

    def _h(self, h_sol, src):
        h = h_sol
        h_p = src.h_p(self.survey.prob)
        if h_p is not None:
            h += h_p
        return h

    def _j_sec(self, h_sol, src): # adjoint=False
        return self._edgeCurl*h_sol

    def _j_secDeriv(self, h_sol, src, v, adjoint=False): 
        return None

    def _j(self, h_sol, src): # adjoint=False
        j = self._j_sec(h_sol,src)
        _,S_e = src.eval(self.survey.prob)
        if S_e is not None:
            j += -S_e
        j_p = src.j_p(self.survey.prob)
        if j_p is not None:
            j += j_p 
        return j

    def _jDeriv(self, h_sol, src, v, adjoint=False):
        _,S_eDeriv = src.getSourceDeriv(self.survey.prob, v, adjoint)
        j_secDeriv = self._j_secDeriv(j,src.freq, v, adjoint)
        if S_eDeriv is None & j_secDeriv is None:
            return None
        elif j_secDeriv is None:
            return - S_eDeriv
        elif S_eDeriv is None:
            return j_secDeriv
        else:
            return - S_eDeriv + j_secDeriv


    # def calcFields(self, sol, freq, fieldType, adjoint=False):
    #     j = sol
    #     if fieldType == 'j':
    #         return j
    #     elif fieldType == 'h':
    #         MeMuI = self._MeMuI
    #         C = self.mesh.edgeCurl
    #         MfSigmai = self._MfSigmai
    #         if not adjoint:
    #             h = -(1./(1j*omega(freq))) * MeMuI * ( C.T * ( MfSigmai * j ) )
    #         else:
    #             h = -(1./(1j*omega(freq))) * MfSigmai.T * ( C * ( MeMuI.T * j ) )
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
    #         sigi = self._MfSigmai
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
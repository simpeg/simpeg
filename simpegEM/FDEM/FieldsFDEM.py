from SimPEG import Survey, Problem, Utils, np, sp 
from simpegEM.Utils.EMUtils import omega


class FieldsFDEM(Problem.Fields):
    """Fancy Field Storage for a FDEM survey."""
    knownFields = {}
    dtype = complex


class FieldsFDEM_e(FieldsFDEM):
    knownFields = {'e':'E'}
    aliasFields = {
                    'b_sec' : ['e','F','_b_sec'],
                    'b' : ['e','F','_b']
                  }

    def __init__(self,mesh,survey,**kwargs):
        FieldsFDEM.__init__(self,mesh,survey,**kwargs)

    def startup(self):
        self.edgeCurl = self.survey.prob.mesh.edgeCurl
        self.getSource = self.survey.prob.getSource
        self.getSourceDeriv = self.survey.prob.getSourceDeriv 

    def _b_sec(self, e, tx): #adjoint=False
        return - 1./(1j*omega(tx.freq)) * (self.edgeCurl * e)

    def _b_secDeriv(self, e, tx, v, adjoint=False): 
        return None

    def _b(self, e, tx): #adjoint=False
        b_sec = self._b_sec(e,tx)
        S_m,_ = self.getSource(tx.freq)
        return b_sec + 1./(1j*omega(tx.freq)) * S_m

    def _bDeriv(self, e, tx, v, adjoint=False):
        S_mDeriv,_ = self.getSourceDeriv(tx.freq, v, adjoint)
        b_secDeriv = self._b_secDeriv(e, tx.freq, v, adjoint)
        if S_mDeriv is None & b_secDeriv is None:
            return None
        elif b_secDeriv is None:
            return 1./(1j*omega(tx.freq)) * S_mDeriv
        elif S_mDeriv is None:
            return b_secDeriv
        else:
            return 1./(1j*omega(tx.freq)) * S_mDeriv + b_secDeriv


class FieldsFDEM_b(FieldsFDEM):
    knownFields = {'b':'F'}
    aliasFields = {
                    'e_sec' : ['b','E','_e_sec'],
                    'e' : ['b','E','_e']
                  }

    def __init__(self,mesh,survey,**kwargs):
        FieldsFDEM.__init__(self,mesh,survey,**kwargs)

    def startup(self):
        self.edgeCurl = self.survey.prob.mesh.edgeCurl
        self.MeSigmaI = self.survey.prob.MeSigmaI
        self.MfMui = self.survey.prob.MfMui
        self.getSource = self.survey.prob.getSource
        self.getSourceDeriv = self.survey.prob.getSourceDeriv 

    def _e_sec(self, b, tx):
        return self.MeSigmaI * ( self.edgeCurl.T * ( self.MfMui * b) )

    def _e_secDeriv(self, b, tx, v, adjoint=False):
        return None

    def _e(self, b, tx):
        e_sec = self._e_sec(b,tx)
        _, S_e = self.getSource(tx.freq)
        return e_sec + S_e

    def _eDeriv(self, b, tx, v, adjoint=False):
        _,S_eDeriv = self.getSourceDeriv(tx.freq, v, adjoint)
        e_secDeriv = self._e_secDeriv(b, tx, v, adjoint)

        if S_eDeriv is None & e_secDeriv is None:
            return None
        elif e_secDeriv is None:
            return -S_eDeriv
        elif S_eDeriv is None:
            return e_secDeriv
        else:
            return e_secDeriv - S_eDeriv


class FieldsFDEM_j(FieldsFDEM):
    knownFields = {'j':'F'}
    aliasFields = {
                    'h_sec' : ['j','E','_h_sec'],
                    'h' : ['j','E','_h']
                  }

    def __init__(self,mesh,survey,**kwargs):
        FieldsFDEM.__init__(self,mesh,survey,**kwargs)

    def startup(self):
        self.edgeCurl = self.survey.prob.mesh.edgeCurl
        self.MeMuI = self.survey.prob.MeMuI
        self.MfSigmai = self.survey.prob.MfSigmai
        self.getSource = self.survey.prob.getSource
        self.getSourceDeriv = self.survey.prob.getSourceDeriv 
        self.curModel = self.survey.prob.curModel

    def _h_sec(self, j, tx): #v, adjoint=False
        return - 1./(1j*omega(tx.freq)) * self.MeMuI * (self.edgeCurl.T * (self.MfSigmai * j) ) 

    def _h_secDeriv(self, j, tx, v, adjoint=False): 
        MeMuI = self.MeMuI
        C = self.edgeCurl
        sig = self.curModel.transform
        sigi = 1/sig
        dsig_dm = self.curModel.transformDeriv
        dsigi_dsig = -Utils.sdiag(sigi)**2
        dMf_dsigi = self.mesh.getFaceInnerProductDeriv(sigi)(j)
        sigi = self.MfSigmai
        if not adjoint:
            return -(1./(1j*omega(freq))) * MeMuI * ( C.T * ( dMf_dsigi * ( dsigi_dsig * ( dsig_dm * v ) ) ) )
        else:
            return -(1./(1j*omega(freq))) * dsig_dm.T * ( dsigi_dsig.T * ( dMf_dsigi.T * ( C * ( MeMuI.T * v ) ) ) )

    def _h(self, j, tx): #v, adjoint=False
        h_sec = self._h_sec(j,tx)
        S_m,_ = self.getSource(tx.freq)
        return h_sec + 1./(1j*omega(tx.freq)) * self.MeMuI * S_m

    def _hDeriv(self, j, tx, v, adjoint=False):
        S_mDeriv,_ = self.getSourceDeriv(tx.freq, v, adjoint)
        h_secDeriv = self._h_secDeriv(j,tx.freq, v, adjoint)
        if S_mDeriv is None & h_secDeriv is None:
            return None
        elif h_secDeriv is None:
            return 1./(1j*omega(tx.freq)) * S_mDeriv
        elif S_mDeriv is None:
            return h_secDeriv
        else:
            return 1./(1j*omega(tx.freq)) * S_mDeriv + h_secDeriv

class FieldsFDEM_h(FieldsFDEM):
    knownFields = {'h':'E'}
    aliasFields = {
                    'j_sec' : ['h','F','_j_sec'],
                    'j' : ['h','F','_j']
                  }

    def __init__(self,mesh,survey,**kwargs):
        FieldsFDEM.__init__(self,mesh,survey,**kwargs)

    def startup(self):
        self.edgeCurl = self.survey.prob.mesh.edgeCurl
        self.MeMuI = self.survey.prob.MeMuI
        self.MfSigmai = self.survey.prob.MfSigmai
        self.getSource = self.survey.prob.getSource
        self.getSourceDeriv = self.survey.prob.getSourceDeriv 

    def _j_sec(self, h, tx): # adjoint=False
        return self.edgeCurl*h

    def _j_secDeriv(self, h, tx, v, adjoint=False): 
        return None

    def _j(self, h, tx): # adjoint=False
        j_sec = self._j_sec(h,tx)
        _,S_e = self.getSource(tx.freq)
        return j_sec - S_e

    def _jDeriv(self, h, tx, v, adjoint=False):
        _,S_eDeriv = self.getSourceDeriv(tx.freq, v, adjoint)
        j_secDeriv = self._j_secDeriv(j,tx.freq, v, adjoint)
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
    #         MeMuI = self.MeMuI
    #         C = self.mesh.edgeCurl
    #         MfSigmai = self.MfSigmai
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
    #         MeMuI = self.MeMuI
    #         C = self.mesh.edgeCurl
    #         sig = self.curModel.transform
    #         sigi = 1/sig
    #         dsig_dm = self.curModel.transformDeriv
    #         dsigi_dsig = -Utils.sdiag(sigi)**2
    #         dMf_dsigi = self.mesh.getFaceInnerProductDeriv(sigi)(j)
    #         sigi = self.MfSigmai
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
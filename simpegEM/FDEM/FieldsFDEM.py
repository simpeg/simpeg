from SimPEG import Survey, Problem, Utils, np, sp 
from simpegEM.Utils.EMUtils import omega


class FieldsFDEM(Problem.Fields):
	"""Fancy Field Storage for a FDEM survey."""
	knownFields = {'b': 'F', 'e': 'E', 'j': 'F', 'h': 'E'} # TODO: a, phi
	dtype = complex

	def calcFields(self,sol,txInd,freqInd,fieldType):
		if fieldType == 'e':
			return self._e(sol,txInd,freqInd)
		elif fieldType == 'e_sec':
			return self._e_sec(sol,txInd,freqInd)
		elif fieldType == 'b':
			return self._b(sol,txInd,freqInd)
		elif fieldType == 'b_sec':
			return self._b_sec(sol,txInd,freqInd)
		else:
			raise NotImplementedError('fieldType "%s" is not implemented.' % fieldType)

	def calcFieldsDeriv(self,sol,txInd,freqInd,fieldType,adjoint=False):
		if fieldType == 'e':
			return self._eDeriv(sol,txInd,freqInd,adjoint)
		elif fieldType == 'e_sec':
			return self._e_secDeriv(sol,txInd,freqInd,adjoint)
		elif fieldType == 'b':
			return self._bDeriv(sol,txInd,freqInd,adjoint,adjoint)
		elif fieldType == 'b_sec':
			return self._b_secDeriv(sol,txInd,freqInd,adjoint,adjoint)
		else:
			raise NotImplementedError('fieldType "%s" is not implemented.' % fieldType)



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
		self.freqs = self.survey.freqs
		self.getSource = self.survey.prob.getSource
		self.getSourceDeriv = self.survey.prob.getSourceDeriv 

	def _e(self, e, txInd, freqInd):
		return e

	def _eDeriv(self, e, txInd, freqInd, adjoint=False):
		return None

	def _b_sec(self, e, txInd, freqInd): #adjoint=False
		iomegainv = 1./(1j*omega(self.freqs[freqInd]))
		return -iomegainv * (self.edgeCurl * e)

	def _b_secDeriv(self, e, txInd, freqInd, adjoint=False): 
		return None

	def _b(self, e, txInd, freqInd): #adjoint=False
		freq = self.freqs[freqInd]
		b_sec = self._bsec(e,txInd,freqInd)
		j_m,_ = self.getSource(freq)
		return 1./(1j*omega(freq)) + b_sec

	def _bDreiv(self, e, txInd, freqInd, adjoint=False):
		freq = self.freqs[freqInd]
		j_mDeriv,_ = self.getSourceDeriv(freq, adjoint)
		if j_mDeriv is None:
			return None
		else:
			return 1./(1j*omega(freq)) * j_mDeriv


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
		self.freqs = self.survey.freqs
		self.getSource = self.survey.prob.getSource
		self.getSourceDeriv = self.survey.prob.getSourceDeriv 

	def _b(self, b, txInd, freqInd):
		return b

	def _bDeriv(self, b, txInd, freqInd, adjoint=False):
		return None

	def _e_sec(self, b, txInd, freqInd):
		return self.MeSigmaI * ( self.edgeCurl.T * ( self.MfMui * b) )

	def _e_secDeriv(self, b, txInd, freqInd, adjoint=False):
		return None

	def _e(self, b, txInd, freqInd):
		e_sec = _e_sec(self, b, txInd, freqInd)
		_, j_g = self.getSource(self.freqs[freqInd])
		return e_s - j_g

	def _eDeriv(self, b, txInd, freqInd, adjoint=False):
		_,j_gDeriv = self.getSourceDeriv(self.freqs[freqInd], adjoint)
		if j_gDeriv is None:
			return None
		else:
			return -j_gDeriv	
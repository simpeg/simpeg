from SimPEG import Survey, Problem, Utils, np, sp 
from simpegEM.Utils.EMUtils import omega


class FieldsFDEM(Problem.Fields):
	"""Fancy Field Storage for a FDEM survey."""
	# knownFields = {'b': 'F', 'e': 'E', 'b_sec' : 'F', 'e_sec':'E' ,'j': 'F', 'h': 'E'} # TODO: a, phi
	dtype = complex

	# def calcFields(self,sol,tx,fieldType):
	# 	if fieldType == 'e':
	# 		return self._e(sol,tx)
	# 	elif fieldType == 'e_sec':
	# 		return self._e_sec(sol,tx)
	# 	elif fieldType == 'b':
	# 		return self._b(sol,tx)
	# 	elif fieldType == 'b_sec':
	# 		return self._b_sec(sol,tx)
	# 	else:
	# 		raise NotImplementedError('fieldType "%s" is not implemented.' % fieldType)

	# def calcFieldsDeriv(self,sol,tx,fieldType,adjoint=False):
	# 	if fieldType == 'e':
	# 		return self._eDeriv(sol,tx,adjoint)
	# 	elif fieldType == 'e_sec':
	# 		return self._e_secDeriv(sol,tx,adjoint)
	# 	elif fieldType == 'b':
	# 		return self._bDeriv(sol,tx,adjoint)
	# 	elif fieldType == 'b_sec':
	# 		return self._b_secDeriv(sol,tx,adjoint)
	# 	else:
	# 		raise NotImplementedError('fieldType "%s" is not implemented.' % fieldType)



class FieldsFDEM_e(FieldsFDEM):
	knownFields = {'e':'E'}
	aliasFields = {
					'b_sec' : ['e','F','_b_sec'],
					'b' : ['b_sec','F','_b']
				  }

	def __init__(self,mesh,survey,**kwargs):
		FieldsFDEM.__init__(self,mesh,survey,**kwargs)

	def startup(self):
		self.edgeCurl = self.survey.prob.mesh.edgeCurl
		self.getSource = self.survey.prob.getSource
		self.getSourceDeriv = self.survey.prob.getSourceDeriv 

	def calcFields(self,sol,tx,fieldType):
		if fieldType == 'e':
			return self._e(sol,tx)
		elif fieldType == 'b':
			return self._b(sol,tx)
		elif fieldType == 'b_sec':
			return self._b_sec(sol,tx)
		else:
			raise NotImplementedError('fieldType "%s" is not implemented.' % fieldType)

	def calcFieldsDeriv(self,sol,tx,fieldType,adjoint=False):
		if fieldType == 'e':
			return self._eDeriv(sol,tx,adjoint)
		elif fieldType == 'b':
			return self._bDeriv(sol,tx,adjoint)
		elif fieldType == 'b_sec':
			return self._b_secDeriv(sol,tx,adjoint)
		else:
			raise NotImplementedError('fieldType "%s" is not implemented.' % fieldType)

	def _e(self, e, tx):
		return e

	def _eDeriv(self, e, tx, adjoint=False):
		return None

	def _b_sec(self, e, tx): #adjoint=False
		iomegainv = 1./(1j*omega(tx.freq))
		return -iomegainv * (self.edgeCurl * e)

	def _b_secDeriv(self, e, tx, adjoint=False): 
		return None

	def _b(self, b_sec, tx): #adjoint=False
		j_m,_ = self.getSource(tx.freq)
		return 1./(1j*omega(tx.freq)) + b_sec

	def _bDreiv(self, e, tx, adjoint=False):
		j_mDeriv,_ = self.getSourceDeriv(tx.freq, adjoint)
		if j_mDeriv is None:
			return None
		else:
			return 1./(1j*omega(tx.freq)) * j_mDeriv


class FieldsFDEM_b(FieldsFDEM):
	knownFields = {'b':'F'}
	aliasFields = {
					'e_sec' : ['b','E','_e_sec'],
					'e' : ['e_sec','E','_e']
				  }

	def __init__(self,mesh,survey,**kwargs):
		FieldsFDEM.__init__(self,mesh,survey,**kwargs)

	def startup(self):
		self.edgeCurl = self.survey.prob.mesh.edgeCurl
		self.MeSigmaI = self.survey.prob.MeSigmaI
		self.MfMui = self.survey.prob.MfMui
		self.getSource = self.survey.prob.getSource
		self.getSourceDeriv = self.survey.prob.getSourceDeriv 

	def calcFields(self,sol,tx,fieldType):
		if fieldType == 'e':
			return self._e(sol,tx)
		elif fieldType == 'e_sec':
			return self._e_sec(sol,tx)
		elif fieldType == 'b':
			return self._b(sol,tx)
		else:
			raise NotImplementedError('fieldType "%s" is not implemented.' % fieldType)

	def calcFieldsDeriv(self,sol,tx,fieldType,adjoint=False):
		if fieldType == 'e':
			return self._eDeriv(sol,tx,adjoint)
		elif fieldType == 'e_sec':
			return self._e_secDeriv(sol,tx,adjoint)
		elif fieldType == 'b':
			return self._bDeriv(sol,tx,adjoint)
		else:
			raise NotImplementedError('fieldType "%s" is not implemented.' % fieldType)

	def _b(self, b, tx):
		return b

	def _bDeriv(self, b, tx, adjoint=False):
		return None

	def _e_sec(self, b, tx):
		return self.MeSigmaI * ( self.edgeCurl.T * ( self.MfMui * b) )

	def _e_secDeriv(self, b, tx, adjoint=False):
		return None

	def _e(self, e_sec, tx):
		_, j_g = self.getSource(tx.freq)
		return e_s - j_g

	def _eDeriv(self, b, tx, adjoint=False):
		_,j_gDeriv = self.getSourceDeriv(tx.freq, adjoint)
		if j_gDeriv is None:
			return None
		else:
			return -j_gDeriv	
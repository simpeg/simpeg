from SimPEG import Survey, Problem, Utils, np, sp 
from simpegEM.Utils.EMUtils import omega


class FieldsFDEM(Problem.Fields):
	"""Fancy Field Storage for a FDEM survey."""
	knownFields = None
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
					'b' : ['e','F','_b']
				  }

	def __init__(self,mesh,survey,**kwargs):
		FieldsFDEM.__init__(self,mesh,survey,**kwargs)

	def startup(self):
		self.edgeCurl = self.survey.prob.mesh.edgeCurl
		self.getSource = self.survey.prob.getSource
		self.getSourceDeriv = self.survey.prob.getSourceDeriv 

	# def _e(self, e, tx):
	# 	return e

	# def _eDeriv(self, e, tx, adjoint=False):
	# 	return None

	def _b_sec(self, e, tx): #adjoint=False
		return - 1./(1j*omega(tx.freq)) * (self.edgeCurl * e)

	def _b_secDeriv(self, e, tx, adjoint=False): 
		return None

	def _b(self, e, tx): #adjoint=False
		b = self._b_sec(e,tx)
		print b.shape
		j_m,_ = self.getSource(tx.freq)
		if j_m[0] is not None:
			b += 1./(1j*omega(tx.freq)) * np.array([j_m[0]]).T
		return b

	def _bDreiv(self, e, tx, adjoint=False):
		j_mDeriv,_ = self.getSourceDeriv(tx.freq, adjoint)
		b_secDeriv = self._b_secDeriv(e,tx.freq,adjoint)
		if j_mDeriv is None & b_secDeriv is None:
			return None
		elif b_secDeriv is None:
			return 1./(1j*omega(tx.freq)) * j_mDeriv
		elif j_mDeriv is None:
			return b_secDeriv
		else:
			return 1./(1j*omega(tx.freq)) * j_mDeriv + b_secDeriv


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

	# def _b(self, b, tx):
	# 	return b

	# def _bDeriv(self, b, tx, adjoint=False):
	# 	return None

	def _e_sec(self, b, tx):
		return self.MeSigmaI * ( self.edgeCurl.T * ( self.MfMui * b) )

	def _e_secDeriv(self, b, tx, adjoint=False):
		return None

	def _e(self, b, tx):
		e = self._e_sec(b,tx)
		_, j_g = self.getSource(tx.freq)
		if j_g[0] is not None:
			e += -np.array([j_g[0]]).T
		return e

	def _eDeriv(self, b, tx, adjoint=False):
		_,j_gDeriv = self.getSourceDeriv(tx.freq, adjoint)
		e_secDeriv = self._e_secDeriv(b,tx,adjoint)

		if j_gDeriv is None & e_secDeriv is None:
			return None
		elif e_secDeriv is None:
			return -j_gDeriv
		elif j_gDeriv is None:
			return e_secDeriv
		else:
			return e_secDeriv - j_gDeriv
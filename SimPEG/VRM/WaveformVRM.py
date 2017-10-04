import numpy as np




###################################################
# 			STEP OFF WAVEFORM
###################################################

class StepOff():

	def __init__(self, t0 = 0.):

		self.t0 = t0

	def getCharDecay(self, dtype, times):

		assert dtype in ["dhdt","dbdt"], "For step-off, dtype must be one of 'dhdt' or 'dbdt' and cannot be 'h' or 'b'"
		assert self.t0 < np.min(times), "Earliest time channel must be after beginning of off-time"
		
		t0 = self.t0

		if dtype is "dbdt":
			mu0 = 4*np.pi*1e-7
			eta = -1/(times-t0)
		elif dtype is "dhdt":
			eta = -1/(times-t0)

		return eta

	# def getTrueDecay(self, dtype, times, chi0, dchi, tau1, tau2):



###################################################
# 			SQUARE PULSE WAVEFORM
###################################################


class SquarePulse():

	def __init__(self, delt, t0 = 0.):

		self.delt = delt
		self.t0 = t0

	def getCharDecay(self, dtype, times):

		assert dtype in ["h","dhdt","b","dbdt"], "For square-pulse, dtype must be one of 'h', 'dhdt', 'b' or 'dbdt'"
		assert self.t0 < np.min(times), "Earliest time channel must be after beginning of off-time"
		
		t0 = self.t0
		delt = self.delt
		mu0 = 4*np.pi*1e-7

		if dtype is "h":
			eta = np.log(1 + delt/(times-t0))
		elif dtype is "b":
			eta = mu0*np.log(1 + delt/(times-t0))
		elif dtype is "dhdt":
			eta = -(1/(times-t0) - 1/(times-t0+delt))
		elif dtype is "dbdt":
			eta = -mu0*(1/(times-t0) - 1/(times-t0+delt))

		return eta





###################################################
# 			ARBITRARY WAVEFORM
###################################################

class Arbitrary():

	def __init__(self, t, I):

		assert I[0] < 1e-10 and I[-1] < 1e-10, "Current at t0 and tmax should be 0"
		assert len(t) == len(I), "Time values and current values must have same length"

		self.t = t
		self.I = I

	def getCharDecay(self, dtype, times):

		assert dtype in ["h","dhdt","b","dbdt"], "dtype must be one of 'h', 'dhdt', 'b' or 'dbdt'"
		assert np.max(self.t) < np.min(times), "Earliest time channel must be after beginning of off-time"

		k = np.nonzero(self.I)
		j = k[0][0]
		k = k[0][-1]

		t = self.t[j:k]
		I = self.I[j:k]/np.max(np.abs(self.I[j:k]))

		N = int(np.ceil(10*(np.max(t)-np.min(t))/np.min(times)))

		if N > 10000:
			N = 10000

		dt = (np.max(t) - np.min(t))/N
		tvec = np.linspace(np.min(t),np.max(t)-dt,N)
		g = np.interp(tvec+dt/2,t,I) # g evaluated at middle of pulses

		eta = np.zeros(len(times))

		if dtype in ["h","b"]:
			for tt in range(0,len(eta)):
				eta[tt] = np.sum(g*np.log(1 + dt/(times[tt] - t + dt)))
		elif dtype in ["dhdt","dbdt"]:
			for tt in range(0,len(eta)):
				eta[tt] = np.sum(g*(1/(times[tt] - t + dt) - 1/(times[tt] - t)))

		if dtype in ["b","dbdt"]:
			mu0 = 4*np.pi*1e-7
			eta = mu0*eta

		return eta












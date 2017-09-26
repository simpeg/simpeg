from SimPEG import Problem
from SimPEG.VRM.SurveyVRM import SurveyVRM
# from SimPEG.VRM.FieldsVRM import Fields_LinearFWD
import numpy as np
import scipy.sparse as sp







############################################
# BASE VRM PROBLEM CLASS
############################################

class BaseProblemVRM(Problem.BaseProblem):

	def __init__(self, mesh, **kwargs):

		assert len(mesh.h) == 3, 'Problem requires 3D tensor or OcTree mesh'

		refFact = 3
		refRadius = 1.25*np.mean(np.r_[np.min(mesh.h[0]),np.min(mesh.h[1]),np.min(mesh.h[2])])*np.r_[1.,2.,3.]

		assert len(refRadius) == refFact, 'Number of refinement radii must equal refinement factor'

		super(BaseProblemVRM,self).__init__(mesh, **kwargs)


		self.surveyPair = SurveyVRM
		self.refFact = refFact
		self.refRadius = refRadius
		self.A = None
		self.Tb = None
		self.Tdbdt = None

	def getH0matrix(self, xyz, pp):

		# Creates sparse matrix containing inducing field components for source pp
		# 
		# INPUTS
		# xyz: [x,y,z] list of locations to predict field
		# pp: Source index

		SrcObj = self.survey.srcList[pp]

		H0 = SrcObj.getH0(xyz)

		Hx0 = sp.diags(H0[0])
		Hy0 = sp.diags(H0[1])
		Hz0 = sp.diags(H0[1])

		H0 = sp.vstack([Hx0,Hy0,Hz0])

		return H0

	def getGeometryMatrix(self, xyzc, xyzh, pp):

		# Creates dense geometry matrix mapping from magentized voxel cells to the receivers for source pp
		#
		# INPUTS:
		# xyzc: N by 3 array containing cell center locations
		# xyzh: [hx,hy,hz] list containing cell dimensions
		# pp: Source index

		srcObj = self.survey.srcList[pp]

		N = np.shape(xyzc)[0] # Number of cells
		K = srcObj.nD # Number of receiver in all rxList

		ax = np.reshape(xyzc[:,0] - xyzh[0]/2, (1,N))
		bx = np.reshape(xyzc[:,0] + xyzh[0]/2, (1,N))
		ay = np.reshape(xyzc[:,1] - xyzh[1]/2, (1,N))
		by = np.reshape(xyzc[:,1] + xyzh[1]/2, (1,N))
		az = np.reshape(xyzc[:,2] - xyzh[2]/2, (1,N))
		bz = np.reshape(xyzc[:,2] + xyzh[2]/2, (1,N))

		G = np.zeros((K,3*N))
		C = -(1/(4*np.pi))
		eps = 1e-10

		COUNT = 0

		for qq in range(0,len(srcObj.rxList)):

			rxObj = srcObj.rxList[qq]
			dComp = rxObj.fieldComp
			locs = rxObj.locs
			M = np.shape(locs)[0]

			if dComp is 'x':
				for rr in range(0,M):
					u1 = locs[rr,0] - ax
					u1[u1==0] =  np.min(xyzh[:,0])/1000 
					u2 = locs[rr,0] - bx 
					u2[u2==0] = -np.min(xyzh[:,0])/1000 
					v1 = locs[rr,1] - ay 
					v1[v1==0] =  np.min(xyzh[:,1])/1000 
					v2 = locs[rr,1] - by 
					v2[v2==0] = -np.min(xyzh[:,1])/1000 
					w1 = locs[rr,2] - az 
					w1[w1==0] =  np.min(xyzh[:,2])/1000 
					w2 = locs[rr,2] - bz 
					w2[w2==0] = -np.min(xyzh[:,2])/1000 

					Gxx = np.arctan((v1*w1)/(u1*np.sqrt(u1**2+v1**2+w1**2)+eps)) \
					- np.arctan((v1*w1)/(u2*np.sqrt(u2**2+v1**2+w1**2)+eps)) \
					+ np.arctan((v2*w1)/(u2*np.sqrt(u2**2+v2**2+w1**2)+eps)) \
					- np.arctan((v2*w1)/(u1*np.sqrt(u1**2+v2**2+w1**2)+eps)) \
					+ np.arctan((v2*w2)/(u1*np.sqrt(u1**2+v2**2+w2**2)+eps)) \
					- np.arctan((v1*w2)/(u1*np.sqrt(u1**2+v1**2+w2**2)+eps)) \
					+ np.arctan((v1*w2)/(u2*np.sqrt(u2**2+v1**2+w2**2)+eps)) \
					- np.arctan((v2*w2)/(u2*np.sqrt(u2**2+v2**2+w2**2)+eps))

					Gyx = log(np.sqrt(u1**2+v1**2+w1**2)-w1) \
					- np.log(np.sqrt(u2**2+v1**2+w1**2)-w1) \
					+ np.log(np.sqrt(u2**2+v2**2+w1**2)-w1) \
					- np.log(np.sqrt(u1**2+v2**2+w1**2)-w1) \
					+ np.log(np.sqrt(u1**2+v2**2+w2**2)-w2) \
					- np.log(np.sqrt(u1**2+v1**2+w2**2)-w2) \
					+ np.log(np.sqrt(u2**2+v1**2+w2**2)-w2) \
					- np.log(np.sqrt(u2**2+v2**2+w2**2)-w2)

					Gzx = log(np.sqrt(u1**2+v1**2+w1**2)-v1) \
					- np.log(np.sqrt(u2**2+v1**2+w1**2)-v1) \
					+ np.log(np.sqrt(u2**2+v2**2+w1**2)-v2) \
					- np.log(np.sqrt(u1**2+v2**2+w1**2)-v2) \
					+ np.log(np.sqrt(u1**2+v2**2+w2**2)-v2) \
					- np.log(np.sqrt(u1**2+v1**2+w2**2)-v1) \
					+ np.log(np.sqrt(u2**2+v1**2+w2**2)-v1) \
					- np.log(np.sqrt(u2**2+v2**2+w2**2)-v2)

					G[COUNT,:] = C*np.c_[Gxx,Gyx,Gzx]
					COUNT = COUNT + 1

			elif dComp is 'y':
				for rr in range(0,M):
					u1 = locs[rr,0] - ax
					u1[u1==0] =  np.min(xyzh[:,0])/1000 
					u2 = locs[rr,0] - bx 
					u2[u2==0] = -np.min(xyzh[:,0])/1000 
					v1 = locs[rr,1] - ay 
					v1[v1==0] =  np.min(xyzh[:,1])/1000 
					v2 = locs[rr,1] - by 
					v2[v2==0] = -np.min(xyzh[:,1])/1000 
					w1 = locs[rr,2] - az 
					w1[w1==0] =  np.min(xyzh[:,2])/1000 
					w2 = locs[rr,2] - bz 
					w2[w2==0] = -np.min(xyzh[:,2])/1000 

					Gxy = log(np.sqrt(u1**2+v1**2+w1**2)-w1) \
					- np.log(np.sqrt(u2**2+v1**2+w1**2)-w1) \
					+ np.log(np.sqrt(u2**2+v2**2+w1**2)-w1) \
					- np.log(np.sqrt(u1**2+v2**2+w1**2)-w1) \
					+ np.log(np.sqrt(u1**2+v2**2+w2**2)-w2) \
					- np.log(np.sqrt(u1**2+v1**2+w2**2)-w2) \
					+ np.log(np.sqrt(u2**2+v1**2+w2**2)-w2) \
					- np.log(np.sqrt(u2**2+v2**2+w2**2)-w2)

					Gyy = np.arctan((u1*w1)/(v1*sqrt(u1**2+v1**2+w1**2)+eps)) \
					- np.arctan((u2*w1)/(v1*sqrt(u2**2+v1**2+w1**2)+eps)) \
					+ np.arctan((u2*w1)/(v2*sqrt(u2**2+v2**2+w1**2)+eps)) \
					- np.arctan((u1*w1)/(v2*sqrt(u1**2+v2**2+w1**2)+eps)) \
					+ np.arctan((u1*w2)/(v2*sqrt(u1**2+v2**2+w2**2)+eps)) \
					- np.arctan((u1*w2)/(v1*sqrt(u1**2+v1**2+w2**2)+eps)) \
					+ np.arctan((u2*w2)/(v1*sqrt(u2**2+v1**2+w2**2)+eps)) \
					- np.arctan((u2*w2)/(v2*sqrt(u2**2+v2**2+w2**2)+eps)) 

					Gzy = log(sqrt(u1**2+v1**2+w1**2)-u1) \
					- log(sqrt(u2**2+v1**2+w1**2)-u2) \
					+ log(sqrt(u2**2+v2**2+w1**2)-u2) \
					- log(sqrt(u1**2+v2**2+w1**2)-u1) \
					+ log(sqrt(u1**2+v2**2+w2**2)-u1) \
					- log(sqrt(u1**2+v1**2+w2**2)-u1) \
					+ log(sqrt(u2**2+v1**2+w2**2)-u2) \
					- log(sqrt(u2**2+v2**2+w2**2)-u2)

					G[COUNT,:] = C*np.c_[Gxx,Gyx,Gzx]
					COUNT = COUNT + 1

			elif dComp is 'z':
				for rr in range(0,M):
					u1 = locs[rr,0] - ax
					u1[u1==0] =  np.min(xyzh[:,0])/1000 
					u2 = locs[rr,0] - bx 
					u2[u2==0] = -np.min(xyzh[:,0])/1000 
					v1 = locs[rr,1] - ay 
					v1[v1==0] =  np.min(xyzh[:,1])/1000 
					v2 = locs[rr,1] - by 
					v2[v2==0] = -np.min(xyzh[:,1])/1000 
					w1 = locs[rr,2] - az 
					w1[w1==0] =  np.min(xyzh[:,2])/1000 
					w2 = locs[rr,2] - bz 
					w2[w2==0] = -np.min(xyzh[:,2])/1000 

					Gxz = log(sqrt(u1**2+v1**2+w1**2)-v1) \
					- log(sqrt(u2**2+v1**2+w1**2)-v1) \
					+ log(sqrt(u2**2+v2**2+w1**2)-v2) \
					- log(sqrt(u1**2+v2**2+w1**2)-v2) \
					+ log(sqrt(u1**2+v2**2+w2**2)-v2) \
					- log(sqrt(u1**2+v1**2+w2**2)-v1) \
					+ log(sqrt(u2**2+v1**2+w2**2)-v1) \
					- log(sqrt(u2**2+v2**2+w2**2)-v2) 

					Gyz = log(sqrt(u1**2+v1**2+w1**2)-u1) \
					- log(sqrt(u2**2+v1**2+w1**2)-u2) \
					+ log(sqrt(u2**2+v2**2+w1**2)-u2) \
					- log(sqrt(u1**2+v2**2+w1**2)-u1) \
					+ log(sqrt(u1**2+v2**2+w2**2)-u1) \
					- log(sqrt(u1**2+v1**2+w2**2)-u1) \
					+ log(sqrt(u2**2+v1**2+w2**2)-u2) \
					- log(sqrt(u2**2+v2**2+w2**2)-u2) 

					Gzz = - np.arctan((v1*w1)/(u1*np.sqrt(u1**2+v1**2+w1**2)+eps)) \
					+ np.arctan((v1*w1)/(u2*np.sqrt(u2**2+v1**2+w1**2)+eps)) \
					- np.arctan((v2*w1)/(u2*np.sqrt(u2**2+v2**2+w1**2)+eps)) \
					+ np.arctan((v2*w1)/(u1*np.sqrt(u1**2+v2**2+w1**2)+eps)) \
					- np.arctan((v2*w2)/(u1*np.sqrt(u1**2+v2**2+w2**2)+eps)) \
					+ np.arctan((v1*w2)/(u1*np.sqrt(u1**2+v1**2+w2**2)+eps)) \
					- np.arctan((v1*w2)/(u2*np.sqrt(u2**2+v1**2+w2**2)+eps)) \
					+ np.arctan((v2*w2)/(u2*np.sqrt(u2**2+v2**2+w2**2)+eps))

					Gzz = Gzz - np.arctan((u1*w1)/(v1*sqrt(u1**2+v1**2+w1**2)+eps)) \
					+ np.arctan((u2*w1)/(v1*sqrt(u2**2+v1**2+w1**2)+eps)) \
					- np.arctan((u2*w1)/(v2*sqrt(u2**2+v2**2+w1**2)+eps)) \
					+ np.arctan((u1*w1)/(v2*sqrt(u1**2+v2**2+w1**2)+eps)) \
					- np.arctan((u1*w2)/(v2*sqrt(u1**2+v2**2+w2**2)+eps)) \
					+ np.arctan((u1*w2)/(v1*sqrt(u1**2+v1**2+w2**2)+eps)) \
					- np.arctan((u2*w2)/(v1*sqrt(u2**2+v1**2+w2**2)+eps)) \
					+ np.arctan((u2*w2)/(v2*sqrt(u2**2+v2**2+w2**2)+eps))

					G[COUNT,:] = C*np.c_[Gxx,Gyx,Gzx]
					COUNT = COUNT + 1











#######################################################################################
# VRM CHARACTERISTIC DECAY FORMULATION (SINGLE MODEL PARAMETER AND ALLOWS INVERSION)
#######################################################################################


class LinearFWD(BaseProblemVRM):

	def __init__(self, mesh, **kwargs):
		super(LinearFWD,self).__init__(mesh, **kwargs)



	def fields(self, mod, **kwargs):

		topoInd = mod != 0 # Only predict data from non-zero model values

		
		# Get A matrix
		A = np.array([])
		
		srcList = self.survey.srcList




		# Get T matrix































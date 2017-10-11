from SimPEG import Problem, mkvc, Maps, Props, Solver as simpegSolver
from SimPEG.VRM.SurveyVRM import SurveyVRM
# from SimPEG.VRM.FieldsVRM import Fields_LinearFWD
import numpy as np
import scipy.sparse as sp
import discretize # <-- only for temporary gridH function







############################################
# BASE VRM PROBLEM CLASS
############################################

class BaseProblemVRM(Problem.BaseProblem):

    """
    Base class for VRM problem.

    REQUIRED ARGUMENTS:

        mesh: 3D tensor or OcTree mesh

    KWARGS:

        refFact: Maximum refinement factor for sensitivities (default = 3)
        
        refRadius: Distances from source in which cell sensitivities are refined.
                   Must be an array or list with elements equal to the refFact.
                   (default based on minimum cell size)
        
        topoMap: A Maps object which operates from mesh cells to cells which are
                 computed in the forward model (default is all mesh cells)

    """

    # SET ATTRIBUTES
    refFact = None
    refRadius = None
    topoMap = None
    surveyPair = SurveyVRM

    def __init__(self, mesh, **kwargs):

        assert len(mesh.h) == 3, 'Problem requires 3D tensor or OcTree mesh'

        super(BaseProblemVRM,self).__init__(mesh, **kwargs)

        self.refFact = kwargs.get('refFact', 3)
        self.refRadius = kwargs.get('refRadius', 1.25*np.mean(np.r_[np.min(mesh.h[0]),np.min(mesh.h[1]),np.min(mesh.h[2])])*np.r_[1.,2.,3.])
        self.topoMap = kwargs.get('topoMap', Maps.InjectActiveCells(mesh, np.ones(mesh.nC, dtype=bool), np.array([])) )

        assert len(self.refRadius) == self.refFact, 'Number of refinement radii must equal refinement factor'
        assert isinstance(self.topoMap, Maps.InjectActiveCells), "topoMap must be an instance of class Maps.InjectActiveCells" 

    def _getH0matrix(self, xyz, pp):

        """
        Creates sparse matrix containing inducing field components for source pp
        
        INPUTS:
        
            xyz: N X 3 array of locations to predict field
            pp: Source index

        """

        SrcObj = self.survey.srcList[pp]

        H0 = SrcObj.getH0(xyz)

        Hx0 = sp.diags(H0[:,0], format="csr")
        Hy0 = sp.diags(H0[:,1], format="csr")
        Hz0 = sp.diags(H0[:,2], format="csr")

        H0 = sp.vstack([Hx0,Hy0,Hz0])

        return H0

    def _getGeometryMatrix(self, xyzc, xyzh, pp):

        """
        Creates the dense geometry matrix mapping from magentized voxel cells to the receivers for source pp
        
        INPUTS:
        
            xyzc: N by 3 array containing cell center locations
            xyzh: N by 3 array containing cell dimensions
            pp: Source index
        """

        srcObj = self.survey.srcList[pp]

        N = np.shape(xyzc)[0] # Number of cells
        K = srcObj.nRx # Number of receiver in all rxList

        ax = np.reshape(xyzc[:,0] - xyzh[:,0]/2, (1,N))
        bx = np.reshape(xyzc[:,0] + xyzh[:,0]/2, (1,N))
        ay = np.reshape(xyzc[:,1] - xyzh[:,1]/2, (1,N))
        by = np.reshape(xyzc[:,1] + xyzh[:,1]/2, (1,N))
        az = np.reshape(xyzc[:,2] - xyzh[:,2]/2, (1,N))
        bz = np.reshape(xyzc[:,2] + xyzh[:,2]/2, (1,N))

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
                    u1[np.abs(u1) < 1e-10] =  np.min(xyzh[:,0])/1000 
                    u2 = locs[rr,0] - bx 
                    u2[np.abs(u2) < 1e-10] = -np.min(xyzh[:,0])/1000 
                    v1 = locs[rr,1] - ay 
                    v1[np.abs(v1) < 1e-10] =  np.min(xyzh[:,1])/1000 
                    v2 = locs[rr,1] - by 
                    v2[np.abs(v2) < 1e-10] = -np.min(xyzh[:,1])/1000 
                    w1 = locs[rr,2] - az 
                    w1[np.abs(w1) < 1e-10] =  np.min(xyzh[:,2])/1000 
                    w2 = locs[rr,2] - bz 
                    w2[np.abs(w2) < 1e-10] = -np.min(xyzh[:,2])/1000

                    Gxx = np.arctan((v1*w1)/(u1*np.sqrt(u1**2+v1**2+w1**2)+eps)) \
                    - np.arctan((v1*w1)/(u2*np.sqrt(u2**2+v1**2+w1**2)+eps)) \
                    + np.arctan((v2*w1)/(u2*np.sqrt(u2**2+v2**2+w1**2)+eps)) \
                    - np.arctan((v2*w1)/(u1*np.sqrt(u1**2+v2**2+w1**2)+eps)) \
                    + np.arctan((v2*w2)/(u1*np.sqrt(u1**2+v2**2+w2**2)+eps)) \
                    - np.arctan((v1*w2)/(u1*np.sqrt(u1**2+v1**2+w2**2)+eps)) \
                    + np.arctan((v1*w2)/(u2*np.sqrt(u2**2+v1**2+w2**2)+eps)) \
                    - np.arctan((v2*w2)/(u2*np.sqrt(u2**2+v2**2+w2**2)+eps))

                    Gyx = np.log(np.sqrt(u1**2+v1**2+w1**2)-w1) \
                    - np.log(np.sqrt(u2**2+v1**2+w1**2)-w1) \
                    + np.log(np.sqrt(u2**2+v2**2+w1**2)-w1) \
                    - np.log(np.sqrt(u1**2+v2**2+w1**2)-w1) \
                    + np.log(np.sqrt(u1**2+v2**2+w2**2)-w2) \
                    - np.log(np.sqrt(u1**2+v1**2+w2**2)-w2) \
                    + np.log(np.sqrt(u2**2+v1**2+w2**2)-w2) \
                    - np.log(np.sqrt(u2**2+v2**2+w2**2)-w2)

                    Gzx = np.log(np.sqrt(u1**2+v1**2+w1**2)-v1) \
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
                    u1[np.abs(u1) < 1e-10] =  np.min(xyzh[:,0])/1000 
                    u2 = locs[rr,0] - bx 
                    u2[np.abs(u2) < 1e-10] = -np.min(xyzh[:,0])/1000 
                    v1 = locs[rr,1] - ay 
                    v1[np.abs(v1) < 1e-10] =  np.min(xyzh[:,1])/1000 
                    v2 = locs[rr,1] - by 
                    v2[np.abs(v2) < 1e-10] = -np.min(xyzh[:,1])/1000 
                    w1 = locs[rr,2] - az 
                    w1[np.abs(w1) < 1e-10] =  np.min(xyzh[:,2])/1000 
                    w2 = locs[rr,2] - bz 
                    w2[np.abs(w2) < 1e-10] = -np.min(xyzh[:,2])/1000 

                    Gxy = np.log(np.sqrt(u1**2+v1**2+w1**2)-w1) \
                    - np.log(np.sqrt(u2**2+v1**2+w1**2)-w1) \
                    + np.log(np.sqrt(u2**2+v2**2+w1**2)-w1) \
                    - np.log(np.sqrt(u1**2+v2**2+w1**2)-w1) \
                    + np.log(np.sqrt(u1**2+v2**2+w2**2)-w2) \
                    - np.log(np.sqrt(u1**2+v1**2+w2**2)-w2) \
                    + np.log(np.sqrt(u2**2+v1**2+w2**2)-w2) \
                    - np.log(np.sqrt(u2**2+v2**2+w2**2)-w2)

                    Gyy = np.arctan((u1*w1)/(v1*np.sqrt(u1**2+v1**2+w1**2)+eps)) \
                    - np.arctan((u2*w1)/(v1*np.sqrt(u2**2+v1**2+w1**2)+eps)) \
                    + np.arctan((u2*w1)/(v2*np.sqrt(u2**2+v2**2+w1**2)+eps)) \
                    - np.arctan((u1*w1)/(v2*np.sqrt(u1**2+v2**2+w1**2)+eps)) \
                    + np.arctan((u1*w2)/(v2*np.sqrt(u1**2+v2**2+w2**2)+eps)) \
                    - np.arctan((u1*w2)/(v1*np.sqrt(u1**2+v1**2+w2**2)+eps)) \
                    + np.arctan((u2*w2)/(v1*np.sqrt(u2**2+v1**2+w2**2)+eps)) \
                    - np.arctan((u2*w2)/(v2*np.sqrt(u2**2+v2**2+w2**2)+eps)) 

                    Gzy = np.log(np.sqrt(u1**2+v1**2+w1**2)-u1) \
                    - np.log(np.sqrt(u2**2+v1**2+w1**2)-u2) \
                    + np.log(np.sqrt(u2**2+v2**2+w1**2)-u2) \
                    - np.log(np.sqrt(u1**2+v2**2+w1**2)-u1) \
                    + np.log(np.sqrt(u1**2+v2**2+w2**2)-u1) \
                    - np.log(np.sqrt(u1**2+v1**2+w2**2)-u1) \
                    + np.log(np.sqrt(u2**2+v1**2+w2**2)-u2) \
                    - np.log(np.sqrt(u2**2+v2**2+w2**2)-u2)

                    G[COUNT,:] = C*np.c_[Gxy,Gyy,Gzy]
                    COUNT = COUNT + 1

            elif dComp is 'z':
                for rr in range(0,M):
                    u1 = locs[rr,0] - ax
                    u1[np.abs(u1) < 1e-10] =  np.min(xyzh[:,0])/1000 
                    u2 = locs[rr,0] - bx 
                    u2[np.abs(u2) < 1e-10] = -np.min(xyzh[:,0])/1000 
                    v1 = locs[rr,1] - ay 
                    v1[np.abs(v1) < 1e-10] =  np.min(xyzh[:,1])/1000 
                    v2 = locs[rr,1] - by 
                    v2[np.abs(v2) < 1e-10] = -np.min(xyzh[:,1])/1000 
                    w1 = locs[rr,2] - az 
                    w1[np.abs(w1) < 1e-10] =  np.min(xyzh[:,2])/1000 
                    w2 = locs[rr,2] - bz 
                    w2[np.abs(w2) < 1e-10] = -np.min(xyzh[:,2])/1000

                    Gxz = np.log(np.sqrt(u1**2+v1**2+w1**2)-v1) \
                    - np.log(np.sqrt(u2**2+v1**2+w1**2)-v1) \
                    + np.log(np.sqrt(u2**2+v2**2+w1**2)-v2) \
                    - np.log(np.sqrt(u1**2+v2**2+w1**2)-v2) \
                    + np.log(np.sqrt(u1**2+v2**2+w2**2)-v2) \
                    - np.log(np.sqrt(u1**2+v1**2+w2**2)-v1) \
                    + np.log(np.sqrt(u2**2+v1**2+w2**2)-v1) \
                    - np.log(np.sqrt(u2**2+v2**2+w2**2)-v2) 

                    Gyz = np.log(np.sqrt(u1**2+v1**2+w1**2)-u1) \
                    - np.log(np.sqrt(u2**2+v1**2+w1**2)-u2) \
                    + np.log(np.sqrt(u2**2+v2**2+w1**2)-u2) \
                    - np.log(np.sqrt(u1**2+v2**2+w1**2)-u1) \
                    + np.log(np.sqrt(u1**2+v2**2+w2**2)-u1) \
                    - np.log(np.sqrt(u1**2+v1**2+w2**2)-u1) \
                    + np.log(np.sqrt(u2**2+v1**2+w2**2)-u2) \
                    - np.log(np.sqrt(u2**2+v2**2+w2**2)-u2) 

                    Gzz = - np.arctan((v1*w1)/(u1*np.sqrt(u1**2+v1**2+w1**2)+eps)) \
                    + np.arctan((v1*w1)/(u2*np.sqrt(u2**2+v1**2+w1**2)+eps)) \
                    - np.arctan((v2*w1)/(u2*np.sqrt(u2**2+v2**2+w1**2)+eps)) \
                    + np.arctan((v2*w1)/(u1*np.sqrt(u1**2+v2**2+w1**2)+eps)) \
                    - np.arctan((v2*w2)/(u1*np.sqrt(u1**2+v2**2+w2**2)+eps)) \
                    + np.arctan((v1*w2)/(u1*np.sqrt(u1**2+v1**2+w2**2)+eps)) \
                    - np.arctan((v1*w2)/(u2*np.sqrt(u2**2+v1**2+w2**2)+eps)) \
                    + np.arctan((v2*w2)/(u2*np.sqrt(u2**2+v2**2+w2**2)+eps))

                    Gzz = Gzz - np.arctan((u1*w1)/(v1*np.sqrt(u1**2+v1**2+w1**2)+eps)) \
                    + np.arctan((u2*w1)/(v1*np.sqrt(u2**2+v1**2+w1**2)+eps)) \
                    - np.arctan((u2*w1)/(v2*np.sqrt(u2**2+v2**2+w1**2)+eps)) \
                    + np.arctan((u1*w1)/(v2*np.sqrt(u1**2+v2**2+w1**2)+eps)) \
                    - np.arctan((u1*w2)/(v2*np.sqrt(u1**2+v2**2+w2**2)+eps)) \
                    + np.arctan((u1*w2)/(v1*np.sqrt(u1**2+v1**2+w2**2)+eps)) \
                    - np.arctan((u2*w2)/(v1*np.sqrt(u2**2+v1**2+w2**2)+eps)) \
                    + np.arctan((u2*w2)/(v2*np.sqrt(u2**2+v2**2+w2**2)+eps))

                    G[COUNT,:] = C*np.c_[Gxz,Gyz,Gzz]
                    COUNT = COUNT + 1

        return np.matrix(G)







#######################################################################################
# VRM CHARACTERISTIC DECAY FORMULATION (SINGLE MODEL PARAMETER AND ALLOWS INVERSION)
#######################################################################################


class LinearFWD(BaseProblemVRM):

    """
    Problem class for linear VRM problem. The the solution to this problem
is a time-approximate solution which uses the characteristic decay of
the VRM response. The solution is only capable of providing the VRM
response during the off-time. For background theory, see Cowan (2016).

REQUIRED ARGUMENTS:

    mesh: 3D tensor or OcTree mesh

KWARGS:

    refFact: Maximum refinement factor for sensitivities (default = 3)
        
    refRadius: Distances from source in which cell sensitivities are refined. Must be an array or list with number of elements equal to refFact.
        
    topoMap: A Maps object which relates mesh cells to cells which are computed in the forward model (default is all mesh cells)

    xiMap: A Maps object which relates mesh cells to active cells

    """



    xi, xiMap, xiDeriv = Props.Invertible("Amalgamated Viscous Remanent Magnetization Parameter xi = dchi/ln(tau2/tau1)")


    def __init__(self, mesh, **kwargs):
        super(LinearFWD,self).__init__(mesh, **kwargs)

        self.A = None
        self.T = None


    def fields(self, m, fType = None):

        assert self.ispaired, "Problem must be paired with survey to predict data"

        if self.A is None:
            A = self.getA()
        
        if self.T is not None:
            T = self.T
        else:
            T = self.getT(fType)

        m = np.matrix(m).T

        # Project to full mesh
        if self.xiMap is not None:
            m = self.xiMap.P*m + np.matrix(self.xiMap.valInactive).T

        # Project to topography cells
        if self.topoMap is not None:
            m = self.topoMap.P.T*m

        return sp.coo_matrix.dot(T, self.A*m)


    def getA(self):

        """
        This function computes the geometric sensitivity matrix for the linear VRM problem. |
        This function requires that the problem be paired with a survey object.
        """

        assert self.ispaired, "Problem must be paired with survey to generate A matrix"

        topoInd = self.topoMap.indActive

        # GET CELL INFORMATION FOR FORWARD MODELING
        meshObj = self.mesh
        xyzc = meshObj.gridCC[topoInd,:]
        xyzh = meshObj.gridH[topoInd,:]

        # GET A MATRIX
        A = []
        for pp in range(0,self.survey.nSrc):

            # Create initial A matrix
            G   = self._getGeometryMatrix(xyzc, xyzh, pp)
            H0  = self._getH0matrix(xyzc, pp)
            A.append(G*H0)

            # Refine A matrix
            refFact = 0 # self.refFact
            refRadius = self.refRadius

            if refFact > 0:

                srcObj = self.survey.srcList[pp]
                refFlag = srcObj._getRefineFlags(xyzc, refFact, refRadius)

                for qq in range(1,refFact):

                    A[pp][:,refFlag==qq] = self._getSubsetAcolumns(xyzc, xyzh, pp, qq, refFlag)

        # COLLAPSE ALL A MATRICIES INTO SINGLE OPERATOR
        A = np.vstack(A)

        if self.A is None:
            self.A = A

        return A

    def getT(self, fType = None):

        """
        This function returns the characteristic decay matrix. This function \
        requires that the problem has been paired with a survey object.
        """

        assert self.ispaired, "Problem must be paired with survey to generate T matrix"

        srcList = self.survey.srcList
        nSrc = len(srcList)
        T = []

        if fType is None:

            for pp in range(0,nSrc):

                rxList = srcList[pp].rxList
                nRx = len(rxList)
                waveObj = srcList[pp].waveform

                for qq in range(0,nRx):

                    times = rxList[qq].times
                    nLoc = np.shape(rxList[qq].locs)[0]
                    
                    I = sp.diags(np.ones(nLoc))
                    eta = waveObj.getCharDecay(rxList[qq].fieldType, times)
                    eta = np.matrix(eta).T

                    T.append(sp.kron(I, eta))

            T = sp.block_diag(T)

            if self.T is None:
                self.T = T

        elif fType in ['h','b','dhdt','dbdt']:

            for pp in range(0,nSrc):

                rxList = srcList[pp].rxList
                nRx = len(rxList)
                waveObj = srcList[pp].waveform

                for qq in range(0,nRx):

                    times = rxList[qq].times
                    nLoc = np.shape(rxList[qq].locs)[0]
                    
                    I = sp.diags(np.ones(nLoc))
                    eta = waveObj.getCharDecay(fType, times)
                    eta = np.matrix(eta).T

                    T.append(sp.kron(I, eta))

            T = sp.block_diag(T)

        return T

    def Jvec(self, v):

        """Compute T*A*Pt'*Pm*v"""

        assert self.ispaired, "Problem must be paired with survey to predict data"

        if self.A is None:
            A = self.getA()
        
        if self.T is not None:
            T = self.T
        else:
            T = self.getT(fType)

        v = np.matrix(mkvc(v)).T

        # Project to full mesh
        if self.xiMap is not None:
            v = self.xiMap.P*v

        # Project to topography cells
        if self.topoMap is not None:
            v = self.topoMap.P.T*v

        return sp.coo_matrix.dot(T, self.A*v)

    def Jtvec(self, v):

        """Compute Pm'*Pt*A'*T'*v"""

        assert self.ispaired, "Problem must be paired with survey to predict data"

        if self.A is None:
            A = self.getA()
        
        if self.T is not None:
            T = self.T
        else:
            T = self.getT(fType)

        # Get v'
        v = np.matrix(mkvc(v))
        # Get v'*T
        v = sp.coo_matrix.dot(v, T)
        # Get A'*T'*v
        v = A.dot(v).T

        # Project to topography cells
        if self.topoMap is not None:
            v = self.topoMap.P*v

        # Project to full mesh
        if self.xiMap is not None:
            v = self.xiMap.P.T*v

        return v

    def unpair(self):
        """Unbind a survey from this problem instance."""
        if not self.ispaired:
            return
        self.survey._prob = None
        self._survey = None
        self.A = None
        self.T = None



    def _getSubsetAcolumns(self, xyzc, xyzh, pp, qq, refFlag):

        """
        This method returns the refined sensitivities for columns that will \
        be replaced in the A matrix for source pp and refinement factor qq
        
        INPUTS:

            xyzc: Cell centers of topo mesh cells N X 3 array
            xyzh: Cell widths of topo mesh cells N X 3 array
            pp: Source ID
            qq: Mesh refinement factor
            refFlag: refinement factors for all topo mesh cells
        """

        # GET SUBMESH GRID
        n = 2**qq
        [nx,ny,nz] = np.meshgrid(np.linspace(1,n,n)-0.5, np.linspace(1,n,n)-0.5, np.linspace(1,n,n)-0.5)
        nxyz_sub = np.c_[mkvc(nx), mkvc(ny), mkvc(nz)]

        xyzh_sub = xyzh[refFlag==qq,:] # Get widths of cells to be refined
        xyzc_sub = xyzc[refFlag==qq,:] - xyzh[refFlag==qq,:]/2 # Get bottom southwest corners of cells to be refined
        m = np.shape(xyzc_sub)[0]
        xyzc_sub = np.kron(xyzc_sub, np.ones((n**3,1))) # Kron for n**3 refined cells
        xyzh_sub = np.kron(xyzh_sub/n, np.ones((n**3,1))) # Kron for n**3 refined cells with widths h/n
        nxyz_sub = np.kron(np.ones((m,1)),nxyz_sub) # Kron for n**3 refined cells
        xyzc_sub = xyzc_sub + xyzh_sub*nxyz_sub

        # GET SUBMESH A MATRIX AND COLLAPSE TO COLUMNS
        G   = self._getGeometryMatrix(xyzc_sub, xyzh_sub, pp)
        H0  = self._getH0matrix(xyzc_sub, pp)
        Acols = (G*H0)*sp.kron(sp.diags(np.ones(m,1)),np.ones((n**3,1)))

        return Acols
































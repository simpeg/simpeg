from SimPEG import *



class FieldsDC_CC(Problem.Fields):
    knownFields = {'phi_sol':'CC'}
    aliasFields = {
                    'phi' : ['phi_sol','CC','_phi'],
                    'e' : ['phi_sol','F','_e'],
                    'j' : ['phi_sol','F','_j']
                  }

    def __init__(self,mesh,survey,**kwargs):
        super(FieldsDC_CC, self).__init__(mesh, survey, **kwargs)

    def startup(self):
        self._cellGrad = self.survey.prob.mesh.cellGrad

    def _phi(self, phi_sol, srcList):
        phi = phi_sol
        for i, src in enumerate(srcList):
            phi_p = src.phi_p(self.survey.prob)
            if phi_p is not None:
                phi[:,i] += phi_p     
        return phi

    def _e(self, phi_sol, srcList):
        e = -self._cellGrad*phi_sol
        for i, src in enumerate(srcList):
            e_p = src.e_p(self.survey.prob)
            if e_p is not None:
                e[:,i] += e_p     
        return e
    
    def _j(self, phi_sol, srcList):
        j = -self.survey.prob.Msig*self._cellGrad*phi_sol
        for i, src in enumerate(srcList):
            j_p = src.j_p(self.survey.prob)
            if j_p is not None:
                j[:,i] += j_p
        return j



class SrcDipole(Survey.BaseSrc):
    """A dipole source, locA and locB are moved to the closest cell-centers"""

    current = 1
    loc  = None
    # _rhsDict = None

    def __init__(self, rxList, locA, locB, **kwargs):
        self.loc = (locA, locB)
        super(SrcDipole, self).__init__(rxList, **kwargs)

    def eval(self, prob):
        # Recompute rhs 
        # if getattr(self, '_rhsDict', None) is None:
        #     self._rhsDict = {}
        # if mesh not in self._rhsDict:
        pts = [self.loc[0], self.loc[1]]
        inds = Utils.closestPoints(prob.mesh, pts)
        q = np.zeros(prob.mesh.nC)
        q[inds] = - self.current * ( np.r_[1., -1.] / prob.mesh.vol[inds] )
        # self._rhsDict[mesh] = q
        # return self._rhsDict[mesh]
        return q


class RxDipole(Survey.BaseRx):
    """A dipole source, locA and locB are moved to the closest cell-centers"""
    def __init__(self, locsM, locsN, **kwargs):
        locs = (locsM, locsN)
        assert locsM.shape == locsN.shape, 'locs must be the same shape.'
        super(RxDipole, self).__init__(locs, 'dipole', storeProjections=False, **kwargs)

    @property
    def nD(self):
        """Number of data in the receiver."""
        return self.locs[0].shape[0]

    def getP(self, mesh):
        P0 = mesh.getInterpolationMat(self.locs[0], self.projGLoc)
        P1 = mesh.getInterpolationMat(self.locs[1], self.projGLoc)
        return P0 - P1


class SurveyDC(Survey.BaseSurvey):
    """
        **SurveyDC**

        Geophysical DC resistivity data.

    """

    def __init__(self, srcList, **kwargs):
        self.srcList = srcList
        Survey.BaseSurvey.__init__(self, **kwargs)
        # self._rhsDict = {}
        self._Ps = {}

    def projectFields(self, u):
        """
            Predicted data.

            .. math::
                d_\\text{pred} = Pu(m)
        """
        P = self.getP(self.prob.mesh)
        return P*mkvc(u[self.srcList, 'phi_sol'])

    def getP(self, mesh):
        if mesh in self._Ps:
            return self._Ps[mesh]

        P_src = [sp.vstack([rx.getP(mesh) for rx in src.rxList]) for src in self.srcList]

        self._Ps[mesh] = sp.block_diag(P_src)
        return self._Ps[mesh]


class ProblemDC_CC(Problem.BaseProblem):
    """
        **ProblemDC**

        Geophysical DC resistivity problem.

    """

    surveyPair = SurveyDC
    Solver     = Solver
    fieldsPair = FieldsDC_CC
    Ainv = None

    def __init__(self, mesh, **kwargs):
        Problem.BaseProblem.__init__(self, mesh)
        self.mesh.setCellGradBC('neumann')
        Utils.setKwargs(self, **kwargs)


    deleteTheseOnModelUpdate = ['_A', '_Msig', '_dMdsig']

    @property
    def Msig(self):
        if getattr(self, '_Msig', None) is None:
            sigma = self.curModel.transform
            Av = self.mesh.aveF2CC
            self._Msig = Utils.sdiag(1/(self.mesh.dim * Av.T * (1/sigma)))
        return self._Msig

    @property
    def dMdsig(self):
        if getattr(self, '_dMdsig', None) is None:
            sigma = self.curModel.transform
            Av = self.mesh.aveF2CC
            dMdprop = self.mesh.dim * Utils.sdiag(self.Msig.diagonal()**2) * Av.T * Utils.sdiag(1./sigma**2)
            self._dMdsig = lambda Gu: Utils.sdiag(Gu) * dMdprop
        return self._dMdsig

    @property
    def A(self):
        """
            Makes the matrix A(m) for the DC resistivity problem.

            :param numpy.array m: model
            :rtype: scipy.csc_matrix
            :return: A(m)

            .. math::
                c(m,u) = A(m)u - q = G\\text{sdiag}(M(mT(m)))Du - q = 0

            Where M() is the mass matrix and mT is the model transform.
        """
        if getattr(self, '_A', None) is None:
            D = self.mesh.faceDiv
            G = self.mesh.cellGrad
            self._A = D*self.Msig*G
            # Remove the null space from the matrix.
            self._A[-1,-1] = 1. #self.mesh.vol[-1]
            self._A = self._A.tocsc()
        return self._A

    def getRHS(self):
        # if self.mesh not in self._rhsDict:
        RHS = np.array([src.eval(self) for src in self.survey.srcList]).T
        # self._rhsDict[mesh] = RHS
        # return self._rhsDict[mesh]
        return RHS

    def fields(self, m):

        F = self.fieldsPair(self.mesh, self.survey)        
        self.curModel = m
        A    = self.A
        self.Ainv = self.Solver(A, **self.solverOpts)
        RHS    = self.getRHS()
        Phi  = self.Ainv * RHS   
        Srcs = self.survey.srcList
        F[Srcs, 'phi_sol'] = Phi

        return F

    def Jvec(self, m, v, u=None):
        """
            :param numpy.array m: model
            :param numpy.array v: vector to multiply
            :param numpy.array u: fields
            :rtype: numpy.array
            :return: Jv

            .. math::
                c(m,u) = A(m)u - q = G\\text{sdiag}(M(mT(m)))Du - q = 0

                \\nabla_u (A(m)u - q) = A(m)

                \\nabla_m (A(m)u - q) = G\\text{sdiag}(Du)\\nabla_m(M(mT(m)))

            Where M() is the mass matrix and mT is the model transform.

            .. math::
                J = - P \left( \\nabla_u c(m, u) \\right)^{-1} \\nabla_m c(m, u)

                J(v) = - P ( A(m)^{-1} ( G\\text{sdiag}(Du)\\nabla_m(M(mT(m))) v ) )
        """
        # Set current model; clear dependent property $\mathbf{A(m)}$
        self.curModel = m
        sigma = self.curModel.transform # $\sigma = \mathcal{M}(\m)$
        if u is None:
            # Run forward simulation if $u$ not provided
            u = self.fields(self.curModel)[self.survey.srcList, 'phi_sol']
        else:
            u = u[self.survey.srcList, 'phi_sol']

        D = self.mesh.faceDiv
        G = self.mesh.cellGrad
        # Derivative of model transform, $\deriv{\sigma}{\m}$
        dsigdm_x_v = self.curModel.transformDeriv * v

        # Take derivative of $C(m,u)$ w.r.t. $m$
        dCdm_x_v = np.empty_like(u)
        # loop over fields for each source
        for i in range(self.survey.nSrc):
            # Derivative of inner product, $\left(\mathbf{M}_{1/\sigma}^f\right)^{-1}$
            dAdsig         = D * self.dMdsig( G * u[:,i] )
            dCdm_x_v[:, i] = dAdsig *  dsigdm_x_v

        # Take derivative of $C(m,u)$ w.r.t. $u$
        dA_du = self.A
        # Solve for $\deriv{u}{m}$
        # dCdu_inv = self.Solver(dCdu, **self.solverOpts)
        if self.Ainv is None:
            self.Ainv = self.Solver(dA_du, **self.solverOpts)

        P        = self.survey.getP(self.mesh)
        Jv    = - P * mkvc( self.Ainv * dCdm_x_v )
        return Jv

    def Jtvec(self, m, v, u=None):

        self.curModel = m
        sigma = self.curModel.transform # $\sigma = \mathcal{M}(\m)$
        if u is None:
            # Run forward simulation if $u$ not provided
            u = self.fields(self.curModel)[self.survey.srcList, 'phi_sol']
        else:
            u = u[self.survey.srcList, 'phi_sol']

        shp = u.shape
        P = self.survey.getP(self.mesh)
        PT_x_v = (P.T*v).reshape(shp, order='F')

        D = self.mesh.faceDiv
        G = self.mesh.cellGrad
        dA_du = self.A
        mT_dm = self.mapping.deriv(m)

        # We probably always need this due to the linesearch .. (?)
        self.Ainv = self.Solver(dA_du.T, **self.solverOpts)
        # if self.Ainv is None:
        #     self.Ainv = self.Solver(dCdu, **self.solverOpts)

        w = self.Ainv * PT_x_v

        Jtv = 0
        for i, ui in enumerate(u.T):  # loop over each column
            Jtv += self.dMdsig( G * ui ).T * ( D.T * w[:,i] )

        Jtv = - mT_dm.T * ( Jtv )
        return Jtv

def readUBC_DC2DModel(fileName):

    from SimPEG import np, mkvc
    """
        Read UBC GIF 2DTensor model and generate 2D Tensor model in simpeg

        Input:
        :param fileName, path to the UBC GIF 2D model file

        Output:
        :param SimPEG TensorMesh 2D object
        :return
        
        Created on Thu Nov 12 13:14:10 2015

        @author: dominiquef

    """
    
    # Open fileand skip header... assume that we know the mesh already

    obsfile = np.genfromtxt(fileName,delimiter=' \n',dtype=np.str,comments='!')
    
    dim = np.array(obsfile[0].split(),dtype=float)
    
    temp = np.array(obsfile[1].split(),dtype=float)
    
    if len(temp) > 1:
        model = np.zeros(dim)
        
        for ii in range(len(obsfile)-1):
            mm = np.array(obsfile[ii+1].split(),dtype=float)
            model[:,ii] = mm
            
        model = model[:,::-1]
        
    else:
        
        if len(obsfile[1:])==1:
            mm = np.array(obsfile[1:].split(),dtype=float)
            
        else:
            mm = np.array(obsfile[1:],dtype=float)
            
        # Permute the second dimension to flip the order
        model = mm.reshape(dim[1],dim[0])
    
        model = model[::-1,:]
        model = np.transpose(model, (1, 0))
        
    model = mkvc(model)


    return model

def plot_pseudoSection(Tx,Rx,data,z0, stype):
    
    from SimPEG import np, mkvc
    from scipy.interpolate import griddata
    from matplotlib.colors import LogNorm
    import pylab as plt
    import re
    """
        Read list of 2D tx-rx location and plot a speudo-section of apparent
        resistivity.
        
        Assumes flat topo for now...
    
        Input:
        :param d2D, z0
        :switch stype -> Either 'pdp' (pole-dipole) | 'dpdp' (dipole-dipole)
    
        Output:
        :figure scatter plot overlayed on image
        
        Created on Mon December 7th, 2015
    
        @author: dominiquef
    
    """
    #d2D = np.asarray(d2D)
    
    midl = []
    midz = []
    rho = []
    
    for ii in range(len(Tx)):
        # Get distances between each poles
        rC1P1 = np.abs(Tx[ii][0] - Rx[ii][:,0]) 
        rC2P1 = np.abs(Tx[ii][1] - Rx[ii][:,0])
        rC1P2 = np.abs(Tx[ii][1] - Rx[ii][:,1])
        rC2P2 = np.abs(Tx[ii][0] - Rx[ii][:,1])
        rP1P2 = np.abs(Rx[ii][:,1] - Rx[ii][:,0])    
    
        # Compute apparent resistivity
        if re.match(stype,'pdp'):
            rho = np.hstack([rho, data[ii] * 2*np.pi  * rC1P1 * ( rC1P1 + rP1P2 ) / rP1P2] )
            
        elif re.match(stype,'dpdp'):
            rho = np.hstack([rho, data[ii] * 2*np.pi / ( 1/rC1P1 - 1/rC2P1 - 1/rC1P2 + 1/rC2P2 ) ])
    
        Cmid = (Tx[ii][0] + Tx[ii][1])/2
        Pmid = (Rx[ii][:,0] + Rx[ii][:,1])/2
    
        midl = np.hstack([midl, ( Cmid + Pmid )/2 ])
        midz = np.hstack([midz, -np.abs(Cmid-Pmid)/2 + z0 ])
    
   
    # Grid points
    grid_x, grid_z = np.mgrid[np.min(midl):np.max(midl), np.min(midz):np.max(midz)]
    grid_rho = griddata(np.c_[midl,midz], np.log10(abs(1/rho.T)), (grid_x, grid_z), method='linear')
    
    
    #plt.subplot(2,1,2)
    plt.imshow(grid_rho.T, extent = (np.min(midl),np.max(midl),np.min(midz),np.max(midz)), origin='lower', alpha=0.8)
    cbar = plt.colorbar(format = '%.2f',fraction=0.02)
    cmin,cmax = cbar.get_clim()
    ticks = np.linspace(cmin,cmax,3)
    cbar.set_ticks(ticks)
    
    # Plot apparent resistivity
    plt.scatter(midl,midz,s=50,c=np.log10(abs(1/rho.T)))

def gen_DCIPsurvey(endl, mesh, stype, a, b, n):
    
    from SimPEG import np
    import re
    """
        Load in endpoints and survey specifications to generate Tx, Rx location
        stations.
        
        Assumes flat topo for now...
    
        Input:
        :param endl -> input endpoints [x1, y1, z1, x2, y2, z2]
        :object mesh -> SimPEG mesh object
        :switch stype -> "dpdp" (dipole-dipole) | "pdp" (pole-dipole) | 'gradient'
        : param a, n -> pole seperation, number of rx dipoles per tx
        
        Output:
        :param Tx, Rx -> List objects for each tx location
            Lines: P1x, P1y, P1z, P2x, P2y, P2z
        
        Created on Wed December 9th, 2015
    
        @author: dominiquef
    
    """
    def xy_2_r(x1,x2,y1,y2):
        r = np.sqrt( np.sum((x2 - x1)**2 + (y2 - y1)**2) )
        return r 
        
    ## Evenly distribute electrodes and put on surface
    # Mesure survey length and direction
    dl_len = xy_2_r(endl[0,0],endl[1,0],endl[0,1],endl[1,1])
    
    dl_x = ( endl[1,0] - endl[0,0] ) / dl_len
    dl_y = ( endl[1,1] - endl[0,1] ) / dl_len
       
    nstn = np.floor( dl_len / a )
    
    # Compute discrete pole location along line
    stn_x = endl[0,0] + np.array(range(int(nstn)))*dl_x*a
    stn_y = endl[0,1] + np.array(range(int(nstn)))*dl_y*a
    
    # Create line of P1 locations
    M = np.c_[stn_x, stn_y, np.ones(nstn).T*mesh.vectorNz[-1]]
    
    # Create line of P2 locations
    N = np.c_[stn_x+a*dl_x, stn_y+a*dl_y, np.ones(nstn).T*mesh.vectorNz[-1]]
    
    ## Build list of Tx-Rx locations depending on survey type
    # Dipole-dipole: Moving tx with [a] spacing -> [AB a MN1 a MN2 ... a MNn]
    # Pole-dipole: Moving pole on one end -> [A a MN1 a MN2 ... MNn a B]
    Tx = []
    Rx = []
    
    if not re.match(stype,'gradient'):
        
        for ii in range(0, int(nstn)-1): 
            
            
            if re.match(stype,'dpdp'):
                tx = np.c_[M[ii,:],N[ii,:]]
            elif re.match(stype,'pdp'):
                tx = np.c_[M[ii,:],M[ii,:]]
                
            #Rx.append(np.c_[M[ii+1:indx,:],N[ii+1:indx,:]])
            
            # Current elctrode seperation
            AB = xy_2_r(tx[0,1],endl[1,0],tx[1,1],endl[1,1])
            
            # Number of receivers to fit
            nstn = np.min([np.floor( (AB - b) / a ) , n])
            
            # Check if there is enough space, else break the loop
            if nstn <= 0:
                continue
            
            # Compute discrete pole location along line
            stn_x = N[ii,0] + dl_x*b + np.array(range(int(nstn)))*dl_x*a
            stn_y = N[ii,1] + dl_y*b + np.array(range(int(nstn)))*dl_y*a
            
            # Create receiver poles
            # Create line of P1 locations
            P1 = np.c_[stn_x, stn_y, np.ones(nstn).T*mesh.vectorNz[-1]]
            
            # Create line of P2 locations
            P2 = np.c_[stn_x+a*dl_x, stn_y+a*dl_y, np.ones(nstn).T*mesh.vectorNz[-1]]
            
            Rx.append(np.c_[P1,P2])
            Tx.append(tx)            
            
#==============================================================================
#     elif re.match(stype,'dpdp'):
#         
#         for ii in range(0, int(nstn)-2):  
#             
#             indx = np.min([ii+n+1,nstn])
#             Tx.append(np.c_[M[ii,:],N[ii,:]])
#             Rx.append(np.c_[M[ii+2:indx,:],N[ii+2:indx,:]])
#==============================================================================
            
    elif re.match(stype,'gradient'):
        
        # Gradient survey only requires Tx at end of line and creates a square
        # grid of receivers at in the middle at a pre-set minimum distance
        Tx.append(np.c_[M[0,:],N[-1,:]])
              
        # Get the edge limit of survey area
        min_x = endl[0,0] + dl_x * b
        min_y = endl[0,1] + dl_y * b
            
        max_x = endl[1,0] - dl_x * b
        max_y = endl[1,1] - dl_y * b
        
        box_l = np.sqrt( (min_x - max_x)**2 + (min_y - max_y)**2 )
        box_w = box_l/2.
        
        nstn = np.floor( box_l / a )
        
        # Compute discrete pole location along line
        stn_x = min_x + np.array(range(int(nstn)))*dl_x*a
        stn_y = min_y + np.array(range(int(nstn)))*dl_y*a
        
        # Define number of cross lines
        nlin = int(np.floor( box_w / a ))
        lind = range(-nlin,nlin+1) 
        
        ngrad = nstn * len(lind)
        
        rx = np.zeros([ngrad,6])
        for ii in range( len(lind) ):
            
            # Move line in perpendicular direction by dipole spacing
            lxx = stn_x - lind[ii]*a*dl_y
            lyy = stn_y + lind[ii]*a*dl_x
            
            
            M = np.c_[ lxx, lyy , np.ones(nstn).T*mesh.vectorNz[-1]]
            N = np.c_[ lxx+a*dl_x, lyy+a*dl_y, np.ones(nstn).T*mesh.vectorNz[-1]]
            
            rx[(ii*nstn):((ii+1)*nstn),:] = np.c_[M,N]
            
        Rx.append(rx)
        
    else:
        print """stype must be either 'pdp', 'dpdp' or 'gradient'. """

        
   
    return Tx, Rx  

def writeUBC_DCobs(fileName,Tx,Rx,d,wd, dtype):
    
    from SimPEG import np, mkvc
    import re
    """
        Read UBC GIF DCIP 3D observation file and generate arrays for tx-rx location
    
        Input:
        :param fileName, path to the UBC GIF 3D obs file
    
        Output:
        :param rx, tx, d, wd
        :return
        
        Created on Mon December 7th, 2015
    
        @author: dominiquef
    
    """
    fid = open(fileName,'w')
    fid.write('! GENERAL FORMAT\n')  
    
    for ii in range(len(Tx)):
        
        tx = np.asarray(Tx[ii])
        rx = np.asarray(Rx[ii])
        nrx = rx.shape[0]
        
        fid.write('\n')
        
        if re.match(dtype,'2D'):
            
            for jj in range(nrx):
                
                fid.writelines("%e " % ii for ii in mkvc(tx))
                fid.writelines("%e " % ii for ii in mkvc(rx[jj]))
                fid.write('%e %e\n'% (d[ii][jj],wd[ii][jj]))
                #np.savetxt(fid, np.c_[ rx ,np.asarray(d[ii]), np.asarray(wd[ii]) ], fmt='%e',delimiter=' ',newline='\n') 
        
        elif re.match(dtype,'3D'):
                                
            fid.write('\n')
            fid.writelines("%e " % ii for ii in mkvc(tx))
            fid.write('%i\n'% nrx)
            np.savetxt(fid, np.c_[ rx ,np.asarray(d[ii]), np.asarray(wd[ii]) ], fmt='%e',delimiter=' ',newline='\n') 
        
        
    fid.close()  

def convertObs_DC3D_to_2D(Tx,Rx):
    
    from SimPEG import np
    import numpy.matlib as npm
    """
        Read list of 3D Tx Rx location and change coordinate system to distance
        along line assuming all data is acquired along line
        First transmitter pole is assumed to be at the origin

        Assumes flat topo for now...
    
        Input:
        :param Tx, Rx
        
        Output:
        :figure Tx2d, Rx2d
        
        Created on Mon December 7th, 2015
    
        @author: dominiquef
    
    """
    
                
    Tx2d = []
    Rx2d = []

    for ii in range(len(Tx)):
        
        if ii == 0:
            endp = Tx[0][0:2,0]
        
        nrx = Rx[ii].shape[0]
                  
        rP1 = np.sqrt( np.sum( ( endp - Tx[ii][0:2,0] )**2 , axis=0))
        rP2 = np.sqrt( np.sum( ( endp - Tx[ii][0:2,1] )**2 , axis=0))
        rC1 = np.sqrt( np.sum( ( npm.repmat(endp.T,nrx,1) - Rx[ii][:,0:2] )**2 , axis=1))
        rC2 = np.sqrt( np.sum( ( npm.repmat(endp.T,nrx,1) - Rx[ii][:,3:5] )**2 , axis=1))
        
        Tx2d.append( np.r_[rP1, rP2] )
        Rx2d.append( np.c_[rC1, rC2] )
            #np.savetxt(fid, data, fmt='%e',delimiter=' ',newline='\n')

    return Tx2d, Rx2d

def readUBC_DC3Dobs(fileName):
    
    from SimPEG import np
    """
        Read UBC GIF DCIP 3D observation file and generate arrays for tx-rx location
    
        Input:
        :param fileName, path to the UBC GIF 3D obs file
    
        Output:
        :param rx, tx, d, wd
        :return
        
        Created on Mon December 7th, 2015
    
        @author: dominiquef
    
    """
       
    # Load file
    obsfile = np.genfromtxt(fileName,delimiter=' \n',dtype=np.str,comments='!')
    
    # Pre-allocate
    Tx = []
    Rx = []
    d = []
    wd = []
    
    # Countdown for number of obs/tx
    count = 0
    for ii in range(obsfile.shape[0]):
        
        if not obsfile[ii]:
            continue
        
        # First line is transmitter with number of receivers
        if count==0:
    
            temp = (np.fromstring(obsfile[ii], dtype=float,sep=' ').T)
            count = int(temp[-1])
            temp = np.reshape(temp[0:-1],[2,3]).T
            
            Tx.append(temp)
            rx = []
            continue
        
        temp = np.fromstring(obsfile[ii], dtype=float,sep=' ')
        
            
        rx.append(temp)          
        
        count = count -1        
        
        # Reach the end of  
        if count == 0:
            temp = np.asarray(rx)
            Rx.append(temp[:,0:6])
            
            # Check for data + uncertainties
            if temp.shape[1]==8:
                d.append(temp[:,6])
                wd.append(temp[:,7])
                
            # Check for data only    
            elif temp.shape[1]==7:
                d.append(temp[:,6])
            
    return Tx, Rx, d, wd

def readUBC_DC2DLoc(fileName):

    from SimPEG import np
    """
        Read UBC GIF 2D observation file and generate arrays for tx-rx location

        Input:
        :param fileName, path to the UBC GIF 2D model file

        Output:
        :param rx, tx
        :return
        
        Created on Thu Nov 12 13:14:10 2015

        @author: dominiquef

    """
    
    # Open fileand skip header... assume that we know the mesh already
#==============================================================================
#     fopen = open(fileName,'r')
#     lines = fopen.readlines()
#     fopen.close()
#==============================================================================

    # Load file
    obsfile = np.genfromtxt(fileName,delimiter=' \n',dtype=np.str,comments='!')
    
    # Check first line and figure out if 2D or 3D file format
    line = np.array(obsfile[0].split(),dtype=float)   
    
    tx_A  = []
    tx_B  = []
    rx_M  = []
    rx_N  = []
    d   = []
    wd  = []
    
    for ii in range(obsfile.shape[0]):
        
        # If len==3, then simple format where tx-rx is listed on each line
        if len(line) == 4:
        
            temp = np.fromstring(obsfile[ii], dtype=float,sep=' ')
            tx_A = np.hstack((tx_A,temp[0]))
            tx_B = np.hstack((tx_B,temp[1]))
            rx_M = np.hstack((rx_M,temp[2]))
            rx_N = np.hstack((rx_N,temp[3]))
            
        
    rx = np.transpose(np.array((rx_M,rx_N)))    
    tx = np.transpose(np.array((tx_A,tx_B)))
    
    return tx, rx, d, wd

def readUBC_DC2DMesh(fileName):

    from SimPEG import np
    """
        Read UBC GIF 2DTensor mesh and generate 2D Tensor mesh in simpeg

        Input:
        :param fileName, path to the UBC GIF mesh file

        Output:
        :param SimPEG TensorMesh 2D object
        :return
        
        Created on Thu Nov 12 13:14:10 2015

        @author: dominiquef

    """
    
    # Open file 
    fopen = open(fileName,'r')
    
    # Read down the file and unpack dx vector
    def unpackdx(fid,nrows):
        for ii in range(nrows):
            
            line = fid.readline()
            var = np.array(line.split(),dtype=float)
            
            if ii==0:
                x0= var[0]
                xvec = np.ones(int(var[2])) * (var[1] - var[0]) / int(var[2])
                xend = var[1]            
                
            else:
                xvec = np.hstack((xvec,np.ones(int(var[1])) * (var[0] - xend) / int(var[1])))
                xend = var[0] 
                
        return x0, xvec
        
    #%% Start with dx block
    # First line specifies the number of rows for x-cells
    line = fopen.readline()
    nl = np.array(line.split(),dtype=float)

    [x0, dx] = unpackdx(fopen,nl) 
    

    #%% Move down the file until reaching the z-block
    line = fopen.readline()
    if not line:
        line = fopen.readline()
        
    #%% End with dz block
    # First line specifies the number of rows for z-cells
    line = fopen.readline()
    nl = np.array(line.split(),dtype=float)

    [z0, dz] = unpackdx(fopen,nl) 
    
    # Flip z0 to be the bottom of the mesh for SimPEG
    z0 = z0 - sum(dz)
    dz = dz[::-1]
    #%% Make the mesh using SimPEG
    
    from SimPEG import Mesh
    tensMsh = Mesh.TensorMesh([dx,dz],(x0, z0))
    return tensMsh


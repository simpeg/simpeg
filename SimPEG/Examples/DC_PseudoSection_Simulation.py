from SimPEG import *
# import simpegDCIP as DC
from SimPEG import DCIP as DC
import scipy.interpolate as interpolation
import matplotlib.pyplot as plt
import time
import re

def run(loc=np.c_[[-50.,0.,-50.],[50.,0.,-50.]], sig=np.r_[1e-2,1e-1,1e-3], radi=np.r_[25.,25.], param = np.r_[30.,30.,5], stype = 'dpdp',  plotIt=True):
    """

        DC Forward Simulation
        =====================

        Forward model conductive spheres in a half-space and plot a pseudo-section

        Created on Mon Feb 01 19:28:06 2016

        @fourndo

    """

    def getIndicesSphere(center,radius,ccMesh):
        """
            Creates a vector containing the sphere indices in the cell centers mesh.
            Returns a tuple

            The sphere is defined by the points

            p0, describe the position of the center of the cell

            r, describe the radius of the sphere.

            ccMesh represents the cell-centered mesh

            The points p0 must live in the the same dimensional space as the mesh.

        """

        # Validation: mesh and point (p0) live in the same dimensional space
        dimMesh = np.size(ccMesh[0,:])
        assert len(center) == dimMesh, "Dimension mismatch. len(p0) != dimMesh"

        if dimMesh == 1:
           # Define the reference points

            ind  = np.abs(center[0] - ccMesh[:,0]) < radius

        elif dimMesh == 2:
           # Define the reference points

            ind = np.sqrt( ( center[0] - ccMesh[:,0] )**2 + ( center[1] - ccMesh[:,1] )**2 ) < radius

        elif dimMesh == 3:
            # Define the points
            ind = np.sqrt( ( center[0] - ccMesh[:,0] )**2 + ( center[1] - ccMesh[:,1] )**2 + ( center[2] - ccMesh[:,2] )**2 ) < radius

        # Return a tuple
        return ind
    # First we need to create a mesh and a model.

    # This is our mesh
    dx    = 5.

    hxind = [(dx,15,-1.3), (dx, 75), (dx,15,1.3)]
    hyind = [(dx,15,-1.3), (dx, 10), (dx,15,1.3)]
    hzind = [(dx,15,-1.3),(dx, 15)]

    mesh = Mesh.TensorMesh([hxind, hyind, hzind], 'CCN')


    # Set background conductivity
    model = np.ones(mesh.nC) * sig[0]

    # First anomaly
    ind = getIndicesSphere(loc[:,0],radi[0],mesh.gridCC)
    model[ind] = sig[1]

    # Second anomaly
    ind = getIndicesSphere(loc[:,1],radi[1],mesh.gridCC)
    model[ind] = sig[2]

    # Get index of the center
    indy = int(mesh.nCy/2)


    # Plot the model for reference
    # Define core mesh extent
    xlim = 200
    zlim = 125

    # Specify the survey type: "pdp" | "dpdp"


    # Then specify the end points of the survey. Let's keep it simple for now and survey above the anomalies, top of the mesh
    ends = [(-175,0),(175,0)]
    ends = np.c_[np.asarray(ends),np.ones(2).T*mesh.vectorNz[-1]]

    # Snap the endpoints to the grid. Easier to create 2D section.
    indx = Utils.closestPoints(mesh, ends )
    locs = np.c_[mesh.gridCC[indx,0],mesh.gridCC[indx,1],np.ones(2).T*mesh.vectorNz[-1]]

    # We will handle the geometry of the survey for you and create all the combination of tx-rx along line
    [Tx, Rx] = DC.gen_DCIPsurvey(locs, mesh, stype, param[0], param[1], param[2])

    # Define some global geometry
    dl_len = np.sqrt( np.sum((locs[0,:] - locs[1,:])**2) )
    dl_x = ( Tx[-1][0,1] - Tx[0][0,0] ) / dl_len
    dl_y = ( Tx[-1][1,1] - Tx[0][1,0]  ) / dl_len
    azm =  np.arctan(dl_y/dl_x)

    #Set boundary conditions
    mesh.setCellGradBC('neumann')

    # Define the differential operators needed for the DC problem
    Div = mesh.faceDiv
    Grad = mesh.cellGrad
    Msig = Utils.sdiag(1./(mesh.aveF2CC.T*(1./model)))

    A = Div*Msig*Grad

    # Change one corner to deal with nullspace
    A[0,0] = 1
    A = sp.csc_matrix(A)

    # We will solve the system iteratively, so a pre-conditioner is helpful
    # This is simply a Jacobi preconditioner (inverse of the main diagonal)
    dA = A.diagonal()
    P = sp.spdiags(1/dA,0,A.shape[0],A.shape[0])

    # Now we can solve the system for all the transmitters
    # We want to store the data
    data = []

    # There is probably a more elegant way to do this, but we can just for-loop through the transmitters
    for ii in range(len(Tx)):

        start_time = time.time() # Let's time the calculations

        #print("Transmitter %i / %i\r" % (ii+1,len(Tx)))

        # Select dipole locations for receiver
        rxloc_M = np.asarray(Rx[ii][:,0:3])
        rxloc_N = np.asarray(Rx[ii][:,3:])


        # For usual cases "dpdp" or "gradient"
        if not re.match(stype,'pdp'):
            inds = Utils.closestPoints(mesh, np.asarray(Tx[ii]).T )
            RHS = mesh.getInterpolationMat(np.asarray(Tx[ii]).T, 'CC').T*( [-1,1] / mesh.vol[inds] )

        else:

            # Create an "inifinity" pole
            tx =  np.squeeze(Tx[ii][:,0:1])
            tinf = tx + np.array([dl_x,dl_y,0])*dl_len*2
            inds = Utils.closestPoints(mesh, np.c_[tx,tinf].T)
            RHS = mesh.getInterpolationMat(np.asarray(Tx[ii]).T, 'CC').T*( [-1] / mesh.vol[inds] )


        # Iterative Solve
        Ainvb = sp.linalg.bicgstab(P*A,P*RHS, tol=1e-5)

        # We now have the potential everywhere
        phi = mkvc(Ainvb[0])

        # Solve for phi on pole locations
        P1 = mesh.getInterpolationMat(rxloc_M, 'CC')
        P2 = mesh.getInterpolationMat(rxloc_N, 'CC')

        # Compute the potential difference
        dtemp = (P1*phi - P2*phi)*np.pi

        data.append( dtemp )
        print '\rTransmitter {0} of {1} -> Time:{2} sec'.format(ii,len(Tx),time.time()- start_time),

    print 'Transmitter {0} of {1}'.format(ii,len(Tx))
    print 'Forward completed'


    # Let's just convert the 3D format into 2D (distance along line) and plot
    [Tx2d, Rx2d] = DC.convertObs_DC3D_to_2D(Tx,Rx)


    # Here is an example for the first tx-rx array
    if plotIt:
        fig = plt.figure()
        ax = plt.subplot(2,1,1, aspect='equal')
        mesh.plotSlice(np.log10(model), ax =ax, normal = 'Y', ind = indy,grid=True)
        ax.set_title('E-W section at '+str(mesh.vectorCCy[indy])+' m')
        plt.gca().set_aspect('equal', adjustable='box')

        plt.scatter(Tx[0][0,:],Tx[0][2,:],s=40,c='g', marker='v')
        plt.scatter(Rx[0][:,0::3],Rx[0][:,2::3],s=40,c='y')
        plt.xlim([-xlim,xlim])
        plt.ylim([-zlim,mesh.vectorNz[-1]+dx])


        ax = plt.subplot(2,1,2, aspect='equal')

        # Plot the location of the spheres for reference
        circle1=plt.Circle((loc[0,0]-Tx[0][0,0],loc[2,0]),radi[0],color='w',fill=False, lw=3)
        circle2=plt.Circle((loc[0,1]-Tx[0][0,0],loc[2,1]),radi[1],color='k',fill=False, lw=3)
        ax.add_artist(circle1)
        ax.add_artist(circle2)

        # Add the speudo section
        DC.plot_pseudoSection(Tx2d,Rx2d,data,mesh.vectorNz[-1],stype)

        plt.scatter(Tx2d[0][:],Tx[0][2,:],s=40,c='g', marker='v')
        plt.scatter(Rx2d[0][:],Rx[0][:,2::3],s=40,c='y')
        plt.plot(np.r_[Tx2d[0][0],Rx2d[-1][-1,-1]],np.ones(2)*mesh.vectorNz[-1], color='k')
        plt.ylim([-zlim,mesh.vectorNz[-1]+dx])

        plt.show()

        return fig, ax

if __name__ == '__main__':
    run()

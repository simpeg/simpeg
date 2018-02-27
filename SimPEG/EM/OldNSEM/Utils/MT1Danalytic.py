from __future__ import print_function
# Analytic solution of EM fields due to a plane wave

import numpy as np, SimPEG as simpeg
from scipy.constants import mu_0, epsilon_0 as eps_0

def getEHfields(m1d,sigma,freq,zd,scaleUD=True,scaleValue=1):
    '''Analytic solution for MT 1D layered earth. Returns E and H fields.

    :param SimPEG.mesh, object m1d: Mesh object with the 1D spatial information.
    :param numpy.array, vector sigma: Physical property of conductivity corresponding with the mesh.
    :param float, freq: Frequency to calculate data at.
    :param numpy array, vector zd: location to calculate EH fields at
    :param boolean, scaleUD: scales the output to be scaleValue at the top, increases numerical stability.

    Assumes a halfspace with the same conductive as the deepest cell.

    '''
    # Note add an error check for the mesh and sigma are the same size.

    # Constants: Assume constant
    mu = mu_0*np.ones((m1d.nC+1))
    eps = eps_0*np.ones((m1d.nC+1))
    # Angular freq
    w = 2*np.pi*freq
    # Add the halfspace value to the property
    sig = np.concatenate((np.array([sigma[0]]),sigma))
    # Calculate the wave number
    k = np.sqrt(eps*mu*w**2-1j*mu*sig*w)

    # Initiate the propagation matrix, in the order down up.
    UDp = np.zeros((2,m1d.nC+1),dtype=complex)
    UDp[1,0] = scaleValue # Set the wave amplitude as 1 into the half-space at the bottom of the mesh
    # Loop over all the layers, starting at the bottom layer
    for lnr, h in enumerate(m1d.hx): # lnr-number of layer, h-thickness of the layer
        # Calculate
        yp1 = k[lnr]/(w*mu[lnr]) # Admittance of the layer below the current layer
        zp = (w*mu[lnr+1])/k[lnr+1] # Impedance in the current layer
        # Build the propagation matrix

        # Convert fields to down/up going components in layer below current layer
        Pj1 = np.array([[1,1],[yp1,-yp1]],dtype=complex)
        # Convert fields to down/up going components in current layer
        Pjinv = 1./2*np.array([[1,zp],[1,-zp]],dtype=complex)
        # Propagate down and up components through the current layer
        elamh = np.array([[np.exp(-1j*k[lnr+1]*h),0],[0,np.exp(1j*k[lnr+1]*h)]])

        # The down and up component in current layer.
        UDp[:,lnr+1] = elamh.dot(Pjinv.dot(Pj1)).dot(UDp[:,lnr])

        if scaleUD:
            # Scale the values such that 1 at the top
            scaleVal = UDp[:,lnr+1::-1]/UDp[1,lnr+1]
            if np.any(~np.isfinite(scaleVal)):
                # If there is a nan (thickness very great), rebuild the move up cell
                scaleVal = np.zeros_like(UDp[:,lnr+1::-1],dtype=complex)
                scaleVal[1,0] = scaleValue

            UDp[:,lnr+1::-1] = scaleVal

    # Calculate the fields
    Ed = np.empty((zd.size,),dtype=complex)
    Eu = np.empty((zd.size,),dtype=complex)
    Hd = np.empty((zd.size,),dtype=complex)
    Hu = np.empty((zd.size,),dtype=complex)

    # Loop over the layers and calculate the fields
    # In the halfspace below the mesh
    dup = m1d.vectorNx[0]
    dind = dup >= zd
    Ed[dind] = UDp[1,0]*np.exp(-1j*k[0]*(dup-zd[dind]))
    Eu[dind] = UDp[0,0]*np.exp(1j*k[0]*(dup-zd[dind]))
    Hd[dind] = (k[0]/(w*mu[0]))*UDp[1,0]*np.exp(-1j*k[0]*(dup-zd[dind]))
    Hu[dind] = -(k[0]/(w*mu[0]))*UDp[0,0]*np.exp(1j*k[0]*(dup-zd[dind]))
    for ki,mui,epsi,dlow,dup,Up,Dp in zip(k[1::],mu[1::],eps[1::],m1d.vectorNx[:-1],m1d.vectorNx[1::],UDp[0,1::],UDp[1,1::]):
        dind = np.logical_and(dup >= zd, zd > dlow)
        Ed[dind] = Dp*np.exp(-1j*ki*(dup-zd[dind]))
        Eu[dind] = Up*np.exp(1j*ki*(dup-zd[dind]))
        Hd[dind] = (ki/(w*mui))*Dp*np.exp(-1j*ki*(dup-zd[dind]))
        Hu[dind] = -(ki/(w*mui))*Up*np.exp(1j*ki*(dup-zd[dind]))

    # Return return the fields
    return Ed, Eu, Hd, Hu

def getImpedance(m1d,sigma,freq):
    """Analytic solution for MT 1D layered earth. Returns the impedance at the surface.

    :param SimPEG.mesh, object m1d: Mesh object with the 1D spatial information.
    :param numpy.array, vector sigma: Physical property corresponding with the mesh.
    :param numpy.array, vector freq: Frequencies to calculate data at.


    """

    # Initiate the impedances
    Z1d = np.empty(len(freq) , dtype='complex')
    h = m1d.hx   #vectorNx[:-1]
    # Start the process
    for nrFr, fr in enumerate(freq):
        om = 2*np.pi*fr
        Zall = np.empty(len(h)+1,dtype='complex')
        # Calculate the impedance for the bottom layer
        Zall[0] = (mu_0*om)/np.sqrt(mu_0*eps_0*(om)**2 - 1j*mu_0*sigma[0]*om)

        for nr,hi in enumerate(h):
            # Calculate the wave number
            # print(nr,sigma[nr])
            k = np.sqrt(mu_0*eps_0*om**2 - 1j*mu_0*sigma[nr]*om)
            Z = (mu_0*om)/k

            Zall[nr+1] = Z *((Zall[nr] + Z*np.tanh(1j*k*hi))/(Z + Zall[nr]*np.tanh(1j*k*hi)))

        #pdb.set_trace()
        Z1d[nrFr] = Zall[-1]

    return Z1d

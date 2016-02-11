from SimPEG import *
from SimPEG import EM
from pymatsolver import MumpsSolver
from scipy.constants import mu_0

def run(plotIt=True):
    """
        FDEM: Effects of susceptibility
        ===============================

        When airborne freqeuncy domain EM (AFEM) survey is flown over
        the earth including significantly susceptible bodies (magnetite-rich rocks),
        negative data is often observed in the real part of the lowest frequency
        (e.g. Dighem system 900 Hz). This phenomenon mostly based upon magnetization
        occurs due to a susceptible body when the magnetic field applied.

        To clarify what is happening in the earth when we are exciting the earth with
        a loop source in the frequency domain we run three forward modelling:

            - F[:math:`\sigma`, :math:`\mu`]: Anomalous conductivity and susceptibility
            - F[:math:`\sigma`, :math:`\mu_0`]: Anomalous conductivity
            - F[:math:`\sigma_{air}`, :math:`\mu_0`]: primary field

        We plot vector magnetic fields in the earth. For secondary fields we provide
        F[:math:`\sigma`, :math:`\mu`]-F[:math:`\sigma`, :math:`\mu_0`]. Following
        figure show only real part, since that is our interest.

    """
    # Generate Cylindrical mesh
    cs, ncx, ncz, npad = 5, 25, 24, 20.
    hx = [(cs,ncx), (cs,npad,1.3)]
    hz = [(cs,npad,-1.3), (cs,ncz), (cs,npad,1.3)]
    mesh = Mesh.CylMesh([hx,1,hz], '00C')
    sighalf = 1e-3
    sigma = np.ones(mesh.nC)*1e-8
    sigmahomo = sigma.copy()
    mu = np.ones(mesh.nC)*mu_0
    sigma[mesh.gridCC[:,-1]<0.] = sighalf
    blkind  = np.logical_and(mesh.gridCC[:,0]<30., (mesh.gridCC[:,2]<0)&(mesh.gridCC[:,2]>-150)&(mesh.gridCC[:,2]<-50))
    sigma[blkind] = 1e-1
    mu[blkind] = mu_0*1.1
    offset = 0.
    frequency = np.r_[10., 100., 1000.]
    rx0 = EM.FDEM.Rx(np.array([[8., 0., 30.]]), 'bzr')
    rx1 = EM.FDEM.Rx(np.array([[8., 0., 30.]]), 'bzi')
    srcLists = []
    nfreq = frequency.size
    for ifreq in range(nfreq):
        src = EM.FDEM.Src.CircularLoop([rx0, rx1], frequency[ifreq], np.array([[0., 0., 30.]]), radius=5.)
        srcLists.append(src)
    survey = EM.FDEM.Survey(srcLists)
    iMap = Maps.IdentityMap(nP=int(mesh.nC))
    # Use PhysPropMap
    maps = [('sigma', iMap), ('mu', iMap)]
    prob = EM.FDEM.Problem_b(mesh, mapping=maps)
    prob.Solver = MumpsSolver
    survey.pair(prob)
    m = np.r_[sigma, mu]
    survey0 = EM.FDEM.Survey(srcLists)
    prob0 = EM.FDEM.Problem_b(mesh, mapping=maps)
    prob0.Solver = MumpsSolver
    survey0.pair(prob0)
    m = np.r_[sigma, mu]
    m0 = np.r_[sigma, np.ones(mesh.nC)*mu_0]
    m00 = np.r_[np.ones(mesh.nC)*1e-8, np.ones(mesh.nC)*mu_0]
    # Anomalous conductivity and susceptibility
    F = prob.fields(m)
    # Only anomalous conductivity
    F0 = prob.fields(m0)
    # Primary field
    F00 = prob.fields(m00)

    if plotIt:
        import matplotlib.pyplot as plt
        def vizfields(ifreq=0, primsec="secondary",realimag="real"):

            titles = ["F[$\sigma$, $\mu$]", "F[$\sigma$, $\mu_0$]", "F[$\sigma$, $\mu$]-F[$\sigma$, $\mu_0$]"]
            actind  = np.logical_and(mesh.gridCC[:,0]<200., (mesh.gridCC[:,2]>-400)&(mesh.gridCC[:,2]<200))

            if primsec=="secondary":
                bCCprim = (mesh.aveF2CCV*F00[:,'b'][:,ifreq]).reshape(mesh.nC, 2, order='F')
                bCC = (mesh.aveF2CCV*F[:,'b'][:,ifreq]).reshape(mesh.nC, 2, order='F')-bCCprim
                bCC0 = (mesh.aveF2CCV*F0[:,'b'][:,ifreq]).reshape(mesh.nC, 2, order='F')-bCCprim
            elif primsec=="primary":
                bCC = (mesh.aveF2CCV*F[:,'b'][:,ifreq]).reshape(mesh.nC, 2, order='F')
                bCC0 = (mesh.aveF2CCV*F0[:,'b'][:,ifreq]).reshape(mesh.nC, 2, order='F')

            XYZ = mesh.gridCC[actind,:]
            X = XYZ[:,0].reshape((31,43), order='F')
            Z = XYZ[:,2].reshape((31,43), order='F')
            bx = bCC[actind,0].reshape((31,43), order='F')
            bz = bCC[actind,1].reshape((31,43), order='F')
            bx0 = bCC0[actind,0].reshape((31,43), order='F')
            bz0 = bCC0[actind,1].reshape((31,43), order='F')

            bxsec = (bCC[actind,0]-bCC0[actind,0]).reshape((31,43), order='F')
            bzsec = (bCC[actind,1]-bCC0[actind,1]).reshape((31,43), order='F')

            absbreal = np.sqrt(bx.real**2+bz.real**2)
            absbimag = np.sqrt(bx.imag**2+bz.imag**2)
            absb0real = np.sqrt(bx0.real**2+bz0.real**2)
            absb0imag = np.sqrt(bx0.imag**2+bz0.imag**2)

            absbrealsec = np.sqrt(bxsec.real**2+bzsec.real**2)
            absbimagsec = np.sqrt(bxsec.imag**2+bzsec.imag**2)

            fig = plt.figure(figsize=(15,5))
            ax1 = plt.subplot(131)
            ax2 = plt.subplot(132)
            ax3 = plt.subplot(133)
            typefield="real"
            scale=20
            if realimag=="real":
                ax1.contourf(X, Z,np.log10(absbreal), 100)
                ax1.quiver(X, Z,bx.real/absbreal,bz.real/absbreal,scale=scale,width=0.005, alpha = 0.5)
                ax2.contourf(X, Z,np.log10(absb0real), 100)
                ax2.quiver(X, Z,bx0.real/absb0real,bz0.real/absb0real,scale=scale,width=0.005, alpha = 0.5)
                ax3.contourf(X, Z,np.log10(absbrealsec), 100)
                ax3.quiver(X, Z,bxsec.real/absbrealsec,bzsec.real/absbrealsec,scale=scale,width=0.005, alpha = 0.5)
            elif realimag=="imag":
                ax1.contourf(X, Z,np.log10(absbimag), 100)
                ax1.quiver(X, Z,bx.imag/absbimag,bz.imag/absbimag,scale=scale,width=0.005, alpha = 0.5)
                ax2.contourf(X, Z,np.log10(absb0imag), 100)
                ax2.quiver(X, Z,bx0.imag/absb0imag,bz0.imag/absb0imag,scale=scale,width=0.005, alpha = 0.5)
                ax3.contourf(X, Z,np.log10(absbimagsec), 100)
                ax3.quiver(X, Z,bxsec.imag/absbimagsec,bzsec.imag/absbimagsec,scale=scale,width=0.005, alpha = 0.5)

            ax = [ax1, ax2, ax3]
            ax3.text(30, 50, ("Frequency=%5.2f Hz")%(frequency[ifreq]), color="k", fontsize=18)
            ax2.text(30, 50, primsec, color="k", fontsize=18)
            for i, axtemp in enumerate(ax):
                axtemp.plot(np.r_[0, 29.75], np.r_[-50, -50], 'w', lw=3)

                axtemp.plot(np.r_[29.5, 29.5], np.r_[-50, -142.5], 'w', lw=3)
                axtemp.plot(np.r_[0, 29.5], np.r_[-142.5, -142.5], 'w', lw=3)
                axtemp.plot(np.r_[0, 100.], np.r_[0, 0], 'w', lw=3)
                axtemp.set_ylim(-200, 100.)
                axtemp.set_xlim(10, 100.)
                axtemp.set_title(titles[i])
            plt.show()
        vizfields(1, primsec="primary", realimag="real")
        vizfields(1, primsec="secondary", realimag="real")

if __name__ == '__main__':
    run()

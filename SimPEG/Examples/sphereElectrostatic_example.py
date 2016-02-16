from scipy.constants import epsilon_0
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
from SimPEG.Utils import ndgrid, mkvc

'''
Authors: Thibaut Astic, Lindsey Heagy, Sanna Tyrvainen, Ronghua Peng
Date: December 2015

This code defines function to resolve analytically the electrostatic sphere problem.
We first define a problem configuration, with a conductive or resistive sphere in a
wholespace background.
We then calculate the potential, then the electric field, then the current density and
finally the charges accumulation.

Several plotting functions are defined for data visualisation.


'''

# Plot options
ftsize_title = 18      #font size for titles
ftsize_axis  = 14      #font size for axis ticks
ftsize_label = 14      #font size for axis labels

# Radius function, useful sigma ratio, and log scale converter
r  = lambda x,y,z: np.sqrt(x**2.+y**2.+z**2.)
sigf = lambda sig0,sig1: (sig1-sig0)/(sig1+2.*sig0)

#tools to convert log conductivity in conductivity
def conductivity_log_wrapper(log_sig0,log_sig1):
    sig0 = 10.**log_sig0
    sig1 = 10.**log_sig1
    
    return sig0,sig1

# Examples
#Plot the configuration. Label=False is used to generate a general case figure
def get_Setup(XYZ,sig0,sig1,R,E0,ax,label,colorsphere):
    '''
    XYZ: ndgrid
    sig0: conductivity of the background
    sig1: conductivity of the sphere
    R: radius of the sphere
    E0: Amplitude of the uniform electrostatic field
    ax: ax where to plot the configuration
    label: True: plot real values, False: plot general case
    colorsphere: color of the sphere, format [x,x,x]
    '''

    xplt = np.linspace(-R, R, num=100)
    xr,yr,zr = np.unique(XYZ[:,0]),np.unique(XYZ[:,1]),np.unique(XYZ[:,2])
    dx = xr[1]-xr[0]
    top = np.sqrt(R**2-xplt**2)
    bot = -np.sqrt(R**2-xplt**2)
    
    if R != 0:
        ax.plot(xplt, top, xplt, bot, color=colorsphere,linewidth=1.5)
        ax.fill_between(xplt,bot,top,color=colorsphere,alpha=0.5 )
        ax.arrow(0.,0.,np.sqrt(2.)*R/2.,np.sqrt(2.)*R/2.,head_width=0.,head_length=0.)

        if label:
            ax.annotate(("$\sigma_1$=%3.3f mS/m")%(sig1*10.**(3.)),
            xy=(0.,-R/2.), xycoords='data',
            xytext=(0.,-R/2.), textcoords='data',
            fontsize=14.)
            ax.annotate(("$\sigma_0$= %3.3f mS/m")%(sig0*10.**(3.)),
            xy=(0.,-1.5*R), xycoords='data',
            xytext=(0.,-1.5*R), textcoords='data',
            fontsize=14.)
            ax.annotate(('$\mathbf{E_0} = %1i \mathbf{\hat{x}}$ V/m')%(E0),
            xy=(xr.min()+np.abs(xr.max()-xr.min())/20.,0), xycoords='data',
            xytext=(xr.min()+np.abs(xr.max()-xr.min())/20.,0), textcoords='data',
            fontsize=14.)
            ax.annotate(('$R$ = %1i m')%(R),
            xy=(R/4.+(xr[1]-xr[0]),R/4.), xycoords='data',
            xytext=(R/4.+(xr[1]-xr[0]),R/4.), textcoords='data',
            fontsize=14.)
            ax.set_ylabel('Y coordinate ($m$)',fontsize = ftsize_label)
            ax.set_xlabel('X coordinate ($m$)',fontsize = ftsize_label)
            ax.tick_params(labelsize=ftsize_axis)

        else:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.text(-1.,-np.sqrt(R)/2.-10.,'$\sigma_1$',fontsize=14)
            ax.text(-0.05,-R-10,'$\sigma_0$',fontsize=14)  
            ax.annotate(('$\mathbf{E_0} = E_0 \mathbf{\hat{x}}$ V/m'),
            xy=(xr.min()+np.abs(xr.max()-xr.min())/20.,0), xycoords='data',
            xytext=(xr.min()+np.abs(xr.max()-xr.min())/20.,0), textcoords='data',
            fontsize=14.)
            ax.annotate(('$R$'),
            xy=(R/4.+(xr[1]-xr[0]),R/4.), xycoords='data',
            xytext=(R/4.+(xr[1]-xr[0]),R/4.), textcoords='data',
            fontsize=14.)
            ax.set_xlabel('x',fontsize=12)
            ax.set_ylabel('y',fontsize=12)
    
    else:
        if label:
            ax.annotate(("$\sigma_0$= %3.3f mS/m")%(sig0*10.**(3.)),
            xy=(0.,-1.5*R), xycoords='data',
            xytext=(0.,-1.5*R), textcoords='data',
            fontsize=14.)
            ax.annotate(('$\mathbf{E_0} = %1i  \mathbf{\hat{x}}$ V/m')%(E0),
            xy=(xr.min()+np.abs(xr.max()-xr.min())/20.,0), xycoords='data',
            xytext=(xr.min()+np.abs(xr.max()-xr.min())/20.,0), textcoords='data',
            fontsize=14.)
            ax.set_ylabel('Y coordinate ($m$)',fontsize = ftsize_label)
            ax.set_xlabel('X coordinate ($m$)',fontsize = ftsize_label)
            ax.tick_params(labelsize=ftsize_axis)

        else:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.text(-0.05,-10,'$\sigma_0$',fontsize=14)  
            ax.text(xr.min()+np.abs(xr.max()-xr.min())/20., 0, '$\mathbf{E_0} = E_0 \mathbf{\hat{x}}$ V/m', fontsize=14)
            ax.set_xlabel('x',fontsize=12)
            ax.set_ylabel('y',fontsize=12)


    ax.set_xlim([xr.min(),xr.max()])
    ax.set_ylim([yr.min(),yr.max()])
    [ax.arrow(xr.min(),_,np.abs(xr.max()-xr.min())/20.,0.,head_width=5.,head_length=2.,color='k') for _ in np.linspace(yr.min(),yr.max(),num=10)]
    ax.patch.set_facecolor([0.4,0.7,0.4])
    ax.patch.set_alpha(0.2)

    ax.set_aspect('equal')

    
    
    return ax

def get_Conductivity(XYZ,sig0,sig1,R):
    '''
    Define the conductivity for each point of the space
    '''
    x,y,z = XYZ[:,0],XYZ[:,1],XYZ[:,2]
    r_view=r(x,y,z)
    
    ind0= (r_view>R)
    ind1= (r_view<=R)
    
    assert (ind0 + ind1).all(), 'Some indicies not included'
    
    Sigma = np.zeros_like(x)
    
    Sigma[ind0] = sig0
    Sigma[ind1] = sig1
    
    return Sigma


def get_Potential(XYZ,sig0,sig1,R,E0): 

    '''
    Function that returns the total, the primary and the secondary potentials, assumes an x-oriented inducing field and that the sphere is at the origin
    :input: grid, outer sigma, inner sigma, radius of the sphere, strength of the electric field
    '''
    
    x,y,z = XYZ[:,0],XYZ[:,1],XYZ[:,2]
    
    sig_cur = sigf(sig0,sig1)
    
    r_cur = r(x,y,z)  # current radius
    
    ind0 = (r_cur > R)
    ind1 = (r_cur <= R)
    
    assert (ind0 + ind1).all(), 'Some indicies not included'
    
    Vt = np.zeros_like(x)
    Vp = np.zeros_like(x)
    Vs = np.zeros_like(x)
    
    Vt[ind0] = -E0*x[ind0]*(1.-sig_cur*R**3./r_cur[ind0]**3.) # total potential outside the sphere
    Vt[ind1] = -E0*x[ind1]*3.*sig0/(sig1+2.*sig0)             # inside the sphere
    
    
    Vp = - E0*x  # primary potential
    
    Vs = Vt - Vp # secondary potential
    
    return Vt,Vp,Vs

#plot the primary potential on ax
def Plot_Primary_Potential(XYZ,sig0,sig1,R,E0,ax):
    
    Vt,Vp,Vs = get_Potential(XYZ,sig0,sig1,R,E0)
    
    xr,yr,zr = np.unique(XYZ[:,0]),np.unique(XYZ[:,1]),np.unique(XYZ[:,2])
    
    xcirc = xr[np.abs(xr) <= R]
    
    Pplot = ax.pcolor(xr,yr,Vp.reshape(xr.size,yr.size))
    ax.plot(xcirc,np.sqrt(R**2-xcirc**2),'--k',xcirc,-np.sqrt(R**2-xcirc**2),'--k')
    ax.set_title('Primary Potential',fontsize=ftsize_title)
    cb = plt.colorbar(Pplot,ax=ax)
    cb.set_label(label= 'Potential ($V$)',size=ftsize_label)
    cb.ax.tick_params(labelsize=ftsize_axis)
    ax.set_xlim([xr.min(),xr.max()])
    ax.set_ylim([yr.min(),yr.max()])
    ax.set_ylabel('Y coordinate ($m$)',fontsize = ftsize_label)
    ax.set_xlabel('X coordinate ($m$)',fontsize = ftsize_label)
    ax.set_aspect('equal')
    ax.tick_params(labelsize=ftsize_axis)
    
    return ax

#plot the total potential on ax
def Plot_Total_Potential(XYZ,sig0,sig1,R,E0,ax):
    
    Vt,Vp,Vs = get_Potential(XYZ,sig0,sig1,R,E0)
    
    xr,yr,zr = np.unique(XYZ[:,0]),np.unique(XYZ[:,1]),np.unique(XYZ[:,2])
    
    xcirc = xr[np.abs(xr) <= R]

    
    Pplot = ax.pcolor(xr,yr,Vt.reshape(xr.size,yr.size))
    ax.plot(xcirc,np.sqrt(R**2-xcirc**2),'--k',xcirc,-np.sqrt(R**2-xcirc**2),'--k')
    ax.set_title('Total Potential',fontsize=ftsize_title)
    cb = plt.colorbar(Pplot,ax=ax)
    cb.set_label(label= 'Potential ($V$)',size=ftsize_label)
    cb.ax.tick_params(labelsize=ftsize_axis)
    ax.set_xlim([xr.min(),xr.max()])
    ax.set_ylim([yr.min(),yr.max()])
    ax.set_ylabel('Y coordinate ($m$)',fontsize = ftsize_label)
    ax.set_xlabel('X coordinate ($m$)',fontsize = ftsize_label)
    ax.set_aspect('equal')
    ax.tick_params(labelsize=ftsize_axis)
    
    return ax

#plot the secondary potential on ax
def Plot_Secondary_Potential(XYZ,sig0,sig1,R,E0,ax):
    
    Vt,Vp,Vs = get_Potential(XYZ,sig0,sig1,R,E0)
    
    xr,yr,zr = np.unique(XYZ[:,0]),np.unique(XYZ[:,1]),np.unique(XYZ[:,2])
    
    xcirc = xr[np.abs(xr) <= R]

    Pplot = ax.pcolor(xr,yr,Vs.reshape(xr.size,yr.size))
    ax.plot(xcirc,np.sqrt(R**2-xcirc**2),'--k',xcirc,-np.sqrt(R**2-xcirc**2),'--k')
    ax.set_title('Secondary Potential',fontsize=ftsize_title)
    cb = plt.colorbar(Pplot,ax=ax)
    cb.set_label(label= 'Potential ($V$)',size=ftsize_label)
    cb.ax.tick_params(labelsize=ftsize_axis)
    ax.set_xlim([xr.min(),xr.max()])
    ax.set_ylim([yr.min(),yr.max()])
    ax.set_ylabel('Y coordinate ($m$)',fontsize = ftsize_label)
    ax.set_xlabel('X coordinate ($m$)',fontsize = ftsize_label)
    ax.set_aspect('equal')
    ax.tick_params(labelsize=ftsize_axis)
    
    return ax


def get_ElectricField(XYZ,sig0,sig1,R,E0):
    '''
    Function that returns the total, the primary and the secondary electric fields, 
    input: grid, outer sigma, inner sigma, radius of the sphere, strength of the electric field
    '''
    
    x,y,z= XYZ[:,0], XYZ[:,1], XYZ[:,2]
    
    r_cur=r(x,y,z)  # current radius
    
    ind0= (r_cur>R)
    ind1= (r_cur<=R)
    
    assert (ind0 + ind1).all(), 'Some indicies not included'
        
    Ep = np.zeros(shape=(len(x),3))
    Ep[:,0] = E0
    
    Et = np.zeros(shape=(len(x),3))
    
    Et[ind0,0] = E0 + E0*R**3./(r_cur[ind0]**5.)*sigf(sig0,sig1)*(2.*x[ind0]**2.-y[ind0]**2.-z[ind0]**2.);
    Et[ind0,1] = E0*R**3./(r_cur[ind0]**5.)*3.*x[ind0]*y[ind0]*sigf(sig0,sig1);
    Et[ind0,2] = E0*R**3./(r_cur[ind0]**5.)*3.*x[ind0]*z[ind0]*sigf(sig0,sig1);

    Et[ind1,0] = 3.*sig0/(sig1+2.*sig0)*E0;
    Et[ind1,1] = 0.;
    Et[ind1,2] = 0.;
    
    Es = Et - Ep
    
    return Et, Ep, Es

#plot the total electric field on ax
def Plot_Total_ElectricField(XYZ,sig0,sig1,R,E0,ax):
    
    Et, Ep, Es = get_ElectricField(XYZ,sig0,sig1,R,E0)
    
    xr,yr,zr = np.unique(XYZ[:,0]),np.unique(XYZ[:,1]),np.unique(XYZ[:,2])

    xcirc = xr[np.abs(xr) <= R]

    EtXr = Et[:,0].reshape(xr.size, yr.size)
    EtYr = Et[:,1].reshape(xr.size, yr.size)
    EtAmp = np.sqrt(Et[:,0]**2+Et[:,1]**2 + Et[:,2]**2).reshape(xr.size, yr.size)
    
    ax.set_xlim([xr.min(),xr.max()])
    ax.set_ylim([yr.min(),yr.max()])
    ax.set_ylabel('Y coordinate ($m$)',fontsize = ftsize_label)
    ax.set_xlabel('X coordinate ($m$)',fontsize = ftsize_label)
    ax.plot(xcirc,np.sqrt(R**2-xcirc**2),'--k',xcirc,-np.sqrt(R**2-xcirc**2),'--k')
    ax.tick_params(labelsize=ftsize_axis)
    ax.set_aspect('equal')
    
    Eplot = ax.pcolor(xr,yr,EtAmp)
    cb = plt.colorbar(Eplot,ax=ax)
    cb.set_label(label= 'Amplitude ($V/m$)',size=ftsize_label) #weight='bold')
    cb.ax.tick_params(labelsize=ftsize_axis)
    ax.streamplot(xr,yr,EtXr,EtYr,color='gray',linewidth=2.,density=0.75)#angles='xy',scale_units='xy',scale=0.05)
    ax.set_title('Total Field',fontsize=ftsize_title)
    
    
    return ax
    
#plot the secondary electric field on ax 
def Plot_Secondary_ElectricField(XYZ,sig0,sig1,R,E0,ax):
    
    Et, Ep, Es = get_ElectricField(XYZ,sig0,sig1,R,E0)
    
    xr,yr,zr = np.unique(XYZ[:,0]),np.unique(XYZ[:,1]),np.unique(XYZ[:,2])

    xcirc = xr[np.abs(xr) <= R]

    EsXr = Es[:,0].reshape(xr.size, yr.size)
    EsYr = Es[:,1].reshape(xr.size, yr.size)
    EsAmp = np.sqrt(Es[:,0]**2+Es[:,1]**2+Es[:,2]**2).reshape(xr.size, yr.size)
    
    ax.set_xlim([xr.min(),xr.max()])
    ax.set_ylim([yr.min(),yr.max()])
    ax.set_ylabel('Y coordinate ($m$)',fontsize = ftsize_label)
    ax.set_xlabel('X coordinate ($m$)',fontsize = ftsize_label)
    ax.plot(xcirc,np.sqrt(R**2-xcirc**2),'--k',xcirc,-np.sqrt(R**2-xcirc**2),'--k')
    ax.tick_params(labelsize=ftsize_axis)
    ax.set_aspect('equal')
    
    Eplot = ax.pcolor(xr,yr,EsAmp)
    cb = plt.colorbar(Eplot,ax=ax)
    cb.set_label(label= 'Amplitude ($V/m$)',size=ftsize_label) #weight='bold')
    cb.ax.tick_params(labelsize=ftsize_axis)
    ax.streamplot(xr,yr,EsXr,EsYr,color='gray',linewidth=2.,density=0.75)#,angles='xy',scale_units='xy',scale=0.05)
    ax.plot(xcirc,np.sqrt(R**2-xcirc**2),'--k',xcirc,-np.sqrt(R**2-xcirc**2),'--k')
    ax.set_title('Secondary Field',fontsize=ftsize_title)
    
    return ax


def get_Current(XYZ,sig0,sig1,R,Et,Ep,Es):
    '''
    Function that returns the total, the primary and the secondary current densities, 
    :input: grid, outer sigma, inner sigma, radius of the sphere, total, the primary and the seconadry electric fields,
    '''
    
    x,y,z= XYZ[:,0], XYZ[:,1], XYZ[:,2]
    
    r_cur=r(x,y,z)
    
    ind0= (r_cur>R)
    ind1= (r_cur<=R)
    
    assert (ind0 + ind1).all(), 'Some indicies not included'
    
    Jt = np.zeros(shape=(len(x),3))
    J0 = np.zeros(shape=(len(x),3))
    Js = np.zeros(shape=(len(x),3))
    

    Jp = sig0*Ep
    
    Jt[ind0,:] = sig0*Et[ind0,:]   
    Jt[ind1,:] = sig1*Et[ind1,:]

    Js[ind0,:] = sig0*(Et[ind0,:]-Ep[ind0,:])
    Js[ind1,:] = sig1*Et[ind1,:]-sig0*Ep[ind1,:]
    
    return Jt,Jp,Js

#plot the total currents density on ax
def Plot_Total_Currents(XYZ,sig0,sig1,R,E0,ax):
    
    Et,Ep,Es = get_ElectricField(XYZ,sig0,sig1,R,E0)
    Jt,Jp,Js = get_Current(XYZ,sig0,sig1,R,Et,Ep,Es)
    
    xr,yr,zr = np.unique(XYZ[:,0]),np.unique(XYZ[:,1]),np.unique(XYZ[:,2])
    xcirc = xr[np.abs(xr) <= R]

    JtXr = Jt[:,0].reshape(xr.size, yr.size)
    JtYr = Jt[:,1].reshape(xr.size, yr.size)
    JtAmp = np.sqrt(Jt[:,0]**2+Jt[:,1]**2+Jt[:,2]**2).reshape(xr.size, yr.size)
    
    ax.set_xlim([xr.min(),xr.max()])
    ax.set_ylim([yr.min(),yr.max()])
    ax.plot(xcirc,np.sqrt(R**2-xcirc**2),'--k',xcirc,-np.sqrt(R**2-xcirc**2),'--k')
    ax.set_ylabel('Y coordinate ($m$)',fontsize=ftsize_label)
    ax.set_xlabel('X coordinate ($m$)',fontsize=ftsize_label)
    ax.tick_params(labelsize=ftsize_axis)
    ax.set_aspect('equal')
    
    Jplot = ax.pcolor(xr,yr,JtAmp.reshape(xr.size,yr.size))
    cb = plt.colorbar(Jplot,ax=ax)
    cb.set_label(label= 'Current Density ($A/m^2$)',size=ftsize_label) #weight='bold')
    cb.ax.tick_params(labelsize=ftsize_axis)
    ax.streamplot(xr,yr,JtXr,JtYr,color='gray',linewidth=2.,density=0.75)#,angles='xy',scale_units='xy',scale=1)
    ax.set_title('Total Current Density',fontsize=ftsize_title)
    
    return ax


#plot the secondary currents density on ax
def Plot_Secondary_Currents(XYZ,sig0,sig1,R,E0,ax):
    
    Et,Ep,Es = get_ElectricField(XYZ,sig0,sig1,R,E0)
    Jt,Jp,Js = get_Current(XYZ,sig0,sig1,R,Et,Ep,Es)
    
    xr,yr,zr = np.unique(XYZ[:,0]),np.unique(XYZ[:,1]),np.unique(XYZ[:,2])
    xcirc = xr[np.abs(xr) <= R]
        
    JsXr = Js[:,0].reshape(xr.size, yr.size)
    JsYr = Js[:,1].reshape(xr.size, yr.size)
    JsAmp = np.sqrt(Js[:,1]**2+Js[:,0]**2+Jt[:,2]**2).reshape(xr.size,yr.size)
    
    ax.set_xlim([xr.min(),xr.max()])
    ax.set_ylim([yr.min(),yr.max()])
    ax.plot(xcirc,np.sqrt(R**2-xcirc**2),'--k',xcirc,-np.sqrt(R**2-xcirc**2),'--k')
    ax.set_ylabel('Y coordinate ($m$)',fontsize=ftsize_label)
    ax.set_xlabel('X coordinate ($m$)',fontsize=ftsize_label)
    ax.tick_params(labelsize=ftsize_axis)
    ax.set_aspect('equal')
    
    Jplot = ax.pcolor(xr,yr,JsAmp.reshape(xr.size,yr.size))
    cb = plt.colorbar(Jplot,ax=ax)
    cb.set_label(label= 'Current Density ($A/m^2$)',size=ftsize_label) #weight='bold')
    cb.ax.tick_params(labelsize=ftsize_axis)
    ax.streamplot(xr,yr,JsXr,JsYr,color='gray',linewidth=2.,density=0.75)#,angles='xy',scale_units='xy',scale=1)
    ax.set_title('Secondary Current Density',fontsize=ftsize_title)
    
    return ax


def get_ChargesDensity(XYZ,sig0,sig1,R,Et,Ep):
    '''
    Function that returns the charges accumulation at the background/sphere interface, 
    :input: grid, outer sigma, inner sigma, radius of the sphere, total and the primary electric fields,
    '''

    x,y,z= XYZ[:,0], XYZ[:,1], XYZ[:,2]
    
    dx = x[1]-x[0]
    
    r_cur=r(x,y,z)
    
    ind0 = (r_cur > R)
    ind1 = (r_cur < R)
    ind2 = ((r_cur < (R+dx/2)) & (r_cur > (R-dx/2)) )
    
    assert (ind0 + ind1 + ind2).all(), 'Some indicies not included'
    
    rho = np.zeros_like(x)
    
    rho[ind0] = 0
    rho[ind1] = 0
    rho[ind2] = epsilon_0*3.*Ep[ind2,0]*sigf(sig0,sig1)*x[ind2]/(np.sqrt(x[ind2]**2.+y[ind2]**2.))
    
    return rho

#Plot charges density on ax
def Plot_ChargesDensity(XYZ,sig0,sig1,R,E0,ax):
    
    xr,yr,zr = np.unique(XYZ[:,0]),np.unique(XYZ[:,1]),np.unique(XYZ[:,2])
    xcirc = xr[np.abs(xr) <= R]
    
    Et, Ep, Es = get_ElectricField(XYZ,sig0,sig1,R,E0)
    rho = get_ChargesDensity(XYZ,sig0,sig1,R,Et,Ep)
    
    ax.set_xlim([xr.min(),xr.max()])
    ax.set_ylim([yr.min(),yr.max()])
    ax.set_aspect('equal')
    Cplot = ax.pcolor(xr,yr,rho.reshape(xr.size, yr.size))
    cb1 = plt.colorbar(Cplot,ax=ax)
    cb1.set_label(label= 'Charge Density ($C/m^2$)',size=ftsize_label) #weight='bold')
    cb1.ax.tick_params(labelsize=ftsize_axis)
    ax.plot(xcirc,np.sqrt(R**2-xcirc**2),'--k',xcirc,-np.sqrt(R**2-xcirc**2),'--k')
    ax.set_ylabel('Y coordinate ($m$)',fontsize=ftsize_label)
    ax.set_xlabel('X coordinate ($m$)',fontsize=ftsize_label)
    ax.tick_params(labelsize=ftsize_axis)
    ax.set_title('Charges Density', fontsize=ftsize_title)
    
    return ax

def MN_Potential_total(sig0,sig1,R,E0,start,end,nbmp,mn):
    
    '''
    Function that return array of midpoints electrodes, electrodes positions,
    potentials differences for total and secondary potentials fields, unormalized and
    normalized to electrodes distances.
    sig0: background conductivity
    sig1: sphere conductivity
    R: Sphere's radius
    E0: uniform E field value
    start: start point for the profile start.shape = (2,)
    end: end point for the profile end.shape = (2,)
    nbmp: number of dipoles
    mn: Space between the M and N electrodes
    '''

    #D: total distance from start to end
    D = np.sqrt((start[0]-end[0])**2.+(start[1]-end[1])**2.)
    
    #MP: dipoles'midpoint positions (x,y)
    MP = np.zeros(shape=(nbmp,2))                            
    MP[:,0] = np.linspace(start[0],end[0],nbmp)
    MP[:,1] = np.linspace(start[1],end[1],nbmp)
    
    #Dipoles'Electrodes positions around each midpoints
    EL = np.zeros(shape=(2*nbmp,2))                         
    for n in range(0,len(EL),2):
        EL[n,0]   = MP[n/2,0] - ((end[0]-start[0])/D)*mn/2.
        EL[n+1,0] = MP[n/2,0] + ((end[0]-start[0])/D)*mn/2.
        EL[n,1]   = MP[n/2,1] - ((end[1]-start[1])/D)*mn/2.
        EL[n+1,1] = MP[n/2,1] + ((end[1]-start[1])/D)*mn/2.
    
    VtEL = np.zeros(2*nbmp) #Total Potential (Vt-) at each electrode (-EL)
    VsEL = np.zeros(2*nbmp) #Secondary Potential (Vt-) at each electrode (-EL)
    dVtMP = np.zeros(nbmp)  #Diffence (d-) of Total Potential (Vt-) at each dipole (-MP)
    dVtMPn = np.zeros(nbmp) #Diffence (d-) of Total Potential (Vt-) at each dipole (-MP) normalized for the mn spacing (n)
    dVsMP = np.zeros(nbmp)  #Diffence (d-) of Secondaty Potential (Vt-) at each dipole (-MP)
    dVsMPn = np.zeros(nbmp) #Diffence (d-) of Secondary Potential (Vt-) at each dipole (-MP) normalized for the mn spacing (n)
    dVpMP = np.zeros(nbmp) #Diffence (d-) of Primary Potential (Vt-) at each dipole (-MP)
    dVpMPn = np.zeros(nbmp) #Diffence (d-) of Primary Potential (Vt-) at each dipole (-MP) normalized for the mn spacing (n)
    
    #Computing VtEL 
    for m in range(0,2*nbmp):
        if (r(EL[m,0],EL[m,1],0) > R):
            VtEL[m] = -E0*EL[m,0]*(1.-sigf(sig0,sig1)*R**3./r(EL[m,0],EL[m,1],0)**3.)
        else:
            VtEL[m] = -E0*EL[m,0]*3.*sig0/(sig1+2.*sig0)
    
    #Computing VsEL
    VsEL = VtEL + E0*EL[:,0]
    
    #Computing dVtMP, dVsMP
    for p in range(0,nbmp):
        dVtMP[p] = VtEL[2*p]-VtEL[2*p+1]
        dVtMPn[p] = dVtMP[p]/mn
        dVsMP[p] = VsEL[2*p]-VsEL[2*p+1]
        dVsMPn[p] = dVsMP[p]/mn
    
    return MP,EL,dVtMP,dVtMPn,dVsMP,dVsMPn

#Compare the DC response of two configurations
def two_configurations_comparison(XYZ,sig0,sig1,sig2,R0,R1,E0,xstart,ystart,xend,yend,nb_dipole,electrode_spacing,PlotOpt):#,linearcolor):
    
    #Define the mesh
    xr,yr,zr = np.unique(XYZ[:,0]),np.unique(XYZ[:,1]),np.unique(XYZ[:,2])
    
    #Defining the Profile
    start = np.array([xstart,ystart])
    end = np.array([xend,yend])
    
    #Calculating the data from the defined survey line for Configuration 0 and 1
    MP0,EL0,VtdMP0,VtdMPn0,VsdMP0,VsdMPn0 = MN_Potential_total(sig0,sig1,R0,E0,start,end,nb_dipole,electrode_spacing)
    MP1,EL1,VtdMP1,VtdMPn1,VsdMP1,VsdMPn1 = MN_Potential_total(sig0,sig2,R1,E0,start,end,nb_dipole,electrode_spacing)


    # Initializing the figure
    fig = plt.figure(figsize=(20,20))
    ax0 = plt.subplot2grid((20,12), (0, 0),colspan=6,rowspan=6)
    ax1 = plt.subplot2grid((20,12), (0, 6),colspan=6,rowspan=6)
    ax2 = plt.subplot2grid((20,12), (16, 2), colspan=9,rowspan=4)
    ax3 = plt.subplot2grid((20,12), (8, 0),colspan=6,rowspan=6)
    ax4 = plt.subplot2grid((20,12), (8, 6),colspan=6,rowspan=6)

    #Plotting the Configuration 0
    ax0 = get_Setup(XYZ,sig0,sig1,R0,E0,ax0,True,[0.6,0.1,0.1])
    
    #Plotting the Configuration 1
    ax1   = get_Setup(XYZ,sig0,sig2,R1,E0,ax1,True,[0.1,0.1,0.6])
    
    #Plotting the Data (Legends)
    ax2.set_title('Potential Differences',fontsize=ftsize_title)
    ax2.set_ylabel('Potential difference ($V$)',fontsize=ftsize_label)
    ax2.set_xlabel('Distance from start point ($m$)',fontsize=ftsize_label)
    ax2.tick_params(labelsize=ftsize_axis)
    ax2.grid()

    if PlotOpt == 'Total':
        ax3= Plot_Total_Potential(XYZ,sig0,sig1,R0,E0,ax3)
        ax4= Plot_Total_Potential(XYZ,sig0,sig2,R1,E0,ax4)
           
        #Plot the Data (from Configuration 0)    
        gphy0 = ax2.plot(np.sqrt((MP0[0,0]-MP0[:,0])**2+(MP0[:,1]-MP0[0,1])**2),VtdMP0
                         ,marker='o',color='blue',linewidth=3.,label ='Left Model Response' )

        #Plot the Data (from Configuration 1)
        gphy1 = ax2.plot(np.sqrt((MP1[0,0]-MP1[:,0])**2+(MP1[:,1]-MP1[0,1])**2),VtdMP1
                ,marker='o',color='red',linewidth=2.,label ='Right Model Response' )
        ax2.legend(('Left Model Response','Right Model Response'),loc=4)

    elif PlotOpt == 'Secondary':
        #plot the secondary potentials
        ax3= Plot_Secondary_Potential(XYZ,sig0,sig1,R0,E0,ax3)
        ax4= Plot_Secondary_Potential(XYZ,sig0,sig2,R1,E0,ax4)
               
        #Plot the data(from configuration 0)
        gphy0 = ax2.plot(np.sqrt((MP0[0,0]-MP0[:,0])**2+(MP0[:,1]-MP0[0,1])**2),VsdMP0,color='blue'
                ,marker='o',linewidth=3.,label ='Left Model Response' )

        
        #Plot the Data (from Configuration 1)
        gphy1 = ax2.plot(np.sqrt((MP1[0,0]-MP1[:,0])**2+(MP1[:,1]-MP1[0,1])**2),VsdMP1
                 ,marker='o',color='red',linewidth=2.,label ='Right Model Response' )
        ax2.legend(('Left Model Response','Right Model Response'),loc=4 )
    
    else:
        print('What dont you get? Total or Secondary?')
    
    #Legends
    ax3.plot(MP0[:,0],MP0[:,1],color='gray')           
    Dip_Midpoint0 = ax3.scatter(MP0[:,0],MP0[:,1],color='black')
    Electrodes0 = ax3.scatter(EL0[:,0],EL0[:,1],color='red')
    ax3.legend([Dip_Midpoint0,Electrodes0], ["Dipole Midpoint", "Electrodes"],scatterpoints=1)
    
    ax4.plot(MP1[:,0],MP1[:,1],color='gray')           
    Dip_Midpoint1 = ax4.scatter(MP1[:,0],MP1[:,1],color='black')
    Electrodes1 = ax4.scatter(EL1[:,0],EL1[:,1],color='red')
    ax4.legend([Dip_Midpoint1,Electrodes1], ["Dipole Midpoint", "Electrodes"],scatterpoints=1)
        
    return fig

#Function to visualise and compare any two meaningful plots for the sphere in a uniform backgound with an unifom Electric Field
def interact_conductiveSphere(R,log_sig0,log_sig1,Figure1a,Figure1b,Figure2a,Figure2b):
     
    sig0,sig1 = conductivity_log_wrapper(log_sig0,log_sig1)
    E0   = 1.           # inducing field strength in V/m
    n = 100             #level of discretisation
    xr = np.linspace(-200., 200., n) # X-axis discretization
    yr = xr.copy()      # Y-axis discretization
    zr = np.r_[0]          # identical to saying `zr = np.array([0])`
    XYZ = ndgrid(xr,yr,zr) # Space Definition

    fig, ax = plt.subplots(1,2,figsize=(18,6))

    #Setup figure 1 with options Configuration, Total or Secondary, 
    #then Potential, ElectricField, Current Density or Charges Density
    if Figure1a == 'Configuration':
        ax[0] = get_Setup(XYZ,sig0,sig1,R,E0,ax[0],True,[0.1,0.1,0.6])
        
    elif Figure1a == 'Total':
        
        if Figure1b == 'Potential':
            ax[0] = Plot_Total_Potential(XYZ,sig0,sig1,R,E0,ax[0])

        elif Figure1b == 'ElectricField':
            ax[0] = Plot_Total_ElectricField(XYZ,sig0,sig1,R,E0,ax[0])
            
        elif Figure1b == 'CurrentDensity':
            ax[0] = Plot_Total_Currents(XYZ,sig0,sig1,R,E0,ax[0])
            
        elif Figure1b == 'ChargesDensity':
            ax[0] = Plot_ChargesDensity(XYZ,sig0,sig1,R,E0,ax[0])
            
    elif Figure1a == 'Secondary':
        
        if Figure1b == 'Potential':
            ax[0] = Plot_Secondary_Potential(XYZ,sig0,sig1,R,E0,ax[0])
        
        elif Figure1b == 'ElectricField':
            ax[0] = Plot_Secondary_ElectricField(XYZ,sig0,sig1,R,E0,ax[0])
            
        elif Figure1b == 'CurrentDensity':
            ax[0] = Plot_Secondary_Currents(XYZ,sig0,sig1,R,E0,ax[0])
            
        elif Figure1b == 'ChargesDensity':
            ax[0] = Plot_ChargesDensity(XYZ,sig0,sig1,R,E0,ax[0])
            
            
    if Figure1a== 'Configuration':
        ax[1] = Plot_Primary_Potential(XYZ,sig0,sig1,R,E0,ax[1])
        print 'While figure1 is plotting Configuration, figure2 plots the primary field'
        
    elif Figure2a == 'Total':      
        if Figure2b == 'Potential':
            ax[1] = Plot_Total_Potential(XYZ,sig0,sig1,R,E0,ax[1])

        elif Figure2b == 'ElectricField':
            ax[1] = Plot_Total_ElectricField(XYZ,sig0,sig1,R,E0,ax[1])

        elif Figure2b == 'CurrentDensity':
            ax[1]=Plot_Total_Currents(XYZ,sig0,sig1,R,E0,ax[1])

        elif Figure2b == 'ChargesDensity':
            ax[1] = Plot_ChargesDensity(XYZ,sig0,sig1,R,E0,ax[1])

    
    elif Figure2a == 'Secondary':      
        if Figure2b == 'Potential':
            ax[1] = Plot_Secondary_Potential(XYZ,sig0,sig1,R,E0,ax[1])

        elif Figure2b == 'ElectricField':
            ax[1] = Plot_Secondary_ElectricField(XYZ,sig0,sig1,R,E0,ax[1])

        elif Figure2b == 'CurrentDensity':
            ax[1] = Plot_Secondary_Currents(XYZ,sig0,sig1,R,E0,ax[1])

        elif Figure2b == 'ChargesDensity':
             ax[1] = Plot_ChargesDensity(XYZ,sig0,sig1,R,E0,ax[1])

    plt.tight_layout(True)
    plt.show()
     
#Interactive Visualisation of the responses of two configurations to a (pseudo) DC resistivity survey
def interactive_two_configurations_comparison(log_sig0,log_sig1,log_sig2,R0,R1,xstart,ystart,xend,yend,dipole_number,electrode_spacing,matching_spheres_example):
    
    sig0,sig1 = conductivity_log_wrapper(log_sig0,log_sig1)
    sig2 = 10.**log_sig2
    E0   = 1.           # inducing field strength in V/m
    n = 100             #level of discretisation
    xr = np.linspace(-200., 200., n) # X-axis discretization
    yr = xr.copy()      # Y-axis discretization
    zr = np.r_[0]          # identical to saying `zr = np.array([0])`
    XYZ = ndgrid(xr,yr,zr) # Space Definition
    PlotOpt = 'Total'
    
    if matching_spheres_example:
        sig0 = 10.**(-3)         
        sig1 = 10.**(-2)         
        sig2 = 1.310344828 * 10**(-3)
        R0   = 20.          
        R1   = 40.

        two_configurations_comparison(XYZ,sig0,sig1,sig2,R0,R1,E0,xstart,ystart,xend,yend,dipole_number,electrode_spacing,PlotOpt)

    else:
        two_configurations_comparison(XYZ,sig0,sig1,sig2,R0,R1,E0,xstart,ystart,xend,yend,dipole_number,electrode_spacing,PlotOpt)

    plt.tight_layout(True)
    plt.show()



if __name__ == '__main__':
    sig0 = -3.          # conductivity of the wholespace
    sig1 = -1.         # conductivity of the sphere
    sig0, sig1 = conductivity_log_wrapper(sig0,sig1)
    R    = 50.          # radius of the sphere
    E0   = 1.           # inducing field strength
    n = 100             #level of discretisation
    xr = np.linspace(-2.*R, 2.*R, n) # X-axis discretization
    yr = xr.copy()      # Y-axis discretization
    zr = np.r_[0]          # identical to saying `zr = np.array([0])`
    XYZ = ndgrid(xr,yr,zr) # Space Definition

    fig, ax = plt.subplots(2,5,figsize=(50,10))
    ax[0,0] = get_Setup(XYZ,sig0,sig1,R,E0,ax[0,0],True,[0.6,0.1,0.1])
    ax[1,0] = Plot_Primary_Potential(XYZ,sig0,sig1,R,E0,ax[1,0])
    ax[0,1] = Plot_Total_Potential(XYZ,sig0,sig1,R,E0,ax[0,1])
    ax[1,1] = Plot_Secondary_Potential(XYZ,sig0,sig1,R,E0,ax[1,1])
    ax[0,2] = Plot_Total_ElectricField(XYZ,sig0,sig1,R,E0,ax[0,2])
    ax[1,2] = Plot_Secondary_ElectricField(XYZ,sig0,sig1,R,E0,ax[1,2])
    ax[0,3] = Plot_Total_Currents(XYZ,sig0,sig1,R,E0,ax[0,3])
    ax[1,3] = Plot_Secondary_Currents(XYZ,sig0,sig1,R,E0,ax[1,3])
    ax[0,4] = Plot_Primary_Potential(XYZ,sig0,sig1,R,E0,ax[0,4])
    ax[1,4] = Plot_ChargesDensity(XYZ,sig0,sig1,R,E0,ax[1,4])
    

    plt.show()


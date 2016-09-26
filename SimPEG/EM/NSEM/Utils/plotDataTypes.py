from __future__ import print_function
from __future__ import division

from matplotlib import pyplot as plt, colors, numpy as np

def plotIsoFreqNSimpedance(ax,freq,array,flag,par='abs',colorbar=True,colorNorm='SymLog',cLevel=True,contour=True):

    indUniFreq = np.where(freq==array['freq'])


    x, y = array['x'][indUniFreq],array['y'][indUniFreq]
    if par == 'abs':
        zPlot = np.abs(array[flag][indUniFreq])
        cmap = plt.get_cmap('OrRd_r')#seismic')
        level = np.logspace(0,-5,31)
        clevel = np.logspace(0,-4,5)
        plotNorm = colors.LogNorm()
    elif par == 'real':
        zPlot = np.real(array[flag][indUniFreq])
        cmap = plt.get_cmap('RdYlBu')
        if cLevel:
            level = np.concatenate((-np.logspace(0,-10,31),np.logspace(-10,0,31)))
            clevel = np.concatenate((-np.logspace(0,-8,5),np.logspace(-8,0,5)))
        else:
            level = np.linspace(zPlot.min(),zPlot.max(),100)
            clevel = np.linspace(zPlot.min(),zPlot.max(),10)
        if colorNorm=='SymLog':
            plotNorm = colors.SymLogNorm(1e-10,linscale=2)
        else:
            plotNorm = colors.Normalize()
    elif par == 'imag':
        zPlot = np.imag(array[flag][indUniFreq])
        cmap = plt.get_cmap('RdYlBu')
        level = np.concatenate((-np.logspace(0,-10,31),np.logspace(-10,0,31)))
        clevel = np.concatenate((-np.logspace(0,-8,5),np.logspace(-8,0,5)))
        plotNorm = colors.SymLogNorm(1e-10,linscale=2)
        if cLevel:
            level = np.concatenate((-np.logspace(0,-10,31),np.logspace(-10,0,31)))
            clevel = np.concatenate((-np.logspace(0,-8,5),np.logspace(-8,0,5)))
        else:
            level = np.linspace(zPlot.min(),zPlot.max(),100)
            clevel = np.linspace(zPlot.min(),zPlot.max(),10)
        if colorNorm=='SymLog':
            plotNorm = colors.SymLogNorm(1e-10,linscale=2)
        elif colorNorm=='Lin':
            plotNorm = colors.Normalize()
    if contour:
        cs = ax.tricontourf(x,y,zPlot,levels=level,cmap=cmap,norm=plotNorm)#,extend='both')
    else:
        uniX,uniY = np.unique(x),np.unique(y)
        X,Y = np.meshgrid(np.append(uniX-25,uniX[-1]+25),np.append(uniY-25,uniY[-1]+25))
        cs = ax.pcolor(X,Y,np.reshape(zPlot,(len(uniY),len(uniX))),cmap=cmap,norm=plotNorm)
    if colorbar:
        plt.colorbar(cs,cax=ax.cax,ticks=clevel,format='%1.2e')
        ax.set_title(flag+' '+par,fontsize=8)
    return cs

def plotIsoFreqNSDiff(ax,freq,arrayList,flag,par='abs',colorbar=True,cLevel=True,mask=None,contourLine=True,useLog=False):

    indUniFreq0 = np.where(freq==arrayList[0]['freq'])
    indUniFreq1 = np.where(freq==arrayList[1]['freq'])
    seicmap = plt.get_cmap('RdYlBu')#seismic')
    x, y = arrayList[0]['x'][indUniFreq0],arrayList[0]['y'][indUniFreq0]
    if par == 'abs':
        if useLog:
            zPlot = (np.log10(np.abs(arrayList[0][flag][indUniFreq0])) - np.log10(np.abs(arrayList[1][flag][indUniFreq1])))/np.log10(np.abs(arrayList[1][flag][indUniFreq1]))
        else:
            zPlot = (np.abs(arrayList[0][flag][indUniFreq0]) - np.abs(arrayList[1][flag][indUniFreq1]))/np.abs(arrayList[1][flag][indUniFreq1])
        if mask:
            maskInd = np.logical_or(np.abs(arrayList[0][flag][indUniFreq0])< 1e-3,np.abs(arrayList[1][flag][indUniFreq1]) < 1e-3)
            zPlot = np.ma.array(zPlot)
            zPlot[maskInd] = mask
        if cLevel:
            level = np.arange(-200,201,10)
            clevel = np.arange(-200,201,25)
        else:
            level = np.linspace(zPlot.min(),zPlot.max(),100)
            clevel = np.linspace(zPlot.min(),zPlot.max(),10)
    elif par == 'real':
        if useLog:
            zPlot = (np.log10(np.real(arrayList[0][flag][indUniFreq0])) -np.log10(np.real(arrayList[1][flag][indUniFreq1])))/np.log10(np.abs((np.real(arrayList[1][flag][indUniFreq1]))))
        else:
            zPlot = (np.real(arrayList[0][flag][indUniFreq0]) -np.real(arrayList[1][flag][indUniFreq1]))/np.abs((np.real(arrayList[1][flag][indUniFreq1])))
        if mask:
            maskInd = np.logical_or(np.abs(np.real(arrayList[0][flag][indUniFreq0])) < 1e-3,np.abs(np.real(arrayList[1][flag][indUniFreq1])) < 1e-3)
            zPlot = np.ma.array(zPlot)
            zPlot[maskInd] = mask
        if cLevel:
            level = np.arange(-200,201,10)
            clevel = np.arange(-200,201,25)
        else:
            level = np.linspace(zPlot.min(),zPlot.max(),100)
            clevel = np.linspace(zPlot.min(),zPlot.max(),10)
    elif par == 'imag':
        if useLog:
            zPlot = (np.log10(np.imag(arrayList[0][flag][indUniFreq0])) -np.log10(np.imag(arrayList[1][flag][indUniFreq1])))/np.log10(np.abs((np.imag(arrayList[1][flag][indUniFreq1]))))
        else:
            zPlot = (np.imag(arrayList[0][flag][indUniFreq0]) -np.imag(arrayList[1][flag][indUniFreq1]))/np.abs((np.imag(arrayList[1][flag][indUniFreq1])))
        if mask:
            maskInd = np.logical_or(np.abs(np.imag(arrayList[0][flag][indUniFreq0])) < 1e-3,np.abs(np.imag(arrayList[1][flag][indUniFreq1])) < 1e-3)
            zPlot = np.ma.array(zPlot)
            zPlot[maskInd] = mask
        if cLevel:
            level = np.arange(-200,201,10)
            clevel = np.arange(-200,201,25)
        else:
            level = np.linspace(zPlot.min(),zPlot.max(),100)
            clevel = np.linspace(zPlot.min(),zPlot.max(),10)
    cs = ax.tricontourf(x,y,zPlot*100,levels=level*100,cmap=seicmap,extend='both') #,norm=colors.SymLogNorm(1e-2,linscale=2))
    if contourLine:
        csl = ax.tricontour(x,y,zPlot*100,levels=clevel*100,colors='k')
        plt.clabel(csl, fontsize=7, inline=1,fmt='%1.1e',inline_spacing=10)
    if colorbar:
        cb = plt.colorbar(cs,cax=ax.cax,ticks=clevel*100,format='%1.1e')
        for t in cb.ax.get_yticklabels():
            t.set_rotation(60)
            t.set_fontsize(8)

    ax.set_title(flag+' '+par,fontsize=8)

def plotIsoFreqNStipper(ax,freq,array,flag,par='abs',colorbar=True,colorNorm='SymLog',cLevel=True,contour=True):

    indUniFreq = np.where(freq==array['freq'])

    x, y = array['x'][indUniFreq],array['y'][indUniFreq]
    if par == 'abs':
        cmap = plt.get_cmap('OrRd_r')#seismic')
        zPlot = np.abs(array[flag][indUniFreq])
        if cLevel:
            level = np.logspace(-4,0,33)
            clevel = np.logspace(-4,0,5)
        else:
            level = np.linspace(zPlot.min(),zPlot.max(),100)
            clevel = np.linspace(zPlot.min(),zPlot.max(),10)
        if colorNorm=='SymLog':
            plotNorm = colors.LogNorm()
        else:
            plotNorm = colors.Normalize()
    elif par == 'real':
        cmap = plt.get_cmap('RdYlBu')
        zPlot = np.real(array[flag][indUniFreq])
        if cLevel:
            level = np.concatenate((-np.logspace(0,-4,33),np.logspace(-4,0,33)))
            clevel = np.concatenate((-np.logspace(0,-4,5),np.logspace(-4,0,5)))
        else:
            level = np.linspace(zPlot.min(),zPlot.max(),100)
            clevel = np.linspace(zPlot.min(),zPlot.max(),10)
        if colorNorm=='SymLog':
            plotNorm = colors.SymLogNorm(1e-4,linscale=2)
        else:
            plotNorm = colors.Normalize()
    elif par == 'imag':
        cmap = plt.get_cmap('RdYlBu')
        zPlot = np.imag(array[flag][indUniFreq])
        if cLevel:
            level = np.concatenate((-np.logspace(0,-4,33),np.logspace(-4,0,33)))
            clevel = np.concatenate((-np.logspace(0,-4,5),np.logspace(-4,0,5)))
        else:
            level = np.linspace(zPlot.min(),zPlot.max(),100)
            clevel = np.linspace(zPlot.min(),zPlot.max(),10)
        if colorNorm=='SymLog':
            plotNorm = colors.SymLogNorm(1e-4,linscale=2)
        else:
            plotNorm = colors.Normalize()
    if contour:
        cs = ax.tricontourf(x,y,zPlot,levels=level,cmap=cmap,norm=plotNorm)#,extend='both')
    else:
        uniX,uniY = np.unique(x),np.unique(y)
        X,Y = np.meshgrid(np.append(uniX-25,uniX[-1]+25),np.append(uniY-25,uniY[-1]+25))
        cs = ax.pcolor(X,Y,np.reshape(zPlot,(len(uniY),len(uniX))),levels=level,cmap=cmap,norm=plotNorm,edgecolors='k', linewidths=0.5)
    if colorbar:
        plt.colorbar(cs,cax=ax.cax,ticks=clevel,format='%1.2e')
    ax.set_title(flag+' '+par,fontsize=8)

def plotIsoStaImpedance(ax, loc, array, flag, par='abs',
                        pSym='s', pColor=None, addLabel='', zorder=1):

    appResFact = 1/(8*np.pi**2*10**(-7))
    treshold = 1.0 # 1 meter
    indUniSta = np.sqrt(np.sum((array[['x','y']].view((float,2))-loc)**2,axis=1)) < treshold
    freq = array['freq'][indUniSta]

    if par == 'abs':
        zPlot = np.abs(array[flag][indUniSta])
    elif par == 'real':
        zPlot = np.real(array[flag][indUniSta])
    elif par == 'imag':
        zPlot = np.imag(array[flag][indUniSta])
    elif par == 'res':
        zPlot = (appResFact/freq)*np.abs(array[flag][indUniSta])**2
    elif par == 'phs':
        zPlot = np.arctan2(array[flag][indUniSta].imag,array[flag][indUniSta].real)*(180/np.pi)

    if not pColor:
        if 'xx' in flag:
            lab = 'XX'
            pColor = 'g'
        elif 'xy' in flag:
            lab = 'XY'
            pColor = 'r'
        elif 'yx' in flag:
            lab = 'YX'
            pColor = 'b'
        elif 'yy' in flag:
            lab = 'YY'
            pColor = 'y'

    ax.plot(freq,zPlot,color=pColor,marker=pSym,label=flag+addLabel,zorder=zorder)


def plotPsudoSectNSimpedance(ax,sectDict,array,flag,par='abs',colorbar=True,colorNorm='None',cLevel=None,contour=True):

    indSect = np.where(sectDict.values()[0]==array[sectDict.keys()[0]])

    # Define the plot axes
    if 'x' in sectDict.keys()[0]:
        x = array['y'][indSect]
    else:
        x = array['x'][indSect]
    y = array['freq'][indSect]

    if par == 'abs':
        zPlot = np.abs(array[flag][indSect])
        cmap = plt.get_cmap('OrRd_r')#seismic')
        if cLevel:
            level = np.logspace(0,-5,31,endpoint=True)
            clevel = np.logspace(0,-4,5,endpoint=True)
        else:
            level = np.linspace(zPlot.min(),zPlot.max(),100,endpoint=True)
            clevel = np.linspace(zPlot.min(),zPlot.max(),10,endpoint=True)

    elif par == 'ares':
        zPlot = np.abs(array[flag][indSect])**2/(8*np.pi**2*10**(-7)*array['freq'][indSect])
        cmap = plt.get_cmap('RdYlBu')#seismic)
        if cLevel:
            zMax = np.log10(cLevel[1])
            zMin = np.log10(cLevel[0])
        else:
            zMax = (np.ceil(np.log10(np.abs(zPlot).max())))
            zMin = (np.floor(np.log10(np.abs(zPlot).min())))
        level = np.logspace(zMin,zMax,(zMax-zMin)*8+1,endpoint=True)
        clevel = np.logspace(zMin,zMax,(zMax-zMin)*2+1,endpoint=True)
        plotNorm = colors.LogNorm()

    elif par == 'aphs':
        zPlot = np.arctan2(array[flag][indSect].imag,array[flag][indSect].real)*(180/np.pi)
        cmap = plt.get_cmap('RdYlBu')#seismic)
        if cLevel:
            zMax = cLevel[1]
            zMin = cLevel[0]
        else:
            zMax = (np.ceil(zPlot).max())
            zMin = (np.floor(zPlot).min())
        level = np.arange(zMin,zMax+.1,1)
        clevel = np.arange(zMin,zMax+.1,10)
        plotNorm = colors.Normalize()

    elif par == 'real':
        zPlot = np.real(array[flag][indSect])
        cmap = plt.get_cmap('Spectral') #('RdYlBu')
        if cLevel:
            zMax = np.log10(cLevel[1])
            zMin = np.log10(cLevel[0])
        else:
            zMax = (np.ceil(np.log10(np.abs(zPlot).max())))
            zMin = (np.floor(np.log10(np.abs(zPlot).min())))
        level = np.concatenate((-np.logspace(zMax,zMin-.125,(zMax-zMin)*8+1,endpoint=True),np.logspace(zMin-.125,zMax,(zMax-zMin)*8+1,endpoint=True)))
        clevel = np.concatenate((-np.logspace(zMax,zMin,(zMax-zMin)*1+1,endpoint=True),np.logspace(zMin,zMax,(zMax-zMin)*1+1,endpoint=True)))
        plotNorm = colors.SymLogNorm(np.abs(level).min(),linscale=0.1)
    elif par == 'imag':
        zPlot = np.imag(array[flag][indSect])
        cmap = plt.get_cmap('Spectral') #('RdYlBu')

        if cLevel:
            zMax = np.log10(cLevel[1])
            zMin = np.log10(cLevel[0])
        else:
            zMax = (np.ceil(np.log10(np.abs(zPlot).max())))
            zMin = (np.floor(np.log10(np.abs(zPlot).min())))
        level = np.concatenate((-np.logspace(zMax,zMin-.125,(zMax-zMin)*8+1,endpoint=True),np.logspace(zMin-.125,zMax,(zMax-zMin)*8+1,endpoint=True)))
        clevel = np.concatenate((-np.logspace(zMax,zMin,(zMax-zMin)*1+1,endpoint=True),np.logspace(zMin,zMax,(zMax-zMin)*1+1,endpoint=True)))
        plotNorm = colors.SymLogNorm(np.abs(level).min(),linscale=0.1)

    if colorNorm=='SymLog':
        plotNorm = colors.SymLogNorm(np.abs(level).min(),linscale=0.1)
    elif colorNorm=='Lin':
        plotNorm = colors.Normalize()
    elif colorNorm=='Log':
        plotNorm = colors.LogNorm()
    if contour:
        cs = ax.tricontourf(x,y,zPlot,levels=level,cmap=cmap,norm=plotNorm)#,extend='both')
    else:
        uniX,uniY = np.unique(x),np.unique(y)
        X,Y = np.meshgrid(np.append(uniX-25,uniX[-1]+25),np.append(uniY-25,uniY[-1]+25))
        cs = ax.pcolor(X,Y,np.reshape(zPlot,(len(uniY),len(uniX))),cmap=cmap,norm=plotNorm)
    if colorbar:
        csB = plt.colorbar(cs,cax=ax.cax,ticks=clevel,format='%1.2e')
        # csB.on_mappable_changed(cs)
        ax.set_title(flag+' '+par,fontsize=8)
        return cs, csB
    return cs,None

def plotPsudoSectNSDiff(ax,sectDict,arrayList,flag,par='abs',colorbar=True,colorNorm='SymLog',cLevel=None,contour=True,mask=None,useLog=False):

    def sortInArr(arr):
        return np.sort(arr,order=['freq','x','y','z'])
    # Find the index for the slice
    indSect0 = np.where(sectDict.values()[0]==arrayList[0][sectDict.keys()[0]])
    indSect1 = np.where(sectDict.values()[0]==arrayList[1][sectDict.keys()[0]])
    # Extract and sort the mats
    arr0 = sortInArr(arrayList[0][indSect0])
    arr1 = sortInArr(arrayList[1][indSect1])

    # Define the plot axes
    if 'x' in sectDict.keys()[0]:
        x0 = arr0['y']
        x1 = arr1['y']
    else:
        x0 = arr0['x']
        x1 = arr1['x']
    y0 = arr0['freq']
    y1 = arr1['freq']


    if par == 'abs':
        if useLog:
            zPlot = (np.log10(np.abs(arr0[flag])) - np.log10(np.abs(arr1[flag])))/np.log10(np.abs(arr1[flag]))
        else:
            zPlot = (np.abs(arr0[flag]) - np.abs(arr1[flag]))/np.abs(arr1[flag])
        if mask:
            maskInd = np.logical_or(np.abs(arr0[flag])< 1e-3,np.abs(arr1[flag]) < 1e-3)
            zPlot = np.ma.array(zPlot)
            zPlot[maskInd] = mask
        cmap = plt.get_cmap('RdYlBu')#seismic)
    elif par == 'ares':
        arF = 1/(8*np.pi**2*10**(-7))
        if useLog:
            zPlot = (np.log10((arF/arr0['freq'])*np.abs(arr0[flag])**2) - np.log10((arF/arr1['freq'])*np.abs(arr1[flag])**2))/np.log10((arF/arr1['freq'])*np.abs(arr1[flag])**2)
        else:
            zPlot = ((arF/arr0['freq'])*np.abs(arr0[flag])**2 - (arF/arr1['freq'])*np.abs(arr1[flag])**2)/((arF/arr1['freq'])*np.abs(arr1[flag])**2)
        if mask:
            maskInd = np.logical_or(np.abs(arr0[flag])< 1e-3,np.abs(arr1[flag]) < 1e-3)
            zPlot = np.ma.array(zPlot)
            zPlot[maskInd] = mask
        cmap = plt.get_cmap('Spectral')#seismic)

    elif par == 'aphs':
        if useLog:
            zPlot = (np.log10(np.arctan2(arr0[flag].imag,arr0[flag].real)*(180/np.pi)) - np.log10(np.arctan2(arr1[flag].imag,arr1[flag].real)*(180/np.pi)) )/np.log10(np.arctan2(arr1[flag].imag,arr1[flag].real)*(180/np.pi))
        else:
            zPlot = ( np.arctan2(arr0[flag].imag,arr0[flag].real)*(180/np.pi) - np.arctan2(arr1[flag].imag,arr1[flag].real)*(180/np.pi) )/(np.arctan2(arr1[flag].imag,arr1[flag].real)*(180/np.pi))
        if mask:
            maskInd = np.logical_or(np.abs(arr0[flag])< 1e-3,np.abs(arr1[flag]) < 1e-3)
            zPlot = np.ma.array(zPlot)
            zPlot[maskInd] = mask
        cmap = plt.get_cmap('Spectral')#seismic)
    elif par == 'real':
        if useLog:
            zPlot = (np.log10(arr0[flag].real) - np.log10(arr1[flag].real))/np.log10(arr1[flag].real)
        else:
            zPlot = (arr0[flag].real - arr1[flag].real)/arr1[flag].real
        if mask:
            maskInd = np.logical_or(arr0[flag].real< 1e-3,arr1[flag].real < 1e-3)
            zPlot = np.ma.array(zPlot)
            zPlot[maskInd] = mask
        cmap = plt.get_cmap('Spectral') #('Spectral')

    elif par == 'imag':
        if useLog:
            zPlot = (np.log10(arr0[flag].imag) - np.log10(arr1[flag].imag))/np.log10(arr1[flag].imag)
        else:
            zPlot = (arr0[flag].imag - arr1[flag].imag)/arr1[flag].imag
        if mask:
            maskInd = np.logical_or(arr0[flag].imag< 1e-3,arr1[flag].imag < 1e-3)
            zPlot = np.ma.array(zPlot)
            zPlot[maskInd] = mask
        cmap = plt.get_cmap('Spectral') #('RdYlBu')

    if cLevel:
        zMax = np.log10(cLevel[1])
        zMin = np.log10(cLevel[0])
    else:
        zMax = (np.ceil(np.log10(np.abs(zPlot).max())))
        zMin = (np.floor(np.log10(np.abs(zPlot).min())))


    if colorNorm=='SymLog':
        level = np.concatenate((-np.logspace(zMax,zMin-.125,(zMax-zMin)*8+1,endpoint=True),np.logspace(zMin-.125,zMax,(zMax-zMin)*8+1,endpoint=True)))
        clevel = np.concatenate((-np.logspace(zMax,zMin,(zMax-zMin)*1+1,endpoint=True),np.logspace(zMin,zMax,(zMax-zMin)*1+1,endpoint=True)))
        plotNorm = colors.SymLogNorm(np.abs(level).min(),linscale=0.1)
    elif colorNorm=='Lin':
        if cLevel:
            level = np.arange(cLevel[0],cLevel[1]+.1,(cLevel[1] - cLevel[0])/50.)
            clevel = np.arange(cLevel[0],cLevel[1]+.1,(cLevel[1] - cLevel[0])/10.)
        else:
            level = np.arange(zPlot.min(),zPlot.max(),(zPlot.max() - zPlot.min())/50.)
            clevel = np.arange(zPlot.min(),zPlot.max(),(zPlot.max() - zPlot.min())/10.)
        plotNorm = colors.Normalize()
    elif colorNorm=='Log':
        level = np.logspace(zMin-.125,zMax,(zMax-zMin)*8+1,endpoint=True)
        clevel = np.logspace(zMin,zMax,(zMax-zMin)*2+1,endpoint=True)
        plotNorm = colors.LogNorm()
    if contour:
        cs = ax.tricontourf(x0,y0,zPlot*100,levels=level*100,cmap=cmap,norm=plotNorm,extend='both')#,extend='both')
    else:
        uniX,uniY = np.unique(x0),np.unique(y0)
        X,Y = np.meshgrid(np.append(uniX-25,uniX[-1]+25),np.append(uniY-25,uniY[-1]+25))
        cs = ax.pcolor(X,Y,np.reshape(zPlot,(len(uniY),len(uniX))),cmap=cmap,norm=plotNorm)
    if colorbar:
        csB = plt.colorbar(cs,cax=ax.cax,ticks=clevel*100,format='%1.2e')
        # csB.on_mappable_changed(cs)
        ax.set_title(flag+' '+par + ' diff',fontsize=8)
        return cs, csB
    return cs,None

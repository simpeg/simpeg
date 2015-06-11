# Utils used for the data,
import numpy as np
def getAppRes(MTdata):
    # Make impedance
    zList = []
    for src in MTdata.survey.srcList:
        zc = [src.freq]
        for rx in src.rxList:
            if 'i' in rx.rxType:
                m=1j
            else:
                m = 1
            zc.append(m*MTdata[src,rx])
        zList.append(zc)
    return [appResPhs(zList[i][0],np.sum(zList[i][1:3])) for i in np.arange(len(zList))]


def appResPhs(freq,z):
    app_res = ((1./(8e-7*np.pi**2))/freq)*np.abs(z)**2
    app_phs = np.arctan2(-z.imag,z.real)*(180/np.pi)
    return app_res, app_phs
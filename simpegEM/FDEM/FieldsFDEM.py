import numpy as np


class FieldsFDEM(object):
    """docstring for FieldsFDEM"""

    phi = None #: Electric potential
    A = None #: Magnetic vector potential
    e = None #: Electric field
    b = None #: Magnetic flux density
    j = None #: Current density
    h = None #: Magnetic field

    def __init__(self, mesh, nTx, nFreq, store='e'):

        self.nFreq = nFreq #: Number of times
        self.nTx = nTx #: Number of transmitters
        self.mesh = mesh

    def update(self, newFields, fInd):
        self.set_b(newFields['b'], fInd)
        self.set_e(newFields['e'], fInd)

    ####################################################
    # Get Methods
    ####################################################

    def get_b(self, ind):
        return self.b[ind,:,:]

    def get_e(self, ind):
        return self.e[ind,:,:]

    ####################################################
    # Set Methods
    ####################################################

    def set_b(self, b, ind):
        if self.b is None:
            self.b = np.zeros((self.nFreq, np.sum(self.mesh.nF), self.nTx), dtype=complex)
            self.b[:] = np.nan
        if len(b.shape) == 1:
            b = b[:, np.newaxis]
        self.b[ind,:,:] = b

    def set_e(self, e, ind):
        if self.e is None:
            self.e = np.zeros((self.nFreq, np.sum(self.mesh.nE), self.nTx), dtype=complex)
            self.e[:] = np.nan
        if len(e.shape) == 1:
            e = e[:, np.newaxis]
        self.e[ind,:,:] = e

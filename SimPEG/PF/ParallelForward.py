from SimPEG.Utils import mkvc
from SimPEG import Utils
import numpy as np
import multiprocessing
import scipy.constants as constants

class Gravity(object):
    """
        Add docstring once it works
    """

    progressIndex = -1

    def __init__(self):
        super(Gravity, self).__init__()

    def calcForward(self, rxLoc, Xn, Yn, Zn, n_cpu, forwardOnly, m=None, rxType='z'):

        if n_cpu is None:
            self.n_cpu = multiprocessing.cpu_count()

        self.rxLoc, self.Xn, self.Yn, self.Zn = rxLoc, Xn, Yn, Zn
        self.rxType = rxType
        self.forwardOnly = forwardOnly
        self.nD = rxLoc.shape[0]
        self.model = m

        pool = multiprocessing.Pool(self.n_cpu)

        # rowInd = np.linspace(0, self.nD, self.n_cpu+1).astype(int)

        # job_args = []

        # for ii in range(self.n_cpu):

        #     nRows = int(rowInd[ii+1]-rowInd[ii])
        #     job_args += [(rowInd[ii], nRows, m)]

        # result = pool.map(self.getTblock, job_args)

        result = pool.map(self.calcTrow, [self.rxLoc[ii, :] for ii in range(self.nD)])
        pool.close()
        pool.join()

        if self.forwardOnly:
            return mkvc(np.vstack(result))

        else:
            return np.vstack(result)

    def calcTrow(self, xyzLoc):
        """
        Load in the active nodes of a tensor mesh and computes the gravity tensor
        for a given observation location xyzLoc[obsx, obsy, obsz]

        INPUT:
        Xn, Yn, Zn: Node location matrix for the lower and upper most corners of
                    all cells in the mesh shape[nC,2]
        M
        OUTPUT:
        Tx = [Txx Txy Txz]
        Ty = [Tyx Tyy Tyz]
        Tz = [Tzx Tzy Tzz]

        where each elements have dimension 1-by-nC.
        Only the upper half 5 elements have to be computed since symetric.
        Currently done as for-loops but will eventually be changed to vector
        indexing, once the topography has been figured out.

        """

        NewtG = constants.G*1e+8  # Convertion from mGal (1e-5) and g/cc (1e-3)
        eps = 1e-8  # add a small value to the locations to avoid

        # Pre-allocate space for 1D array
        row = np.zeros((1, self.Xn.shape[0]))

        dz = xyzLoc[2] - self.Zn

        dy = self.Yn - xyzLoc[1]

        dx = self.Xn - xyzLoc[0]

        # Compute contribution from each corners
        for aa in range(2):
            for bb in range(2):
                for cc in range(2):

                    r = (
                            mkvc(dx[:, aa]) ** 2 +
                            mkvc(dy[:, bb]) ** 2 +
                            mkvc(dz[:, cc]) ** 2
                        ) ** (0.50)

                    if self.rxType == 'x':
                        row = row - NewtG * (-1) ** aa * (-1) ** bb * (-1) ** cc * (
                            dy[:, bb] * np.log(dz[:, cc] + r + eps) +
                            dz[:, cc] * np.log(dy[:, bb] + r + eps) -
                            dx[:, aa] * np.arctan(dy[:, bb] * dz[:, cc] /
                                                  (dx[:, aa] * r + eps)))

                    elif self.rxType == 'y':
                        row = row - NewtG * (-1) ** aa * (-1) ** bb * (-1) ** cc * (
                            dx[:, aa] * np.log(dz[:, cc] + r + eps) +
                            dz[:, cc] * np.log(dx[:, aa] + r + eps) -
                            dy[:, bb] * np.arctan(dx[:, aa] * dz[:, cc] /
                                                  (dy[:, bb] * r + eps)))

                    else:
                        row -= NewtG * (-1) ** aa * (-1) ** bb * (-1) ** cc * (
                            dx[:, aa] * np.log(dy[:, bb] + r + eps) +
                            dy[:, bb] * np.log(dx[:, aa] + r + eps) -
                            dz[:, cc] * np.arctan(dx[:, aa] * dy[:, bb] /
                                                  (dz[:, cc] * r + eps)))

        if self.forwardOnly:
            return np.dot(row, self.model)
        else:
            return row

    # def getTblock(self, args):
    #     """
    #         Calculate rows of sensitivity
    #     """
    #     indStart, nRows, m = args
    #     block = []

    #     if self.forwardOnly:
    #         for ii in range(nRows):
    #             block += [np.dot(self.calcTrow(self.rxLoc[indStart+ii, :]), m)]

    #             # Monitor progress of first thread
    #             # if indStart == 0:
    #             #     self.progress(ii, nRows)

    #         return mkvc(np.vstack(block))

    #     else:
    #         for ii in range(nRows):
    #             block += [self.calcTrow(self.rxLoc[indStart+ii, :])]

    #             # Monitor progress of first thread
    #             # if indStart == 0:
    #             #     self.progress(ii, nRows)

    #         return np.vstack(block)

    # def progress(self, iter, nRows):
    #     """
    #     progress(iter,prog,final)

    #     Function measuring the progress of a process and print to screen the %.
    #     Useful to estimate the remaining runtime of a large problem.

    #     Created on Dec, 20th 2015

    #     @author: dominiquef
    #     """
    #     arg = np.floor(iter/nRows*10.)
    #     if arg > self.progressIndex:
    #         print("Done " + str(arg*10) + " %")
    #         self.progressIndex = arg

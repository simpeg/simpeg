from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import properties
import numpy as np
import multiprocessing
from ..simulation import LinearSimulation
from scipy.sparse import csr_matrix as csr
from SimPEG.utils import mkvc, sdiag
from .. import props
from dask import delayed, array, config
from dask.diagnostics import ProgressBar

try:
    from pymatsolver import Pardiso as Solver
except ImportError:
    from SimPEG import SolverLU as Solver

###############################################################################
#                                                                             #
#                             Base Potential Fields Problem                   #
#                                                                             #
###############################################################################


class BasePFSimulation(LinearSimulation):

    store_sensitivity = properties.Bool(
        "Store the sensitivity to disk",
        default=True
    )

    actInd = properties.Array(
        "Array of active cells (ground)",
        dtype=(bool, int),
        default=None
    )

    n_cpu = properties.Integer(
        "Number of processors used for the forward simulation",
        default=int(multiprocessing.cpu_count())
    )

    store_sensitivities = properties.StringChoice(
        "Compute and store G",
        choices=['disk', 'ram', 'forward_only'],
        default='disk'
    )

    chunk_format = properties.StringChoice(
        "Apply memory chunks along rows of G",
        choices=['equal', 'row', 'auto'],
        default='equal'
    )

    sensitivity_path = properties.String(
        "Directory used to store the sensitivity matrix on disk",
        default="./Inversion/sensitivity.zarr"
    )

    def __init__(self, mesh, **kwargs):

        LinearSimulation.__init__(self, mesh, **kwargs)

        # Find non-zero cells
        if getattr(self, 'actInd', None) is not None:
            if self.actInd.dtype == 'bool':
                indices = np.where(self.actInd)[0]
            else:
                indices = self.actInd

        else:

            indices = np.asarray(range(self.mesh.nC))

        self.nC = len(indices)

        # Create active cell projector
        projection = csr(
            (np.ones(self.nC), (indices, range(self.nC))),
            shape=(self.mesh.nC, self.nC)
        )

        # Create vectors of nodal location for the lower and upper corners
        bsw = (self.mesh.gridCC - self.mesh.h_gridded/2.)
        tne = (self.mesh.gridCC + self.mesh.h_gridded/2.)

        xn1, xn2 = bsw[:, 0], tne[:, 0]
        yn1, yn2 = bsw[:, 1], tne[:, 1]

        self.Yn = projection.T*np.c_[mkvc(yn1), mkvc(yn2)]
        self.Xn = projection.T*np.c_[mkvc(xn1), mkvc(xn2)]

        # Allows for 2D mesh where Zn is defined by user
        if self.mesh.dim > 2:
            zn1, zn2 = bsw[:, 2], tne[:, 2]
            self.Zn = projection.T*np.c_[mkvc(zn1), mkvc(zn2)]

    def linear_operator(self):

        self.nC = self.modelMap.shape[0]

        n_data_comp = len(self.survey.components)

        components = np.array(list(self.survey.components.keys()))
        active_components = np.hstack([np.c_[values] for values in self.survey.components.values()]).tolist()

        if self.store_sensitivities != 'ram':

            row = delayed(self.evaluate_integral, pure=True)

            rows = [
                array.from_delayed(
                    row(receiver_location, components[component]), dtype=np.float32, shape=(n_data_comp,  self.nC)
                )
                for receiver_location, component in zip(self.survey.receiver_locations.tolist(), active_components)
            ]
            stack = array.vstack(rows)

            # Chunking options
            if self.chunk_format == 'row' or self.store_sensitivities == 'forward_only':
                config.set({'array.chunk-size': f'{self.max_chunk_size}MiB'})
                # Autochunking by rows is faster and more memory efficient for
                # very large problems sensitivty and forward calculations
                stack = stack.rechunk({0: 'auto', 1: -1})

            elif self.chunk_format == 'equal':
                # Manual chunks for equal number of blocks along rows and columns.
                # Optimal for Jvec and Jtvec operations
                n_chunks_col = 1
                n_chunks_row = 1
                row_chunk = int(np.ceil(stack.shape[0]/n_chunks_row))
                col_chunk = int(np.ceil(stack.shape[1]/n_chunks_col))
                chunk_size = row_chunk*col_chunk*8*1e-6  # in Mb

                # Add more chunks along either dimensions until memory falls below target
                while chunk_size >= self.max_chunk_size:

                    if row_chunk > col_chunk:
                        n_chunks_row += 1
                    else:
                        n_chunks_col += 1

                    row_chunk = int(np.ceil(stack.shape[0]/n_chunks_row))
                    col_chunk = int(np.ceil(stack.shape[1]/n_chunks_col))
                    chunk_size = row_chunk*col_chunk*8*1e-6  # in Mb

                stack = stack.rechunk((row_chunk, col_chunk))
            else:
                # Auto chunking by columns is faster for Inversions
                config.set({'array.chunk-size': f'{self.max_chunk_size}MiB'})
                stack = stack.rechunk({0: -1, 1: 'auto'})

            if self.store_sensitivities == 'forward_only':

                with ProgressBar():
                    print("Forward calculation: ")
                    pred = array.dot(stack, self.model).compute()

                return pred

            else:
                if os.path.exists(self.sensitivity_path):

                    kernel = array.from_zarr(self.sensitivity_path)

                    if np.all(np.r_[
                            np.any(np.r_[kernel.chunks[0]] == stack.chunks[0]),
                            np.any(np.r_[kernel.chunks[1]] == stack.chunks[1]),
                            np.r_[kernel.shape] == np.r_[stack.shape]]):
                        # Check that loaded kernel matches supplied data and mesh
                        print("Zarr file detected with same shape and chunksize ... re-loading")

                        return kernel
                    else:
                        print("Zarr file detected with wrong shape and chunksize ... over-writing")

                with ProgressBar():
                    print("Saving kernel to zarr: " + self.sensitivity_path)
                    kernel = array.to_zarr(stack, self.sensitivity_path, compute=True, return_stored=True, overwrite=True)

        else:
            # TODO
            # Process in parallel using multiprocessing
            # pool = multiprocessing.Pool(self.n_cpu)
            # kernel = pool.map(
            #   self.evaluate_integral, [
            #       receiver for receiver in self.survey.receiver_locations.tolist()
            # ])
            # pool.close()
            # pool.join()

            # Single threaded
            kernel = np.vstack([
                self.evaluate_integral(receiver, components[component])
                for receiver, component in zip(self.survey.receiver_locations.tolist(), active_components)
            ])

        return kernel

    def evaluate_integral(self):
        """
        evaluate_integral

        Compute the forward linear relationship between the model and the physics at a point.
        :param self:
        :return:
        """

        raise RuntimeError(f"Integral calculations must implemented by the subclass {self}.")

def progress(iter, prog, final):
    """
    progress(iter,prog,final)
    Function measuring the progress of a process and print to screen the %.
    Useful to estimate the remaining runtime of a large problem.
    Created on Dec, 20th 2015
    @author: dominiquef
    """
    arg = np.floor(float(iter)/float(final)*10.)

    if arg > prog:

        print("Done " + str(arg*10) + " %")
        prog = arg

    return prog

def get_dist_wgt(mesh, receiver_locations, actv, R, R0):
    """
    get_dist_wgt(xn,yn,zn,receiver_locations,R,R0)

    Function creating a distance weighting function required for the magnetic
    inverse problem.

    INPUT
    xn, yn, zn : Node location
    receiver_locations       : Observation locations [obsx, obsy, obsz]
    actv        : Active cell vector [0:air , 1: ground]
    R           : Decay factor (mag=3, grav =2)
    R0          : Small factor added (default=dx/4)

    OUTPUT
    wr       : [nC] Vector of distance weighting

    Created on Dec, 20th 2015

    @author: dominiquef
    """

    # Find non-zero cells
    if actv.dtype == 'bool':
        inds = np.asarray(
            [
                inds for inds, elem in enumerate(actv, 1) if elem
            ],
            dtype=int
        ) - 1
    else:
        inds = actv

    nC = len(inds)

    # Create active cell projector
    P = csr((np.ones(nC), (inds, range(nC))),
                      shape=(mesh.nC, nC))

    # Geometrical constant
    p = 1 / np.sqrt(3)

    # Create cell center location
    Ym, Xm, Zm = np.meshgrid(mesh.vectorCCy, mesh.vectorCCx, mesh.vectorCCz)
    hY, hX, hZ = np.meshgrid(mesh.hy, mesh.hx, mesh.hz)

    # Remove air cells
    Xm = P.T * mkvc(Xm)
    Ym = P.T * mkvc(Ym)
    Zm = P.T * mkvc(Zm)

    hX = P.T * mkvc(hX)
    hY = P.T * mkvc(hY)
    hZ = P.T * mkvc(hZ)

    V = P.T * mkvc(mesh.vol)
    wr = np.zeros(nC)

    ndata = receiver_locations.shape[0]
    count = -1
    print("Begin calculation of distance weighting for R= " + str(R))

    for dd in range(ndata):

        nx1 = (Xm - hX * p - receiver_locations[dd, 0])**2
        nx2 = (Xm + hX * p - receiver_locations[dd, 0])**2

        ny1 = (Ym - hY * p - receiver_locations[dd, 1])**2
        ny2 = (Ym + hY * p - receiver_locations[dd, 1])**2

        nz1 = (Zm - hZ * p - receiver_locations[dd, 2])**2
        nz2 = (Zm + hZ * p - receiver_locations[dd, 2])**2

        R1 = np.sqrt(nx1 + ny1 + nz1)
        R2 = np.sqrt(nx1 + ny1 + nz2)
        R3 = np.sqrt(nx2 + ny1 + nz1)
        R4 = np.sqrt(nx2 + ny1 + nz2)
        R5 = np.sqrt(nx1 + ny2 + nz1)
        R6 = np.sqrt(nx1 + ny2 + nz2)
        R7 = np.sqrt(nx2 + ny2 + nz1)
        R8 = np.sqrt(nx2 + ny2 + nz2)

        temp = (R1 + R0)**-R + (R2 + R0)**-R + (R3 + R0)**-R + \
            (R4 + R0)**-R + (R5 + R0)**-R + (R6 + R0)**-R + \
            (R7 + R0)**-R + (R8 + R0)**-R

        wr = wr + (V * temp / 8.)**2.

        count = progress(dd, count, ndata)

    wr = np.sqrt(wr) / V
    wr = mkvc(wr)
    wr = np.sqrt(wr / (np.max(wr)))

    print("Done 100% ...distance weighting completed!!\n")

    return wr

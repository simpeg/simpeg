from ....electromagnetics.natural_source.simulation import BaseNSEMSimulation as Sim
from ....electromagnetics.natural_source.fields import Fields3DPrimarySecondary
from ....utils import Zero

from ...utils import compute_chunk_sizes
from ....utils import mkvc
from ....survey import Data
import multiprocessing
import dask
import dask.array as da
import os
import shutil
import numpy as np
import zarr
from pyMKL import mkl_set_num_threads


Sim.n_cpu = int(multiprocessing.cpu_count())
Sim.j_path = './sensitivity/'
Sim._jtjdiag = None
Sim._Jmatrix_pointers = None

def dask_Jvec(self, m, v, f=None):
    """
    Function to calculate the data sensitivities dD/dm times a vector.
    :param numpy.ndarray m: conductivity model (nP,)
    :param numpy.ndarray v: vector which we take sensitivity product with (nP,)
    :param SimPEG.EM.NSEM.FieldsNSEM (optional) u: NSEM fields object, if not given it is calculated
    :rtype: numpy.ndarray
    :return: Jv (nData,) Data sensitivities wrt m
    """

    # Calculate the fields if not given as input
    # if f is None:
    #     # f = self.fields(m)
    #     F = self.fields2(m)
    # Set current model
    self.model = m
    # Initiate the Jv list
    Jv = []
    number_of_frequencies = len(self.survey.frequencies)
    number_of_components = len(self.survey.get_sources_by_frequency(self.survey.frequencies[0])[0].receiver_list)
    n_dim = number_of_frequencies * number_of_components
    m_dim = int(self.survey.nD / (number_of_components * number_of_frequencies))

    # Loop all the frequenies
    F = self.fieldsPair(self)
    for nF, freq in enumerate(self.survey.frequencies):
        # calculating fields on the fly ===
        Src = self.survey.get_sources_by_frequency(freq)[0]
        e_s = da.from_delayed(self.fieldByFrequency(freq, nF), (self.mesh.nE, 2), dtype=complex).compute()
        F[Src, 'e_pxSolution'] = e_s[:, 0]
        F[Src, 'e_pySolution'] = e_s[:, 1]
        # Get the system
        for src in self.survey.get_sources_by_frequency(freq):
            # need fDeriv_m = df/du*du/dm + df/dm
            # Construct du/dm, it requires a solve
            u_src = F[src, :]
            dA_dm_v = self.getADeriv(freq, u_src, v)
            dRHS_dm_v = self.getRHSDeriv(freq, v)

            # Calculate du/dm*v
            du_dm_v = self.Ainv[nF] * (-dA_dm_v + dRHS_dm_v)
            # Calculate the projection derivatives
            for rx in src.receiver_list:
                # Calculate dP/du*du/dm*v
                Jv.append(da.from_delayed(dask.delayed(rx.evalDeriv)(src, self.mesh, F, mkvc(du_dm_v)), shape=(m_dim,), dtype=float))
        # when running full inversion clearing the fields creates error and inversion crashes
        self.Ainv[nF].clean()
    # return Jv.flatten('F')
    return da.concatenate(Jv, axis=0).compute()
Sim.Jvec = dask_Jvec


def dask_Jtvec(self, m, v, f=None):
    """
    Function to calculate the transpose of the data sensitivities (dD/dm)^T times a vector.
    :param numpy.ndarray m: inversion model (nP,)
    :param numpy.ndarray v: vector which we take adjoint product with (nP,)
    :param SimPEG.EM.NSEM.FieldsNSEM f (optional): NSEM fields object, if not given it is calculated
    :rtype: numpy.ndarray
    :return: Jtv (nP,) Data sensitivities wrt m
    """

    # if f is None:
    #     f = self.fields(m)
    self.model = m

    # Ensure v is a data object.
    if not isinstance(v, Data):
        v = Data(self.survey, v)
    # initiate dask array for Jtv
    Jtv = da.zeros(m.size)
    f = self.fieldsPair(self)
    for nF, freq in enumerate(self.survey.frequencies):
        # calculating fields on the fly ===
        Src = self.survey.get_sources_by_frequency(freq)[0]
        e_s = da.from_delayed(self.fieldByFrequency(freq, nF), (self.mesh.nE, 2), dtype=complex).compute()
        f[Src, 'e_pxSolution'] = e_s[:, 0]
        f[Src, 'e_pySolution'] = e_s[:, 1]

        for src in self.survey.get_sources_by_frequency(freq):
            # u_src needs to have both polarizations
            u_src = f[src, :]
            PTv = np.zeros((self.mesh.nE, 2), dtype=complex)
            for rx in src.receiver_list:
                # Get the adjoint evalDeriv
                real_or_imag = rx.component
                if real_or_imag == 'real':
                    PTv += rx.evalDeriv(src, self.mesh, f, mkvc(v[src, rx]), adjoint=True) # wrt f, need possibility wrt m
                elif real_or_imag == 'imag':
                    PTv += -rx.evalDeriv(src, self.mesh, f, mkvc(v[src, rx]), adjoint=True) # wrt f, need possibility wrt m
                else:
                    raise Exception('Must be real or imag')
            dA_duIT = da.from_delayed(dask.delayed(mkvc(self.Ainv[nF] * PTv)),
                                      shape=(self.mesh.nE * 2,), dtype=float)  # Force (nU,) shape

            dA_dmT = dask.delayed(self.getADeriv)(freq, u_src, dA_duIT, adjoint=True)
            dRHS_dmT = dask.delayed(self.getRHSDeriv)(freq, dA_duIT, adjoint=True)
            du_dmT = da.from_delayed(-dA_dmT, shape=(self.model.size,),
                                     dtype=complex)
            # Make du_dmT
            du_dmT += da.from_delayed(dRHS_dmT, shape=(self.model.size,), dtype=complex)
            Jtv += du_dmT.real
            # when running full inversion clearing the fields creates error and inversion crashes
            self.Ainv[nF].clean()
    return Jtv.compute()
Sim.Jtvec = dask_Jtvec


def dask_getJ(self, m, f=None):
    """
    Function to calculate the sensitivity matrix.
    :param numpy.ndarray m: inversion model (nP,)
    :param SimPEG.EM.NSEM.FieldsNSEM f (optional): NSEM fields object, if not given it is calculated
    :rtype: numpy.ndarray
    :return: J (nD, nP) Data sensitivities wrt m
    """
    self.model = m

    f = self.fieldsPair(self)
    fcount = 0
    Jmatrix_pointers = []
    nf = 0
    for freq in self.survey.frequencies:
        # calculating fields on the fly ===
        Src = self.survey.get_sources_by_frequency(freq)[0]
        e_s = da.from_delayed(self.fieldByFrequency(freq, nf), (self.mesh.nE, 2), dtype=complex).compute()
        f[Src, 'e_pxSolution'] = e_s[:, 0]
        f[Src, 'e_pySolution'] = e_s[:, 1]
        ATinv = self.Ainv[nf]

        # construct all the PT matricies dasked
        count = 0
        for src in self.survey.get_sources_by_frequency(freq):
            for rx in src.receiver_list:
                # Get the adjoint evalDeriv
                # Need to make PT
                # PT = np.empty((self.mesh.nE, rx.nD*2), dtype=complex, order='F')
                PT_ = []
                for i in range(rx.nD):
                    v = np.zeros(rx.nD)
                    v[i] = 1.0
                    PT_.append(da.from_delayed(dask.delayed(rx.evalDeriv)
                        (src, self.mesh, f, v, adjoint=True),
                        shape=(self.mesh.nE, 2), dtype=complex)
                    )
                PT = da.hstack(PT_)
                blockName = self.j_path + "F" + str(fcount) + "_P" + str(count) + ".zarr"
                da.to_zarr(PT.rechunk('auto'), blockName)
                count += 1

        # run solver in serial due to pardiso not being thread safe
        dA_duIT_list = []
        for ii in range(count):
            blockName = self.j_path + "F" + str(fcount) + "_P" + str(count) + ".zarr"
            dA_duIT = ATinv * da.from_zarr(blockName).compute()
            dA_duIT_list.append(dA_duIT.reshape(-1, rx.nD, order='F')) # shape now nUxnD
        # [Clean the factorization, clear memory.
        ATinv.clean()

        # construct all the sub J matricies of the total J matrix - dasked
        count = 0
        for src in self.survey.get_sources_by_frequency(freq):
            # u_src needs to have both polarizations
            for rx in src.receiver_list:
                # Jsub = da.from_delayed(self.constructJsubMatrix(freq, src, rx, f[src, :], dA_duIT_list[count]), shape=(rx.nD, self.model.size), dtype=float)
                Jsub = dask.delayed(self.constructJsubMatrix)(freq, src, rx, f[src, :], dA_duIT_list[count])
                blockName = self.j_path + "F" + str(fcount) + "_J" + str(count) + ".zarr"
                da.to_zarr(Jsub.rechunk('auto'), blockName)
                Jmatrix_pointers.append(blockName)
                count += 1
        fcount += 1
        nf += 1
    self._Jmatrix_pointers = Jmatrix_pointers
    return Jmatrix_pointers
Sim.getJ = dask_getJ


def constructJsubMatrix(self, freq, src, rx, u_src, dA_duIT):
    # getADeriv and getRHSDeriv should be updated to accept and return
    # matrices, but for now this works.
    # They should also be update to return the real parts as well.
    dA_dmT = np.empty((rx.nD, self.model.size))
    dRHS_dmT = np.empty((rx.nD, self.model.size))
    for i in range(rx.nD):
        dA_dmT[i, :] = self.getADeriv(freq, u_src, dA_duIT[:, i], adjoint=True).real
        dRHS_dmT[i, :] = self.getRHSDeriv(freq, dA_duIT[:, i], adjoint=True).real
    # Make du_dmT
    du_dmT = -dA_dmT + dRHS_dmT
    # Now just need to put it in the right spot.....
    real_or_imag = rx.component
    if real_or_imag == 'real':
        J_rows = du_dmT
    elif real_or_imag == 'imag':
        J_rows = -du_dmT
    else:
        raise Exception('Must be real or imag')
    return J_rows
Sim.constructJsubMatrix = constructJsubMatrix


def dask_getJtJdiag(self, m=None):
    if m is not None:
        self._Jmatrix_pointers = self.getJ(m)

    # now loop through the martricies
    jtjdiag = da.zeros(self.model.size)
    for subJ in self._Jmatrix_pointers:
        jtjdiag += da.sum(da.from_zarr(subJ)**2)

    jtjdiag = jtjdiag**0.5
    self.jtjdiag = jtjdiag / jtjdiag.max()
    return self._jtjdiag.compute()
Sim.getJtJdiag = dask_getJtJdiag


@dask.delayed(pure=True)
def fieldByFrequency(self, freq, freq_index):
    """
    Function to calculate all the fields for the model m.
    :param numpy.ndarray (nC,) m: Conductivity model
    :rtype: SimPEG.EM.NSEM.FieldsNSEM
    :return: Fields object with of the solution
    """
    if self.verbose:
        print('Starting work for {:.3e}'.format(freq))
        sys.stdout.flush()
    mkl_set_num_threads(self.n_cpu)
    if self.Ainv[int(freq_index)] is not None:
        self.Ainv[int(freq_index)].clean()

    A = self.getA(freq)
    rhs = self.getRHS(freq)
    # Solve the system
    self.Ainv[int(freq_index)] = self.Solver(A, **self.solver_opts)
    e_s = self.Ainv[int(freq_index)] * rhs

    return e_s
    # Ainv.clean()
Sim.fieldByFrequency = fieldByFrequency


def fields2(self, m=None):
    """
    Function to calculate all the fields for the model m.
    :param numpy.ndarray (nC,) m: Conductivity model
    :rtype: SimPEG.EM.NSEM.FieldsNSEM
    :return: Fields object with of the solution
    """
    self.Ainv = [None for i in range(self.survey.num_frequencies)]
    F = self.fieldsPair(self)
    e_s = []
    # ds = []
    for nf, freq in enumerate(self.survey.frequencies):
        # ==== Xarray stuff for when it is ready to handle in fields =====
        # Src = self.survey.get_sources_by_frequency(freq)[0]
        # e_s = da.from_delayed(self.fieldByFrequency(freq, nf), (self.mesh.nE, 2), dtype=complex)
        # ds.append(xr.DataArray(e_s, coords=[np.arange(0, self.mesh.nE, 1), ['e_pxSolution', 'e_pySolution']], dims=['index', 'space']))

        # ==== regular implementation =====
        e_s.append(da.from_delayed(self.fieldByFrequency(freq, nf), (self.mesh.nE, 2), dtype=complex))

    F_ = da.hstack(e_s).compute()
    # da.hstack(e_s).visualize()

    F[:, 'e_pxSolution'] = F_[:, ::2]
    F[:, 'e_pySolution'] = F_[:, 1::2]

    # index = np.arange(0, len(self.survey.frequencies), 1)
    # # self.fieldByFrequency(self.survey.frequencies[0], index[0])
    # pool = multiprocessing.Pool()
    # pool.map(self.fieldByFrequency, (index))
    # pool.close()
    # pool.join()
    return F
Sim.fields = fields2


def fields(self, m=None):
    """
    Function to calculate all the fields for the model m.
    :param numpy.ndarray (nC,) m: Conductivity model
    :rtype: SimPEG.EM.NSEM.FieldsNSEM
    :return: Fields object with of the solution
    """
    # Set the current model
    if m is not None:
        self.model = m

    F = self.fieldsPair(self)
    try:
        if self.Ainv[0] is not None:
            for i in range(self.survey.num_frequencies):
                self.Ainv[i].clean()
    except:
        pass
    self.Ainv = [None for i in range(self.survey.num_frequencies)]

    for nf, freq in enumerate(self.survey.frequencies):
        A = self.getA(freq)
        rhs = self.getRHS(freq)
        # Solve the system
        self.Ainv[nf] = self.Solver(A, **self.solver_opts)
        e_s = self.Ainv[nf] * rhs

        # Store the fields
        Src = self.survey.get_sources_by_frequency(freq)[0]
        # Store the fields
        # Use self._solutionType
        F[Src, 'e_pxSolution'] = e_s[:, 0]
        F[Src, 'e_pySolution'] = e_s[:, 1]
        # Note curl e = -iwb so b = -curl/iw
        # Ainv.clean()
    return F
# Sim.fields = fields

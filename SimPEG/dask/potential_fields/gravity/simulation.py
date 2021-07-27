import numpy as np
from ....potential_fields.gravity import Simulation3DIntegral as Sim
from ....potential_fields.gravity.simulation import evaluate_integral
from ....utils import sdiag, mkvc
from dask import array, delayed, config
from scipy.sparse import csr_matrix as csr
from dask.distributed import Future, get_client
from dask.diagnostics import ProgressBar
from ...utils import compute_chunk_sizes


def dask_fields(self, m):
    """
    Fields computed from a linear operation
    """
    self.model = m

    if self.store_sensitivities == "forward_only":
        self.model = m
        # Compute the linear operation without forming the full dense G
        fields = self.linear_operator()
    else:
        if hasattr(self, "G"): # Trigger calculations
            fields = self.G @ (self.rhoMap @ m).astype(np.float32)

    return fields


Sim.fields = dask_fields


def dask_getJtJdiag(self, m, W=None):
    """
        Return the diagonal of JtJ
    """

    self.model = m

    if isinstance(self.Jmatrix, Future):
        self.Jmatrix  # Wait to finish

    if W is None:
        W = np.ones(self.nD)
    else:
        W = W.diagonal()

    if getattr(self, "_gtg_diagonal", None) is None:
        diag = ((W[:, None] * self.Jmatrix) ** 2).sum(axis=0).compute()
        self._gtg_diagonal = diag
    else:
        diag = self._gtg_diagonal
    return mkvc((sdiag(np.sqrt(diag)) @ self.rhoDeriv).power(2).sum(axis=0))


Sim.getJtJdiag = dask_getJtJdiag


def dask_Jvec(self, _, v, f=None):
    """
    Sensitivity times a vector
    """

    if isinstance(self.Jmatrix, Future):
        self.Jmatrix  # Wait to finish

    dmu_dm_v = self.rhoDeriv @ v
    return array.dot(self.Jmatrix, dmu_dm_v.astype(np.float32))


Sim.Jvec = dask_Jvec


def dask_Jtvec(self, _, v, f=None):
    """
    Sensitivity transposed times a vector
    """

    if isinstance(self.Jmatrix, Future):
        self.Jmatrix  # Wait to finish

    Jtvec = array.dot(v.astype(np.float32), self.Jmatrix)
    Jtjvec_dmudm = delayed(csr.dot)(Jtvec, self.rhoDeriv)
    h_vec = array.from_delayed(
        Jtjvec_dmudm, dtype=float, shape=[self.rhoDeriv.shape[1]]
    )

    return h_vec


Sim.Jtvec = dask_Jtvec


@property
def dask_G(self):
    """
    The linear forward operator
    """
    return self.Jmatrix


Sim.G = dask_G


def linear_operator(self):
    self.nC = self.modelMap.shape[0]

    hx, hy, hz = self.mesh.h[0].min(), self.mesh.h[1].min(), self.mesh.h[2].min()

    n_data_comp = len(self.survey.components)
    components = np.array(list(self.survey.components.keys()))
    active_components = np.hstack(
        [np.c_[values] for values in self.survey.components.values()]
    ).tolist()

    client = get_client()

    Xn, Yn, Zn = client.scatter([self.Xn, self.Yn, self.Zn], workers=self.workers)
    row = delayed(evaluate_integral, pure=True)
    rows = [
        array.from_delayed(
            row(Xn, Yn, Zn, hx, hy, hz, receiver_location, components[component]),
            dtype=np.float32,
            shape=(n_data_comp, self.nC),
        )
        for receiver_location, component in zip(
            self.survey.receiver_locations.tolist(), active_components
        )
    ]
    stack = array.vstack(rows)

    # Chunking options
    if self.chunk_format == "row" or self.store_sensitivities == "forward_only":
        config.set({"array.chunk-size": f"{self.max_chunk_size}MiB"})
        # Autochunking by rows is faster and more memory efficient for
        # very large problems sensitivty and forward calculations
        stack = stack.rechunk({0: "auto", 1: -1})

    elif self.chunk_format == "equal":
        # Manual chunks for equal number of blocks along rows and columns.
        # Optimal for Jvec and Jtvec operations
        row_chunk, col_chunk = compute_chunk_sizes(*stack.shape, self.max_chunk_size)
        stack = stack.rechunk((row_chunk, col_chunk))
    else:
        # Auto chunking by columns is faster for Inversions
        config.set({"array.chunk-size": f"{self.max_chunk_size}MiB"})
        stack = stack.rechunk({0: -1, 1: "auto"})

    if self.store_sensitivities == "disk":
        sens_name = self.sensitivity_path
        if os.path.exists(sens_name):
            kernel = array.from_zarr(sens_name)
            if np.all(
                np.r_[
                    np.any(np.r_[kernel.chunks[0]] == stack.chunks[0]),
                    np.any(np.r_[kernel.chunks[1]] == stack.chunks[1]),
                    np.r_[kernel.shape] == np.r_[stack.shape],
                ]
            ):
                # Check that loaded kernel matches supplied data and mesh
                print("Zarr file detected with same shape and chunksize ... re-loading")
                return kernel

        print("Writing Zarr file to disk")

        # with ProgressBar():
        print("Saving kernel to zarr: " + sens_name)

        kernel = array.to_zarr(
                stack, sens_name,
                compute=True, return_stored=True, overwrite=True
        )
        return kernel

    elif self.store_sensitivities == "forward_only":
        # with ProgressBar():
        print("Forward calculation (DASK): ")
        pred = stack @ self.model.astype(np.float32)
        return pred

    # with ProgressBar():
    #     print("Computing sensitivities to local ram")
    return array.asarray(stack)


Sim.linear_operator = linear_operator
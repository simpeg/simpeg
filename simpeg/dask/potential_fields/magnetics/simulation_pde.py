from dask import array, compute, delayed
import numpy as np
from ....potential_fields.magnetics import Simulation3DDifferential as Sim
from ....utils import sdiag, mkvc


def distance_weights(
    indices: np.ndarray,
    locations: np.ndarray,
    uncertainties: np.ndarray,
    cell_centers: np.ndarray,
    cell_volumes: np.ndarray,
    exponent: float = 3.0,
    threshold: float = 1e-2,
) -> np.ndarray:
    """
    Compute sum of weighted inverse distance weighting to
    mimic sensitivities of magnetic data.

    :param indices: Indices of data to process
    :param locations: All receiver locations
    :param uncertainties: Data uncertainties per components
    :param cell_centers: Cell center locations of the inversion mesh
    :param cell_volumes: Cell volumes of the inversion mesh
    :param exponent: Exponent of the decay
    :param threshold: Threshold to avoid zero division

    :return: Weighted inverse distance sum over the locations for
        each cell in the mesh.
    """
    weights = np.zeros(len(cell_centers))

    for ind in indices:
        distance = np.linalg.norm(cell_centers - locations[ind], axis=1)
        weights += (
            uncertainties[ind] ** 2.0
            * cell_volumes**2.0
            * (distance + threshold) ** (-2 * exponent)
        )

    return weights


def dask_getJtJdiag(self, m: np.ndarray, W=None, f=None):
    """
    Return the diagonal of JtJ
    """

    self.model = m

    if W is None:
        uncertainties = np.ones(self.Jmatrix.shape[0])
    else:
        uncertainties = W.diagonal()

    if getattr(self, "_gtg_diagonal", None) is None:

        client, worker = self._get_client_worker()

        n_threads = self.n_threads(client=client, worker=worker)
        n_data = self.survey.receiver_locations.shape[0]
        chunks = np.array_split(np.arange(n_data), n_threads)
        # Collapse the components
        uncertainties = np.sum(uncertainties.reshape(n_data, -1) ** 2.0, axis=1) ** 0.5
        cell_centers = self.mesh.cell_centers.copy()
        cell_volumes = self.mesh.cell_volumes.copy()
        locations = self.survey.receiver_locations.copy()

        if client:
            cell_centers = client.scatter(cell_centers, workers=worker)
            cell_volumes = client.scatter(cell_volumes, workers=worker)
            locations = client.scatter(locations, workers=worker)
            uncertainties = client.scatter(uncertainties, workers=worker)
        else:
            delayed_distance_weights = delayed(distance_weights)

        futures = []
        for block in chunks:
            if client:
                futures.append(
                    client.submit(
                        distance_weights,
                        block,
                        locations,
                        uncertainties,
                        cell_centers,
                        cell_volumes,
                        workers=worker,
                    )
                )
            else:
                futures.append(
                    array.from_delayed(
                        delayed_distance_weights(
                            block,
                            locations,
                            uncertainties,
                            cell_centers,
                            cell_volumes,
                        ),
                        dtype=np.float32,
                        shape=(len(cell_centers),),
                    )
                )

        if client:
            diag = client.gather(futures)
        else:
            diag = compute(futures)[0]

        diag = np.tile(np.sum(diag, axis=0), 3)

        self._gtg_diagonal = diag

    else:
        diag = self._gtg_diagonal

    return mkvc((sdiag(np.sqrt(diag)) @ self.remDeriv).power(2).sum(axis=0))


Sim.getJtJdiag = dask_getJtJdiag

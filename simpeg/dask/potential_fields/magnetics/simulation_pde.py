import numpy as np
from ....potential_fields.magnetics import Simulation3DDifferential as Sim
from ....utils import sdiag, mkvc


def distance_weights(locations, cell_centers, exponent=3, threshold=1e-2):
    distance_weights = np.zeros(len(cell_centers))
    for ind, loc in enumerate(locations):
        distance = np.linalg.norm(cell_centers - loc, axis=1)
        distance_weights += (distance + threshold) ** (-2 * exponent)

    return distance_weights

def dask_getJtJdiag(self, m, W=None, f=None):
    """
    Return the diagonal of JtJ
    """

    self.model = m

    self.model = m
    if W is None:
        W = np.ones(self.Jmatrix.shape[0])
    else:
        W = W.diagonal()

    client, worker = self._get_client_worker()

    n_threads = self.n_threads(client=client, worker=worker)

    chunks = np.array_split(self.survey.receiver_locations, n_threads)
    cell_centers = self.mesh.cell_centers.copy()

    if client:
        cell_centers = client.scatter(cell_centers, workers=worker)
    else:
        delayed_distance_weights = delayed(distance_weights)

    futures = []
    for block in chunks:
        if client:
            futures.append(
                client.submit(
                    distance_weights,
                    block,
                    cell_centers,
                    workers=worker,
                )
            )
        else:
            futures.append(
                array.from_delayed(
                    delayed_compute_rows(
                        block,
                        cell_centers,
                    ),
                    dtype=np.float32,
                    shape=(
                        len(block),
                        self.active_cells.sum(),
                    ),
                )
            )

    if client:
        diag = client.gather(futures)
    else:
        diag = compute(futures)

    diag = np.tile(np.vstack(diag).sum(axis=0) * self.mesh.cell_volumes**2.,3)**0.5
    return mkvc((sdiag(np.sqrt(diag)) @ self.remDeriv).power(2).sum(axis=0))


Sim.getJtJdiag = dask_getJtJdiag
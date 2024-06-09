from ..objective_function import ComboObjectiveFunction, BaseObjectiveFunction

import dask.array as da

import numpy as np
from dask.distributed import Future, get_client, Client
from ..data_misfit import L2DataMisfit

BaseObjectiveFunction._workers = None


@property
def client(self):
    if getattr(self, "_client", None) is None:
        self._client = get_client()

    return self._client


@client.setter
def client(self, client):
    assert isinstance(client, Client)
    self._client = client


BaseObjectiveFunction.client = client


@property
def workers(self):
    return self._workers


@workers.setter
def workers(self, workers):
    self._workers = workers


BaseObjectiveFunction.workers = workers


def dask_call(self, m, f=None):
    fcts = []
    multipliers = []
    for i, phi in enumerate(self):
        multiplier, objfct = phi
        if multiplier == 0.0:  # don't evaluate the fct
            continue
        else:

            if f is not None and objfct._has_fields:
                fct = objfct(m, f=f[i])
            else:
                fct = objfct(m)

            if isinstance(fct, Future):
                future = self.client.compute(
                    self.client.submit(da.multiply, multiplier, fct).result()
                )
                fcts += [future]
            else:
                fcts += [fct]

            multipliers += [multiplier]

    if isinstance(fcts[0], Future):
        phi = self.client.submit(
            da.sum, self.client.submit(da.vstack, fcts), axis=0
        ).result()
        return phi
    else:
        return np.sum(np.r_[multipliers][:, None] * np.vstack(fcts), axis=0).squeeze()


ComboObjectiveFunction.__call__ = dask_call


def dask_deriv(self, m, f=None):
    """
    First derivative of the composite objective function is the sum of the
    derivatives of each objective function in the list, weighted by their
    respective multplier.

    :param numpy.ndarray m: model
    :param SimPEG.Fields f: Fields object (if applicable)
    """

    g = []
    multipliers = []
    for i, phi in enumerate(self):
        multiplier, objfct = phi
        if multiplier == 0.0:  # don't evaluate the fct
            continue
        else:

            if f is not None and isinstance(objfct, L2DataMisfit):
                fct = objfct.deriv(m, f=f[i])
            else:
                fct = objfct.deriv(m)

            if isinstance(fct, Future):
                future = self.client.compute(
                    self.client.submit(da.multiply, multiplier, fct)
                )
                g += [future]
            else:
                g += [fct]

            multipliers += [multiplier]

    if isinstance(g[0], Future):
        big_future = self.client.submit(
            da.sum, self.client.submit(da.vstack, g), axis=0
        ).result()
        return self.client.compute(big_future).result()

    else:
        return np.sum(np.r_[multipliers][:, None] * np.vstack(g), axis=0).squeeze()


ComboObjectiveFunction.deriv = dask_deriv


def dask_deriv2(self, m, v=None, f=None):
    """
    Second derivative of the composite objective function is the sum of the
    second derivatives of each objective function in the list, weighted by
    their respective multplier.

    :param numpy.ndarray m: model
    :param numpy.ndarray v: vector we are multiplying by
    :param SimPEG.Fields f: Fields object (if applicable)
    """

    H = []
    multipliers = []
    for phi in self:
        multiplier, objfct = phi
        if multiplier == 0.0:  # don't evaluate the fct
            continue
        else:
            fct = objfct.deriv2(m, v)

            if isinstance(fct, Future):
                future = self.client.submit(da.multiply, multiplier, fct)
                H += [future]
            else:
                H += [fct]

            multipliers += [multiplier]

    if isinstance(H[0], Future):
        big_future = self.client.submit(
            da.sum, self.client.submit(da.vstack, H), axis=0
        ).result()

        return np.asarray(big_future)

    else:
        phi_deriv2 = 0
        for multiplier, h in zip(multipliers, H):
            phi_deriv2 += multiplier * h

        return phi_deriv2


ComboObjectiveFunction.deriv2 = dask_deriv2

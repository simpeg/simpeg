from ..optimization import ProjectedGNCG
import dask.array as da
import numpy as np
from time import time


def dask_findSearchDirection(self):
    """
        findSearchDirection()
        Finds the search direction based on either CG or steepest descent.
    """
    Active = self.activeSet(self.xc)
    temp = sum(np.ones_like(self.xc.size) - Active)

    delx = np.zeros(self.g.size)
    resid = -(1 - Active) * self.g
    r = resid - (1 - Active) * self.H(delx)
    p = self.approxHinv * r
    sold = np.dot(r, p)
    count = 0
    ct = time()
    while np.all([np.linalg.norm(r) > self.tolCG, count < self.maxIterCG]):
        count += 1
        Hp = self.H(p)
        q = (1 - Active) * Hp
        alpha = sold / np.dot(p, q.T)
        delx = delx + alpha * p
        r = np.asarray(r - alpha * q)
        h = self.approxHinv * r
        snew = np.dot(r, h)
        p = (h + (snew / sold * p))
        sold = snew
    self.cg_runtime = time()-ct
    delx = np.asarray(delx)
    # Take a gradient step on the active cells if exist
    if temp != self.xc.size:
        rhs_a = np.asarray((Active) * -self.g)
        dm_i = max(abs(delx))
        dm_a = max(abs(rhs_a))

        # perturb inactive set off of bounds so that they are included
        # in the step
        delx = delx + self.stepOffBoundsFact * (rhs_a * dm_i / dm_a)

    # Only keep gradients going in the right direction on the active
    # set
    indx = ((self.xc <= self.lower) & (delx < 0)) | (
            (self.xc >= self.upper) & (delx > 0)
    )
    delx[indx] = 0.0

    return delx


ProjectedGNCG.findSearchDirection = dask_findSearchDirection

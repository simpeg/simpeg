import numpy as np
from ...utils import mkvc
import discretize
import warnings


def line(a, t, l):
    """
        Linear interpolation between a and b
        0 <= t <= 1
    """
    return a + t * l


def edge_basis_function(t, a1, l1, h1, a2, l2, h2):
    """
        Edge basis functions
    """
    x1 = line(a1, t, l1)
    x2 = line(a2, t, l2)
    w0 = (1.0 - x1 / h1) * (1.0 - x2 / h2)
    w1 = (x1 / h1) * (1.0 - x2 / h2)
    w2 = (1.0 - x1 / h1) * (x2 / h2)
    w3 = (x1 / h1) * (x2 / h2)
    return np.r_[w0, w1, w2, w3]


# TODO: Extend this when current is defined on cell-face
def getStraightLineCurrentIntegral(hx, hy, hz, ax, ay, az, bx, by, bz):
    """
      Compute integral int(W . J dx^3) in brick of size hx x hy x hz
      where W denotes the 12 local bilinear edge basis functions
      and where J prescribes a unit line current
      between points (ax,ay,az) and (bx,by,bz).
    """

    # length of line segment
    lx = bx - ax
    ly = by - ay
    lz = bz - az
    l = np.sqrt(lx ** 2 + ly ** 2 + lz ** 2)

    if l == 0:
        sx = np.zeros((4, 1))
        sy = np.zeros((4, 1))
        sz = np.zeros((4, 1))

    # integration using Simpson's rule
    wx0 = edge_basis_function(0.0, ay, ly, hy, az, lz, hz)
    wx0_5 = edge_basis_function(0.5, ay, ly, hy, az, lz, hz)
    wx1 = edge_basis_function(1.0, ay, ly, hy, az, lz, hz)

    wy0 = edge_basis_function(0.0, ax, lx, hx, az, lz, hz)
    wy0_5 = edge_basis_function(0.5, ax, lx, hx, az, lz, hz)
    wy1 = edge_basis_function(1.0, ax, lx, hx, az, lz, hz)

    wz0 = edge_basis_function(0.0, ax, lx, hx, ay, ly, hy)
    wz0_5 = edge_basis_function(0.5, ax, lx, hx, ay, ly, hy)
    wz1 = edge_basis_function(1.0, ax, lx, hx, ay, ly, hy)

    sx = (wx0 + 4.0 * wx0_5 + wx1) * (lx / 6.0)

    sy = (wy0 + 4.0 * wy0_5 + wy1) * (ly / 6.0)
    sz = (wz0 + 4.0 * wz0_5 + wz1) * (lz / 6.0)

    return sx, sy, sz


def findlast(x):
    if x.sum() == 0:
        return -1
    else:
        return np.arange(x.size)[x][-1]


def segmented_line_current_source_term(mesh, locs):
    """Calculate a source term for a line current source on a mesh

    Given a discretize mesh, compute the source vector for a unit current flowing
    along the segmented line path with vertices defined by `locs`.

    Parameters
    ----------
    mesh : discretize.TreeMesh or discretize.TensorMesh
        The Mesh (3D) for the system.
    locs : numpy.ndarray
        The array of locations of consecutive points along the polygonal path.
        in a shape of (n_points, 3)

    Returns
    -------
    numpy.array of length (mesh.nE)
        Contains the source term for all x, y, and z edges of the OcTreeMesh.

    Notes
    -----
    You can create a closed loop by setting the first and end point to be the same.
    """
    if isinstance(mesh, discretize.TensorMesh):
        return _poly_line_source_tens(mesh, locs)
    elif isinstance(mesh, discretize.TreeMesh):
        return _poly_line_source_tree(mesh, locs)


def _poly_line_source_tens(mesh, locs):
    """
        Given a tensor product mesh with origin at (x0,y0,z0) and cell sizes
        hx, hy, hz, compute the source vector for a unit current flowing along
        the polygon with vertices px, py, pz.
        The 3-D arrays sx, sy, sz contain the source terms for all x/y/z-edges
        of the tensor product mesh.

        Modified from matlab code:

            getSourceTermLineCurrentPolygon(x0,y0,z0,hx,hy,hz,px,py,pz)
            Christoph Schwarzbach, February 2014

    """
    # number of cells
    xorig = mesh.x0
    hx, hy, hz = mesh.h
    px = locs[:, 0]
    py = locs[:, 1]
    pz = locs[:, 2]
    nx = len(hx)
    ny = len(hy)
    nz = len(hz)
    x0, y0, z0 = xorig[0], xorig[1], xorig[2]
    # nodal grid
    x = np.r_[x0, x0 + np.cumsum(hx)]
    y = np.r_[y0, y0 + np.cumsum(hy)]
    z = np.r_[z0, z0 + np.cumsum(hz)]

    # discrete edge function
    sx = np.zeros((nx, ny + 1, nz + 1))
    sy = np.zeros((nx + 1, ny, nz + 1))
    sz = np.zeros((nx + 1, ny + 1, nz))

    # number of line segments
    nP = len(px) - 1

    # check that all polygon vertices are inside the mesh
    for ip in range(nP + 1):
        ax = px[ip]
        ay = py[ip]
        az = pz[ip]
        ix = findlast(np.logical_and(ax >= x[: nx - 1], ax <= x[1:nx]))
        iy = findlast(np.logical_and(ay >= y[: ny - 1], ay <= y[1:ny]))
        iz = findlast(np.logical_and(az >= z[: nz - 1], az <= z[1:nz]))

        if (ix < 0) or (iy < 0) or (iz < 0):
            msg = "Polygon vertex (%.1f, %.1f, %.1f) is outside the mesh"
            print((msg) % (ax, ay, az))

    # integrate each line segment
    for ip in range(nP):
        # start and end vertices
        ax = px[ip]
        ay = py[ip]
        az = pz[ip]
        bx = px[ip + 1]
        by = py[ip + 1]
        bz = pz[ip + 1]

        # find intersection with mesh planes
        dx = bx - ax
        dy = by - ay
        dz = bz - az
        d = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

        tol = d * np.finfo(float).eps

        if abs(dx) > tol:
            tx = (x - ax) / dx
            tx = tx[np.logical_and(tx >= 0, tx <= 1)]
        else:
            tx = []

        if abs(dy) > tol:
            ty = (y - ay) / dy
            ty = ty[np.logical_and(ty >= 0, ty <= 1)]
        else:
            ty = []

        if abs(dz) > tol:
            tz = (z - az) / dz
            tz = tz[np.logical_and(tz >= 0, tz <= 1)]
        else:
            tz = []

        t = np.unique(np.r_[0.0, tx, ty, tz, 1.0])
        nq = len(t) - 1
        tc = 0.5 * (t[:nq] + t[1 : nq + 1])

        for iq in range(nq):

            cx = ax + tc[iq] * dx
            cy = ay + tc[iq] * dy
            cz = az + tc[iq] * dz

            # locate cell id

            ix = findlast(np.logical_and(cx >= x[: nx - 1], cx <= x[1:nx]))
            iy = findlast(np.logical_and(cy >= y[: ny - 1], cy <= y[1:ny]))
            iz = findlast(np.logical_and(cz >= z[: nz - 1], cz <= z[1:nz]))

            # local coordinates
            hxloc = hx[ix]
            hyloc = hy[iy]
            hzloc = hz[iz]
            axloc = ax + t[iq] * dx - x[ix]
            ayloc = ay + t[iq] * dy - y[iy]
            azloc = az + t[iq] * dz - z[iz]
            bxloc = ax + t[iq + 1] * dx - x[ix]
            byloc = ay + t[iq + 1] * dy - y[iy]
            bzloc = az + t[iq + 1] * dz - z[iz]
            # integrate
            sxloc, syloc, szloc = getStraightLineCurrentIntegral(
                hxloc, hyloc, hzloc, axloc, ayloc, azloc, bxloc, byloc, bzloc
            )
            # integrate
            sx[ix, iy : iy + 2, iz : iz + 2] += np.reshape(sxloc, (2, 2), order="F")
            sy[ix : ix + 2, iy, iz : iz + 2] += np.reshape(syloc, (2, 2), order="F")
            sz[ix : ix + 2, iy : iy + 2, iz] += np.reshape(szloc, (2, 2), order="F")


def _poly_line_source_tree(mesh, locs):
    """Calculate a source term for a line current source on a OctTreeMesh

    Given an OcTreeMesh compute the source vector for a unit current flowing
    along the polygon with vertices px, py, pz.

    Parameters
    ----------
    mesh : discretize.TreeMesh
        The OctTreeMesh (3D) for the system.
    px, py, pz : 1D numpy.array
        The 1D arrays contain the x, y, and z, locations of consecutive points
        along the polygonal path

    Returns
    -------
    numpy.array of length (mesh.nE)
        Contains the source term for all x, y, and z edges of the OcTreeMesh.
    """

    px = locs[:, 0]
    py = locs[:, 1]
    pz = locs[:, 2]

    # discrete edge vectors
    sx = np.zeros(mesh.ntEx)
    sy = np.zeros(mesh.ntEy)
    sz = np.zeros(mesh.ntEz)

    points = np.c_[px, py, pz]
    # number of line segments
    nP = len(points) - 1
    x0 = mesh.x0
    dim = mesh.dim
    for ip in range(nP + 1):
        A = points[0]
        xF = np.array([mesh.vectorNx[-1], mesh.vectorNy[-1], mesh.vectorNz[-1]])
        if np.any(A < x0) or np.any(A > xF):
            msg = "Polygon vertex ({.1f}, {.1f}, {.1f}) is outside the mesh".format(*A)
            raise ValueError(msg)

    # Loop over each line segment
    for ip in range(nP):
        # Start and end vertices
        A = points[ip]
        B = points[ip + 1]

        # Components of vector (dx, dy, dz) along the wirepath
        ds = B - A

        # Find indices of all cells intersected by the wirepath
        srcCellIds = mesh.get_cells_along_line(A, B)
        levels = mesh.cell_levels_by_index(srcCellIds)
        if np.any(levels != levels[0]):
            warnings.warn("Warning! Line path crosses a cell level change.")

        # Starts at point A!
        p0 = A
        for cell_id in srcCellIds:
            cell = mesh[cell_id]

            x0 = cell.x0
            h = cell.h
            xF = x0 + h

            edges = cell.edges
            edges_x = edges[0:4]
            edges_y = edges[4:8]
            edges_z = edges[8:12]

            # find next intersection along path
            ts = np.ones(dim)
            for i in range(dim):
                if ds[i] > 0:
                    ts[i] = (xF[i] - A[i]) / ds[i]
                elif ds[i] < 0:
                    ts[i] = (x0[i] - A[i]) / ds[i]
                else:
                    ts[i] = np.inf
            t = min(*ts, 1)  # the last value should be 1
            p1 = A + t * ds  # the next intersection point

            cA = p0 - x0
            cB = p1 - x0

            cell_s = getStraightLineCurrentIntegral(*h, *cA, *cB)

            sx[edges_x] += cell_s[0]
            sy[edges_y] += cell_s[1]
            sz[edges_z] += cell_s[2]

            p0 = p1
    s = np.r_[sx, sy, sz]
    R = mesh._deflate_edges()
    s = R.T.dot(s)

    return s


def getSourceTermLineCurrentPolygon(xorig, hx, hy, hz, px, py, pz):
    warnings.warn(
        "getSourceTermLineCurrentPolygon has been deprecated and will be"
        "removed in SimPEG 0.15.0. Please use segmented_line_current_source_term.",
        DeprecationWarning,
    )
    mesh = discretize.TensorMesh((hx, hy, hz), x0=xorig)
    locs = np.c_[px, py, pz]
    return segmented_line_current_source_term(mesh, locs)

import numpy as np
from ...utils import mkvc
import discretize
import warnings


def edge_basis_function(t, a1, l1, h1, a2, l2, h2):
    """Edge basis function

    Parameters
    ----------
    t : float
        Parameterized distance along wire
    a1 : float
        Start of wire in first coordinate
    l1 : float
        Length of wire in first coordinate
    h1 : float
        Cell dimension in first coordinate
    a2 : float
        Start of wire in second coordinate
    l2 : float
        Length of wire in second coordinate
    h2 : float
        Cell dimension in second coordinate

    Returns
    -------
    (4) numpy.ndarray
        Edge weights
    """
    x1 = a1 + t * l1
    x2 = a2 + t * l2
    w0 = (1.0 - x1 / h1) * (1.0 - x2 / h2)
    w1 = (x1 / h1) * (1.0 - x2 / h2)
    w2 = (1.0 - x1 / h1) * (x2 / h2)
    w3 = (x1 / h1) * (x2 / h2)
    return np.r_[w0, w1, w2, w3]


def _simpsons_rule(a1, l1, h1, a2, l2, h2):
    """Return weights for Simpson's rule."""
    wl = edge_basis_function(0.0, a1, l1, h1, a2, l2, h2)
    wc = edge_basis_function(0.5, a1, l1, h1, a2, l2, h2)
    wr = edge_basis_function(1.0, a1, l1, h1, a2, l2, h2)
    return (wl + 4.0 * wc + wr) / 6.0


# TODO: Extend this when current is defined on cell-face
def getStraightLineCurrentIntegral(hx, hy, hz, ax, ay, az, bx, by, bz):
    """Compute straight current line integral

    Compute integral int(W . J dx^3) in brick of size hx x hy x hz
    where W denotes the 12 local bilinear edge basis functions
    and where J prescribes a unit line current
    between points (ax,ay,az) and (bx,by,bz).

    Parameters
    ----------
    hx : float
        x cell dimension
    hy : float
        y cell dimension
    hz : float
        z cell dimension
    ax : float
        x-location of start of wire segement
    ay :float
        y-location of start of wire segement
    az :float
        z-location of start of wire segement
    bx :float
        x-location of end of wire segement
    by :float
        y-location of end of wire segement
    bz :float
        z-location of end of wire segement

    Returns
    -------
    sx : float
        x weight
    sy : float
        y weight
    sz : flat
        z weight
    """

    # length of line segment
    lx = bx - ax
    ly = by - ay
    lz = bz - az

    # integration using Simpson's rule
    sx = _simpsons_rule(ay, ly, hy, az, lz, hz) * lx
    sy = _simpsons_rule(ax, lx, hx, az, lz, hz) * ly
    sz = _simpsons_rule(ax, lx, hx, ay, ly, hy) * lz

    return sx, sy, sz


def _findlast(x):
    """Returns last element

    Parameters
    ----------
    array_like of int

    Returns
    -------

    """
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
    locs : (n_points, 3) numpy.ndarray
        The array of locations of consecutive points along the polygonal path.

    Returns
    -------
    (mesh.n_edges) numpy.ndarray
        Contains the source term for all x, y, and z edges of the mesh.

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
    # Get some mesh properties
    nx, ny, nz = mesh.shape_cells
    hx, hy, hz = mesh.h
    x = mesh.nodes_x
    y = mesh.nodes_y
    z = mesh.nodes_z

    # Source points
    px = locs[:, 0]
    py = locs[:, 1]
    pz = locs[:, 2]

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
        ix = _findlast(np.logical_and(ax >= x[: nx - 1], ax <= x[1:nx]))
        iy = _findlast(np.logical_and(ay >= y[: ny - 1], ay <= y[1:ny]))
        iz = _findlast(np.logical_and(az >= z[: nz - 1], az <= z[1:nz]))

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

            ix = _findlast(np.logical_and(cx >= x[: nx - 1], cx <= x[1:nx]))
            iy = _findlast(np.logical_and(cy >= y[: ny - 1], cy <= y[1:ny]))
            iz = _findlast(np.logical_and(cz >= z[: nz - 1], cz <= z[1:nz]))

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

    return np.r_[mkvc(sx), mkvc(sy), mkvc(sz)]


def _poly_line_source_tree(mesh, locs):
    """Calculate a source term for a line current source on a OctTreeMesh

    Given an OcTreeMesh compute the source vector for a unit current flowing
    along the polygon with vertices px, py, pz.

    Parameters
    ----------
    mesh : discretize.TreeMesh
        The OctTreeMesh (3D) for the system.
    locs : (n, 3) numpy.array
        Columns contain the x, y, and z locations of consecutive points
        along the polygonal path

    Returns
    -------
    (n_edges) numpy.ndarray
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


def line_through_faces(
    mesh,
    locations,
    normalize_by_area=True,
    check_divergence=True,
    tolerance_divergence=1e-9,
):
    """Define line current through cell faces

    Define the current through cell faces given path locations. Note that this
    will perform best of your path locations are at cell centers. Only paths
    that align with the mesh (e.g. a straight line in x, y, z) are currently
    supported

    Parameters
    ----------
    mesh : discretize.TreeMesh
        The OctTreeMesh (3D) for the system.
    normalize_by_area : bool, default: ``True``
        If ``True``, normalize by face area
    check_divergence : bool, default: ``True``
        If ``True``, carry out divergence check
    tolerance_divergence: float, default: 1e-9
        Tolerance for divergence check

    Returns
    -------
    (mesh.n_faces) numpy.ndarray
        Line current source on faces
    """

    current = np.zeros(mesh.n_faces)

    def not_aligned_error(i):
        raise NotImplementedError(
            "With the current implementation, the line between points "
            "must align with the axes of the mesh. The points "
            f"{locations[i, :]} and {locations[i+1, :]} do not."
        )

    # pre-processing step: find closest cell centers
    cell_centers = discretize.utils.closest_points_index(mesh, locations, "CC")
    locations = mesh.gridCC[cell_centers, :]

    # next step: find segments between lines
    for i in range(locations.shape[0] - 1):

        dimension = np.nonzero(np.abs(locations[i, :] - locations[i + 1, :]))[0]
        if len(dimension) > 1:
            not_aligned_error(i)
        dimension = dimension[0]
        direction = np.sign(locations[i, dimension] - locations[i + 1, dimension])

        if dimension == 0:
            grid_loc = "x"
            current_inds = slice(0, mesh.n_faces_x)
        elif dimension == 1:
            grid_loc = "y"
            start = mesh.n_faces_x
            current_inds = slice(start, start + mesh.n_faces_y)
        elif dimension == 2:
            grid_loc = "z"
            start = mesh.n_faces_x + mesh.n_faces_y
            current_inds = slice(start, start + mesh.n_faces_z)

        # interpolate to closest face
        loca = discretize.utils.closest_points_index(
            mesh, locations[i, :], f"F{grid_loc}"
        )
        locb = discretize.utils.closest_points_index(
            mesh, locations[i + 1, :], f"F{grid_loc}"
        )

        if len(loca) > 1 or len(locb) > 1:
            raise Exception(
                "Current across multiple faces is not implemented. "
                "Please put path through a cell rather than along edges"
            )

        grid = getattr(mesh, f"faces_{grid_loc}")
        loca = grid[loca[0]]
        locb = grid[locb[0]]

        # find all faces between these points
        if dimension == 0:
            xlocs = np.r_[locations[i, 0], locations[i + 1, 0]]

            if not (np.allclose(loca[1], locb[1]) and np.allclose(loca[2], locb[2])):
                not_aligned_error(i)

            ylocs = loca[1] + mesh.hy.min() / 4 * np.r_[-1, 1]
            zlocs = loca[2] + mesh.hz.min() / 4 * np.r_[-1, 1]

        elif dimension == 1:
            ylocs = np.r_[locations[i, 1], locations[i + 1, 1]]

            if not (np.allclose(loca[0], locb[0]) and np.allclose(loca[2], locb[2])):
                not_aligned_error(i)

            xlocs = loca[0] + mesh.hx.min() / 4 * np.r_[-1, 1]
            zlocs = loca[2] + mesh.hz.min() / 4 * np.r_[-1, 1]

        elif dimension == 2:
            zlocs = np.r_[locations[i, 2], locations[i + 1, 2]]

            if not (np.allclose(loca[0], locb[0]) and np.allclose(loca[1], locb[1])):
                not_aligned_error(i)

            xlocs = loca[0] + mesh.hx.min() / 4 * np.r_[-1, 1]
            ylocs = loca[1] + mesh.hy.min() / 4 * np.r_[-1, 1]

        src_inds = (
            (grid[:, 0] >= xlocs.min())
            & (grid[:, 0] <= xlocs.max())
            & (grid[:, 1] >= ylocs.min())
            & (grid[:, 1] <= ylocs.max())
            & (grid[:, 2] >= zlocs.min())
            & (grid[:, 2] <= zlocs.max())
        )

        current[current_inds][src_inds] = direction

    if normalize_by_area:
        current = current / mesh.area

    # check that there is only a divergence at the ends if not a loop
    if check_divergence:
        div = mesh.vol * mesh.face_divergence * current
        nonzero = np.abs(div) > tolerance_divergence

        # check if the source is a loop or grounded
        if not np.allclose(locations[0, :], locations[-1, :]):  # grounded source
            if nonzero.sum() > 2:
                raise Exception(
                    "The source path is not connected. Check that all points go through cell centers"
                )
            if np.abs(div.sum()) > tolerance_divergence:
                raise Exception(
                    "The integral of the divergence is not zero. Something is wrong :("
                )
        else:  # loop source
            if nonzero.sum() > 0:
                raise Exception(
                    "The source path is not connected. Check that all points go through cell centers"
                )

    return current


def getSourceTermLineCurrentPolygon(xorig, hx, hy, hz, px, py, pz):
    """getSourceTermLineCurrentPolygon is deprecated. Use :func:`segmented_line_current_source_term`"""
    warnings.warn(
        "getSourceTermLineCurrentPolygon has been deprecated and will be"
        "removed in SimPEG 0.17.0. Please use segmented_line_current_source_term.",
        FutureWarning,
    )
    mesh = discretize.TensorMesh((hx, hy, hz), x0=xorig)
    locs = np.c_[px, py, pz]
    return segmented_line_current_source_term(mesh, locs)

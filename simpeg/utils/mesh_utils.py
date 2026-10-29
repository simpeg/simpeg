import numpy as np
from scipy.spatial import cKDTree

from discretize import TensorMesh

from discretize.utils import (  # noqa: F401
    unpack_widths,
    closest_points_index,
    extract_core_mesh,
    active_from_xyz,
)

from .code_utils import validate_active_indices


def surface2inds(vrtx, trgl, mesh, boundaries=True, internal=True):
    """Takes a triangulated surface and determines which mesh cells it intersects.

    Parameters
    ----------
    vrtx : (n_nodes, 3) numpy.ndarray of float
        The location of the vertices of the triangles
    trgl : (n_triang, 3) numpy.ndarray of int
        Each row describes the 3 indices into the `vrtx` array that make up a
        triangle's vertices.
    mesh : TensorMesh
    boundaries : bool, optional
    internal : bool, optional

    Returns
    -------
    numpy.ndarray of bool

    Notes
    -----
    Requires `vtk`.
    """
    import vtk
    import vtk.util.numpy_support as npsup

    # Adjust the index
    trgl = trgl - 1

    # Make vtk pts
    ptsvtk = vtk.vtkPoints()
    ptsvtk.SetData(npsup.numpy_to_vtk(vrtx, deep=1))

    # Make the polygon connection
    polys = vtk.vtkCellArray()
    for face in trgl:
        poly = vtk.vtkPolygon()
        poly.GetPointIds().SetNumberOfIds(len(face))
        for nrv, vert in enumerate(face):
            poly.GetPointIds().SetId(nrv, vert)
        polys.InsertNextCell(poly)

    # Make the polydata, structure of connections and vrtx
    polyData = vtk.vtkPolyData()
    polyData.SetPoints(ptsvtk)
    polyData.SetPolys(polys)

    # Make implicit func
    ImpDistFunc = vtk.vtkImplicitPolyDataDistance()
    ImpDistFunc.SetInput(polyData)

    # Convert the mesh
    vtkMesh = vtk.vtkRectilinearGrid()
    vtkMesh.SetDimensions(*mesh.shape_nodes)
    vtkMesh.SetXCoordinates(npsup.numpy_to_vtk(mesh.nodes_x, deep=1))
    vtkMesh.SetYCoordinates(npsup.numpy_to_vtk(mesh.nodes_y, deep=1))
    vtkMesh.SetZCoordinates(npsup.numpy_to_vtk(mesh.nodes_z, deep=1))
    # Add indexes
    vtkInd = npsup.numpy_to_vtk(np.arange(mesh.nC), deep=1)
    vtkInd.SetName("Index")
    vtkMesh.GetCellData().AddArray(vtkInd)

    extractImpDistRectGridFilt = vtk.vtkExtractGeometry()  # Object constructor
    extractImpDistRectGridFilt.SetImplicitFunction(ImpDistFunc)  #
    extractImpDistRectGridFilt.SetInputData(vtkMesh)

    if boundaries is True:
        extractImpDistRectGridFilt.ExtractBoundaryCellsOn()

    else:
        extractImpDistRectGridFilt.ExtractBoundaryCellsOff()

    if internal is True:
        extractImpDistRectGridFilt.ExtractInsideOn()

    else:
        extractImpDistRectGridFilt.ExtractInsideOff()

    print("Extracting indices from grid...")
    # Executing the pipe
    extractImpDistRectGridFilt.Update()

    # Get index inside
    insideGrid = extractImpDistRectGridFilt.GetOutput()
    insideGrid = npsup.vtk_to_numpy(insideGrid.GetCellData().GetArray("Index"))

    # Return the indexes inside
    return insideGrid


def _closest_grid_indices(grid, pts):
    """Return indices of closest gridded points for a set of input points.

    Parameters
    ----------
    grid : (n, dim) numpy.ndarray
        A gridded set of points.
    pts : (m, dim) numpy.ndarray
        Points being projected to gridded locations.

    Returns
    -------
    (n,) numpy.ndarray
        Indices of the closest gridded points for all *pts* supplied.
    """
    if grid.squeeze().ndim == 1:
        grid_inds = np.asarray(
            [np.abs(pt - grid).argmin() for pt in pts.tolist()], dtype=int
        )
    else:
        tree = cKDTree(grid)
        _, grid_inds = tree.query(pts)

    return grid_inds


def get_discrete_topography(mesh, active_cells, topo_cell_cutoff="top"):
    """
    Generate discrete topography locations from mesh and active cells.

    Parameters
    ----------
    mesh : TensorMesh or discretize.TreeMesh
        A tensor or tree mesh.
    active_cells : numpy.ndarray of bool or int
        Active cells index; i.e. indices of cells below surface
    topo_cell_cutoff : {"top", "center"}, optional
        String to specify the cutoff for ground cells and the locations of
        the discrete topography. For "top", ground cells lie entirely below
        the surface topography and the discrete topography is defined on the
        top faces of surface cells. For "center", only the cell centers must
        lie below the surface topography and the discrete topography is defined
        at the centers of surface cells.

    Returns
    -------
    (n, dim) numpy.ndarray
        Discrete topography locations; i.e. xy[z].
    """
    if mesh._meshType == "TENSOR":
        if mesh.dim == 3:
            mesh2D = TensorMesh([mesh.h[0], mesh.h[1]], mesh.x0[:2])
            zc = mesh.cell_centers[:, 2]
            ACTIND = active_cells.reshape(
                (mesh.vnC[0] * mesh.vnC[1], mesh.vnC[2]), order="F"
            )
            ZC = zc.reshape((mesh.vnC[0] * mesh.vnC[1], mesh.vnC[2]), order="F")
            topoCC = np.zeros(ZC.shape[0])

            for i in range(ZC.shape[0]):
                if topo_cell_cutoff == "top":
                    ind = np.argmax(ZC[i, :][ACTIND[i, :]])
                    dz = mesh.h[2][ACTIND[i, :]][ind] * 0.5
                elif topo_cell_cutoff == "center":
                    dz = 0.0
                else:
                    raise ValueError("'topo_cell_cutoff' must be 'top' or 'center'.")
                topoCC[i] = ZC[i, :][ACTIND[i, :]].max() + dz
            return np.c_[mesh2D.cell_centers, topoCC]

        elif mesh.dim == 2:
            mesh1D = TensorMesh([mesh.h[0]], [mesh.x0[0]])
            yc = mesh.cell_centers[:, 1]
            ACTIND = active_cells.reshape((mesh.vnC[0], mesh.vnC[1]), order="F")
            YC = yc.reshape((mesh.vnC[0], mesh.vnC[1]), order="F")
            topoCC = np.zeros(YC.shape[0])
            for i in range(YC.shape[0]):
                ind = np.argmax(YC[i, :][ACTIND[i, :]])
                if topo_cell_cutoff == "top":
                    dy = mesh.h[1][ACTIND[i, :]][ind] * 0.5
                elif topo_cell_cutoff == "center":
                    dy = 0.0
                else:
                    raise ValueError("'topo_cell_cutoff' must be 'top' or 'center'.")
                topoCC[i] = YC[i, :][ACTIND[i, :]].max() + dy
            return np.c_[mesh1D.cell_centers, topoCC]

    elif mesh._meshType == "TREE":
        inds = mesh.get_boundary_cells(active_cells, direction="zu")[0]

        if topo_cell_cutoff == "top":
            dz = mesh.h_gridded[inds, -1] * 0.5
        elif topo_cell_cutoff == "center":
            dz = 0.0
        return np.c_[mesh.cell_centers[inds, :-1], mesh.cell_centers[inds, -1] + dz]
    else:
        msg = (
            f"Invalid mesh of type '{type(mesh).__name__}'. "
            "`get_discrete_topography` only supports TensorMesh and TreeMesh."
        )
        raise TypeError(msg)


def shift_to_discrete_topography(
    mesh,
    pts,
    active_cells,
    topo_cell_cutoff="top",
    shift_horizontal=True,
    heights=0.0,
):
    """
    Shift locations relative to discrete surface topography.

    Parameters
    ----------
    mesh : discretize.TensorMesh or discretize.TreeMesh
        The mesh (2D or 3D) defining the discrete domain.
    pts : (n, dim) numpy.ndarray
        The original set of points being shifted relative to the discretize
        surface topography.
    active_cells : numpy.ndarray of int or bool
        Index array for all cells lying below the surface topography.
    topo_cell_cutoff : {"top", "center"}, optional
        String to specify the cutoff for ground cells and the locations of the discrete
        topography. For "top", ground cells lie entirely below the surface topography
        and the discrete topography is defined on the top faces of surface cells.
        For "center", only the cell centers must lie below the surface topography and
        the discrete topography is defined at the centers of surface cells.
        The topography is defined using the ``active_cells`` parameter: the topography
        is obtained as the topmost surface of active cells.
    heights : float or (n,) numpy.ndarray, optional
        Height(s) relative to the true surface topography. Used to preserve flight
        heights or borehole depths.
    shift_horizontal : bool, optional
        When True, locations are shifted to the closest surface cell center
        (horizontally), on the discrete topography of that cell. When False, the
        original horizontal locations are preserved, on the discrete topography
        of the column they lie in.

    Returns
    -------
    (n, dim) numpy.ndarray
        The set of points shifted relative to the discretize surface topography.

    See also
    --------
    :func:`simpeg.utils.get_discrete_topography`
    """
    if mesh._meshType != "TENSOR" and mesh._meshType != "TREE":
        msg = (
            f"Invalid mesh of type '{type(mesh).__name__}'. "
            "`shift_to_discrete_topography` only supports TensorMesh and TreeMesh."
        )
        raise TypeError(msg)

    if not isinstance(heights, (int, float)) and len(pts) != len(heights):
        raise ValueError(
            (
                "If supplied as a `numpy.ndarray`, the number of heights must ",
                "equal the number of points.",
            )
        )

    if topo_cell_cutoff not in ("top", "center"):
        raise ValueError("'topo_cell_cutoff' must be 'top' or 'center'.")
    active_cells = validate_active_indices("active_cells", active_cells, mesh.n_cells)

    pts = np.asarray(pts, dtype=float)
    if pts.ndim == 2 and pts.shape[1] == mesh.dim:
        horizontal_pts = pts[:, :-1]
    else:
        horizontal_pts = pts.reshape(-1, mesh.dim - 1)  # in case (n,) or (n, 1) array

    if shift_horizontal:
        # the closest surface cell center, and the top (or center) of that cell
        discrete_topography = get_discrete_topography(
            mesh, active_cells, topo_cell_cutoff=topo_cell_cutoff
        )
        topo_inds = _closest_grid_indices(discrete_topography[:, :-1], horizontal_pts)
        out = discrete_topography[topo_inds]
    else:
        # Walk down each location's column from the top of the mesh, one cell at a
        # time, to its highest active cell (on a tree mesh the closest surface cell
        # center can belong to the neighboring column). Locations outside the mesh
        # use the closest column.
        lower = mesh.origin
        upper = mesh.origin + np.array([h.sum() for h in mesh.h])
        eps = 1e-6 * mesh.h[-1].min()
        columns = np.clip(horizontal_pts, lower[:-1] + eps, upper[:-1] - eps)
        bottoms = mesh.cell_centers[:, -1] - 0.5 * mesh.h_gridded[:, -1]

        z = np.full(len(columns), upper[-1] - eps)
        surface_cells = np.empty(len(columns), dtype=int)
        remaining = np.arange(len(columns))
        while remaining.size:
            points = np.c_[columns[remaining], z[remaining]]
            cells = np.atleast_1d(mesh.point2index(points))
            is_active = active_cells[cells]
            surface_cells[remaining[is_active]] = cells[is_active]
            remaining, cells = remaining[~is_active], cells[~is_active]
            z[remaining] = bottoms[cells] - eps
            if np.any(z[remaining] < lower[-1]):
                no_active = remaining[z[remaining] < lower[-1]]
                raise ValueError(
                    "No active cells below the locations "
                    f"{horizontal_pts[no_active]}."
                )

        elevations = mesh.cell_centers[surface_cells, -1]
        if topo_cell_cutoff == "top":
            elevations = elevations + 0.5 * mesh.h_gridded[surface_cells, -1]
        out = np.c_[horizontal_pts, elevations]

    out[:, -1] += heights

    return out

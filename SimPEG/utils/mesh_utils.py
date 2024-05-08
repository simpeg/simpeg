import numpy as np
from .code_utils import deprecate_function

from discretize.utils import (  # noqa: F401
    unpack_widths,
    closest_points_index,
    extract_core_mesh,
)


def surface2inds(vrtx, trgl, mesh, boundaries=True, internal=True):
    """Takes a triangulated surface and determine which mesh cells it intersects.

    Parameters
    ----------
    vrtx : (n_nodes, 3) numpy.ndarray of float
        The location of the vertices of the triangles
    trgl : (n_triang, 3) numpy.ndarray of int
        Each row describes the 3 indices into the `vrtx` array that make up a triangle's
        vertices.
    mesh : discretize.TensorMesh
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


################################################
#             DEPRECATED FUNCTIONS
################################################
meshTensor = deprecate_function(
    unpack_widths, "meshTensor", removal_version="0.19.0", error=True
)
closestPoints = deprecate_function(
    closest_points_index, "closestPoints", removal_version="0.19.0", error=True
)
ExtractCoreMesh = deprecate_function(
    extract_core_mesh, "ExtractCoreMesh", removal_version="0.19.0", error=True
)

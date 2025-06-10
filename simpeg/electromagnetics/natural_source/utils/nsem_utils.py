from ...static.utils import drapeTopotoLoc


def shift_to_discrete_topography(
    locations, mesh, active_cells, option="top", height=0.0
):
    """Shift locations relative to discrete surface topography.

    Parameters
    ----------
    locations : (n, dim) numpy.ndarray
        The original locations.
    mesh : discretize.TensorMesh or discretize.TreeMesh
        The mesh defining the discrete domain.
    active_cells : numpy.ndarray of int or bool
        Active topography cells.
    option : {"top", "center"}
        Define discrete topography at tops of cells or cell centers.
    height : float or (n,) numpy.ndarray
        Location height relative to surface topography.

    Returns
    -------
    (n, dim) numpy.ndarray
        The shifted locations.
    """

    locations_shifted = drapeTopotoLoc(
        mesh, locations, active_cells=active_cells, option=option
    )

    locations_shifted[:, -1] += height

    return locations_shifted

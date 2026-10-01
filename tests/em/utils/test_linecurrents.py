import numpy as np
from simpeg.electromagnetics.utils import (
    getStraightLineCurrentIntegral,
    segmented_line_current_source_term,
    line_through_faces,
)
import discretize
import unittest
import pytest
from simpeg.utils import download


class LineCurrentTests(unittest.TestCase):
    def setUp(self):
        url = "https://storage.googleapis.com/simpeg/tests/em_utils/currents.npy"
        self.basePath = download(url)

    def test_deprecated(self):
        hx, hy, hz = 2, 2, 2
        ax, ay, az = 0.1, 0.3, 0.4
        bx, by, bz = 1, 0.8, 0.7

        sx_true = np.r_[0.475875, 0.176625, 0.176625, 0.070875]
        sy_true = np.r_[0.265625, 0.096875, 0.096875, 0.040625]
        sz_true = np.r_[0.1605, 0.057, 0.057, 0.0255]

        sx, sy, sz = getStraightLineCurrentIntegral(hx, hy, hz, ax, ay, az, bx, by, bz)

        s_true = np.r_[sx_true, sy_true, sz_true]
        s = np.r_[sx, sy, sz]
        np.testing.assert_allclose(s_true, s)

    def test_segmented_tensor(self):
        hx = np.ones(10) * 1.0
        hy = np.ones(10) * 2.0
        hz = np.ones(10) * 3.0

        px = np.r_[0, 3, 5]
        py = np.r_[0, 3, 5]
        pz = np.r_[2.0, 3.0, 4.0]

        xorig = np.r_[0.0, 0.0, 0.0]

        mesh = discretize.TensorMesh((hx, hy, hz), x0=xorig)
        locs = np.c_[px, py, pz]
        out = segmented_line_current_source_term(mesh, locs)
        fname = self.basePath
        out_true = np.load(fname)
        np.testing.assert_allclose(out, out_true)


class TreeMeshLineCurrentTest(unittest.TestCase):
    def test_segmented_tree(self):
        hx = np.ones(16) * 1.0
        hy = np.ones(16) * 2.0
        hz = np.ones(16) * 3.0

        px = np.r_[0, 3, 5]
        py = np.r_[0, 3, 5]
        pz = np.r_[2.0, 3.0, 4.0]

        xorig = np.r_[0.0, 0.0, 0.0]

        tensor_mesh = discretize.TensorMesh((hx, hy, hz), x0=xorig)
        tree_mesh = discretize.TreeMesh((hx, hy, hz), x0=xorig)
        tree_mesh.refine(4)

        locs = np.c_[px, py, pz]
        out1 = segmented_line_current_source_term(tensor_mesh, locs)
        out2 = segmented_line_current_source_term(tree_mesh, locs)

        sort1 = np.lexsort(tensor_mesh.edges.T)
        sort2 = np.lexsort(tree_mesh.edges.T)
        np.testing.assert_allclose(out1[sort1], out2[sort2])


class LineCurrentFacesTest(unittest.TestCase):
    def setUp(self):
        dh = 1
        nc = 10
        npad = 4
        hx = [(dh, npad, -1.3), (dh, nc + 1), (dh, npad, 1.3)]
        hy = [(dh, npad, -1.3), (dh, nc), (dh, npad, 1.3)]
        hz = [(dh, npad, -1.3), (dh, nc - 1), (dh, npad, 1.3)]
        mesh = discretize.TensorMesh([hx, hy, hz], x0="CCC")
        self.mesh = mesh

    def test_not_aligned(self):
        line = np.array([[-5, 0, 0], [5, 2, 0]])

        with self.assertRaises(NotImplementedError):
            line_through_faces(mesh=self.mesh, locations=line)

    def test_line_orientations(self):
        def create_line(points):
            line_vertices = np.array(
                [
                    [
                        self.mesh.cell_centers_x[points[0, 0]],
                        self.mesh.cell_centers_y[points[0, 1]],
                        self.mesh.cell_centers_z[points[0, 2]],
                    ],
                    [
                        self.mesh.cell_centers_x[points[1, 0]],
                        self.mesh.cell_centers_y[points[1, 1]],
                        self.mesh.cell_centers_z[points[1, 2]],
                    ],
                ]
            )
            return line_vertices

        # x
        points = np.array([[5, 7, 7], [10, 7, 7]])
        linex = create_line(points)
        src = line_through_faces(self.mesh, linex)
        self.assertTrue(len(src) == self.mesh.nF)

        nonzerox = (
            (self.mesh.faces_x[:, 0] >= linex[0, 0])
            & (self.mesh.faces_x[:, 0] <= linex[1, 0])
            & (self.mesh.faces_x[:, 1] >= linex[0, 1] - 0.1)
            & (self.mesh.faces_x[:, 1] <= linex[1, 1] + 0.1)
            & (self.mesh.faces_x[:, 2] >= linex[0, 2] - 0.1)
            & (self.mesh.faces_x[:, 2] <= linex[1, 2] + 0.1)
        )
        self.assertTrue(np.all(np.nonzero(src)[0] == np.nonzero(nonzerox)[0]))

        # y
        points = np.array([[7, 5, 7], [7, 10, 7]])
        liney = create_line(points)
        src = line_through_faces(self.mesh, liney)

        nonzeroy = (
            (self.mesh.faces_y[:, 0] >= liney[0, 0] - 0.1)
            & (self.mesh.faces_y[:, 0] <= liney[1, 0] + 0.1)
            & (self.mesh.faces_y[:, 1] >= liney[0, 1])
            & (self.mesh.faces_y[:, 1] <= liney[1, 1])
            & (self.mesh.faces_y[:, 2] >= liney[0, 2] - 0.1)
            & (self.mesh.faces_y[:, 2] <= liney[1, 2] + 0.1)
        )
        self.assertTrue(
            np.all(
                np.nonzero(src)[0]
                == np.nonzero(np.hstack([np.zeros(self.mesh.nFx), nonzeroy]))[0]
            )
        )

        # z
        points = np.array([[7, 7, 5], [7, 7, 10]])
        linez = create_line(points)
        src = line_through_faces(self.mesh, linez)

        nonzeroz = (
            (self.mesh.faces_z[:, 0] >= linez[0, 0] - 0.1)
            & (self.mesh.faces_z[:, 0] <= linez[1, 0] + 0.1)
            & (self.mesh.faces_z[:, 1] >= linez[0, 1] - 0.1)
            & (self.mesh.faces_z[:, 1] <= linez[1, 1] + 0.1)
            & (self.mesh.faces_z[:, 2] >= linez[0, 2])
            & (self.mesh.faces_z[:, 2] <= linez[1, 2])
        )
        self.assertTrue(
            np.all(
                np.nonzero(src)[0]
                == np.nonzero(
                    np.hstack([np.zeros(self.mesh.nFx + self.mesh.nFy), nonzeroz])
                )[0]
            )
        )


class TestLineCurrentDirection:
    """
    Test that line_through_faces and segmented_line_current_source_term agree
    on the direction of the current: from ``locations[i]`` to ``locations[i + 1]``.
    """

    mesh = discretize.TensorMesh([np.full(8, 10.0)] * 3, origin="CCC")

    @pytest.mark.parametrize("axis", [0, 1, 2])
    @pytest.mark.parametrize("reverse", [False, True])
    def test_straight_wire(self, axis, reverse):
        mesh = self.mesh
        # wire through cell centers
        locations = np.full((2, 3), 5.0)
        locations[:, axis] = [-25.0, 25.0]
        expected_sign = 1.0
        if reverse:
            locations = locations[::-1]
            expected_sign = -1.0

        faces = line_through_faces(mesh, locations, normalize_by_area=True)
        edges = segmented_line_current_source_term(mesh, locations)

        faces_split = np.split(faces, np.cumsum(mesh.n_faces_per_direction)[:-1])
        edges_split = np.split(edges, np.cumsum(mesh.n_edges_per_direction)[:-1])

        for i in range(3):
            if i == axis:
                nonzero = faces_split[i][faces_split[i] != 0]
                assert len(nonzero) > 0
                np.testing.assert_array_equal(np.sign(nonzero), expected_sign)
                assert np.sign(edges_split[i].sum()) == expected_sign
            else:
                np.testing.assert_array_equal(faces_split[i], 0.0)
                np.testing.assert_allclose(edges_split[i], 0.0, atol=1e-12)

    def test_loop(self):
        mesh = self.mesh
        # counter-clockwise loop (viewed from +z)
        a, z = 25.0, 5.0
        locations = np.array(
            [[-a, -a, z], [a, -a, z], [a, a, z], [-a, a, z], [-a, -a, z]]
        )
        faces = line_through_faces(mesh, locations, normalize_by_area=True)
        edges = segmented_line_current_source_term(mesh, locations)

        faces_split = np.split(faces, np.cumsum(mesh.n_faces_per_direction)[:-1])
        edges_split = np.split(edges, np.cumsum(mesh.n_edges_per_direction)[:-1])
        faces_grid = [mesh.faces_x, mesh.faces_y, mesh.faces_z]
        edges_grid = [mesh.edges_x, mesh.edges_y, mesh.edges_z]

        # the current on faces and edges should circulate counter-clockwise:
        # (component, coordinate of the segment, position, expected sign)
        segments = [
            (0, 1, -a, 1.0),  # bottom: +x
            (1, 0, a, 1.0),  # right: +y
            (0, 1, a, -1.0),  # top: -x
            (1, 0, -a, -1.0),  # left: -y
        ]
        for component, coordinate, position, expected_sign in segments:
            f = faces_split[component]
            face_inds = np.isclose(faces_grid[component][:, coordinate], position) & (
                f != 0
            )
            assert face_inds.sum() > 0
            np.testing.assert_array_equal(np.sign(f[face_inds]), expected_sign)

            # edges within a cell of this segment
            e = edges_split[component]
            edge_inds = np.abs(edges_grid[component][:, coordinate] - position) < 10
            assert np.sign(e[edge_inds].sum()) == expected_sign

        # no vertical current
        np.testing.assert_array_equal(faces_split[2], 0.0)


if __name__ == "__main__":
    unittest.main()

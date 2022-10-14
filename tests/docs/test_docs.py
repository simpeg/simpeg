import unittest
import os
from sphinx.application import Sphinx
import multiprocessing

n_cpu = multiprocessing.cpu_count()


class Doc_Test(unittest.TestCase):
    @property
    def path_to_docs(self):
        dirname, file_name = os.path.split(os.path.abspath(__file__))
        return os.path.sep.join(dirname.split(os.path.sep)[:-2] + ["docs"])

    def test_html(self):
        src_dir = config_dir = self.path_to_docs
        output_dir = os.path.sep.join([src_dir, "_build", "html"])
        doctree_dir = os.path.sep.join([src_dir, "_build", "doctrees"])
        app = Sphinx(
            src_dir,
            config_dir,
            output_dir,
            doctree_dir,
            buildername="html",
            warningiserror=False,
            parallel=n_cpu,
            confoverrides={"plot_gallery": 0},
        )
        app.build()

    def test_linkcheck(self):
        src_dir = config_dir = self.path_to_docs
        output_dir = os.path.sep.join([src_dir, "_build", "linkcheck"])
        doctree_dir = os.path.sep.join([src_dir, "_build", "doctrees"])
        app = Sphinx(
            src_dir,
            config_dir,
            output_dir,
            doctree_dir,
            buildername="linkcheck",
            warningiserror=False,
            parallel=n_cpu,
            confoverrides={"plot_gallery": 0},
        )
        app.build()


if __name__ == "__main__":
    unittest.main()

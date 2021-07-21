import subprocess
import unittest
import os
import platform


class Doc_Test(unittest.TestCase):
    @property
    def path_to_docs(self):
        dirname, filename = os.path.split(os.path.abspath(__file__))
        return dirname.split(os.path.sep)[:-1] + ["docs"]

    def test_html(self):
        wd = os.getcwd()
        os.chdir(os.path.sep.join(self.path_to_docs))

        if platform.system() != "Windows":
            response = subprocess.run(["make", "html-noplot"])
            self.assertTrue(response.returncode == 0)
        else:
            response = subprocess.call(["make", "html"], shell=True)
            self.assertTrue(response == 0)

        os.chdir(wd)

    def test_linkcheck(self):
        wd = os.getcwd()
        os.chdir(os.path.sep.join(self.path_to_docs))

        if platform.system() != "Windows":
            response = subprocess.run(["make", "linkcheck-noplot"])
            self.assertTrue(response.returncode == 0)
        else:
            response = subprocess.call(["make", "linkcheck"], shell=True)
            self.assertTrue(response == 0)

        os.chdir(wd)


if __name__ == "__main__":
    unittest.main()

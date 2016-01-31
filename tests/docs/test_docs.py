import subprocess
import unittest
import os


class Doc_Test(unittest.TestCase):

    def test_html(self,rmdir=True):
        check = subprocess.call(["sphinx-build", "-nW", "-b", "html", "-d", "../../docs/_build/doctrees", "../../docs", "../../docs/_build/html"])
        assert check == 0

    def test_latex(self,rmdir=True):
        check = subprocess.call(["sphinx-build", "-nW", "-b", "latex", "-d", "../../docs/_build/doctrees", "../../docs", "../../docs/_build/latex"])
        assert check == 0

    def test_linkcheck(self,rmdir=True):
        check = subprocess.call(["sphinx-build", "-nW", "-b", "linkcheck", "-d", "../../docs/_build/doctrees", "../../docs", "../../docs/_build"])
        assert check == 0 

if __name__ == '__main__':
    unittest.main()
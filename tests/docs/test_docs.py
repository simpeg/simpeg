import subprocess
import unittest
import os

class Doc_Test(unittest.TestCase):

    @property
    def path_to_docs(self):
        dirname, filename = os.path.split(os.path.abspath(__file__))
        return os.path.sep.join(dirname.split(os.path.sep)[:-2] + ['docs'])

    def test_html(self):
        doctrees_path = os.path.sep.join(self.path_to_docs.split(os.path.sep) + ['_build']+['doctrees'])
        html_path = os.path.sep.join(self.path_to_docs.split(os.path.sep) + ['_build']+['html'])

        check = subprocess.Popen(("sphinx-build", "-nW", "-b", "html", "-d",
            "{0!s}".format(doctrees_path),
            "{0!s}".format(self.path_to_docs),
            "{0!s}".format(html_path)),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = check.communicate()
        assert check == 0, 'Test failed with stdout:\n{}\n and stderr:\n{}\n'.format(out,err)

    # def test_latex(self):
    #     doctrees_path = os.path.sep.join(self.path_to_docs.split(os.path.sep) + ['_build']+['doctrees'])
    #     latex_path = os.path.sep.join(self.path_to_docs.split(os.path.sep) + ['_build']+['latex'])

    #     check = subprocess.call(["sphinx-build", "-nW", "-b", "latex", "-d",
    #         "%s"%(doctrees_path),
    #         "%s"%(self.path_to_docs),
    #         "%s"%(latex_path)])
    #     assert check == 0

    def test_linkcheck(self):
        doctrees_path = os.path.sep.join(self.path_to_docs.split(os.path.sep) + ['_build']+['doctrees'])
        link_path = os.path.sep.join(self.path_to_docs.split(os.path.sep) + ['_build'])

        check = subprocess.Popen(("sphinx-build", "-nW", "-b", "linkcheck", "-d",
            "{}".format(doctrees_path),
            "{}".format(self.path_to_docs),
            "{}".format(link_path)),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = check.communicate()
        assert check == 0, 'Test failed with stdout:\n{}\n and stderr:\n{}\n'.format(out,err)


if __name__ == '__main__':
    unittest.main()

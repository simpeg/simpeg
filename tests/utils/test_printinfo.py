import pip
import unittest
import sys
import io

# Optional imports
try:
    import IPython
except ImportError:
    IPython = False

from SimPEG import Versions


class TestVersion(unittest.TestCase):

    def catch_version_stdout(self, *args, **kwargs):

        # Check the default
        stdout = sys.stdout
        sys.stdout = io.StringIO()

        # Print text version
        print(Versions(*args, **kwargs).__repr__())

        # catch the output
        out1 = sys.stdout.getvalue()
        sys.stdout = stdout

        # Check the default
        stdout = sys.stdout
        sys.stdout = io.StringIO()

        # Print html version
        print(Versions(*args, **kwargs)._repr_html_())

        # catch the output
        out2 = sys.stdout.getvalue()
        sys.stdout = stdout

        return out1, out2

    def test_version_defaults(self):

        # Check the default
        out1_text, out1_html = self.catch_version_stdout(pip)

        # Check one of the standard packages
        assert 'numpy' in out1_text
        assert 'numpy' in out1_html

        # Providing a package as a tuple
        out2_text, out2_html = self.catch_version_stdout(add_pckg=(pip,))

        # Check the provided package, with number
        assert pip.__version__ + ' : pip' in out2_text
        assert pip.__version__  in out2_html
        assert ">pip</td>" in out2_html

        # Providing a package as a list
        out3_text, out3_html = self.catch_version_stdout(add_pckg=[pip])

        assert 'numpy' in out3_text
        assert 'td style=' in out3_html

        # Check row of provided package, with number
        teststr = "<td style='text-align: right; background-color: #ccc; "
        teststr += "border: 2px solid #fff;'>"
        teststr += pip.__version__
        teststr += "</td>\n    <td style='"
        teststr += "text-align: left; border: 2px solid #fff;'>pip</td>"
        assert teststr in out3_html

if __name__ == '__main__':
    unittest.main()

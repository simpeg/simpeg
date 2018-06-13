import pip

# Optional imports
try:
    import IPython
except ImportError:
    IPython = False

from SimPEG import versions
from SimPEG.Utils import printinfo


def test_versions(capsys):

    # Check the default
    versions()
    out1, _ = capsys.readouterr()

    # Check one of the standard packages
    assert 'numpy' in out1

    # Check the 'auto'-version, providing a package
    versions(add_pckg=pip)
    out1b, _ = capsys.readouterr()

    # Check the provided package, with number
    assert pip.__version__ + ' : pip' in out1b

    # Check the 'text'-version, providing a package as tuple
    versions('print', add_pckg=(pip, ))
    out2, _ = capsys.readouterr()

    # They have to be the same, except time (run at slightly different times)
    assert out1b[75:] == out2[75:]

    # Check the 'Pretty'/'plain'-version, providing a package as list
    out3 = versions('plain', add_pckg=[pip, ])
    out3b = printinfo.versions_text(add_pckg=[pip, ])
    out3c = versions('Pretty', add_pckg=[pip, ])

    # They have to be the same, except time (run at slightly different times)
    assert out3[75:] == out3b[75:]
    if IPython:
        assert out3[75:] == out3c.data[75:]
    else:
        assert out3c is None

    # Check one of the standard packages
    assert 'numpy' in out3

    # Check the provided package, with number
    assert pip.__version__ + ' : pip' in out3

    # Check 'HTML'/'html'-version, providing a package as a list
    out4 = versions('html', add_pckg=[pip])
    out4b = printinfo.versions_html(add_pckg=[pip])
    out4c = versions('HTML', add_pckg=[pip])

    assert 'numpy' in out4
    assert 'td style=' in out4

    # They have to be the same, except time (run at slightly different times)
    assert out4[50:] == out4b[50:]
    if IPython:
        assert out4[50:] == out4c.data[50:]
    else:
        assert out4c is None

    # Check row of provided package, with number
    teststr = "<td style='text-align: right; background-color: #ccc; "
    teststr += "border: 2px solid #fff;'>"
    teststr += pip.__version__
    teststr += "</td>\n    <td style='"
    teststr += "text-align: left; border: 2px solid #fff;'>pip</td>"
    assert teststr in out4

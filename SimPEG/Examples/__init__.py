from __future__ import print_function
# Run this file to add imports.

##### AUTOIMPORTS #####
from . import DC_Analytic_Dipole
from . import EM_FDEM_1D_Inversion
from . import EM_FDEM_Analytic_MagDipoleWholespace
from . import EM_Schenkel_Morrison_Casing
from . import EM_TDEM_1D_Inversion
from . import FLOW_Richards_1D_Celia1990
from . import Inversion_IRLS
from . import Inversion_Linear
from . import Maps_ComboMaps
from . import Maps_Mesh2Mesh
from . import Mesh_Basic_ForwardDC
from . import Mesh_Basic_PlotImage
from . import Mesh_Basic_Types
from . import Mesh_Operators_CahnHilliard
from . import Mesh_QuadTree_Creation
from . import Mesh_QuadTree_FaceDiv
from . import Mesh_QuadTree_HangingNodes
from . import Mesh_Tensor_Creation
from . import MT_1D_ForwardAndInversion
from . import MT_3D_Foward
from . import PF_Gravity_Inversion_Linear
from . import PF_Magnetics_Analytics
from . import PF_Magnetics_Inversion_Linear
from . import Utils_surface2ind_topo

__examples__ = ["DC_Analytic_Dipole", "EM_FDEM_1D_Inversion", "EM_FDEM_Analytic_MagDipoleWholespace", "EM_Schenkel_Morrison_Casing", "EM_TDEM_1D_Inversion", "FLOW_Richards_1D_Celia1990", "Inversion_IRLS", "Inversion_Linear", "Maps_ComboMaps", "Maps_Mesh2Mesh", "Mesh_Basic_ForwardDC", "Mesh_Basic_PlotImage", "Mesh_Basic_Types", "Mesh_Operators_CahnHilliard", "Mesh_QuadTree_Creation", "Mesh_QuadTree_FaceDiv", "Mesh_QuadTree_HangingNodes", "Mesh_Tensor_Creation", "MT_1D_ForwardAndInversion", "MT_3D_Foward", "PF_Gravity_Inversion_Linear", "PF_Magnetics_Analytics", "PF_Magnetics_Inversion_Linear", "Utils_surface2ind_topo"]

##### AUTOIMPORTS #####

if __name__ == '__main__':
    """

        Run the following to create the examples documentation and add to the imports at the top.

    """

    import shutil, os
    from SimPEG import Examples

    # Create the examples dir in the docs folder.
    fName = os.path.realpath(__file__)
    docExamplesDir = os.path.sep.join(fName.split(os.path.sep)[:-3] + ['docs', 'content', 'examples'])
    shutil.rmtree(docExamplesDir)
    os.makedirs(docExamplesDir)

    # Get all the python examples in this folder
    thispath = os.path.sep.join(fName.split(os.path.sep)[:-1])
    exfiles  = [f[:-3] for f in os.listdir(thispath) if os.path.isfile(os.path.join(thispath, f)) and f.endswith('.py') and not f.startswith('_')]

    # Add the imports to the top in the AUTOIMPORTS section
    f = file(fName, 'r')
    inimports = False
    out = ''
    for line in f:
        if not inimports:
            out += line

        if line == "##### AUTOIMPORTS #####\n":
            inimports = not inimports
            if inimports:
                out += '\n'.join(["import {0!s}".format(_) for _ in exfiles])
                out += '\n\n__examples__ = ["' + '", "'.join(exfiles)+ '"]\n'
                out += '\n##### AUTOIMPORTS #####\n'
    f.close()

    f = file(fName, 'w')
    f.write(out)
    f.close()


    def _makeExample(filePath, runFunction):
        """Makes the example given a path of the file and the run function."""
        filePath = os.path.realpath(filePath)
        name = filePath.split(os.path.sep)[-1].rstrip('.pyc').rstrip('.py')

        docstr = runFunction.__doc__
        if docstr is None:
            doc = '{0!s}\n{1!s}'.format(name.replace('_',' '), '='*len(name))
        else:
            doc = '\n'.join([_[8:].rstrip() for _ in docstr.split('\n')])

        out = """.. _examples_{0!s}:

.. --------------------------------- ..
..                                   ..
..    THIS FILE IS AUTO GENEREATED   ..
..                                   ..
..    SimPEG/Examples/__init__.py    ..
..                                   ..
.. --------------------------------- ..

{1!s}

.. plot::

    from SimPEG import Examples
    Examples.{2!s}.run()

.. literalinclude:: ../../../SimPEG/Examples/{3!s}.py
    :language: python
    :linenos:
""".format(name, doc, name, name)

        rst = os.path.sep.join((filePath.split(os.path.sep)[:-3] + ['docs', 'content', 'examples', name + '.rst']))

        print('Creating: {0!s}.rst'.format(name))
        f = open(rst, 'w')
        f.write(out)
        f.close()

    for ex in dir(Examples):
        if ex.startswith('_'): continue
        E = getattr(Examples,ex)
        _makeExample(E.__file__, E.run)

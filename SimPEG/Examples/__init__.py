# Run this file to add imports.

##### AUTOIMPORTS #####
import EM_FDEM_1D_Inversion
import EM_FDEM_Analytic_MagDipoleWholespace
import EM_TDEM_1D_Inversion
import FLOW_Richards_1D_Celia1990
import Forward_BasicDirectCurrent
import Inversion_Linear
import Mesh_Basic_PlotImage
import Mesh_Basic_Types
import Mesh_Operators_CahnHilliard
import Mesh_QuadTree_Creation
import Mesh_QuadTree_FaceDiv
import Mesh_QuadTree_HangingNodes
import Mesh_Tensor_Creation

__examples__ = ["EM_FDEM_1D_Inversion", "EM_FDEM_Analytic_MagDipoleWholespace", "EM_TDEM_1D_Inversion", "FLOW_Richards_1D_Celia1990", "Forward_BasicDirectCurrent", "Inversion_Linear", "Mesh_Basic_PlotImage", "Mesh_Basic_Types", "Mesh_Operators_CahnHilliard", "Mesh_QuadTree_Creation", "Mesh_QuadTree_FaceDiv", "Mesh_QuadTree_HangingNodes", "Mesh_Tensor_Creation"]

##### AUTOIMPORTS #####

if __name__ == '__main__':
    """

        Run the following to create the examples documentation and add to the imports at the top.

    """

    import shutil, os
    from SimPEG import Examples

    # Create the examples dir in the docs folder.
    fName = os.path.realpath(__file__)
    docExamplesDir = os.path.sep.join(fName.split(os.path.sep)[:-3] + ['docs', 'examples'])
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
                out += '\n'.join(["import %s"%_ for _ in exfiles])
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
            doc = '%s\n%s'%(name.replace('_',' '),'='*len(name))
        else:
            doc = '\n'.join([_[8:].rstrip() for _ in docstr.split('\n')])

        out = """.. _examples_%s:

.. --------------------------------- ..
..                                   ..
..    THIS FILE IS AUTO GENEREATED   ..
..                                   ..
..    SimPEG/Examples/__init__.py    ..
..                                   ..
.. --------------------------------- ..

%s

.. plot::

    from SimPEG import Examples
    Examples.%s.run()

.. literalinclude:: ../../SimPEG/Examples/%s.py
    :language: python
    :linenos:
"""%(name,doc,name,name)

        rst = os.path.sep.join((filePath.split(os.path.sep)[:-3] + ['docs', 'examples', name + '.rst']))

        print 'Creating: %s.rst'%name
        f = open(rst, 'w')
        f.write(out)
        f.close()

    for ex in dir(Examples):
        if ex.startswith('_'): continue
        E = getattr(Examples,ex)
        _makeExample(E.__file__, E.run)

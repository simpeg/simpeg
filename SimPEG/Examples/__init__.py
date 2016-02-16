# Run this file to add imports.

##### AUTOIMPORTS #####
import Mesh_QuadTree_Creation
import EM_TDEM_1D_Inversion
import Mesh_QuadTree_FaceDiv
import Mesh_Tensor_Creation
import FLOW_Richards_1D_Celia1990
import Mesh_Operators_CahnHilliard
import Mesh_Basic_Types
import Inversion_Linear
import MT_3D_Foward
import MT_1D_ForwardAndInversion
import Forward_BasicDirectCurrent
import EM_FDEM_Analytic_MagDipoleWholespace
import Mesh_Basic_PlotImage
import Mesh_QuadTree_HangingNodes

__examples__ = ["Mesh_QuadTree_Creation", "EM_TDEM_1D_Inversion", "Mesh_QuadTree_FaceDiv", "Mesh_Tensor_Creation", "FLOW_Richards_1D_Celia1990", "Mesh_Operators_CahnHilliard", "Mesh_Basic_Types", "Inversion_Linear", "MT_3D_Foward", "MT_1D_ForwardAndInversion", "Forward_BasicDirectCurrent", "EM_FDEM_Analytic_MagDipoleWholespace", "Mesh_Basic_PlotImage", "Mesh_QuadTree_HangingNodes"]

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

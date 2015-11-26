# This will import everything in the directory into this file
from os import path as p
from glob import glob
__all__ = []
for x in glob(p.join(p.dirname(__file__), '*.py')):
    if not p.basename(x).startswith('__'):
        __import__(p.basename(x)[:-3], globals(), locals())
        __all__ += [p.basename(x)[:-3]]
del glob, p, x

if __name__ == '__main__':
    """

        Run the following to create the examples documentation.

    """

    import shutil, os
    from SimPEG import Examples

    def _makeExample(filePath, runFunction):
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

        f = open(rst, 'w')
        f.write(out)
        f.close()


    docExamplesDir = os.path.sep.join(os.path.realpath(__file__).split(os.path.sep)[:-3] + ['docs', 'examples'])
    shutil.rmtree(docExamplesDir)
    os.makedirs(docExamplesDir)

    for ex in dir(Examples):
        if ex.startswith('_'): continue
        E = getattr(Examples,ex)
        _makeExample(E.__file__, E.run)

# This will import everything in the directory into this file
from os import path as p
from glob import glob
__all__ = []
for x in glob(p.join(p.dirname(__file__), '*.py')):
    if not p.basename(x).startswith('__'):
        __import__(p.basename(x)[:-3], globals(), locals())
        __all__ += [p.basename(x)]
del glob, p, x

from distutils.core import setup
from Cython.Build import cythonize

setup(
  name = 'Matrix vector product in C',
  ext_modules = cythonize("prodAvec.pyx"),
)

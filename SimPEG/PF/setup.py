
from Cython.Build import cythonize
from Cython.Distutils import build_ext

try:
    from setuptools import setup
    from setuptools import Extension
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension
# setup(name="calcTmat", ext_modules=cythonize('calcTmat.pyx'),)

ext_modules=[
    Extension("calcTmat",
              ["calcTmat.pyx"],
              libraries=["m"],
              extra_compile_args = ["-O3", "-ffast-math", "-march=native", "/openmp" ],
              extra_link_args=['/openmp']
              )
]


setup(
  name = "calcTmat",
  cmdclass = {"build_ext": build_ext},
  ext_modules = ext_modules
)

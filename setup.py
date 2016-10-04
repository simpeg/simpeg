#!/usr/bin/env python
from __future__ import print_function
"""SimPEG: Simulation and Parameter Estimation in Geophysics

SimPEG is a python package for simulation and gradient based
parameter estimation in the context of geophysical applications.
"""

import os
import sys
import subprocess

from distutils.core import setup
from distutils.command.build_ext import build_ext
from setuptools import find_packages
from distutils.extension import Extension



CLASSIFIERS = [
'Development Status :: 4 - Beta',
'Intended Audience :: Developers',
'Intended Audience :: Science/Research',
'License :: OSI Approved :: MIT License',
'Programming Language :: Python',
'Topic :: Scientific/Engineering',
'Topic :: Scientific/Engineering :: Mathematics',
'Topic :: Scientific/Engineering :: Physics',
'Operating System :: Microsoft :: Windows',
'Operating System :: POSIX',
'Operating System :: Unix',
'Operating System :: MacOS',
'Natural Language :: English',
]

args = sys.argv[1:]

# Make a `cleanall` rule to get rid of intermediate and library files
if "cleanall" in args:
    print("Deleting cython files...")
    # Just in case the build directory was created by accident,
    # note that shell=True should be OK here because the command is constant.
    subprocess.Popen("rm -rf build", shell=True, executable="/bin/bash")
    subprocess.Popen("find . -name \*.c -type f -delete", shell=True, executable="/bin/bash")
    subprocess.Popen("find . -name \*.so -type f -delete", shell=True, executable="/bin/bash")
    # Now do a normal clean
    sys.argv[sys.argv.index('cleanall')] = "clean"

# We want to always use build_ext --inplace
if args.count("build_ext") > 0 and args.count("--inplace") == 0:
    sys.argv.insert(sys.argv.index("build_ext")+1, "--inplace")

try:
    from Cython.Build import cythonize
    from Cython.Distutils import build_ext
    USE_CYTHON = True
except Exception as e:
    USE_CYTHON = False

class NumpyBuild(build_ext):
    def finalize_options(self):
        build_ext.finalize_options(self)
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())

ext = '.pyx' if USE_CYTHON else '.c'

cython_files = [
                    "SimPEG/Utils/interputils_cython",
                    "SimPEG/Mesh/TreeUtils"
               ]
extensions = [Extension(f, [f+ext]) for f in cython_files]
scripts = [f+'.pyx' for f in cython_files]

if USE_CYTHON and "cleanall" not in args:
    from Cython.Build import cythonize
    extensions = cythonize(extensions)

import os, os.path

with open("README.rst") as f:
    LONG_DESCRIPTION = ''.join(f.readlines())

setup(
    name = "SimPEG",
    version = "0.3.0",
    packages = find_packages(),
    install_requires = ['numpy>=1.7',
                        'scipy>=0.13',
                        'Cython'
                       ],
    author = "Rowan Cockett",
    author_email = "rowanc1@gmail.com",
    description = "SimPEG: Simulation and Parameter Estimation in Geophysics",
    long_description = LONG_DESCRIPTION,
    license = "MIT",
    keywords = "geophysics inverse problem",
    url = "http://simpeg.xyz/",
    download_url = "http://github.com/simpeg/simpeg",
    classifiers=CLASSIFIERS,
    platforms = ["Windows", "Linux", "Solaris", "Mac OS-X", "Unix"],
    use_2to3 = False,
    cmdclass={'build_ext':NumpyBuild},
    setup_requires=['numpy'],
    ext_modules = extensions,
    scripts=scripts,
)

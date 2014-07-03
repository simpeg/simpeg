#!/usr/bin/env python
"""SimPEG: Simulation and Parameter Estimation for Geophysics

SimPEG is a python package for simulation and gradient based
parameter estimation in the context of geophysical applications.
"""

from distutils.core import setup
from setuptools import find_packages
from Cython.Build import cythonize
import numpy as np

CLASSIFIERS = [
'Intended Audience :: Developers',
'License :: OSI Approved :: MIT License',
'Programming Language :: Python',
'Topic :: Scientific/Engineering',
'Topic :: Scientific/Engineering :: Mathematics',
'Operating System :: Microsoft :: Windows',
'Operating System :: POSIX',
'Operating System :: Unix',
'Operating System :: MacOS',
]

import os, os.path

setup(
    name = "SimPEG",
    version = "0.1.1",
    packages = find_packages(),
    install_requires = ['numpy>=1.7',
                        'scipy>=0.13',
                        'matplotlib>=1.3',
                       ],
    author = "Rowan Cockett",
    author_email = "rowanc1@gmail.com",
    description = "SimPEG: Simulation and Parameter Estimation for Geophysics",
    license = "MIT",
    keywords = "geophysics inverse problem",
    url = "http://simpeg.3ptscience.com/",
    download_url = "http://github.com/simpeg/simpeg",
    classifiers=CLASSIFIERS,
    platforms = ["Windows", "Linux", "Solaris", "Mac OS-X", "Unix"],
    use_2to3 = False,
    include_dirs=[np.get_include()],
    ext_modules = cythonize('SimPEG/Utils/interputils_cython.pyx')
)

#!/usr/bin/env python
"""SimPEG: Simulation and Parameter Estimation for Geophysics

SimPEG is a python package for simulation and gradient based
parameter estimation in the context of geophysical applications.
"""

import ez_setup
ez_setup.use_setuptools()
from setuptools import setup, find_packages

CLASSIFIERS = [
'Development Status :: 0.0.1 - Alpha',
'Intended Audience :: Science/Research',
'Intended Audience :: Developers',
'License :: MIT License',
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
    version = "0.1dev",
    packages = find_packages(),
    install_requires = ['numpy>=1.7',
                        'scipy>=0.12',
                        'matplotlib>=1.3',
                       ],
    author = "Rowan Cockett",
    author_email = "rowanc1@gmail.com",
    description = "SimPEG: Simulation and Parameter Estimation for Geophysics",
    license = "MIT",
    keywords = "geophysics inverse problem",
    url = "http://simeg.rtfd.org/",
    download_url = "http://github.com/simpeg",
    classifiers=CLASSIFIERS,
    platforms = ["Windows", "Linux", "Solaris", "Mac OS-X", "Unix"],
    use_2to3 = False
)

#!/usr/bin/env python
"""SimPEG: Simulation and Parameter Estimation in Geophysics

SimPEG is a python package for simulation and gradient based
parameter estimation in the context of geophysical applications.
"""

from distutils.core import setup
from setuptools import find_packages

CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python",
    "Topic :: Scientific/Engineering",
    "Topic :: Scientific/Engineering :: Mathematics",
    "Topic :: Scientific/Engineering :: Physics",
    "Operating System :: Microsoft :: Windows",
    "Operating System :: POSIX",
    "Operating System :: Unix",
    "Operating System :: MacOS",
    "Natural Language :: English",
]

with open("README.rst") as f:
    LONG_DESCRIPTION = "".join(f.readlines())

setup(
    name="SimPEG",
    version="0.19.0",
    packages=find_packages(exclude=["tests*", "examples*", "tutorials*"]),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20",
        "scipy>=1.8",
        "scikit-learn>=1.2",
        "pymatsolver>=0.2",
        "matplotlib",
        "discretize>=0.8",
        "geoana>=0.4.0",
        "empymod>=2.0.0",
        "pandas",
    ],
    author="Rowan Cockett",
    author_email="rowanc1@gmail.com",
    description="SimPEG: Simulation and Parameter Estimation in Geophysics",
    long_description=LONG_DESCRIPTION,
    license="MIT",
    keywords="geophysics inverse problem",
    url="http://simpeg.xyz/",
    download_url="http://github.com/simpeg/simpeg",
    classifiers=CLASSIFIERS,
    platforms=["Windows", "Linux", "Solaris", "Mac OS-X", "Unix"],
    use_2to3=False,
)

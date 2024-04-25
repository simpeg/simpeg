#!/usr/bin/env python
"""SimPEG: Simulation and Parameter Estimation in Geophysics

SimPEG is a python package for simulation and gradient based
parameter estimation in the context of geophysical applications.
"""

from setuptools import setup, find_packages
import os

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
    name="simpeg",
    packages=find_packages(exclude=["tests*", "examples*", "tutorials*"]),
    py_modules=["SimPEG"],
    python_requires=">=3.8",
    setup_requires=[
        "setuptools_scm",
    ],
    install_requires=[
        "numpy>=1.20",
        "scipy>=1.8",
        "scikit-learn>=1.2",
        "pymatsolver>=0.2",
        "matplotlib",
        "discretize>=0.10",
        "geoana>=0.5.0",
        "empymod>=2.0.0",
        "pandas",
    ],
    author="Rowan Cockett",
    author_email="rowanc1@gmail.com",
    description="SimPEG: Simulation and Parameter Estimation in Geophysics",
    long_description=LONG_DESCRIPTION,
    license="MIT",
    keywords="geophysics inverse problem",
    url="https://simpeg.xyz/",
    download_url="https://github.com/simpeg/simpeg",
    classifiers=CLASSIFIERS,
    platforms=["Windows", "Linux", "Solaris", "Mac OS-X", "Unix"],
    use_2to3=False,
    use_scm_version={
        "write_to": os.path.join("simpeg", "version.py"),
    },
)

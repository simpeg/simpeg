#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cross-gradient Joint Inversion of Gravity and Magnetic Anomaly Data
===============================================

Here we simultaneously invert gravity and magentic data using cross-gradient 
constraint. The recovered density and susceptibility models are supposed to have
structural similarity. For this tutorial, we focus on the following:
    
    - Defining the survey from xyz formatted data
    - Generating a mesh based on survey geometry
    - Including surface topography
    - Defining the inverse problem via combmaps (2 data misfit terms, 
        3 regularization terms including the coupling term, optimization)
    - Specifying directives for the inversion
    - Plotting the recovered model and data misfit


Although we consider gravity and magnetic anomaly data in this tutorial, 
the same approach can be used to invert gradiometry and other types of geophysical data.



"""

#########################################################################
# Import modules
# --------------
#

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import tarfile

from discretize import TensorMesh

from SimPEG.utils import plot2Ddata, surface2ind_topo
from SimPEG.potential_fields import gravity
from SimPEG import (
    maps,
    data,
    data_misfit,
    inverse_problem,
    regularization,
    optimization,
    directives,
    inversion,
    utils,
)
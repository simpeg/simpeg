.. image:: https://raw.github.com/simpeg/simpeg/main/docs/images/simpeg-logo.png
    :alt: simpeg Logo

SimPEG
******

.. image:: https://img.shields.io/pypi/v/simpeg.svg
    :target: https://pypi.python.org/pypi/simpeg
    :alt: Latest PyPI version

.. image:: https://img.shields.io/conda/v/conda-forge/simpeg.svg
    :target: https://anaconda.org/conda-forge/simpeg
    :alt: Latest conda-forge version

.. image:: https://img.shields.io/github/license/simpeg/simpeg.svg
    :target: https://github.com/simpeg/simpeg/blob/main/LICENSE
    :alt: MIT license

.. image:: https://dev.azure.com/simpeg/simpeg/_apis/build/status/simpeg.simpeg?branchName=main
    :target: https://dev.azure.com/simpeg/simpeg/_build/latest?definitionId=2&branchName=main
    :alt: Azure pipeline

.. image:: https://codecov.io/gh/simpeg/simpeg/branch/main/graph/badge.svg
    :target: https://codecov.io/gh/simpeg/simpeg
    :alt: Coverage status

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.596373.svg
   :target: https://doi.org/10.5281/zenodo.596373

.. image:: https://img.shields.io/discourse/users?server=http%3A%2F%2Fsimpeg.discourse.group%2F
    :target: https://simpeg.discourse.group/

.. image:: https://img.shields.io/badge/simpeg-purple?logo=mattermost&label=Mattermost
    :target: https://mattermost.softwareunderground.org/simpeg

.. image:: https://img.shields.io/badge/Youtube%20channel-GeoSci.xyz-FF0000.svg?logo=youtube
    :target: https://www.youtube.com/channel/UCBrC4M8_S4GXhyHht7FyQqw

Simulation and Parameter Estimation in Geophysics  -  A python package for simulation and gradient based parameter estimation in the context of geophysical applications.

The vision is to create a package for finite volume simulation with applications to geophysical imaging and subsurface flow. To enable the understanding of the many different components, this package has the following features:

* modular with respect to the spacial discretization, optimization routine, and geophysical problem
* built with the inverse problem in mind
* provides a framework for geophysical and hydrogeologic problems
* supports 1D, 2D and 3D problems
* designed for large-scale inversions

You are welcome to join our forum and engage with people who use and develop SimPEG at: https://simpeg.discourse.group/.

Weekly meetings are open to all. They are generally held on Wednesdays at 10:30am PDT. Please see the calendar (`GCAL <https://calendar.google.com/calendar/b/0?cid=ZHVhamYzMWlibThycWdkZXM5NTdoYXV2MnNAZ3JvdXAuY2FsZW5kYXIuZ29vZ2xlLmNvbQ>`_, `ICAL <https://calendar.google.com/calendar/ical/duajf31ibm8rqgdes957hauv2s%40group.calendar.google.com/public/basic.ics>`_) for information on the next meeting.

Overview Video
==============

.. image:: https://img.youtube.com/vi/yUm01YsS9hQ/0.jpg
    :target: https://www.youtube.com/watch?v=yUm01YsS9hQ
    :alt: All of the Geophysics But Backwards

Working towards all the Geophysics, but Backwards - SciPy 2016


Citing SimPEG
=============

There is a paper about SimPEG!


    Cockett, R., Kang, S., Heagy, L. J., Pidlisecky, A., & Oldenburg, D. W. (2015). SimPEG: An open source framework for simulation and gradient based parameter estimation in geophysical applications. Computers & Geosciences.

**BibTex:**

.. code::

    @article{cockett2015simpeg,
      title={SimPEG: An open source framework for simulation and gradient based parameter estimation in geophysical applications},
      author={Cockett, Rowan and Kang, Seogi and Heagy, Lindsey J and Pidlisecky, Adam and Oldenburg, Douglas W},
      journal={Computers \& Geosciences},
      year={2015},
      publisher={Elsevier}
    }

Electromagnetics
----------------

If you are using the electromagnetics module of SimPEG, please cite:

    Lindsey J. Heagy, Rowan Cockett, Seogi Kang, Gudni K. Rosenkjaer, Douglas W. Oldenburg, A framework for simulation and inversion in electromagnetics, Computers & Geosciences, Volume 107, 2017, Pages 1-19, ISSN 0098-3004, http://dx.doi.org/10.1016/j.cageo.2017.06.018.

**BibTex:**

.. code::

    @article{heagy2017,
        title= "A framework for simulation and inversion in electromagnetics",
        author= "Lindsey J. Heagy and Rowan Cockett and Seogi Kang and Gudni K. Rosenkjaer and Douglas W. Oldenburg",
        journal= "Computers & Geosciences",
        volume = "107",
        pages = "1 - 19",
        year = "2017",
        note = "",
        issn = "0098-3004",
        doi = "http://dx.doi.org/10.1016/j.cageo.2017.06.018"
    }

Questions
=========

If you have a question regarding a specific use of SimPEG, the fastest way
to get a response is by posting on our Discourse discussion forum:
https://simpeg.discourse.group/. Alternatively, if you prefer real-time chat,
you can join our Mattermost Team at
https://mattermost.softwareunderground.org/simpeg.
Please do not create an issue to ask a question.


Meetings
========

SimPEG hosts weekly meetings for users to interact with each other,
for developers to discuss upcoming changes to the code base, and for
discussing topics related to geophysics in general.
Currently our meetings are held every Wednesday, alternating between
a mornings (10:30 am pacific time) and afternoons (3:00 pm pacific time)
on even numbered Wednesdays. Find more info on our
`Mattermost <https://mattermost.softwareunderground.org/simpeg>`_.


Links
=====

Website:
https://simpeg.xyz

Forums:
https://simpeg.discourse.group/


Mattermost (real time chat):
https://mattermost.softwareunderground.org/simpeg


Documentation:
https://docs.simpeg.xyz


Code:
https://github.com/simpeg/simpeg


Tests:
https://dev.azure.com/simpeg/simpeg/_build


Bugs & Issues:
https://github.com/simpeg/simpeg/issues

Contributing
============

We always welcome contributions towards SimPEG whether they are adding
new code, suggesting improvements to existing codes, identifying bugs,
providing examples, or anything that will improve SimPEG.
Please checkout the `contributing guide <https://docs.simpeg.xyz/content/getting_started/contributing/index.html>`_
for more information on how to contribute.

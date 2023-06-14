.. _contributing:

Contributing to SimPEG
=======================

First of all, we are glad you are here! We welcome contributions and input
from the community.

This document is a set of guidelines for contributing to the repositories
hosted in the `SimPEG <https://github.com/simpeg>`_ organization on GitHub.
These repositories are maintained on a volunteer basis.

.. _questions:

Questions
=========

If you have a question regarding a specific use of SimPEG, the fastest way
to get a response is by posting on our Discourse discussion forum:
https://simpeg.discourse.group/. Alternatively, if you prefer real-time chat,
you can join our slack group at http://slack.simpeg.xyz.
Please do not create an issue to ask a question.

.. _Issues:

Issues
======

Issues are a place for you to suggest enhancements or raise problems you are
having with the package to the developers.

.. _bugs:

Bugs
----

When reporting an issue, please be as descriptive and provide sufficient
detail to reproduce the error. Whenever possible, if you can include a small
example that produces the error, this will help us resolve issues faster.


.. _suggest enhancements:

Suggesting enhancements
-----------------------

We welcome ideas for improvements on SimPEG! When writing an issue to suggest
an improvement, please

- use a descriptive title,
- explain where the gap in current functionality is,
- include a pseudocode sketch of the intended functionality

We will use the issue as a place to discuss and provide feedback. Please
remember that SimPEG is maintained on a volunteer basis. If you suggest an
enhancement, we certainly appreciate if you are also willing to take action
and start a pull request!


.. _pull_requests:

Pull Requests
=============

We welcome contributions to SimPEG in the form of pull requests (PR)

.. _contributing_new_code:

Contributing new code
---------------------

.. _getting started: https://docs.simpeg.xyz/content/basic/installing_for_developers.html

.. _practices: https://docs.simpeg.xyz/content/basic/practices.html

.. _testing: https://docs.simpeg.xyz/content/basic/practices.html#testing

.. _documentation: https://docs.simpeg.xyz/content/basic/practices.html#documentation

.. _code style: https://docs.simpeg.xyz/content/basic/practices.html#style

If you have an idea for how to improve SimPEG, please first create an issue
and `suggest enhancements`_. We will use the
issue as a place to discuss and make decisions on the suggestion. Once you are
ready to take action and commit some code to SimPEG, please check out
`getting started`_ for
tips on setting up a development environment and `practices`_
for a description of the development practices we aim to follow. In particular,

- `testing`_
- `documentation`_
- `code style`_

are aspects we look for in all pull requests. We do code reviews on pull
requests, with the aim of promoting best practices and ensuring that new
contributions can be built upon by the SimPEG community.

.. _pr_stages:

Stages of a pull request
------------------------

When first creating a pull request (PR), try to make your suggested changes as tightly
scoped as possible (try to solve one problem at a time). The fewer changes you make, the faster
your branch will be merged!

If your pull request is not ready for final review, but you still want feedback
on your current coding process please mark it as a draft pull request. Once you
feel the pull request is ready for final review, you can convert the draft PR to
an open PR by selecting the ``Ready for review`` button at the bottom of the page.

Once a pull request is in ``open`` status and you are ready for review, please ping
the simpeg developers in a github comment ``@simpeg/simpeg-developers`` to request a
review. At minimum for a PR to be eligible to merge, we look for

- 100% (or as close as possible) difference testing. Meaning any new code is completely tested.
- All tests are passing.
- All reviewer comments (if any) have been addressed.
- A developer approves the PR.

After all these steps are satisfied, a ``@simpeg/simpeg-admin`` will merge your pull request into
the main branch (feel free to ping one of us on github).

This being said, all simpeg developers and admins are essentially volunteers
providing their time for the benefit of the community. This does mean that
it might take some time for us to get your PR.


Licensing
=========

All code contributed to SimPEG is licensed under the `MIT license
<https://github.com/simpeg/simpeg/blob/main/LICENSE>`_ which allows open
and commercial use and extension of SimPEG. If you did not write
the code yourself, it is your responsibility to ensure that the existing
license is compatible and included in the contributed files or you can obtain
permission from the original author to relicense the code.

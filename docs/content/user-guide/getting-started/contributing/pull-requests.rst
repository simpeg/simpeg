.. _pull-requests:

Pull Requests
=============

We welcome contributions to SimPEG in the form of pull requests (PR).

Stages of a pull request
------------------------

When first creating a pull request (PR), try to make your suggested changes as
tightly scoped as possible (try to solve one problem at a time). The fewer
changes you make, the faster your branch will be merged!

If your pull request is not ready for final review, but you still want feedback
on your current coding process please mark it as a draft pull request. Once you
feel the pull request is ready for final review, you can convert the draft PR to
an open PR by selecting the ``Ready for review`` button at the bottom of the page.

Once a pull request is in ``open`` status and you are ready for review, please
ping the simpeg developers in a github comment ``@simpeg/simpeg-developers`` to
request a review. At minimum for a PR to be eligible to merge, we look for

- 100% (or as close as possible) difference testing. Meaning any new code is
  completely tested.
- All tests are passing.
- All reviewer comments (if any) have been addressed.
- A developer approves the PR.

After all these steps are satisfied, a ``@simpeg/simpeg-admin`` will merge your
pull request into the main branch (feel free to ping one of us on Github).

This being said, all SimPEG developers and admins are essentially volunteers
providing their time for the benefit of the community. This does mean that
it might take some time for us to get your PR.

Merging a Pull Request
----------------------

The ``@simpeg/simpeg-admin`` will merge a Pull Request to the `main` branch
using the `Squash and Merge
<https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/incorporating-changes-from-a-pull-request/about-pull-request-merges#squash-and-merge-your-commits>`_
strategy: all commits made to the PR branch will be _squashed_ to a single
commit that will be added to `main`.

SimPEG admins will ensure that the commit message is descriptive and
comprehensive. Contributors can help by providing a descriptive and
comprehensive PR description of the changes that were applied and the reasons
behind them. This will be greatly appreciated.

Admins will mention other authors that made significant contributions to
the PR in the commit message, following GitHub's approach for `Creating
co-authored commits
<https://docs.github.com/en/pull-requests/committing-changes-to-your-project/creating-and-editing-commits/creating-a-commit-with-multiple-authors#creating-co-authored-commits-using-github-desktop>`_.

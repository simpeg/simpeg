---
name: Release checklist
about: "Maintainers only: Checklist for making a new release"
title: "Release vX.Y.Z"
labels: "maintenance"
assignees: ""
---

<!-- Optional -->

**Target date:** YYYY/MM/DD

## Generate release notes

### Autogenerate release notes with GitHub

- [ ] Generate a draft for a new Release in GitHub.
- [ ] Create a new tag for it (the version number with a leading `v`).
- [ ] Generate release notes automatically.
- [ ] Copy those notes and paste them into a `notes.md` file.
- [ ] Discard the draft (we'll generate a new one later on).

### Add release notes to the docs

- [ ] Convert the Markdown file to RST with: `pandoc notes.md -o notes.rst`.
- [ ] Generate list of contributors from the release notes with:
  ```bash
  grep -Eo "@[[:alnum:]-]+" notes.rst | sort -u | sed -E 's/^/* /'
  ```
  Paste the list into the file under a new `Contributors` category.
- [ ] Check if every contributor that participated in the release is in the
  list. Generate a list of authors and co-authors from the git log with (update
  the `last_release`):
  ```bash
  export last_release="v0.20.0"
  git shortlog HEAD...$last_release -sne > contributors
  git log HEAD...$last_release | grep "Co-authored-by" | sed 's/Co-authored-by://' | sed 's/^[[:space:]]*/ /' | sort | uniq -c | sort -nr | sed 's/^ //' >> contributors
  sort -rn contributors
  ```
- [ ] Transform GitHub handles into links to their profiles:
  ```bash
  sed -Ei 's/@([[:alnum:]-]+)/`@\1 <https:\/\/github.com\/\1>`__/' notes.rst
  ```
- [ ] Copy the content of `notes.rst` to a new file
  `docs/content/release/<version>-notes.rst`.
- [ ] Edit the release notes file, following the template below and the
  previous release notes.
- [ ] Add the new release notes to the list in `docs/content/release/index.rst`.
- [ ] Open a PR with the new release notes.
- [ ] Manually view the built documentation by downloading the Azure `html_doc`
  artifact and check for formatting and errors.
- [ ] Merge that PR


<details>
<summary>Template for release notes:</summary>

```rst
.. _<VERSION>_notes:

===========================
SimPEG <VERSION> Release Notes
===========================

MONTH DAYth, YEAR

.. contents:: Highlights
    :depth: 3

Updates
=======

New features
------------

..
    list new features under subheadings, include link to related PRs

Documentation
-------------

..
    list improvements to documentation

Bugfixes
--------

..
    list bugfixes, include link to related PRs

Breaking changes
----------------

..
    list breaking changes introduced in this new release, include link to
    releated PRs

Contributors
============

..
    paste list of contributors that was generated in `notes.rst`

Pull Requests
=============

..
    paste list of PRs that were copied to `notes.rst`
```

</details>


## Make the new release

- [ ] Draft a new GitHub Release
- [ ] Create a new tag for it (the version number with a leading `v`).
- [ ] Target the release on `main` or on a particular commit from `main`
- [ ] Generate release notes automatically.
- [ ] Publish the release

## Extra tasks

After publishing the release, Azure will automatically push the new version to
PyPI, and build and deploy the docs. You can check the progress of these tasks
in: https://dev.azure.com/simpeg/simpeg/_build

After they finish:

- [ ] Check the new version is available in PyPI: https://pypi.org/project/SimPEG/
- [ ] Check the new documentation is online: https://docs.simpeg.xyz

For the new version to be available in conda-forge, we need to update the
[conda-forge/simpeg-feedstock](https://github.com/conda-forge/simpeg-feedstock)
repository. Within the same day of the release a new PR will be automatically
open in that repository. So:

- [ ] Follow the steps provided in the checklist in that PR and merge it.
- [ ] Make sure the new version is available through conda-forge: https://anaconda.org/conda-forge/simpeg

Lastly, we would need to update the SimPEG version used in
[`simpeg/user-tutorials`](https://github.com/simpeg/user-tutorials) and rerun
its notebooks:

- [ ] Open issue in
  [`simpeg/user-tutorials`](https://github.com/simpeg/user-tutorials) for
  rerunning the notebooks using the new released version of SimPEG

Finally:

- [ ] Close this issue

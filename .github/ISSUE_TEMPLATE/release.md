---
name: Release checklist
about: 'Maintainers only: Checklist for making a new release'
title: 'Release vX.Y.Z'
labels: 'maintenance'
assignees: ''
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
      ```
      grep -Eo "@[[:alnum:]-]+" notes.rst | sort -u | sed -E 's/^/* /'
      ```
      Paste the list into the file under a new `Contributors` category.
- [ ] Check if every contributor that participated in the release is in the list. Generate a list of authors and co-authors from the git log with (update the `last_release`):
      ```
      export last_release="v0.20.0"
      git shortlog HEAD...$last_release -sne > contributors
      git log HEAD...$last_release | grep "Co-authored-by" | sed 's/Co-authored-by://' | sed 's/^[[:space:]]*/ /' | sort | uniq -c | sort -nr | sed 's/^ //' >> contributors
      sort -rn contributors
      ```
- [ ] Transform GitHub handles into links to their profiles:
      ```
      sed -Ei 's/@([[:alnum:]-]+)/`@\1 <https:\/\/github.com\/\1>`__/' notes.rst
      ```
- [ ] Copy the content of `notes.rst` to a new file `docs/content/release/<version>-notes.rst`.
- [ ] Add the new release notes to the list in `docs/content/release/index.rst`.
- [ ] Open a PR with the new release notes.
- [ ] Merge that PR

## Make the new release

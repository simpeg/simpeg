# -*- coding: utf-8 -*-
#
# SimPEG documentation build configuration file, created by
# sphinx-quickstart on Fri Aug 30 18:42:44 2013.
#
# This file is execfile()d with the current directory set to its containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

import sys
import os
from sphinx_gallery.sorting import FileNameSortKey
import glob
import simpeg
import plotly.io as pio
from importlib.metadata import version

pio.renderers.default = "sphinx_gallery"

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
# sys.path.insert(0, os.path.abspath('.'))

# sys.path.append(os.path.abspath("..{}".format(os.path.sep)))
# sys.path.append(os.path.abspath(".{}_ext".format(os.path.sep)))

# -- General configuration -----------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be extensions
# coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    "sphinx.ext.autodoc",
    "numpydoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx_gallery.gen_gallery",
    "sphinx.ext.todo",
    "sphinx.ext.linkcode",
    "matplotlib.sphinxext.plot_directive",
]

# Autosummary pages will be generated by sphinx-autogen instead of sphinx-build
autosummary_generate = True

numpydoc_attributes_as_param_list = False
numpydoc_show_inherited_class_members = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix of source filenames.
source_suffix = ".rst"

# The encoding of source files.
# source_encoding = 'utf-8-sig'

# The master toctree document.
master_doc = "index"

# General information about the project.
project = "SimPEG"
copyright = "2013 - 2023, SimPEG Team, https://simpeg.xyz"

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The full version, including alpha/beta/rc tags.
release = version("simpeg")
# The short X.Y version.
version = ".".join(release.split(".")[:2])

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
# language = None

# There are two options for replacing |today|: either, you set today to some
# non-false value, then it is used:
# today = ''
# Else, today_fmt is used as the format for a strftime call.
# today_fmt = '%B %d, %Y'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ["_build"]

linkcheck_ignore = [
    r"https://github.com/simpeg/simpeg*",
    "/content/examples/*",
    "/content/tutorials/*",
    r"https://www.pardiso-project.org",
    r"https://docs.github.com/*",
    # GJI refuses the connexion during the check
    r"https://doi.org/10.1093/gji/*",
]

linkcheck_retries = 3
linkcheck_timeout = 500

# The reST default role (used for this markup: `text`) to use for all documents.
# default_role = None

# If true, '()' will be appended to :func: etc. cross-reference text.
# add_function_parentheses = True

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
# add_module_names = True

# If true, sectionauthor and moduleauthor directives will be shown in the
# output. They are ignored by default.
# show_authors = False

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# A list of ignored prefixes for module index sorting.
# modindex_common_prefix = []

# -- Edit on Github Extension ---------------------------------------------

# edit_on_github_project = "simpeg/simpeg"
# edit_on_github_branch = "main/docs"
# check_meta = False

# source code links
link_github = True
# You can build old with link_github = False

if link_github:
    import inspect
    from os.path import relpath, dirname

    extensions.append("sphinx.ext.linkcode")

    def linkcode_resolve(domain, info):
        if domain != "py":
            return None

        modname = info["module"]
        fullname = info["fullname"]

        submod = sys.modules.get(modname)
        if submod is None:
            return None

        obj = submod
        for part in fullname.split("."):
            try:
                obj = getattr(obj, part)
            except Exception:
                return None

        try:
            unwrap = inspect.unwrap
        except AttributeError:
            pass
        else:
            obj = unwrap(obj)

        try:
            fn = inspect.getsourcefile(obj)
        except Exception:
            fn = None
        if not fn:
            return None

        try:
            source, lineno = inspect.getsourcelines(obj)
        except Exception:
            lineno = None

        if lineno:
            linespec = "#L%d-L%d" % (lineno, lineno + len(source) - 1)
        else:
            linespec = ""

        try:
            fn = relpath(fn, start=dirname(simpeg.__file__))
        except ValueError:
            return None

        return f"https://github.com/simpeg/simpeg/blob/main/simpeg/{fn}{linespec}"

else:
    extensions.append("sphinx.ext.viewcode")


# Make numpydoc to generate plots for example sections
numpydoc_use_plots = True
plot_pre_code = """
import numpy as np
np.random.seed(0)
"""
plot_include_source = True
plot_formats = [("png", 100), "pdf"]

import math

phi = (math.sqrt(5) + 1) / 2

plot_rcparams = {
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.figsize": (3 * phi, 3),
    "figure.subplot.bottom": 0.2,
    "figure.subplot.left": 0.2,
    "figure.subplot.right": 0.9,
    "figure.subplot.top": 0.85,
    "figure.subplot.wspace": 0.4,
    "text.usetex": False,
}

# -- Options for HTML output ---------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
external_links = [
    dict(name="User Tutorials", url="https://simpeg.xyz/user-tutorials"),
    dict(name="SimPEG", url="https://simpeg.xyz"),
    dict(name="Contact", url="https://mattermost.softwareunderground.org/simpeg"),
]

try:
    import pydata_sphinx_theme

    html_theme = "pydata_sphinx_theme"

    # If false, no module index is generated.
    html_use_modindex = True

    html_theme_options = {
        "external_links": external_links,
        "icon_links": [
            {
                "name": "GitHub",
                "url": "https://github.com/simpeg/simpeg",
                "icon": "fab fa-github",
            },
            {
                "name": "Mattermost",
                "url": "https://mattermost.softwareunderground.org/simpeg",
                "icon": "fas fa-comment",
            },
            {
                "name": "Discourse",
                "url": "https://simpeg.discourse.group/",
                "icon": "fab fa-discourse",
            },
            {
                "name": "Youtube",
                "url": "https://www.youtube.com/c/geoscixyz",
                "icon": "fab fa-youtube",
            },
        ],
        "use_edit_page_button": False,
        "collapse_navigation": True,
        "analytics": {
            "plausible_analytics_domain": "docs.simpeg.xyz",
            "plausible_analytics_url": "https://plausible.io/js/script.js",
        },
        "navbar_align": "left",  # make elements closer to logo on the left
    }
    html_logo = "images/simpeg-logo.png"

    html_static_path = ["_static"]

    html_css_files = [
        "css/custom.css",
    ]

    # Define SimPEG version for generating links to sources in GitHub
    github_version = SimPEG.__version__
    if "dev" in github_version:
        github_version = "main"
    # hardcoded github_version to check if it works well. DON'T MERGE!
    github_version = "v0.21.1"

    html_context = {
        "github_user": "simpeg",
        "github_repo": "simpeg",
        "github_version": github_version,
        "doc_path": "docs",
    }
except Exception:
    html_theme = "default"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
# html_theme_options = {}

# Add any paths that contain custom themes here, relative to this directory.
# html_theme_path = []

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
# html_title = None

# A shorter title for the navigation bar.  Default is the same as html_title.
# html_short_title = None

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
# html_logo = None

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
html_favicon = os.path.sep.join([".", "images", "logo-block.ico"])

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = []

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
# html_last_updated_fmt = '%b %d, %Y'

# If true, SmartyPants will be used to convert quotes and dashes to
# typographically correct entities.
# html_use_smartypants = True

# Custom sidebar templates, maps document names to template names.
# html_sidebars = {}

# Additional templates that should be rendered to pages, maps page names to
# template names.
# html_additional_pages = {}

# If false, no module index is generated.
# html_domain_indices = True

# If false, no index is generated.
# html_use_index = True

# If true, the index is split into individual pages for each letter.
# html_split_index = False

# If true, links to the reST sources are added to the pages.
# html_show_sourcelink = True

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
html_show_sphinx = False

# If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.
html_show_copyright = False

# If true, an OpenSearch description file will be output, and all pages will
# contain a <link> tag referring to it.  The value of this option must be the
# base URL from which the finished HTML is served.
# html_use_opensearch = ''

# This is the file name suffix for HTML files (e.g. ".xhtml").
# html_file_suffix = None

# Output file base name for HTML help builder.
htmlhelp_basename = "simpegdoc"


# -- Options for LaTeX output --------------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    #'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    #'preamble': '',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass [howto/manual]).
latex_documents = [
    ("index", "simpeg.tex", "SimPEG Documentation", "SimPEG Team", "manual"),
]

# The name of an image file (relative to this directory) to place at the top of
# the title page.
# latex_logo = None

# For "manual" documents, if this is true, then toplevel headings are parts,
# not chapters.
# latex_use_parts = False

# If true, show page references after internal links.
# latex_show_pagerefs = False

# If true, show URL addresses after external links.
# latex_show_urls = False

# Documents to append as an appendix to all manuals.
# latex_appendices = []

# If false, no module index is generated.
# latex_domain_indices = True


# -- Options for manual page output --------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [("index", "SimPEG", "SimPEG Documentation", ["SimPEG Team"], 1)]

# If true, show URL addresses after external links.
# man_show_urls = False

# Intersphinx
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "properties": ("https://propertiespy.readthedocs.io/en/latest/", None),
    "discretize": ("https://discretize.simpeg.xyz/en/main/", None),
    "pymatsolver": ("https://pymatsolver.readthedocs.io/en/latest/", None),
}
numpydoc_xref_param_type = True

# -- Options for Texinfo output ------------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        "index",
        "SimPEG",
        "SimPEG Documentation",
        "SimPEG Team",
        "SimPEG",
        "Simulation and parameter estimation in geophyiscs.",
        "Miscellaneous",
    ),
]

tutorial_dirs = glob.glob("../tutorials/[!_]*")
tut_gallery_dirs = ["content/tutorials/" + os.path.basename(f) for f in tutorial_dirs]

# Scaping images to generate on website
from plotly.io._sg_scraper import plotly_sg_scraper
import pyvista

# Make sure off screen is set to true when building locally
pyvista.OFF_SCREEN = True
# necessary when building the sphinx gallery
pyvista.BUILDING_GALLERY = True

image_scrapers = ("matplotlib", plotly_sg_scraper, pyvista.Scraper())

# Sphinx Gallery
sphinx_gallery_conf = {
    # path to your examples scripts
    "examples_dirs": ["../examples"] + tutorial_dirs,
    "gallery_dirs": ["content/examples"] + tut_gallery_dirs,
    "within_subsection_order": FileNameSortKey,
    "filename_pattern": "\.py",
    "backreferences_dir": "content/api/generated/backreferences",
    "doc_module": "simpeg",
    "show_memory": True,
    "image_scrapers": image_scrapers,
}


# Documents to append as an appendix to all manuals.
# texinfo_appendices = []

# If false, no module index is generated.
# texinfo_domain_indices = True

# How to display URL addresses: 'footnote', 'no', or 'inline'.
# texinfo_show_urls = 'footnote'

# graphviz_dot = shutil.which("dot")
# this must be png, because links on SVG are broken
# graphviz_output_format = "png"

autodoc_member_order = "bysource"


# def supress_nonlocal_image_warn():
#     import sphinx.environment

#     sphinx.environment.BuildEnvironment.warn_node = _supress_nonlocal_image_warn


# def _supress_nonlocal_image_warn(self, msg, node, **kwargs):
#     from docutils.utils import get_source_line

#     if not msg.startswith("nonlocal image URI found:"):
#         self._warnfunc(msg, "{0!s}:{1!s}".format(*get_source_line(node)))


# supress_nonlocal_image_warn()

# http://stackoverflow.com/questions/11417221/sphinx-autodoc-gives-warning-pyclass-reference-target-not-found-type-warning

nitpick_ignore = [
    ("py:class", "discretize.base.base_mesh.BaseMesh"),
    ("py:class", "callable"),
    ("py:class", "properties.base.base.HasProperties"),
    ("py:class", "pymatsolver.direct.Pardiso"),
    ("py:class", "matplotlib.axes._axes.Axes"),
    ("py:class", "optional"),
    ("py:class", "builtins.float"),
    ("py:class", "builtins.complex"),
    ("py:meth", "__call__"),
]

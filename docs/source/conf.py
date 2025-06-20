# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
import warnings

sys.path.insert(0, os.path.abspath("../../"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "pvdeg"
copyright = "2023, NREL"
author = "Alliance For Sustainable Energy LLC"

import pvdeg

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

########################################################################################
### INSTALL pydoc with conda NOT PIP and run in same conda environment when building ###
########################################################################################

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.extlinks",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "IPython.sphinxext.ipython_directive",
    "IPython.sphinxext.ipython_console_highlighting",
    # 'sphinx_gallery.gen_gallery',
    "sphinx_gallery.load_style",  # thumbnail gallery for .ipynb
    "nbsphinx",  # convert .ipynb to html, install pandoc using CONDA not pip
    "sphinx_toggleprompt",
]

# sphinx_gallery_conf = {
#      'examples_dirs': '../examples',   # path to your example scripts
#      'gallery_dirs': 'auto_examples',  # path to where to save gallery generated output
# }

# Add or update these if necessary
autodoc_default_options = {
    "member-order": "bysource",
    "show-inheritance": True,
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

# supress warnings in gallery output
# https://sphinx-gallery.github.io/stable/configuration.html
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="Matplotlib is currently using agg, which is a"
    " non-GUI backend, so cannot show the figure.",
)

napoleon_use_rtype = False  # group rtype on same line together with return

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix of source filenames.
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"

# The short X.Y version.
version = "%s" % (pvdeg.__version__)
# The full version, including alpha/beta/rc tags.
release = version

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["**.ipynb_checkpoints", "_build", "Thumbs.db", ".DS_Store"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# List of external link aliases.  Allows use of :pull:`123` to autolink that PR
extlinks = {
    "issue": ("https://github.com/NREL/PVDegradationTools/issues/%s", "issue %s"),
    "pull": ("https://github.com/NREL/PVDegradationTools/pull/%s", "pull %s"),
    "ghuser": ("https://github.com/%s", "ghuser %s"),
}

## Generate autodoc stubs with summaries from code
autosummary_generate = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
# https://pydata-sphinx-theme.rtfd.io/en/latest/user_guide/configuring.html

html_theme_options = {
    "navigation_depth": 4,
    "titles_only": False,
    "extra_nav_links": {
        "User Guide": "user_guide/index",
        "Tutorials": "tutorials/index",
        "API reference": "api",
        "What's New": "whatsnew/index",
    },
    "github_url": "https://github.com/NREL/PVDegradationTools",
    # "favicons": [
    #     {"rel": "icon", "sizes": "16x16", "href": "favicon-16x16.png"}, ### CHECK THIS IMAGE ###
    #     {"rel": "icon", "sizes": "32x32", "href": "favicon-32x32.png"}, ### CHECK THIS IMAGE ###
    # ],
    ### DO WE HAVE ANY OF THESE ###
    # "icon_links": [
    #     {
    #         "name": "StackOverflow",
    #         "url": "https://stackoverflow.com/questions/tagged/pvlib",
    #         "icon": "fab fa-stack-overflow",
    #     },
    #     {
    #         "name": "Google Group",
    #         "url": "https://groups.google.com/g/pvlib-python",
    #         "icon": "fab fa-google",
    #     },
    #     {
    #         "name": "PyPI",
    #         "url": "https://pypi.org/project/pvlib/",
    #         "icon": "fab fa-python",
    # },
    # ],
    # "use_edit_page_button": True, # THIS WAS PROBLEMATIC FOR SOME REASON
    "show_toc_level": 1,
    "footer_items": ["copyright", "sphinx-version", "sidebar-ethical-ads"],
    "left_sidebar_end": [],
}

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = (
    "./_static/logo-vectors/PVdeg-Logo-Horiz-Color.svg"  # logo should work at this path
)

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
html_favicon = "./_static/pvdeg.ico"

html_static_path = ["_static"]

# If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.
html_show_copyright = True

# Output file base name for HTML help builder.
htmlhelp_basename = "pvdeg_pythondoc"


# custom CSS workarounds
def setup(app):
    # A workaround for the responsive tables always having annoying scrollbars.
    app.add_css_file("no_scrollbars.css")
    # Override footnote callout CSS to be normal text instead of superscript
    # In-line links to references as numbers in brackets.
    app.add_css_file("reference_format.css")
    # Add a warning banner at the top of the page if viewing the "latest" docs
    app.add_js_file("version-alert.js")


# -- Options for LaTeX output ---------------------------------------------


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)

# Documents to append as an appendix to all manuals.
# texinfo_appendices = []

# If false, no module index is generated.
# texinfo_domain_indices = True

# How to display URL addresses: 'footnote', 'no', or 'inline'.
# texinfo_show_urls = 'footnote'

# If true, do not generate a @detailmenu in the "Top" node's menu.
# texinfo_no_detailmenu = False

# Example configuration for intersphinx: refer to the Python standard library.
# intersphinx_mapping = {
#     'python': ('https://docs.python.org/3/', None),
#     'numpy': ('https://numpy.org/doc/stable/', None),
#     'scipy': ('https://docs.scipy.org/doc/scipy/reference/', None),
#     'pandas': ('https://pandas.pydata.org/pandas-docs/stable', None),
#     'matplotlib': ('https://matplotlib.org/stable', None),
# }

ipython_warning_is_error = False

# suppress "WARNING: Footnote [1] is not referenced." messages
# https://github.com/pvlib/pvlib-python/issues/837
suppress_warnings = ["ref.footnote"]


# supress warnings in gallery output
# https://sphinx-gallery.github.io/stable/configuration.html
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="Matplotlib is currently using agg, which is a"
    " non-GUI backend, so cannot show the figure.",
)

# %% helper functions for intelligent "View on Github" linking
# based on
# https://gist.github.com/flying-sheep/b65875c0ce965fbdd1d9e5d0b9851ef1


def get_obj_module(qualname):
    """Get a module/class/attribute and its original module by qualname.

    Useful for
    looking up the original location when a function is imported into an __init__.py.

    Examples
    --------
    >>> func, mod = get_obj_module("pvlib.iotools.read_midc")
    >>> mod.__name__
    'pvlib.iotools.midc'
    """
    modname = qualname
    classname = None
    attrname = None
    while modname not in sys.modules:
        attrname = classname
        modname, classname = modname.rsplit(".", 1)

    # retrieve object and find original module name
    if classname:
        cls = getattr(sys.modules[modname], classname)
        modname = cls.__module__
        obj = getattr(cls, attrname) if attrname else cls
    else:
        obj = None

    return obj, sys.modules[modname]


# def get_linenos(obj):
#     """Get an object’s line numbers in its source code file"""
#     try:
#         lines, start = inspect.getsourcelines(obj)
#     except TypeError:  # obj is an attribute or None
#         return None, None
#     except OSError:  # obj listing cannot be found
#         # This happens for methods that are not explicitly defined
#         # such as the __init__ method for a dataclass
#         return None, None
#     else:
#         return start, start + len(lines) - 1


# def make_github_url(file_name):
#     """
#     Generate the appropriate GH link for a given docs page.  This function
#     is intended for use in sphinx template files.

#     The target URL is built differently based on the type of page.  The pydata
#     sphinx theme has a built-in `file_name` variable that looks like
#     "/docs/sphinx/source/api.rst" or "generated/pvlib.atmosphere.alt2pres.rst"
#     """

#     URL_BASE = "https://github.com/pvlib/pvlib-python/blob/main/"

#     # is it a gallery page?
#     if any(d in file_name for d in sphinx_gallery_conf['gallery_dirs']):
#         if file_name.split("/")[-1] == "index":
#             example_file = "README.rst"
#         else:
#             example_file = file_name.split("/")[-1].replace('.rst', '.py')
#         target_url = URL_BASE + "docs/examples/" + example_file

#     # is it an API autogen page?
#     elif "generated" in file_name:
#         # pagename looks like "generated/pvlib.atmosphere.alt2pres.rst"
#         qualname = file_name.split("/")[-1].replace('.rst', '')
#         obj, module = get_obj_module(qualname)
#         path = module.__name__.replace(".", "/") + ".py"
#         target_url = URL_BASE + path
#         # add line numbers if possible:
#         start, end = get_linenos(obj)
#         if start and end:
#             target_url += f'#L{start}-L{end}'

#     # Just a normal source RST page
#     else:
#         target_url = URL_BASE + "docs/sphinx/source/" + file_name

#     return target_url


# variables to pass into the HTML templating engine; these are accessible from
# _templates/breadcrumbs.html
# html_context = {
#     'make_github_url': make_github_url,
#     'edit_page_url_template': '{{ make_github_url(file_name) }}',
# }

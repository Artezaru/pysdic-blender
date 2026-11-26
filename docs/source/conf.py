# -----
# Import the necessary modules
# -----
import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

import datetime

import pydata_sphinx_theme

# -----
# Path setup
# -----

project_directory = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "..")
doc_source_directory = os.path.join(project_directory, "docs", "source")

sys.path.insert(0, project_directory)
sys.path.insert(0, doc_source_directory)

# -----
# Project information
# -----


def read_version():
    version_file = os.path.join(project_directory, "pysdic_blender", "__version__.py")
    namespace = {}
    with open(version_file, "r") as file:
        exec(file.read(), namespace)
    return namespace["__version__"]


project = "pysdic-blender"
copyright = f"2026-{datetime.datetime.now().year}, Artezaru"
author = "Artezaru"
release = read_version()

# -----
# Extensions and general configuration
# -----

html_theme = "pydata_sphinx_theme"
html_static_path = [
    os.path.join(doc_source_directory, "_static"),
]

extensions = [
    "sphinx_copybutton",  # Add copy button to code blocks
    "sphinx.ext.autodoc",  # Automatically document the code
    "sphinx.ext.viewcode",  # Add links to the code
    "sphinx.ext.napoleon",  # Support for Google and Numpy docstring formats
    "sphinx.ext.githubpages",  # Publish the documentation on GitHub
    "sphinx.ext.autosummary",  # Generate summaries of the modules
    "sphinx.ext.mathjax",  # Render math in the documentation
    "sphinx_gallery.gen_gallery",  # Generate galleries of examples
    "sphinx_design",  # Add design elements to the documentation
    "sphinx.ext.intersphinx",  # Link to other projects' documentation
]


# -----
# Autosummary configuration
# -----
autosummary_generate = True
templates_path = ["_templates"]

# -----
# Autodoc configuration
# -----
autodoc_typehints = "description"
add_module_names = False

# -----
# Napoleon configuration
# -----
napoleon_google_docstring = False
napoleon_numpy_docstring = True

# -----
# Intersphinx configuration
# -----
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pysdic": ("https://Artezaru.github.io/pysdic/", None),
}


# Legacy "^^^" / "+++" header and footer separators of the sphinx-design cards
suppress_warnings = ["design.card_legacy"]


# -----
# Sphinx Gallery configuration
# -----
# The examples are executed during the build: the documentation must be built in the Python 3.13 environment
# where ``bpy`` is installed.
gallery_directory = os.path.join(project_directory, "gallery")

sphinx_gallery_conf = {
    # Paths relative to this conf.py: with absolute paths, sphinx-gallery writes absolute
    # paths in the toctrees of _gallery/index.rst, which Sphinx cannot resolve.
    "examples_dirs": [os.path.relpath(gallery_directory, doc_source_directory)],
    "gallery_dirs": ["_gallery"],
    "backreferences_dir": "_gallery_backreferences",
    "doc_module": ("pysdic_blender",),
    "capture_repr": ["__repr__", "__str__", "_repr_html_"],
    "image_scrapers": ("matplotlib",),
    "filename_pattern": r"\.py$",
    # Order of the examples: by file name
    "within_subsection_order": "sphinx_gallery.sorting.FileNameSortKey",
}

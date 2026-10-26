project = "YTM"

extensions = [
    "autoapi.extension",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "numpydoc",
    "myst_parser",
    "sphinx_copybutton",
]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
exclude_patterns = ["_build"]

autoapi_dirs = ["../src/ytm"]
autoapi_ignore = ["*/_o/*", "*/_e/*"]
autoapi_options = ["members", "undoc-members", "show-inheritance", "show-module-summary", "imported-members", "inherited-members"]
numpydoc_show_class_members = False
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

html_theme = "pydata_sphinx_theme"

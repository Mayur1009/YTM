import dataclasses
import types

project = "YTM"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "numpydoc",
    "myst_parser",
    "sphinx_copybutton",
]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
exclude_patterns = ["_build"]
templates_path = ["_templates"]

autosummary_generate = True
autodoc_default_options = {"members": True, "inherited-members": True, "show-inheritance": True, "exclude-members": "__init__"}
autodoc_mock_imports = ["cupy", "pynvml"]
autodoc_preserve_defaults = True
numpydoc_show_class_members = False
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

html_theme = "pydata_sphinx_theme"


def _drop_dataclass_signature(app, what, name, obj, options, signature, return_annotation):
    if what == "class" and dataclasses.is_dataclass(obj):
        return "", return_annotation


def _skip_module_attrs(app, what, name, obj, skip, options):
    if isinstance(obj, types.ModuleType):
        return True


def setup(app):
    app.connect("autodoc-process-signature", _drop_dataclass_signature)
    app.connect("autodoc-skip-member", _skip_module_attrs)

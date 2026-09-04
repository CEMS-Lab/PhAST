"""Sphinx configuration for PhAST."""
import os
from pathlib import Path
import re
import sys

# The package uses a src/ layout.
sys.path.insert(0, os.path.abspath("../src"))

project = "PhAST"
author = "Allamaprabhu Ani"
copyright = "2026, Allamaprabhu Ani"
_pyproject = Path(__file__).parents[1] / "pyproject.toml"
_version_match = re.search(r'^version = "([^"]+)"', _pyproject.read_text(encoding="utf-8"), re.MULTILINE)
version = _version_match.group(1) if _version_match else "unknown"
release = version
commit = os.environ.get("PHAST_COMMIT", os.environ.get("GITHUB_SHA", "unavailable"))
site_description = (
    "PhAST is an open-source PyTorch finite-element solver for two-dimensional "
    "dynamic and quasi-static phase-field fracture."
)
html_context = {
    "phast_version": version,
    "phast_commit": commit,
    "phast_description": site_description,
}

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinx_immaterial",
    "sphinxcontrib.mermaid",
]

# MyST: enable common extensions but keep parsing forgiving for existing .md.
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
]
myst_heading_anchors = 3
myst_nb_render_plugin = "jupyter"
nb_execution_mode = os.environ.get("PHAST_NB_EXECUTION_MODE", "off")
nb_execution_excludepatterns = ["tutorial/problem_setup_walkthrough.ipynb"]
nb_execution_timeout = 180
nb_execution_raise_on_error = True
nb_execution_show_tb = True

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
    ".ipynb": "myst-nb",
}
master_doc = "index"
templates_path = ["_templates"]

# Autodoc: don't fail the build on heavy missing imports.
autodoc_mock_imports = [
    "torch",
    "numpy",
    "scipy",
    "matplotlib",
    "h5py",
    "meshio",
    "gmsh",
    "PIL",
    "yaml",
    "pyamg",
    "pymetis",
    "pyvista",
    "cupy",
]
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# HTML theme.
html_theme = "sphinx_immaterial"
html_title = "PhAST | PyTorch Phase-Field Fracture Solver"
html_short_title = "PhAST"
html_baseurl = "https://cems-lab.github.io/PhAST/"
html_logo = "_static/brand/phast-icon.png"
html_favicon = "_static/brand/phast-icon.png"
html_static_path = ["_static"]
html_extra_path = ["../assets"]
html_css_files = ["phast.css"]
html_meta = {
    "description": site_description,
    "keywords": "PhAST, phase-field fracture, PyTorch, matrix-free solver, differentiable mechanics, fracture mechanics",
    "viewport": "width=device-width, initial-scale=1.0",
}
html_context["canonical_baseurl"] = html_baseurl.rstrip("/")
html_theme_options = {
    "icon": {"repo": "fontawesome/brands/github"},
    "site_url": "https://cems-lab.github.io/PhAST/",
    "repo_url": "https://github.com/CEMS-Lab/PhAST",
    "repo_name": "CEMS-Lab/PhAST",
    "features": [
        "navigation.expand",
        "navigation.sections",
        "navigation.top",
        "search.share",
        "toc.follow",
        "toc.sticky",
    ],
    "palette": [
        {
            "media": "(prefers-color-scheme: light)",
            "scheme": "default",
            "primary": "deep-orange",
            "accent": "orange",
            "toggle": {
                "icon": "material/lightbulb-outline",
                "name": "Switch to dark mode",
            },
        },
        {
            "media": "(prefers-color-scheme: dark)",
            "scheme": "slate",
            "primary": "deep-orange",
            "accent": "orange",
            "toggle": {
                "icon": "material/lightbulb",
                "name": "Switch to light mode",
            },
        },
    ],
}

# Be lenient: missing references are warnings, not errors, so -W only catches real issues.
nitpicky = False

# ASME's DOI destination rejects automated HEAD/GET checks with HTTP 403 even
# though the DOI resolves in an interactive browser. Keep the citation link for
# readers and exclude only this known bot-blocked endpoint from linkcheck.
linkcheck_ignore = [r"https://doi\.org/10\.1115/1\.2900803"]
linkcheck_report_timeouts_as_broken = False

# Wrapper pages may carry an H1 title and include sliced documentation starting
# at an H2. Sphinx can demote the included headings, producing harmless H1->H3
# jumps. The pages render correctly; silence just this cosmetic warning class.
suppress_warnings = ["myst.header"]

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    # Local/generated build artifacts.
    "qs_hpc_results/**",
    # The documentation-native copy is the rendered and executed lesson.
    "tutorial/problem_setup_walkthrough.ipynb",
    "_modules/**",
    "_sources/**",
    "genindex",
    "py-modindex",
    "search",
]


def _exclude_secondary_pages_from_sitemap(app, pagename, templatename, context, doctree):
    """Remove generated and maintainer-facing pages from the public sitemap."""
    excluded = (
        pagename.startswith("_modules/")
        or pagename.startswith("_sources/")
        or pagename.startswith("maintainer/")
        or pagename in {"agent-contribution-guide", "genindex", "py-modindex", "search"}
    )
    if excluded and app.sitemap_links:
        app.sitemap_links.pop()


def setup(app):
    app.connect("html-page-context", _exclude_secondary_pages_from_sitemap)

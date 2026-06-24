"""Sphinx configuration for PhAST."""
import os
import sys

# The package uses a src/ layout.
sys.path.insert(0, os.path.abspath("../src"))

project = "PhAST"
author = "Allamaprabhu Ani"
copyright = "2026, Allamaprabhu Ani"
release = "0.16.2"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinx_immaterial",
    "sphinxcontrib.mermaid",
    "sphinx_sitemap",
]

# MyST: enable common extensions but keep parsing forgiving for existing .md.
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
]
myst_heading_anchors = 3

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
master_doc = "index"

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
    "description": "PhAST is an open-source, matrix-free PyTorch solver for explicit dynamic phase-field fracture, supporting CPU and GPU execution.",
    "keywords": "PhAST, phase-field fracture, PyTorch, matrix-free solver, differentiable mechanics, fracture mechanics",
    "viewport": "width=device-width, initial-scale=1.0",
}
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
]

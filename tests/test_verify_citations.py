"""Unit tests for ``scripts/verify_citations.py``.

Covers:

* DOI parsing across various bib field formattings.
* Placeholder / TODO detection.
* (Network) live DOI resolution against a stable reference DOI; the
  network test is skipped if doi.org is unreachable.

Issue: #243.
"""
from __future__ import annotations

import importlib.util
import socket
import sys
import textwrap
from pathlib import Path

import pytest

REPO_DIR = Path(__file__).resolve().parents[1]
SCRIPT = REPO_DIR / "scripts" / "verify_citations.py"


def _load_module():
    """Import the script as a module without making it importable
    via ``scripts.verify_citations`` (no ``__init__.py`` in scripts/).
    """
    spec = importlib.util.spec_from_file_location(
        "_verify_citations_under_test", SCRIPT
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_verify_citations_under_test"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


vc = _load_module()


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------

def test_parse_doi_field_simple_braces(tmp_path):
    bib = tmp_path / "x.bib"
    bib.write_text(textwrap.dedent("""
        @article{foo2020bar,
          title={Some title},
          author={A. B. and C. D.},
          journal={Journal of Stuff},
          year={2020},
          doi={10.1000/example.123}
        }
    """).strip())
    entries = vc.parse_bib(bib)
    assert len(entries) == 1
    e = entries[0]
    assert e.key == "foo2020bar"
    assert e.entry_type == "article"
    assert e.doi == "10.1000/example.123"
    assert not e.is_arxiv_preprint_doi


def test_parse_doi_field_padded_equals_and_quotes(tmp_path):
    """Mixed quote / brace / spacing styles."""
    bib = tmp_path / "x.bib"
    bib.write_text(textwrap.dedent("""
        @article{spaced2021quoted,
          author  = {Some, A.},
          title   = "A spaced  field with embedded {B}races",
          journal = {J. Things},
          year    = {2021},
          doi     =  {10.1234/abc.987} ,
        }

        @misc{nodoi2019x,
          author = {Q., R.},
          title  = {No DOI here},
          year   = {2019}
        }
    """).strip())
    entries = vc.parse_bib(bib)
    assert {e.key for e in entries} == {"spaced2021quoted", "nodoi2019x"}
    by_key = {e.key: e for e in entries}
    assert by_key["spaced2021quoted"].doi == "10.1234/abc.987"
    assert by_key["nodoi2019x"].doi is None


def test_parse_arxiv_preprint_doi(tmp_path):
    bib = tmp_path / "x.bib"
    bib.write_text(textwrap.dedent("""
        @misc{prep2022stuff,
          author={X.},
          title={A preprint},
          year={2022},
          doi={10.48550/arXiv.2202.12345}
        }
    """).strip())
    e = vc.parse_bib(bib)[0]
    assert e.doi == "10.48550/arXiv.2202.12345"
    assert e.is_arxiv_preprint_doi


def test_parse_multiline_field(tmp_path):
    bib = tmp_path / "x.bib"
    bib.write_text(textwrap.dedent("""
        @article{long2020title,
          author = {Author, One and Author, Two},
          title  = {A very long title that
                    wraps across two source
                    lines for tidy formatting},
          journal = {J.},
          year = {2020},
          doi = {10.1/y.1}
        }
    """).strip())
    e = vc.parse_bib(bib)[0]
    assert "wraps across" in e.fields["title"]
    assert e.doi == "10.1/y.1"


# ---------------------------------------------------------------------------
# Classifier tests
# ---------------------------------------------------------------------------

def test_placeholder_detected_in_journal(tmp_path):
    bib = tmp_path / "x.bib"
    bib.write_text(textwrap.dedent("""
        @article{stub2024missing,
          author = {Stub, A.},
          title = {Real title},
          journal = {TODO --- complete journal record},
          year = {2024}
        }
    """).strip())
    e = vc.parse_bib(bib)[0]
    s = vc.classify_entry(e, check_doi=False)
    assert "PLACEHOLDER" in s.issues
    assert "MISSING_DOI" in s.issues  # also missing DOI


def test_placeholder_detected_in_note(tmp_path):
    bib = tmp_path / "x.bib"
    bib.write_text(textwrap.dedent("""
        @article{accepted2026paper,
          author = {S, A.},
          title = {Real title},
          journal = {Real Journal},
          year = {2026},
          note = {In revision. Update DOI on acceptance.}
        }
    """).strip())
    e = vc.parse_bib(bib)[0]
    s = vc.classify_entry(e, check_doi=False)
    assert "PLACEHOLDER" in s.issues


def test_clean_entry_classified_ok_no_network(tmp_path):
    bib = tmp_path / "x.bib"
    bib.write_text(textwrap.dedent("""
        @article{clean2020entry,
          author = {Author, A.},
          title  = {A clean entry},
          journal = {J.},
          year = {2020},
          doi = {10.1234/clean.entry.2020}
        }
    """).strip())
    e = vc.parse_bib(bib)[0]
    s = vc.classify_entry(e, check_doi=False)
    assert s.issues == []
    assert s.ok


def test_arxiv_preprint_flagged(tmp_path):
    bib = tmp_path / "x.bib"
    bib.write_text(textwrap.dedent("""
        @misc{arx2024paper,
          author = {A, B.},
          title = {Preprint},
          year = {2024},
          doi = {10.48550/arXiv.2401.99999}
        }
    """).strip())
    e = vc.parse_bib(bib)[0]
    s = vc.classify_entry(e, check_doi=False)
    assert "ARXIV_PREPRINT" in s.issues
    assert s.ok  # arxiv-only is non-blocking


# ---------------------------------------------------------------------------
# Network test (skipped if doi.org unreachable)
# ---------------------------------------------------------------------------

def _doi_org_reachable(timeout: float = 5.0) -> bool:
    try:
        with socket.create_connection(("doi.org", 443), timeout=timeout):
            return True
    except OSError:
        return False


@pytest.fixture(scope="module")
def doi_org_reachable():
    return _doi_org_reachable()


def test_curl_resolves_known_doi(doi_org_reachable):
    """A famously stable DOI: ``10.1126/science.1259855``
    (Brown 2014, Science).
    """
    if not doi_org_reachable:
        pytest.skip("doi.org unreachable from this host")
    code, _ = vc.resolve_doi("10.1126/science.1259855", timeout=10.0)
    assert code is not None
    assert 200 <= code < 400, f"unexpected HTTP {code}"


def test_curl_rejects_obviously_broken_doi(doi_org_reachable):
    if not doi_org_reachable:
        pytest.skip("doi.org unreachable from this host")
    # Random-looking DOI almost certain not to resolve.
    code, _ = vc.resolve_doi(
        "10.9999/this-doi-should-not-exist-99999", timeout=10.0
    )
    # Either non-200 or None (network error). Treat both as "not OK".
    assert code is None or not (200 <= code < 400)


# ---------------------------------------------------------------------------
# End-to-end CLI test
# ---------------------------------------------------------------------------

def test_run_audit_produces_markdown_and_exit_code(tmp_path):
    bib = tmp_path / "mini.bib"
    bib.write_text(textwrap.dedent("""
        @article{ok1,
          author = {A},
          title = {Real title},
          journal = {Real J},
          year = {2020},
          doi = {10.1234/clean.1}
        }
        @article{missing1,
          author = {B},
          title = {Title},
          journal = {J},
          year = {2021}
        }
    """).strip())
    md, code = vc.run_audit([bib], strict=False, check_doi=False)
    assert "Citation audit" in md
    assert "missing1" in md
    assert "MISSING_DOI" in md
    assert code == 1


def test_run_audit_clean_returns_zero(tmp_path):
    bib = tmp_path / "clean.bib"
    bib.write_text(textwrap.dedent("""
        @article{ok1,
          author = {A},
          title = {Real title},
          journal = {Real J},
          year = {2020},
          doi = {10.1234/clean.1}
        }
    """).strip())
    md, code = vc.run_audit([bib], strict=False, check_doi=False)
    assert code == 0
    assert "All 1 entries OK" in md


def test_strict_mode_fails_on_arxiv_only(tmp_path):
    bib = tmp_path / "arxiv.bib"
    bib.write_text(textwrap.dedent("""
        @misc{prep2024,
          author = {A},
          title = {T},
          year = {2024},
          doi = {10.48550/arXiv.2401.00001}
        }
    """).strip())
    md_loose, code_loose = vc.run_audit(
        [bib], strict=False, check_doi=False
    )
    assert code_loose == 0
    md_strict, code_strict = vc.run_audit(
        [bib], strict=True, check_doi=False
    )
    assert code_strict == 1

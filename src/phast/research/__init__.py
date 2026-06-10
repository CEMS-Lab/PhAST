"""Experimental / scaffold modules.

These are not wired into the main solver path; each is tracked here so
that their tests stay reproducible while the underlying research
direction is being explored.

Members (one per file):
  anisotropic_length_scale  — direction-dependent ℓ for the PF gradient
                              term (issue #258).
  nows                      — Neural Operator Warm Start for iterative
                              solvers (issue #61).
  process_zone              — FPZ post-processing diagnostics (issue #258).
  two_field_damage          — (α, d) two-field PF formulation scaffold
                              (issue #258).
  volterra_delay            — delayed damage activation via Volterra
                              memory kernel (issue #258).
"""

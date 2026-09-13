"""Convenient entry point for the official practice runner.

Run from the project root with ``python -m question3.main``.  The complete
implementation lives beside the pushed strategy in ``global_policy``.
"""

from .global_policy.tour_optimization.official_main import main


if __name__ == "__main__":
    raise SystemExit(main())

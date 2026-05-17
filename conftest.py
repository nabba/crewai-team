"""Pytest top-level configuration.

Force-load the real psycopg2 module at collection time so that test
modules installing a defensive MagicMock under
``if "psycopg2" not in sys.modules`` (a pattern from tests that
originally ran on hosts without psycopg2-binary installed) see
psycopg2 already present and skip the stub.

The dev .venv now ships psycopg2-binary, so the defensive stub does
more harm than good: it leaks a MagicMock across test files,
breaking any test that asserts on real psycopg2 exception types
(e.g. ``pytest.raises(psycopg2.Error)`` or
``pytest.raises(psycopg2.pool.PoolError)``).

On hosts genuinely without psycopg2 the import fails, the
downstream defensive stubs install as before, and tests that depend
on the real types are skipped by their own resilience guards.
"""
try:
    import psycopg2  # noqa: F401
    import psycopg2.pool  # noqa: F401
except ImportError:
    pass

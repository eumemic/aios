"""Discovery-runner declaration for the Matrix connector type.

Matrix v1 is a **secrets-less** connector type (design doc
``docs/design/matrix-appservice-connector.md`` §4.4): the appservice
credentials (``as_token`` / ``hs_token`` / homeserver coordinates) are
appservice-wide *container config* carried in env — exactly like
``AIOS_URL`` / ``AIOS_RUNTIME_TOKEN`` — and the per-connection encrypted
secrets dict is empty.  Replicating one fleet-wide ``as_token`` into
every connection's secrets would turn rotation into a millions-row
rewrite and buy nothing.

Declaring ``uses_connection_secrets = False`` makes the SDK runner skip
the per-``added`` ``GET /v1/connectors/runtime/secrets`` round-trip
entirely, so a fleet-scale discovery backfill issues **zero** secrets
requests (the M5 gap-4 fix; pinned by
``tests/test_connector.py::test_matrix_backfill_makes_zero_secrets_requests``).

The retirement consumer (#1906) grows this class into the full runtime:
``serve_connection`` populating the ``ghost_localpart → connection_id``
routing index and the persisted-cursor ``tail`` resume.  The HS→AS
receiver process (``__main__`` / ``appservice.py``) is unchanged.
"""

from __future__ import annotations

from aios_connector_http import HttpConnector


class MatrixConnector(HttpConnector):
    """Matrix appservice connector — secrets-less by design (§4.4)."""

    connector = "matrix"
    # Appservice credentials live in container env, not per-connection
    # secrets; the runner must not issue a secrets GET per ``added``.
    uses_connection_secrets = False

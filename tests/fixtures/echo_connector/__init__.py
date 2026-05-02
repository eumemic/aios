"""Tiny stdio MCP server fixture used by the connector supervisor e2e tests.

Hand-written against the MCP Python SDK (NOT the reference SDK that PR3
ships).  Exposes:

* ``ping`` — returns ``"pong"``
* ``echo`` — echoes the ``text`` argument

Declares the ``experimental.aios/connector`` capability that the
:class:`~aios.harness.connector_supervisor.ConnectorSubprocessRegistry`
hard-checks at initialize time.  After ``notifications/initialized``
arrives, emits one ``notifications/aios/accounts`` so the supervisor's
in-memory snapshot is non-empty for the test that inspects it.

Spawned in two ways:

* As a Python module: ``python -m tests.fixtures.echo_connector``.  Used
  by tests that build a :class:`~aios.mcp.stdio_transport.ConnectorSpec`
  directly without entry-point lookup.
* Via the ``aios.connectors`` entry point ``echo`` (declared in
  ``pyproject.toml``).  Used by tests that exercise the supervisor's
  full resolve-spec pipeline.

The fixture deliberately bypasses :meth:`BaseSession.send_notification`
for the ``notifications/aios/`` payloads — that helper validates against
the SDK's closed ``ServerNotification`` union.  PR3's reference SDK
will ship a proper ``send_aios_notification`` helper; here we reach
into ``_write_stream`` directly.
"""

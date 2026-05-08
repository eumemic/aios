# aios-echo-http

Reference connector built on `aios-connector-http`.  The canonical
example for #301: a connector that's a pure HTTP client of the aios
management API.

Three model-facing tools:

* `ping` — returns `pong`.  Verifies the connector is alive.
* `echo` — returns the input `text` back.  Verifies tool dispatch.
* `trigger_inbound` — synthesizes an inbound message.  Useful for
  integration tests that need to drive the inbound path without a
  real chat platform.

"""End-to-end test package configuration.

Deliberately EMPTY of policy relaxation. A process-global
``AIOS_OAUTH_ALLOW_INSECURE_HOSTS`` here would allowlist loopback for the whole
E2E process, disarming BOTH the write-boundary validator
(``aios.models.target_urls.validate_outbound_target_url``) and the
connection-time ``PinnedTransport`` for every test in the suite — including the
tests whose whole job is to prove loopback targets are refused (PR #1931
review). The default policy therefore stays ARMED by default; the handful of
fixtures that genuinely need a local connection opt in explicitly and
narrowly via ``tests.e2e.local_targets.allow_local_targets``.
"""

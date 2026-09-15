# Done

Root cause: the credential DNS request is source-NATed to cross the bridge,
then conntrack restores its original `127.0.0.11` destination on the reply.
Linux performs the non-loopback route check using the receiving interface's
`route_localnet` value. Setting only `net.ipv4.conf.all.route_localnet=1` did
not reliably enable that check for Docker's already-created `eth0`, so the
reply was discarded as a martian. Curl consequently reported
`HTTP_STATUS=000` and the recorder saw no request in all four legs.

The Docker backend now sets both the aggregate and `eth0`-specific sysctls
when `SandboxSpec.route_localnet` is enabled. This keeps the existing
credential-gated behavior and makes the redirected DNS reply routable to the
worker resolver; the existing sentinel DNAT then reaches the TLS proxy and
the swap can complete.

Validation: `uv run pytest -q tests/unit/test_networking.py -k route_localnet`
passed (6 passed, 142 deselected). Docker/e2e validation remains CI's oracle.

Rebased `gvisorgrn` onto `origin/master` (`7df8b5d8`, #2422 landed).

High: periodic egress refresh no longer resolves from tenant-writable
`/etc/hosts`. Both arms consult a worker-baked operator table first;
only the provision arm still reads the netns hosts file (operator
`--add-host` input, no tenant process yet).

Medium: runsc's operator chroot never saw `--add-host aios-worker:host-gateway`.
The worker now probes the daemon's `host-gateway` substitution and bakes
that address into the same operator table, so `$PROXY_IP` resolves on
every sidecar shape.

Focused unit tests: `test_sandbox_dns_resolution.py`,
`test_sandbox_network.py`, `test_egress_refresh.py`,
`test_runsc_operator_shadow.py`.

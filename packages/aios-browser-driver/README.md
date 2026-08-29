# aios-browser-driver

The Chromium driver daemon that runs inside the per-account browser image
("the computer", jarbot#106 Phase 2). It implements the worker ↔ driver wire
contract and is invoked, inside the account's browser container, as:

```
browser-cli invoke '<request JSON>'
```

`browser-cli` is a stdlib-only client that talks to the daemon over a unix
socket and prints exactly one JSON response document to stdout. The daemon
(`aios-browser-driver`) owns a single shared Chromium persistent context, one
page per agent session, and the human-takeover machinery (epoch barrier, live
frames, input spool).

## The one-source-of-truth rule

`aios_browser_driver/browser_protocol.py` is a **byte-identical vendored copy**
of `src/aios/sandbox/browser_protocol.py` in the aios repo. Do **not** edit the
copy: edit the src/ original and re-copy it verbatim —

```
cp src/aios/sandbox/browser_protocol.py \
   packages/aios-browser-driver/aios_browser_driver/browser_protocol.py
```

`tests/test_protocol_drift.py` fails the build if the two ever diverge, and the
browser image COPYs the src/ original over this file at build time so the
running driver's contract can never fork from the worker's.

The package deliberately does **not** depend on `aios` (the copy's own rule:
stdlib + pydantic only), which is what lets the browser image install it
standalone.

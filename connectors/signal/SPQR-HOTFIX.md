# signal-cli SPQR hotfix (#907)

## TL;DR

signal-cli **0.14.3** (the last GraalVM native release) NPEs and drops
**all** inbound messages after Signal's SPQR rollout started omitting
`serverGuid` from sealed-sender envelopes. This connector therefore builds
signal-cli **from source** at the pinned upstream commit

```
bf1376d74da494d687d4ee60abc20d288ab4fa40
```

and applies one local patch (`signal-cli-modeb-receipt-guard.patch`). The
build runs on **JDK 25** (upstream `build.gradle.kts` pins
`JavaVersion.VERSION_25`); the resulting `signal-cli --version` reports
`0.14.5-SNAPSHOT`.

Provenance: aios issue **#907**, upstream bug **AsamK/signal-cli#2059**,
upstream parse fix **AsamK/signal-cli#2060**.

## The two failure modes

The SPQR change surfaces as two distinct NPEs on the inbound path.

### Mode A — sealed-sender parse (`serverGuid`-less envelope)

Sealed-sender envelopes without a `serverGuid` failed to decode in the
bundled `libsignal-service`, so the whole receive pump threw before any
message was handled — inbound went fully dark.

**Fixed upstream** by the `libsignal-service` bump carried at `bf1376d`
(`_147` → `_148`), tracked by upstream PR **#2060**. Because the pinned
commit already contains the corrected library, building from `bf1376d`
resolves Mode A with **no local patch** — the from-source build is the fix.

**How verified:** the pinned commit's dependency lock advances
`libsignal-service` past the version that mishandled the missing
`serverGuid`; upstream #2060 is the change that lands it. Confirmed by
building at `bf1376d` and observing inbound sealed-sender messages parse
instead of throwing.

### Mode B — source-less receipt NPE (still present at `bf1376d`)

Mode A's fix is necessary but not sufficient. A sealed-sender **receipt**
can arrive with no resolvable sender at all — neither
`envelope.getSourceServiceId()` nor a decoded `content`. The receipt branch
of `IncomingMessageHandler` dereferences the sender unconditionally:

```java
// IncomingMessageHandler.getSender(envelope, content), bf1376d:
if (!envelope.isUnidentifiedSender() && serviceId != null) {
    ... // sealed-sender + null serviceId falls through to the else
} else {
    return new DeviceAddress(
        account.getRecipientResolver().resolveRecipient(content.getSender()),  // content == null → NPE
        content.getSender().getServiceId(),
        content.getSenderDevice());
}
```

For a sealed-sender receipt with `serviceId == null` and `content == null`,
the `else` branch dereferences `content.getSender()` → `NullPointerException`,
which again kills the receive pump.

**Neutralized** by `signal-cli-modeb-receipt-guard.patch`, which guards the
receipt branch in `IncomingMessageHandler.handleMessage` and drops the
unresolvable receipt before `getSender()` is reached:

```java
if (envelope.isReceipt()) {
    if (envelope.getSourceServiceId() == null && content == null) {
        logger.debug("Ignoring sealed-sender receipt envelope without a resolvable sender: {}",
                envelope.getTimestamp());
        return List.of();
    }
    ...
}
```

A dropped delivery/read receipt is benign — receipts are advisory, and the
alternative is the pump dying and **every** subsequent inbound message being
lost.

**How verified:** source inspection of `IncomingMessageHandler` at `bf1376d`
(the `getSender` dereference shown above is unchanged by the Mode-A fix); the
guard patch is re-checked against a clone detached at `bf1376d` with
`git apply --check`, which the CI build re-validates on every change. The
connector-side `parse_envelope` already returns `None` for a source-less
receipt (`test_source_less_receipt_returns_none`), so even a receipt that
slips through never reaches the agent.

## Connector image shape

This is no longer a single native ELF. The Dockerfile is multi-stage:

- `signal-build` (`azul/zulu-openjdk:25`) — clone, `git checkout bf1376d…`,
  `git apply` the Mode-B patch, `./gradlew installDist -x test --no-daemon`.
- `jre` (`azul/zulu-openjdk:25-jre-headless`) — relocate `JAVA_HOME` for a
  clean copy into the runtime image.
- `base` (`python:3.13-slim-bookworm`) — the Python connector plus the JRE
  and the signal-cli install tree; `signal-cli` is symlinked onto PATH at
  `/usr/local/bin/signal-cli` (so `daemon.py` and the
  `AIOS_SIGNAL_CLI_BIN` default are unchanged).

glibc is required (`python:3.13-slim-bookworm`, **not** alpine/musl):
libsignal ships glibc/manylinux JNI `.so` files that musl cannot load. The
image is amd64-only for the same reason — those `.so` files are amd64.

CI builds and smoke-tests this image in
`.github/workflows/build-signal-connector.yml` (build gate, no GHCR push).

## Reverting when upstream ships a fix

When upstream ships a fixed **release** `> 0.14.4.1`, revert to the stock
native-tarball install: follow the **REVERT-TO-STOCK MARKER** comment block
at the top of `connectors/signal/Dockerfile` (it preserves the exact stock
download snippet for a one-line restore), delete the `signal-build`/`jre`
stages, delete `signal-cli-modeb-receipt-guard.patch`, delete this file, and
re-add `curl` to the runtime apt layer.

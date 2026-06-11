# Signal connector — runs in its own container, talks HTTP to aios (#301).
#
# signal-cli's daemon is launched as a subprocess (see daemon.py); the
# binary lives in this container, not in the worker image.  Per-account
# state (registered phones, contacts, groups) is in
# /home/aios/.signal-cli; bind-mount a host volume there to persist
# across rebuilds.
#
# Bind-mount ``<workspace_root>`` (host path) at the same path inside the
# container, read-only — outbound attachments from the model arrive as
# ``/workspace/...`` strings the SDK resolves to host paths under
# ``<workspace_root>/<session_id>``.

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  REVERT-TO-STOCK MARKER — signal-cli SPQR hotfix (#907)               ║
# ╠══════════════════════════════════════════════════════════════════════╣
# ║  WHY THIS IS A FROM-SOURCE BUILD INSTEAD OF A NATIVE TARBALL:          ║
# ║  signal-cli 0.14.3 (the last native release) NPEs and drops ALL       ║
# ║  inbound after Signal's SPQR rollout started omitting ``serverGuid``   ║
# ║  from sealed-sender envelopes.  Upstream #2059 tracks it.  The fix     ║
# ║  for the parse path (Mode A) landed on master at the pinned commit     ║
# ║  below via a libsignal-service bump; the receipt-handler NPE (Mode B)  ║
# ║  is neutralized by the local ``modeb`` patch.  See SPQR-HOTFIX.md.     ║
# ║                                                                        ║
# ║  WHEN UPSTREAM SHIPS A FIXED RELEASE > 0.14.4.1:                       ║
# ║    1. Restore the stock native-tarball download (snippet preserved     ║
# ║       below, commented, for a one-line revert).                        ║
# ║    2. Delete the ``signal-build`` and ``jre`` stages.                  ║
# ║    3. Delete ``signal-cli-modeb-receipt-guard.patch`` + SPQR-HOTFIX.md.║
# ║    4. Re-add ``curl`` to the runtime apt layer.                        ║
# ║                                                                        ║
# ║  STOCK SNIPPET (the pre-hotfix install — paste back verbatim):         ║
# ║    ARG SIGNAL_CLI_VERSION=0.14.3                                       ║
# ║    RUN curl -fsSL "https://github.com/AsamK/signal-cli/releases/\      ║
# ║        download/v${SIGNAL_CLI_VERSION}/\                               ║
# ║        signal-cli-${SIGNAL_CLI_VERSION}-Linux-native.tar.gz" \         ║
# ║          -o /tmp/signal-cli.tar.gz \                                   ║
# ║     && tar -xzf /tmp/signal-cli.tar.gz -C /usr/local/bin \             ║
# ║     && chmod 0755 /usr/local/bin/signal-cli \                          ║
# ║     && rm /tmp/signal-cli.tar.gz \                                     ║
# ║     && signal-cli --version                                           ║
# ╚══════════════════════════════════════════════════════════════════════╝

# ─── signal-cli from-source build stage ──────────────────────────────────
# JDK 25 (NOT 21): upstream build.gradle.kts pins JavaVersion.VERSION_25 at
# the hotfix commit; compiling with JDK 21 fails outright.  The full JDK
# image is required here (the ``-jre`` image cannot compile gradle sources).
FROM azul/zulu-openjdk:25 AS signal-build

RUN apt-get update \
 && apt-get install -y --no-install-recommends git ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Shallow-fetch the FULL upstream SHA (never ``master`` — the tree moves).
# bf1376d carries the Mode-A SPQR parse fix (libsignal-service _147→_148);
# the Mode-B receipt-handler NPE is patched in below.  ``git fetch --depth 1``
# of the reachable commit SHA avoids cloning years of history on cold CI.
#
# TRADE-OFF: this build depends on bf1376d remaining reachable on the
# AsamK/signal-cli remote — there is no release artifact to pin to (that
# unreleased fix is the whole reason for this hotfix).  An accepted,
# short-lived risk for the incident response: revert to the stock release
# tarball (see the REVERT-TO-STOCK marker above) once upstream ships a
# fixed release, at which point this from-source fetch goes away entirely.
RUN git init /signal-cli
WORKDIR /signal-cli
RUN git remote add origin https://github.com/AsamK/signal-cli.git \
 && git fetch --depth 1 origin bf1376d74da494d687d4ee60abc20d288ab4fa40 \
 && git checkout FETCH_HEAD

# Mode-B receipt-guard patch — drops sealed-sender receipts that carry no
# resolvable sender before getSender() can NPE.  See SPQR-HOTFIX.md.
COPY connectors/signal/signal-cli-modeb-receipt-guard.patch /tmp/modeb.patch
RUN git apply /tmp/modeb.patch

# installDist produces /signal-cli/build/install/signal-cli/{bin,lib}:
# a ``bin/signal-cli`` shell launcher that runs ``java -jar`` against the
# bundled ``lib/*.jar``.  No tests (slow, network-bound); no gradle daemon
# (one-shot container build).
RUN ./gradlew installDist -x test --no-daemon

# ─── JRE relocation stage ────────────────────────────────────────────────
# This is now a JVM build (not a native ELF), so the runtime image needs a
# JRE 25.  Copy the env-resolved JAVA_HOME to a fixed path so the runtime
# COPY --from doesn't depend on the base image's internal layout.
FROM azul/zulu-openjdk:25-jre-headless AS jre
# Copy CONTENTS (``/.``) into a pre-created /opt/jre with ``cp -a`` so the
# result is /opt/jre/bin/java whether or not /opt/jre pre-exists, and so
# the JRE's symlinks + perms are preserved (``cp -r`` would nest into an
# existing dir and dereference symlinks).
RUN mkdir -p /opt/jre && cp -a "$JAVA_HOME/." /opt/jre/

# ─── Python runtime stage ────────────────────────────────────────────────
# glibc base (python:3.13-slim-bookworm) is mandatory: libsignal bundles
# glibc/manylinux JNI ``.so`` files — alpine/musl would fail to load them.
FROM python:3.13-slim-bookworm AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# OS deps: ca-certificates for TLS.  (``curl`` was dropped with the native
# tarball download — restore it if reverting to stock; see marker above.)
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# uv binary (pinned).  Faster + reproducible vs `pip install uv`.
COPY --from=ghcr.io/astral-sh/uv:0.5.18 /uv /usr/local/bin/uv

# JRE 25 from the relocation stage — signal-cli's launcher runs ``java``.
COPY --from=jre /opt/jre /opt/java
ENV JAVA_HOME=/opt/java \
    PATH=/opt/java/bin:$PATH

# signal-cli install tree (bin/ launcher + lib/*.jar) from the build stage.
# Symlink the launcher onto PATH so daemon.py and the
# ``AIOS_SIGNAL_CLI_BIN`` default (/usr/local/bin/signal-cli) are unchanged.
# The trailing ``--version`` is a build-time smoke test — a broken jar or
# missing JRE fails the build instead of shipping quietly-broken.
COPY --from=signal-build /signal-cli/build/install/signal-cli /opt/signal-cli
RUN ln -s /opt/signal-cli/bin/signal-cli /usr/local/bin/signal-cli \
 && signal-cli --version

WORKDIR /app

# SDK packages first so the layer cache stays warm across connector tweaks.
# aios-connector-http depends on aios-sdk (the workspace-published typed SDK).
COPY packages/aios-sdk /app/packages/aios-sdk
COPY packages/aios-connector-http /app/packages/aios-connector-http
COPY connectors/signal /app/connectors/signal

# ``--no-sources`` tells uv to ignore each package's ``[tool.uv.sources]``
# block — those declare workspace-relative deps (e.g.
# ``aios-sdk = { workspace = true }``) that only resolve from the
# repo root.  In this image we install the three packages by path in
# one invocation; uv satisfies the inter-package deps by name from
# the install set itself.
RUN uv pip install --system --no-cache --no-sources \
    /app/packages/aios-sdk \
    /app/packages/aios-connector-http \
    /app/connectors/signal

# Entrypoint chowns the shared state dir so the api (uid 1000) can read
# signal-cli's attachment writes from the same host bind mount.  Worker
# and connector containers run as root so the chown succeeds; the actual
# signal-cli process inside aios_signal still runs as root.
RUN chmod 0755 /app/connectors/signal/entrypoint.sh
ENTRYPOINT ["/app/connectors/signal/entrypoint.sh"]

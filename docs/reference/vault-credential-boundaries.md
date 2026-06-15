# Vault credential accepted boundaries

Vault credentials let a session use placeholder material in the sandbox while the worker swaps the real secret onto matching outbound HTTPS requests. These boundaries are ratified decisions, not open design questions.

## Inbound boundary (#881)

**What's protected:** the real secret value never enters the sandbox. It is swapped onto a request only at the egress boundary, and only for an outbound HTTPS request whose TLS SNI host is on the credential's allow-set. That confinement of the secret *value* to allowed hosts holds in **every** networking mode.

**What isn't protected:** inbound responses are not scrubbed. If an allowed HTTPS host reflects authorization material, tokens, signed headers, or other credential-derived values in its response, the model can read them. The egress swap is outbound-only: it governs where the secret *value* may go, not whether a reflected secret is redacted on the way back.

**Where a reflected secret can then go depends on the networking mode (#1153):**

- **Limited** — the network lockdown (filter `-P OUTPUT DROP` plus allowlisted `ACCEPT`s) bounds where the model can send anything it learned, including a reflected secret. This allowlist is the exfiltration boundary.
- **Unrestricted** — env-var credentials are permitted under a permit-with-warning posture (provision emits `sandbox.envvar_creds_open_egress`). Only the credential-host → proxy DNAT is installed; the filter policy stays at `ACCEPT` with **no** `-P OUTPUT DROP`. General egress remains open, so there is **no allowlist containment** — a reflected secret the model reads can be sent anywhere. The only protections that remain are the secret *value*'s confinement to allowed hosts (the SNI gate, above) and the forthcoming content/reflector-host denylist (#976). Running Unrestricted with credentials accepts this trade.

**What to do instead:** scope vault credentials only to hosts you trust not to reflect secrets; prefer a Limited environment so its allowlist bounds any follow-on exfiltration; and treat Unrestricted-with-credentials as an explicit, warned trade-off rather than a default.

## HTTPS-only scope (#887)

**What's protected:** env-var credential materialization plus the egress swap covers HTTPS hosts by construction.

**What isn't protected:** non-HTTP secret consumers are out of scope for this mechanism, including database passwords, SSH keys, raw-TCP protocols, and other clients that do not traverse the HTTPS egress swap.

**What to do instead:** keep those secrets on the worker side, or use a future named-consumer broker once one exists for non-HTTP consumers.

## Request-signing API boundary (#974)

**What's protected:** bearer-style HTTPS credentials whose final outbound request can be rewritten by the egress swap are supported.

**What isn't protected:** request-signing APIs such as AWS SigV4, HMAC-over-body schemes, and OAuth1 do not work with placeholder credentials. The request body and headers are signed before the egress swap sees them, so replacing placeholder material afterward invalidates the signature.

**What to do instead:** route signing workflows through the worker-side converged-endpoint path (#1007), where the real credential is available at signing time without exposing it to the model.

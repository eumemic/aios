# Vault credential accepted boundaries

Vault credentials let a session use placeholder material in the sandbox while the worker swaps the real secret onto matching outbound HTTPS requests. These boundaries are ratified decisions, not open design questions.

## Inbound boundary (#881)

**What's protected:** credential material is withheld from the model until an outbound request crosses the egress swap for an allowed destination.

**What isn't protected:** inbound responses are not scrubbed. If an HTTPS host reflects authorization material, tokens, signed headers, or other credential-derived values in its response, the model can read them. Containment is the Limited-env network allowlist: it controls where the model can send what it learns, not whether reflected secrets are redacted from the response.

**What to do instead:** scope vault credentials only to hosts you trust not to reflect secrets, and use Limited-env allowlists to restrict any follow-on exfiltration paths.

## HTTPS-only scope (#887)

**What's protected:** env-var credential materialization plus the egress swap covers HTTPS hosts by construction.

**What isn't protected:** non-HTTP secret consumers are out of scope for this mechanism, including database passwords, SSH keys, raw-TCP protocols, and other clients that do not traverse the HTTPS egress swap.

**What to do instead:** keep those secrets on the worker side, or use a future named-consumer broker once one exists for non-HTTP consumers.

## Request-signing API boundary (#974)

**What's protected:** bearer-style HTTPS credentials whose final outbound request can be rewritten by the egress swap are supported.

**What isn't protected:** request-signing APIs such as AWS SigV4, HMAC-over-body schemes, and OAuth1 do not work with placeholder credentials. The request body and headers are signed before the egress swap sees them, so replacing placeholder material afterward invalidates the signature.

**What to do instead:** route signing workflows through the worker-side converged-endpoint path (#1007), where the real credential is available at signing time without exposing it to the model.

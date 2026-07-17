# Inference credential posture

Paid inference credentials are tenant secrets. Store each provider's API key and proxy base URL together in an account-scoped `model_providers` row. Platform infrastructure secrets (`AIOS_VAULT_KEY`, database credentials, bootstrap/control-plane tokens, and egress CA keys) remain deployment settings and must not be placed in provider rows.

`AIOS_INFERENCE_CREDENTIAL_POLICY` controls whether LiteLLM may use paid-provider process environment variables:

- `account_only` (default): a call without a resolvable account/ancestor provider row fails with `model_provider_not_configured` before LiteLLM is invoked.
- `observe_legacy_env`: temporary single-operator migration mode; missing rows may use process environment credentials.
- `legacy_env`: explicit legacy single-operator fallback.

`AIOS_TENANCY_POSTURE=external_byok` is valid only with `account_only`; startup rejects either legacy mode. Never admit external accounts while a legacy mode is configured.

The platform root must have no active paid-provider rows. Eumemic and external customers are sibling children of that root. Descendant inheritance is appropriate only inside an intentional shared billing/trust domain.

Before external admission, remove paid-provider variables such as `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` from every API/worker deployment, restart all processes, and verify unconfigured session and workflow calls emit `model_provider_not_configured`. Rollback must restore verified account-scoped rows or stop inference; it must never restore worker-global paid keys.

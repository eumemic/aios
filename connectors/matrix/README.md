# Matrix appservice receiver (Milestone 0)

This is the bare, private-network HS→AS receiver. It has no aios integration. Copy
`registration.yaml`, replace both token placeholders and the example server name in the regex,
then install it in Synapse together with the settings in `synapse.yaml`. The registration and
container environment must contain identical tokens and namespace details.

Required environment: `MATRIX_HS_URL`, `MATRIX_SERVER_NAME`, `MATRIX_AS_TOKEN`,
`MATRIX_HS_TOKEN`, `MATRIX_USER_NAMESPACE_REGEX`, and `MATRIX_DATABASE_URL`. Optional:
`MATRIX_SENDER_LOCALPART` (default `_aios`) and `MATRIX_LISTEN_ADDR` (default
`0.0.0.0:29328`). Keep the endpoint and both tokens on a trusted private network: the AS token
can impersonate every namespaced ghost and the HS token can forge inbound transactions.

The Postgres-backed mautrix state store is opened before the listener and survives restarts.
The image healthcheck calls the authenticated AS ping endpoint itself, so a live process with a
stopped or wedged listener becomes unhealthy.

## Rollout acceptance

`forget_rooms_on_leave: true` is deliberately accepted for the fleet homeserver. It is
homeserver-wide: every local human also loses pre-leave room history on rejoin. Deploy humans
to a separate homeserver if that behavior is unacceptable. `forgotten_room_retention_period:
28d` reclaims fully forgotten rooms in the background.

Before production rollout, perform and record these checks against the target Synapse:

1. Registering a matching ghost as a human fails with `M_EXCLUSIVE`.
2. POST `/_matrix/app/v1/ping` with the HS token and PUT an empty transaction; both return 200,
   while the same transaction with a wrong token returns 401.
3. Restart the receiver and verify its membership state remains in Postgres.
4. Stop only the AS listener and verify the container becomes unhealthy.
5. As a ghost, call client `POST /account/deactivate` using AS-token masquerade and no UIA;
   then attempt a masqueraded operation as that ghost. Record whether it is inert or errors
   cleanly. If either deactivation check fails, omit deactivation from retirement. Do **not** add
   a server-admin credential: leave plus forget remains the required cleanup.

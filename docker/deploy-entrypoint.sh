#!/bin/sh
# Repository-owned API candidate admission gate.
#
# The image's API CMD opts into this gate with ``--candidate``.  Keeping the
# opt-in out of the ordinary ``aios api`` command is important: an older image
# selected for application rollback must remain able to start after a newer
# image has advanced the database.  The worker never owns or races migrations.
set -eu

if [ "${1:-}" = "--candidate" ]; then
    shift
    if [ "${1:-}" != "aios" ] || [ "${2:-}" != "api" ]; then
        echo "--candidate is only valid for 'aios api'" >&2
        exit 64
    fi
    # A failed migration (including exhausted lock retries) exits this
    # candidate before the API can become healthy or be promoted.
    aios migrate
fi

exec "$@"

#!/bin/sh
# Repository-owned deployment gate.
#
# Coolify's post-deployment command is not an ordering boundary: the candidate
# can already have been selected by the time that command runs, and some
# versions report the deployment successful even when it exits non-zero.  Put
# the migration in the candidate process instead.  `set -e` makes a failed
# migration (including exhausted lock retries) terminate the candidate before
# either service can start or become healthy; the old healthy image remains the
# only promotable one.  Migrations are forward-only: this gate never rolls them
# back when an image is rolled back.
set -eu

if [ "${1:-}" = "aios" ]; then
    case "${2:-}" in
        api|worker)
            aios migrate
            ;;
    esac
fi

exec "$@"

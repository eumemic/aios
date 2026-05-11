#!/bin/sh
# Chown signal-cli's config_dir to uid 1000 so the api can read attachment
# bytes that signal-cli downloads here.  Both containers share this dir via
# a bind/named-volume mount; the api runs as USER aios (uid 1000) and would
# otherwise hit EACCES traversing a root-owned dir tree.  signal-cli itself
# runs as root inside this container and continues to write new files as
# root with mode 0644, which the api can read+unlink because the parent
# dir is uid-1000-owned.
#
# The ``|| true`` is so a read-only mount (someone running outside compose
# with the volume mode wrong) surfaces as a normal startup error from
# aios_signal rather than a chown failure here.

set -e

config_dir="${AIOS_SIGNAL_CONFIG_DIR:-/var/lib/aios/signal-data}"
mkdir -p "$config_dir"
chown -R 1000:1000 "$config_dir" 2>/dev/null || true

exec python -m aios_signal

#!/bin/sh
# Chown the WhatsApp data dir to uid 1000 so the api can read the
# inbound-media bytes the daemon decrypts here.  Both containers
# share this dir via a bind/named-volume mount; the api runs as
# USER aios (uid 1000) and would otherwise hit EACCES traversing a
# root-owned dir tree.  The whatsapp-daemon itself runs as root
# inside this container and continues to write new files as root
# with mode 0o600 (commit b3a3... in PR 5 review fix-ups), which
# the api can read+unlink because the parent dir is uid-1000-owned.
#
# The ``|| true`` lets a read-only mount (someone running outside
# compose with the volume mode wrong) surface as a normal startup
# error from aios_whatsapp rather than a chown failure here.

set -e

data_dir="${AIOS_WHATSAPP_DATA_DIR:-/var/lib/aios/whatsapp-data}"
mkdir -p "$data_dir"
chown -R 1000:1000 "$data_dir" 2>/dev/null || true

exec python -m aios_whatsapp

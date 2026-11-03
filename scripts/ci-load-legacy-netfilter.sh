#!/usr/bin/env bash
# Load the host kernel's legacy xtables modules before docker e2e runs.
#
# The egress lockdown selects ``iptables-legacy`` / ``ip6tables-legacy``
# (aios.sandbox.setup ``_IPTABLES_BACKEND_SELECT``; #1022 -- gVisor's netstack
# only speaks the legacy ABI). Under runc that ABI is the HOST kernel's
# ``ip_tables`` / ``iptable_filter`` / ``iptable_nat`` modules, and a container
# cannot load them: the sidecar's ``iptables-legacy`` just exits 3 with
# "can't initialize iptables table `filter': Table does not exist".
#
# ubuntu-24.04 runners had them loaded. ubuntu-26.04 (kernel 7.0, Docker 29
# programming the host through iptables-nft) builds them as modules
# (CONFIG_IP_NF_IPTABLES_LEGACY=m) that nothing loads, so every runc Limited
# lockdown fails closed. Load them explicitly; ``modprobe -a`` fails the step if
# a module is missing, rather than letting the suite fail far from the cause.
#
# A production runc host on such a kernel needs the same modules loaded at boot
# (aios#2567, which also tracks the nft-coexistence fix that retires this).
set -euo pipefail
sudo modprobe -a ip_tables iptable_filter iptable_nat ip6_tables ip6table_filter
lsmod | grep -E '^(ip_tables|iptable_filter|iptable_nat|ip6_tables|ip6table_filter) '

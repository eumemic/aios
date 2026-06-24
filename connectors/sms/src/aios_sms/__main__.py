"""``python -m aios_sms`` entry point.

Reads ``AIOS_URL`` / ``AIOS_RUNTIME_TOKEN`` from env (the SDK does this
inside ``HttpConnector.__init__``) and the ``AIOS_SMS_*`` listener
settings via :class:`aios_sms.config.Settings`. Per-connection Twilio
secrets (``from_number``, ``auth_token``) live on the connection record,
encrypted at rest, fetched per-connection inside ``serve_connection``.
"""

from __future__ import annotations

import asyncio

from .connector import SmsConnector


def main() -> None:
    asyncio.run(SmsConnector().run_until_stopped())


if __name__ == "__main__":
    main()

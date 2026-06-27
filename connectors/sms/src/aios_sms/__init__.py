"""SMS connector for aios (design docs/design/sms-connector.md).

Provider-neutral (``connector = "sms"``); Twilio-first transport arm.
Implements the inbound/transport layer (#1253, §3.1-3.3) and the public
HTTPS ingress + ingress-config slice (#1265, §5.3, §6): the forwarded-host
trust gate, the pinned public port, and the startup ingress self-test.
"""

from __future__ import annotations

from .connector import SmsConnector

__all__ = ["SmsConnector"]

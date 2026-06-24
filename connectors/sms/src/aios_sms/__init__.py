"""SMS connector for aios (#1253, design docs/design/sms-connector.md §3.1-3.3).

Provider-neutral (``connector = "sms"``); Twilio-first transport arm.
This slice implements the inbound/transport layer only.
"""

from __future__ import annotations

from .connector import SmsConnector

__all__ = ["SmsConnector"]

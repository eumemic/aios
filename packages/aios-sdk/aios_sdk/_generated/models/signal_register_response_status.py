from enum import Enum


class SignalRegisterResponseStatus(str, Enum):
    CAPTCHA_REQUIRED = "captcha_required"
    SMS_SENT = "sms_sent"
    VOICE_SENT = "voice_sent"

    def __str__(self) -> str:
        return str(self.value)

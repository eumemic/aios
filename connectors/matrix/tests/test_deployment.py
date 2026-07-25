from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).parents[1]


def test_registration_namespace_is_exact_and_no_e2ee_flags() -> None:
    registration = yaml.safe_load((ROOT / "registration.yaml").read_text())
    assert registration == {
        "id": "aios-matrix",
        "url": "http://matrix:29328",
        "as_token": "CHANGE_ME_AS_TOKEN",
        "hs_token": "CHANGE_ME_HS_TOKEN",
        "sender_localpart": "_aios",
        "rate_limited": True,
        "namespaces": {
            "users": [{"exclusive": True, "regex": r"^@_aios_agent_[a-z0-9]+:your\.server$"}],
            "aliases": [],
            "rooms": [],
        },
    }


def test_synapse_overlay_closes_federation_and_enables_retention() -> None:
    config = yaml.safe_load((ROOT / "synapse.yaml").read_text())
    assert config["app_service_config_files"] == ["/data/aios-matrix-registration.yaml"]
    assert config["federation_domain_whitelist"] == {}
    assert config["allow_profile_lookup_over_federation"] is False
    assert config["forget_rooms_on_leave"] is True
    assert config["forgotten_room_retention_period"] == "28d"

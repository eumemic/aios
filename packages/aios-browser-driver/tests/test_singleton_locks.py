"""Stale Chromium singleton residue must not survive to a launch.

The lock names the dead container's hostname, so Chromium reads it as a live
peer "on another computer" and refuses the profile — wedging every future
browser container for the account (prod, 2026-09-01)."""

from __future__ import annotations

from pathlib import Path

from aios_browser_driver.host import clear_stale_singleton_locks


def _wedged_profile(tmp_path: Path) -> Path:
    profile = tmp_path / "profile"
    profile.mkdir()
    # As Chromium leaves them: symlinks, dangling once the container is gone.
    (profile / "SingletonLock").symlink_to("dbd9172f950a-27")
    (profile / "SingletonCookie").symlink_to("16322985541810594182")
    (profile / "SingletonSocket").symlink_to("/tmp/org.chromium.Chromium.gone/SingletonSocket")
    (profile / "Preferences").write_text("{}")
    return profile


def test_removes_all_three_residue_links(tmp_path: Path) -> None:
    profile = _wedged_profile(tmp_path)
    clear_stale_singleton_locks(profile)
    for name in ("SingletonLock", "SingletonCookie", "SingletonSocket"):
        # lexists: a dangling symlink "exists" as a link even though its
        # target does not — exactly the state that wedges Chromium.
        assert not (profile / name).is_symlink()
    assert (profile / "Preferences").exists()


def test_clean_profile_and_first_boot_are_no_ops(tmp_path: Path) -> None:
    profile = _wedged_profile(tmp_path)
    clear_stale_singleton_locks(profile)
    clear_stale_singleton_locks(profile)  # second pass: nothing to remove
    # First boot: Chromium has not created the profile dir yet.
    clear_stale_singleton_locks(tmp_path / "never-created")

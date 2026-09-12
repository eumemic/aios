# Done

- Commit: `119ed983383395c9c96d6e2e166aae6ea1cf2558`
- Rebased onto `origin/master` before the fix.
- Root cause: BuildKit treats `/etc/resolv.conf` as a daemon-managed build
  mount, so a plain `COPY` can be absent from the committed image layer even
  though the Dockerfile and source file look correct.
- Bake-path fix: changed the final resolver instruction to
  `COPY --link docker/sandbox-resolv.conf /etc/resolv.conf`, which forces the
  resolver bytes into a linked image layer. Updated the unit pin and comments.
- Verification: `uv run pytest -q tests/unit/sandbox/test_sandbox_resolv_conf.py`
  (2 passed). Docker is unavailable in this environment, so the live e2e image
  contract could not be run here.
- Not pushed.

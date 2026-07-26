# Docker E2E retry recovery

The Docker E2E shard runs serially. If the first attempt fails because the shared-runner daemon is contaminated, the workflow restarts Docker, waits for the daemon to become ready, prunes stale resources, and reruns the full shard from a clean session.

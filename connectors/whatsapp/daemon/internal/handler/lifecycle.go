package handler

import (
	"context"
	"encoding/json"

	"aios.dev/connectors/whatsapp/daemon/internal/rpc"
)

// VersionResult is the `version` method's response. Stable across
// daemon versions so the Python side can probe readiness without
// caring what version it's talking to.
type VersionResult struct {
	Name    string `json:"name"`
	Version string `json:"version"`
}

// RegisterLifecycle wires the `version` method into reg.
//
// `version` is the universal readiness probe — side-effect-free,
// always answers immediately once the TCP listener is up. The Python
// daemon-spawn helper polls it in a 200 ms loop until the daemon
// answers (or 30 s elapses), so its existence is load-bearing for
// startup correctness.
func RegisterLifecycle(reg *Registry, name, version string) {
	reg.Register("version", func(_ context.Context, _ json.RawMessage) (any, *rpc.Error) {
		return VersionResult{Name: name, Version: version}, nil
	})
}

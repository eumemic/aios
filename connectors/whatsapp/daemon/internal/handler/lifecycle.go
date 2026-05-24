package handler

import (
	"context"
	"encoding/json"

	"aios.dev/connectors/whatsapp/daemon/internal/rpc"
)

// VersionResult is the `version` method's response.
type VersionResult struct {
	Name    string `json:"name"`
	Version string `json:"version"`
}

// RegisterLifecycle wires the `version` method into reg.
//
// `version` returns the daemon name + build version.  Side-effect-free;
// safe to call before any whatsmeow connection completes.
func RegisterLifecycle(reg *Registry, name, version string) {
	reg.Register("version", func(_ context.Context, _ json.RawMessage) (any, *rpc.Error) {
		return VersionResult{Name: name, Version: version}, nil
	})
}

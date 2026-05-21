// Package handler is the daemon's RPC method registry. Each method is
// registered by name; Dispatch routes incoming requests to the matching
// MethodFunc.
package handler

import (
	"context"
	"encoding/json"

	"aios.dev/connectors/whatsapp/daemon/internal/rpc"
)

// MethodFunc is the signature of an RPC method handler. Returning a
// nil *rpc.Error means success; returning a non-nil Error means the
// daemon rejected the call. params is the raw JSON; handlers
// Unmarshal it into a method-specific struct.
type MethodFunc func(ctx context.Context, params json.RawMessage) (any, *rpc.Error)

// Registry maps method name → handler. Not goroutine-safe to register
// into; Register is only called during process startup before
// rpc.Server.Run accepts connections.
type Registry struct {
	methods map[string]MethodFunc
}

// NewRegistry returns an empty Registry.
func NewRegistry() *Registry {
	return &Registry{methods: make(map[string]MethodFunc)}
}

// Register binds name to fn. Panics on duplicate registration — that's
// a programmer error caught at process startup before the listener binds.
func (r *Registry) Register(name string, fn MethodFunc) {
	if _, exists := r.methods[name]; exists {
		panic("handler: duplicate registration for " + name)
	}
	r.methods[name] = fn
}

// Dispatch implements rpc.Handler.
func (r *Registry) Dispatch(ctx context.Context, method string, params json.RawMessage) (any, *rpc.Error) {
	fn, ok := r.methods[method]
	if !ok {
		return nil, &rpc.Error{Code: -32601, Message: "method not found: " + method}
	}
	return fn(ctx, params)
}

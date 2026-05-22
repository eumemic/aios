package handler

import (
	"context"
	"encoding/json"

	"aios.dev/connectors/whatsapp/daemon/internal/rpc"
)

// Groups is the daemon-internal interface for group operations.
// Mirrors the Pairer / MessageOps pattern: keeps handler decoupled
// from whatsmeow types so tests can stub it and main.go bridges
// concrete shapes via per-call memcopies.
type Groups interface {
	ListGroups(ctx context.Context) ([]GroupSummary, error)
	CreateGroup(ctx context.Context, name string, participantJIDs []string) (*GroupSummary, error)
	RenameGroup(ctx context.Context, groupJID, name string) error
}

// GroupSummary is the wire-shape mirror of wameow.GroupSummary.
type GroupSummary struct {
	JID          string                 `json:"jid"`
	Name         string                 `json:"name"`
	Topic        string                 `json:"topic,omitempty"`
	Participants []GroupParticipantInfo `json:"participants"`
}

type GroupParticipantInfo struct {
	JID     string `json:"jid"`
	IsAdmin bool   `json:"is_admin"`
}

type createGroupArgs struct {
	Name         string   `json:"name"`
	Participants []string `json:"participants"`
}

type renameGroupArgs struct {
	JID  string `json:"jid"`
	Name string `json:"name"`
}

// RegisterGroups wires the listGroups / createGroup / renameGroup
// RPC methods into reg.
func RegisterGroups(reg *Registry, g Groups) {
	reg.Register("listGroups", func(ctx context.Context, _ json.RawMessage) (any, *rpc.Error) {
		groups, err := g.ListGroups(ctx)
		if err != nil {
			return nil, &rpc.Error{Code: rpc.ErrCodeServerError, Message: err.Error()}
		}
		// JSON-RPC requires a non-nil result for success; wrap so an
		// empty group list serializes as ``{"groups": []}`` rather
		// than top-level ``null``.
		return map[string]any{"groups": groups}, nil
	})
	reg.Register("createGroup", func(ctx context.Context, params json.RawMessage) (any, *rpc.Error) {
		var args createGroupArgs
		if err := json.Unmarshal(params, &args); err != nil {
			return nil, &rpc.Error{Code: rpc.ErrCodeInvalidParams, Message: "invalid createGroup params: " + err.Error()}
		}
		if args.Name == "" {
			return nil, &rpc.Error{Code: rpc.ErrCodeInvalidParams, Message: "createGroup: name required"}
		}
		summary, err := g.CreateGroup(ctx, args.Name, args.Participants)
		if err != nil {
			return nil, &rpc.Error{Code: rpc.ErrCodeServerError, Message: err.Error()}
		}
		return summary, nil
	})
	reg.Register("renameGroup", func(ctx context.Context, params json.RawMessage) (any, *rpc.Error) {
		var args renameGroupArgs
		if err := json.Unmarshal(params, &args); err != nil {
			return nil, &rpc.Error{Code: rpc.ErrCodeInvalidParams, Message: "invalid renameGroup params: " + err.Error()}
		}
		if args.JID == "" {
			return nil, &rpc.Error{Code: rpc.ErrCodeInvalidParams, Message: "renameGroup: jid required"}
		}
		if args.Name == "" {
			return nil, &rpc.Error{Code: rpc.ErrCodeInvalidParams, Message: "renameGroup: name required"}
		}
		if err := g.RenameGroup(ctx, args.JID, args.Name); err != nil {
			return nil, &rpc.Error{Code: rpc.ErrCodeServerError, Message: err.Error()}
		}
		return map[string]string{"status": "ok"}, nil
	})
}

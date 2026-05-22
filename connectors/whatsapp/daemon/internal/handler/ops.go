package handler

import (
	"context"
	"encoding/json"

	"aios.dev/connectors/whatsapp/daemon/internal/rpc"
)

// MessageOps is the daemon-internal interface for per-message
// operations (react/edit/revoke).  Each method takes a message_id the
// daemon has previously seen — the daemon-side MessageStore holds the
// MessageKey context (chat/sender/from_me) that WhatsApp's protocol
// requires.
type MessageOps interface {
	React(ctx context.Context, msgID, emoji string) (string, int64, error)
	Edit(ctx context.Context, msgID, newText string) (string, int64, error)
	Revoke(ctx context.Context, msgID string) (string, int64, error)
	IsNotFoundErr(err error) bool
}

type reactArgs struct {
	MessageID string `json:"message_id"`
	Reaction  string `json:"reaction"`
}

type editArgs struct {
	MessageID string `json:"message_id"`
	Text      string `json:"text"`
}

type revokeArgs struct {
	MessageID string `json:"message_id"`
}

type opResult struct {
	MessageID   string `json:"message_id"`
	TimestampMS int64  `json:"timestamp_ms"`
}

// RegisterMessageOps wires sendReaction / editMessage / deleteMessage
// into reg.  Lookup misses surface as ErrCodeInvalidParams so the
// caller (and through it, the model) distinguishes "you targeted a
// message I don't know about" from generic server failures.
func RegisterMessageOps(reg *Registry, ops MessageOps) {
	reg.Register("sendReaction", func(ctx context.Context, params json.RawMessage) (any, *rpc.Error) {
		var args reactArgs
		if err := json.Unmarshal(params, &args); err != nil {
			return nil, &rpc.Error{Code: rpc.ErrCodeInvalidParams, Message: "invalid sendReaction params: " + err.Error()}
		}
		if args.MessageID == "" {
			return nil, &rpc.Error{Code: rpc.ErrCodeInvalidParams, Message: "sendReaction: message_id required"}
		}
		msgID, ts, err := ops.React(ctx, args.MessageID, args.Reaction)
		if err != nil {
			return nil, mapOpError(err, ops)
		}
		return opResult{MessageID: msgID, TimestampMS: ts}, nil
	})
	reg.Register("editMessage", func(ctx context.Context, params json.RawMessage) (any, *rpc.Error) {
		var args editArgs
		if err := json.Unmarshal(params, &args); err != nil {
			return nil, &rpc.Error{Code: rpc.ErrCodeInvalidParams, Message: "invalid editMessage params: " + err.Error()}
		}
		if args.MessageID == "" {
			return nil, &rpc.Error{Code: rpc.ErrCodeInvalidParams, Message: "editMessage: message_id required"}
		}
		msgID, ts, err := ops.Edit(ctx, args.MessageID, args.Text)
		if err != nil {
			return nil, mapOpError(err, ops)
		}
		return opResult{MessageID: msgID, TimestampMS: ts}, nil
	})
	reg.Register("deleteMessage", func(ctx context.Context, params json.RawMessage) (any, *rpc.Error) {
		var args revokeArgs
		if err := json.Unmarshal(params, &args); err != nil {
			return nil, &rpc.Error{Code: rpc.ErrCodeInvalidParams, Message: "invalid deleteMessage params: " + err.Error()}
		}
		if args.MessageID == "" {
			return nil, &rpc.Error{Code: rpc.ErrCodeInvalidParams, Message: "deleteMessage: message_id required"}
		}
		msgID, ts, err := ops.Revoke(ctx, args.MessageID)
		if err != nil {
			return nil, mapOpError(err, ops)
		}
		return opResult{MessageID: msgID, TimestampMS: ts}, nil
	})
}

func mapOpError(err error, ops MessageOps) *rpc.Error {
	if ops.IsNotFoundErr(err) {
		return &rpc.Error{Code: rpc.ErrCodeInvalidParams, Message: "unknown message_id"}
	}
	return &rpc.Error{Code: rpc.ErrCodeServerError, Message: err.Error()}
}

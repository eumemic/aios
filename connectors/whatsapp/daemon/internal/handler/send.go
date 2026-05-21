package handler

import (
	"context"
	"encoding/json"

	"aios.dev/connectors/whatsapp/daemon/internal/rpc"
)

// SendMessageFn is the daemon-internal closure that delivers a text
// message to a WhatsApp JID.  Taking a closure (rather than a
// *whatsmeow.Client) keeps the handler decoupled from whatsmeow types
// so tests can stub it and so main.go can swap the wameow integration
// in independently.
type SendMessageFn func(ctx context.Context, jid, text string) (messageID string, timestampMs int64, err error)

type sendArgs struct {
	JID  string `json:"jid"`
	Text string `json:"text"`
}

type sendResult struct {
	MessageID   string `json:"message_id"`
	TimestampMS int64  `json:"timestamp_ms"`
}

// RegisterSend wires the ``sendMessage`` method into reg.
func RegisterSend(reg *Registry, fn SendMessageFn) {
	reg.Register("sendMessage", func(ctx context.Context, params json.RawMessage) (any, *rpc.Error) {
		var args sendArgs
		if err := json.Unmarshal(params, &args); err != nil {
			return nil, &rpc.Error{Code: -32602, Message: "invalid sendMessage params: " + err.Error()}
		}
		if args.JID == "" {
			return nil, &rpc.Error{Code: -32602, Message: "sendMessage: jid required"}
		}
		msgID, ts, err := fn(ctx, args.JID, args.Text)
		if err != nil {
			return nil, &rpc.Error{Code: -32000, Message: err.Error()}
		}
		return sendResult{MessageID: msgID, TimestampMS: ts}, nil
	})
}

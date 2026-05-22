package handler

import (
	"context"
	"encoding/json"

	"aios.dev/connectors/whatsapp/daemon/internal/rpc"
)

// SendMessageFn is the daemon-internal closure that delivers a
// message (text and/or attachments) to a WhatsApp JID.  Taking a
// closure (rather than a *whatsmeow.Client) keeps the handler
// decoupled from whatsmeow types so tests can stub it and main.go
// can swap the whatsmeow integration in independently.
//
// For multi-attachment sends, the closure returns the FIRST send's
// id+timestamp (the one carrying the caption when text is non-empty);
// the daemon's MessageStore captures every send so subsequent
// attachments are still react/edit/delete-targetable by id.
type SendMessageFn func(ctx context.Context, jid, text string, attachments []Attachment) (messageID string, timestampMs int64, err error)

// Attachment is the wire-shaped media payload received from Python.
// Path is a host-side filesystem path the daemon can os.ReadFile on;
// the Python connector resolves SandboxPath → host path before the
// RPC.  Filename appears on the WhatsApp wire only for documents but
// the daemon carries it for all kinds (passed to whatsmeow's
// DocumentMessage.FileName).
type Attachment struct {
	Path     string `json:"path"`
	Mimetype string `json:"mimetype"`
	Filename string `json:"filename"`
}

type sendArgs struct {
	JID         string       `json:"jid"`
	Text        string       `json:"text"`
	Attachments []Attachment `json:"attachments"`
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
			return nil, &rpc.Error{Code: rpc.ErrCodeInvalidParams, Message: "invalid sendMessage params: " + err.Error()}
		}
		if args.JID == "" {
			return nil, &rpc.Error{Code: rpc.ErrCodeInvalidParams, Message: "sendMessage: jid required"}
		}
		if args.Text == "" && len(args.Attachments) == 0 {
			return nil, &rpc.Error{Code: rpc.ErrCodeInvalidParams, Message: "sendMessage: text or attachments required"}
		}
		msgID, ts, err := fn(ctx, args.JID, args.Text, args.Attachments)
		if err != nil {
			return nil, &rpc.Error{Code: rpc.ErrCodeServerError, Message: err.Error()}
		}
		return sendResult{MessageID: msgID, TimestampMS: ts}, nil
	})
}

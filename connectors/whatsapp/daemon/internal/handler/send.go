package handler

import (
	"context"
	"encoding/json"
	"errors"

	"aios.dev/connectors/whatsapp/daemon/internal/rpc"
)

// SendMessageFn is the daemon-internal closure that delivers a
// message (text and/or attachments) to a WhatsApp JID.  Taking a
// closure (rather than a *whatsmeow.Client) keeps the handler
// decoupled from whatsmeow types so tests can stub it and main.go
// can swap the whatsmeow integration in independently.
//
// Returns the FULL slice of delivered message_ids (in send order);
// the first is the caption-bearing send when applicable.  On a
// mid-loop partial failure, the closure returns the already-delivered
// ids alongside the error — the handler surfaces them via rpc.Error.Data
// so the model can still address each delivered attachment by id.
//
// ``mentionedJIDs`` are attached to the outbound message's ContextInfo
// so WhatsApp clients render @-mentions as pills.  Pass nil for none.
type SendMessageFn func(ctx context.Context, jid, text string, attachments []Attachment, mentionedJIDs []string) (deliveredIDs []string, timestampMs int64, err error)

// PartialSendErrorUnwrapper exposes the partial-delivery ids the
// closure carries on a multi-attachment failure.  handler/send.go
// uses errors.As against a private interface to keep this package's
// public surface decoupled from wameow's concrete type.
type partialSendError interface {
	error
	Partial() (delivered []string, failedIndex int, failedFilename string)
}

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
	JID           string       `json:"jid"`
	Text          string       `json:"text"`
	Attachments   []Attachment `json:"attachments"`
	MentionedJIDs []string     `json:"mentioned_jids,omitempty"`
}

type sendResult struct {
	MessageID           string   `json:"message_id"`
	TimestampMS         int64    `json:"timestamp_ms"`
	DeliveredMessageIDs []string `json:"delivered_message_ids,omitempty"`
}

// partialSendErrorData is the structured Data block on rpc.Error
// when a multi-attachment send delivers some but not all
// attachments.  The model can read these fields off the tool result
// to address the delivered attachments by id.
type partialSendErrorData struct {
	DeliveredMessageIDs []string `json:"delivered_message_ids"`
	FailedIndex         int      `json:"failed_index"`
	FailedFilename      string   `json:"failed_filename"`
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
		ids, ts, err := fn(ctx, args.JID, args.Text, args.Attachments, args.MentionedJIDs)
		if err != nil {
			rpcErr := &rpc.Error{Code: rpc.ErrCodeServerError, Message: err.Error()}
			// Partial-delivery error: surface the ids that landed
			// so the model can still target them by id.
			var partial partialSendError
			if errors.As(err, &partial) {
				delivered, idx, name := partial.Partial()
				rpcErr.Data = partialSendErrorData{
					DeliveredMessageIDs: delivered,
					FailedIndex:         idx,
					FailedFilename:      name,
				}
			}
			return nil, rpcErr
		}
		var firstID string
		if len(ids) > 0 {
			firstID = ids[0]
		}
		return sendResult{MessageID: firstID, TimestampMS: ts, DeliveredMessageIDs: ids}, nil
	})
}

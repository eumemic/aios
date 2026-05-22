package handler

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

	"aios.dev/connectors/whatsapp/daemon/internal/rpc"
)

func TestSendMessageDispatchesAndReturnsResult(t *testing.T) {
	var seenJID, seenText string
	var seenAttachments []Attachment
	fn := func(_ context.Context, jid, text string, atts []Attachment) (string, int64, error) {
		seenJID = jid
		seenText = text
		seenAttachments = atts
		return "MSG-1", 1700000000000, nil
	}
	reg := NewRegistry()
	RegisterSend(reg, fn)

	params := json.RawMessage(`{"jid":"15553334444@s.whatsapp.net","text":"hello"}`)
	result, rpcErr := reg.Dispatch(context.Background(), "sendMessage", params)
	if rpcErr != nil {
		t.Fatalf("unexpected rpc error: %+v", rpcErr)
	}
	if seenJID != "15553334444@s.whatsapp.net" || seenText != "hello" {
		t.Fatalf("send fn args = (%q, %q)", seenJID, seenText)
	}
	if len(seenAttachments) != 0 {
		t.Fatalf("expected no attachments, got %v", seenAttachments)
	}
	out, _ := json.Marshal(result)
	if string(out) != `{"message_id":"MSG-1","timestamp_ms":1700000000000}` {
		t.Fatalf("result json = %s", out)
	}
}

func TestSendMessageRejectsMissingJID(t *testing.T) {
	reg := NewRegistry()
	RegisterSend(reg, func(context.Context, string, string, []Attachment) (string, int64, error) {
		t.Fatalf("send fn should not be called when jid missing")
		return "", 0, nil
	})
	_, rpcErr := reg.Dispatch(context.Background(), "sendMessage", json.RawMessage(`{"text":"hi"}`))
	if rpcErr == nil {
		t.Fatalf("expected rpc error for missing jid")
	}
	if rpcErr.Code != rpc.ErrCodeInvalidParams {
		t.Fatalf("expected ErrCodeInvalidParams, got %d", rpcErr.Code)
	}
}

func TestSendMessageRejectsEmptyBody(t *testing.T) {
	// Bare {jid:...} with neither text nor attachments is a programmer
	// error — the daemon refuses rather than silently sending an empty
	// Conversation message that the peer would see as a blank bubble.
	reg := NewRegistry()
	RegisterSend(reg, func(context.Context, string, string, []Attachment) (string, int64, error) {
		t.Fatalf("send fn should not be called for empty body")
		return "", 0, nil
	})
	_, rpcErr := reg.Dispatch(context.Background(), "sendMessage", json.RawMessage(`{"jid":"x@s.whatsapp.net"}`))
	if rpcErr == nil || rpcErr.Code != rpc.ErrCodeInvalidParams {
		t.Fatalf("expected ErrCodeInvalidParams for empty body, got %+v", rpcErr)
	}
}

func TestSendMessagePropagatesSendError(t *testing.T) {
	reg := NewRegistry()
	RegisterSend(reg, func(context.Context, string, string, []Attachment) (string, int64, error) {
		return "", 0, errors.New("network down")
	})
	params := json.RawMessage(`{"jid":"15553334444@s.whatsapp.net","text":"hi"}`)
	_, rpcErr := reg.Dispatch(context.Background(), "sendMessage", params)
	if rpcErr == nil {
		t.Fatalf("expected rpc error from send failure")
	}
	if rpcErr.Code != rpc.ErrCodeServerError {
		t.Fatalf("expected ErrCodeServerError, got %d", rpcErr.Code)
	}
	if rpcErr.Message != "network down" {
		t.Fatalf("rpc error message = %q", rpcErr.Message)
	}
}

func TestSendMessageRejectsMalformedParams(t *testing.T) {
	reg := NewRegistry()
	RegisterSend(reg, func(context.Context, string, string, []Attachment) (string, int64, error) {
		return "", 0, nil
	})
	_, rpcErr := reg.Dispatch(context.Background(), "sendMessage", json.RawMessage(`not json`))
	if rpcErr == nil || rpcErr.Code != rpc.ErrCodeInvalidParams {
		t.Fatalf("expected ErrCodeInvalidParams for malformed json, got %+v", rpcErr)
	}
}

func TestSendMessageForwardsAttachments(t *testing.T) {
	var seenAttachments []Attachment
	reg := NewRegistry()
	RegisterSend(reg, func(_ context.Context, _ string, _ string, atts []Attachment) (string, int64, error) {
		seenAttachments = atts
		return "MSG-2", 1700000001000, nil
	})
	params := json.RawMessage(`{
		"jid": "15553334444@s.whatsapp.net",
		"text": "caption",
		"attachments": [
			{"path": "/tmp/a.jpg", "mimetype": "image/jpeg", "filename": "a.jpg"},
			{"path": "/tmp/b.pdf", "mimetype": "application/pdf", "filename": "b.pdf"}
		]
	}`)
	_, rpcErr := reg.Dispatch(context.Background(), "sendMessage", params)
	if rpcErr != nil {
		t.Fatalf("unexpected rpc error: %+v", rpcErr)
	}
	if len(seenAttachments) != 2 {
		t.Fatalf("expected 2 attachments, got %d", len(seenAttachments))
	}
	if seenAttachments[0].Path != "/tmp/a.jpg" || seenAttachments[0].Mimetype != "image/jpeg" {
		t.Errorf("attachment[0] = %+v", seenAttachments[0])
	}
	if seenAttachments[1].Path != "/tmp/b.pdf" || seenAttachments[1].Filename != "b.pdf" {
		t.Errorf("attachment[1] = %+v", seenAttachments[1])
	}
}

func TestSendMessageAcceptsAttachmentsWithoutText(t *testing.T) {
	called := false
	reg := NewRegistry()
	RegisterSend(reg, func(context.Context, string, string, []Attachment) (string, int64, error) {
		called = true
		return "MSG-3", 1700000002000, nil
	})
	params := json.RawMessage(`{
		"jid": "x@s.whatsapp.net",
		"attachments": [{"path":"/tmp/a.jpg","mimetype":"image/jpeg","filename":"a.jpg"}]
	}`)
	_, rpcErr := reg.Dispatch(context.Background(), "sendMessage", params)
	if rpcErr != nil {
		t.Fatalf("unexpected rpc error for attachment-only send: %+v", rpcErr)
	}
	if !called {
		t.Fatal("send fn should have been called")
	}
}

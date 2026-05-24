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
	fn := func(_ context.Context, jid, text string, atts []Attachment, _ []string, _ string) ([]string, int64, error) {
		seenJID = jid
		seenText = text
		seenAttachments = atts
		return []string{"MSG-1"}, 1700000000000, nil
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
	want := `{"message_id":"MSG-1","timestamp_ms":1700000000000,"delivered_message_ids":["MSG-1"]}`
	if string(out) != want {
		t.Fatalf("result json = %s\nwant %s", out, want)
	}
}

func TestSendMessageSurfacesAllDeliveredIDs(t *testing.T) {
	// Multi-attachment success returns ALL delivered ids in the
	// result so the model can address each by id for follow-up
	// react/edit/delete.  Previously only the first id was exposed.
	reg := NewRegistry()
	RegisterSend(reg, func(context.Context, string, string, []Attachment, []string, string) ([]string, int64, error) {
		return []string{"M0", "M1", "M2"}, 1700000000000, nil
	})
	params := json.RawMessage(`{
		"jid": "x@s.whatsapp.net", "text": "cap",
		"attachments": [
			{"path":"/tmp/a.jpg","mimetype":"image/jpeg","filename":"a.jpg"},
			{"path":"/tmp/b.jpg","mimetype":"image/jpeg","filename":"b.jpg"},
			{"path":"/tmp/c.jpg","mimetype":"image/jpeg","filename":"c.jpg"}
		]
	}`)
	result, rpcErr := reg.Dispatch(context.Background(), "sendMessage", params)
	if rpcErr != nil {
		t.Fatalf("unexpected rpc error: %+v", rpcErr)
	}
	res, ok := result.(sendResult)
	if !ok {
		t.Fatalf("unexpected result type %T", result)
	}
	if res.MessageID != "M0" {
		t.Errorf("MessageID = %q, want 'M0'", res.MessageID)
	}
	if len(res.DeliveredMessageIDs) != 3 || res.DeliveredMessageIDs[2] != "M2" {
		t.Errorf("DeliveredMessageIDs = %v, want [M0 M1 M2]", res.DeliveredMessageIDs)
	}
}

type stubPartialErr struct {
	cause     error
	delivered []string
	idx       int
	filename  string
}

func (s *stubPartialErr) Error() string { return s.cause.Error() }
func (s *stubPartialErr) Unwrap() error { return s.cause }
func (s *stubPartialErr) Partial() ([]string, int, string) {
	return s.delivered, s.idx, s.filename
}

func TestSendMessagePartialFailureSurfacesDeliveredIDs(t *testing.T) {
	// Mid-loop multi-attachment failure must surface the ids that
	// landed so the model can target them by id for follow-up
	// react/edit/revoke.  Previously these ids were swallowed by
	// the error path, stranding them in the daemon's msgstore.
	partial := &stubPartialErr{
		cause:     errors.New("upload failed"),
		delivered: []string{"M0", "M1"},
		idx:       2,
		filename:  "broken.mp4",
	}
	reg := NewRegistry()
	RegisterSend(reg, func(context.Context, string, string, []Attachment, []string, string) ([]string, int64, error) {
		return []string{"M0", "M1"}, 1700000000000, partial
	})
	params := json.RawMessage(`{"jid":"x@s.whatsapp.net","text":"cap","attachments":[
		{"path":"/tmp/a.jpg","mimetype":"image/jpeg","filename":"a.jpg"},
		{"path":"/tmp/b.jpg","mimetype":"image/jpeg","filename":"b.jpg"},
		{"path":"/tmp/c.mp4","mimetype":"video/mp4","filename":"broken.mp4"}
	]}`)
	_, rpcErr := reg.Dispatch(context.Background(), "sendMessage", params)
	if rpcErr == nil {
		t.Fatalf("expected rpc error")
	}
	data, ok := rpcErr.Data.(partialSendErrorData)
	if !ok {
		t.Fatalf("rpc.Error.Data = %T, want partialSendErrorData", rpcErr.Data)
	}
	if len(data.DeliveredMessageIDs) != 2 || data.DeliveredMessageIDs[1] != "M1" {
		t.Errorf("DeliveredMessageIDs = %v, want [M0 M1]", data.DeliveredMessageIDs)
	}
	if data.FailedIndex != 2 || data.FailedFilename != "broken.mp4" {
		t.Errorf("FailedIndex/Filename = %d %q", data.FailedIndex, data.FailedFilename)
	}
}

func TestSendMessageRejectsMissingJID(t *testing.T) {
	reg := NewRegistry()
	RegisterSend(reg, func(context.Context, string, string, []Attachment, []string, string) ([]string, int64, error) {
		t.Fatalf("send fn should not be called when jid missing")
		return nil, 0, nil
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
	RegisterSend(reg, func(context.Context, string, string, []Attachment, []string, string) ([]string, int64, error) {
		t.Fatalf("send fn should not be called for empty body")
		return nil, 0, nil
	})
	_, rpcErr := reg.Dispatch(context.Background(), "sendMessage", json.RawMessage(`{"jid":"x@s.whatsapp.net"}`))
	if rpcErr == nil || rpcErr.Code != rpc.ErrCodeInvalidParams {
		t.Fatalf("expected ErrCodeInvalidParams for empty body, got %+v", rpcErr)
	}
}

func TestSendMessagePropagatesSendError(t *testing.T) {
	reg := NewRegistry()
	RegisterSend(reg, func(context.Context, string, string, []Attachment, []string, string) ([]string, int64, error) {
		return nil, 0, errors.New("network down")
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
	RegisterSend(reg, func(context.Context, string, string, []Attachment, []string, string) ([]string, int64, error) {
		return []string{"MSG"}, 0, nil
	})
	_, rpcErr := reg.Dispatch(context.Background(), "sendMessage", json.RawMessage(`not json`))
	if rpcErr == nil || rpcErr.Code != rpc.ErrCodeInvalidParams {
		t.Fatalf("expected ErrCodeInvalidParams for malformed json, got %+v", rpcErr)
	}
}

func TestSendMessageForwardsAttachments(t *testing.T) {
	var seenAttachments []Attachment
	reg := NewRegistry()
	RegisterSend(reg, func(_ context.Context, _ string, _ string, atts []Attachment, _ []string, _ string) ([]string, int64, error) {
		seenAttachments = atts
		return []string{"MSG-2"}, 1700000001000, nil
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

func TestSendMessageForwardsMentionedJIDs(t *testing.T) {
	var seenMentions []string
	reg := NewRegistry()
	RegisterSend(reg, func(_ context.Context, _, _ string, _ []Attachment, m []string, _ string) ([]string, int64, error) {
		seenMentions = m
		return []string{"MSG"}, 1700000000000, nil
	})
	params := json.RawMessage(`{
		"jid": "g@g.us", "text": "hey @+15551234567",
		"mentioned_jids": ["15551234567@s.whatsapp.net"]
	}`)
	_, rpcErr := reg.Dispatch(context.Background(), "sendMessage", params)
	if rpcErr != nil {
		t.Fatalf("unexpected rpc error: %+v", rpcErr)
	}
	if len(seenMentions) != 1 || seenMentions[0] != "15551234567@s.whatsapp.net" {
		t.Errorf("seenMentions = %v, want [15551234567@s.whatsapp.net]", seenMentions)
	}
}

func TestSendMessageAcceptsAttachmentsWithoutText(t *testing.T) {
	called := false
	reg := NewRegistry()
	RegisterSend(reg, func(context.Context, string, string, []Attachment, []string, string) ([]string, int64, error) {
		called = true
		return []string{"MSG-3"}, 1700000002000, nil
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

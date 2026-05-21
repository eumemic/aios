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
	fn := func(_ context.Context, jid, text string) (string, int64, error) {
		seenJID = jid
		seenText = text
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
	out, _ := json.Marshal(result)
	if string(out) != `{"message_id":"MSG-1","timestamp_ms":1700000000000}` {
		t.Fatalf("result json = %s", out)
	}
}

func TestSendMessageRejectsMissingJID(t *testing.T) {
	reg := NewRegistry()
	RegisterSend(reg, func(context.Context, string, string) (string, int64, error) {
		t.Fatalf("send fn should not be called when jid missing")
		return "", 0, nil
	})
	_, rpcErr := reg.Dispatch(context.Background(), "sendMessage", json.RawMessage(`{"text":"hi"}`))
	if rpcErr == nil {
		t.Fatalf("expected rpc error for missing jid")
	}
	if rpcErr.Code != -32602 {
		t.Fatalf("expected -32602, got %d", rpcErr.Code)
	}
}

func TestSendMessagePropagatesSendError(t *testing.T) {
	reg := NewRegistry()
	RegisterSend(reg, func(context.Context, string, string) (string, int64, error) {
		return "", 0, errors.New("network down")
	})
	params := json.RawMessage(`{"jid":"15553334444@s.whatsapp.net","text":"hi"}`)
	_, rpcErr := reg.Dispatch(context.Background(), "sendMessage", params)
	if rpcErr == nil {
		t.Fatalf("expected rpc error from send failure")
	}
	if rpcErr.Message != "network down" {
		t.Fatalf("rpc error message = %q", rpcErr.Message)
	}
}

func TestSendMessageRejectsMalformedParams(t *testing.T) {
	reg := NewRegistry()
	RegisterSend(reg, func(context.Context, string, string) (string, int64, error) {
		return "", 0, nil
	})
	_, rpcErr := reg.Dispatch(context.Background(), "sendMessage", json.RawMessage(`not json`))
	if rpcErr == nil || rpcErr.Code != -32602 {
		t.Fatalf("expected -32602 for malformed json, got %+v", rpcErr)
	}
}

// Suppress unused symbol when toolchains pre-1.20 don't recognize rpc.Error.
var _ = rpc.Error{}

package handler

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

	"aios.dev/connectors/whatsapp/daemon/internal/rpc"
)

// stubOps is a MessageOps implementation that lets each test method
// inject a specific return triple.  The first matching call records
// its args so the test can assert what the registry forwarded.
type stubOps struct {
	react  func(ctx context.Context, msgID, emoji string) (string, int64, error)
	edit   func(ctx context.Context, msgID, text string, mentionedJIDs []string) (string, int64, error)
	revoke func(ctx context.Context, msgID string) (string, int64, error)
	// notFoundErr / notOwnErr are what IsNotFoundErr / IsNotOwnMessageErr
	// match against — set in tests that want to verify the
	// ErrCodeInvalidParams mapping for either branch.
	notFoundErr error
	notOwnErr   error
}

func (s *stubOps) React(ctx context.Context, msgID, emoji string) (string, int64, error) {
	return s.react(ctx, msgID, emoji)
}
func (s *stubOps) Edit(ctx context.Context, msgID, text string, mentionedJIDs []string) (string, int64, error) {
	return s.edit(ctx, msgID, text, mentionedJIDs)
}
func (s *stubOps) Revoke(ctx context.Context, msgID string) (string, int64, error) {
	return s.revoke(ctx, msgID)
}
func (s *stubOps) IsNotFoundErr(err error) bool {
	return s.notFoundErr != nil && errors.Is(err, s.notFoundErr)
}
func (s *stubOps) IsNotOwnMessageErr(err error) bool {
	return s.notOwnErr != nil && errors.Is(err, s.notOwnErr)
}

func TestSendReactionDispatchesAndReturnsResult(t *testing.T) {
	var seenID, seenEmoji string
	reg := NewRegistry()
	RegisterMessageOps(reg, &stubOps{
		react: func(_ context.Context, id, emoji string) (string, int64, error) {
			seenID = id
			seenEmoji = emoji
			return "REACT-1", 1700000000000, nil
		},
	})
	params := json.RawMessage(`{"message_id":"M1","reaction":"👍"}`)
	result, rpcErr := reg.Dispatch(context.Background(), "sendReaction", params)
	if rpcErr != nil {
		t.Fatalf("unexpected rpc error: %+v", rpcErr)
	}
	if seenID != "M1" || seenEmoji != "👍" {
		t.Fatalf("react args = (%q, %q)", seenID, seenEmoji)
	}
	res, ok := result.(opResult)
	if !ok {
		t.Fatalf("unexpected result type: %T", result)
	}
	if res.MessageID != "REACT-1" || res.TimestampMS != 1700000000000 {
		t.Errorf("opResult = %+v", res)
	}
}

func TestSendReactionRejectsMissingMessageID(t *testing.T) {
	reg := NewRegistry()
	RegisterMessageOps(reg, &stubOps{
		react: func(context.Context, string, string) (string, int64, error) {
			t.Fatalf("react fn should not be called when message_id missing")
			return "", 0, nil
		},
	})
	_, rpcErr := reg.Dispatch(context.Background(), "sendReaction", json.RawMessage(`{"reaction":"👍"}`))
	if rpcErr == nil {
		t.Fatalf("expected rpc error for missing message_id")
	}
	if rpcErr.Code != rpc.ErrCodeInvalidParams {
		t.Errorf("code = %d, want ErrCodeInvalidParams", rpcErr.Code)
	}
}

func TestEditMessageDispatches(t *testing.T) {
	var seenID, seenText string
	var seenMentions []string
	reg := NewRegistry()
	RegisterMessageOps(reg, &stubOps{
		edit: func(_ context.Context, id, text string, mentions []string) (string, int64, error) {
			seenID = id
			seenText = text
			seenMentions = mentions
			return "EDIT-1", 1700000000001, nil
		},
	})
	params := json.RawMessage(`{"message_id":"M1","text":"corrected","mentioned_jids":["15551234567@s.whatsapp.net"]}`)
	_, rpcErr := reg.Dispatch(context.Background(), "editMessage", params)
	if rpcErr != nil {
		t.Fatalf("unexpected rpc error: %+v", rpcErr)
	}
	if len(seenMentions) != 1 || seenMentions[0] != "15551234567@s.whatsapp.net" {
		t.Errorf("seenMentions = %v, want [15551234567@s.whatsapp.net]", seenMentions)
	}
	if seenID != "M1" || seenText != "corrected" {
		t.Errorf("edit args = (%q, %q)", seenID, seenText)
	}
}

func TestDeleteMessageDispatches(t *testing.T) {
	var seenID string
	reg := NewRegistry()
	RegisterMessageOps(reg, &stubOps{
		revoke: func(_ context.Context, id string) (string, int64, error) {
			seenID = id
			return "REV-1", 1700000000002, nil
		},
	})
	_, rpcErr := reg.Dispatch(context.Background(), "deleteMessage", json.RawMessage(`{"message_id":"M1"}`))
	if rpcErr != nil {
		t.Fatalf("unexpected rpc error: %+v", rpcErr)
	}
	if seenID != "M1" {
		t.Errorf("revoke arg = %q", seenID)
	}
}

func TestMessageOpNotOwnMessageMapsToInvalidParams(t *testing.T) {
	// Edit/Revoke on a peer's message returns ErrNotOwnMessage; the
	// handler must surface it as a precondition refusal so the model
	// distinguishes "you targeted a foreign message you can't edit"
	// from a transient infra failure that should retry.
	sentinel := errors.New("sentinel-not-own")
	reg := NewRegistry()
	RegisterMessageOps(reg, &stubOps{
		edit: func(context.Context, string, string, []string) (string, int64, error) {
			return "", 0, sentinel
		},
		notOwnErr: sentinel,
	})
	_, rpcErr := reg.Dispatch(context.Background(), "editMessage", json.RawMessage(`{"message_id":"FOREIGN","text":"x"}`))
	if rpcErr == nil {
		t.Fatalf("expected rpc error")
	}
	if rpcErr.Code != rpc.ErrCodeInvalidParams {
		t.Errorf("code = %d, want ErrCodeInvalidParams", rpcErr.Code)
	}
}

func TestMessageOpNotFoundMapsToInvalidParams(t *testing.T) {
	// Lookup miss must surface as ErrCodeInvalidParams so callers can
	// distinguish "unknown message_id" from generic server failures —
	// the model would retry the wrong thing otherwise.
	sentinel := errors.New("sentinel-not-found")
	reg := NewRegistry()
	RegisterMessageOps(reg, &stubOps{
		react: func(context.Context, string, string) (string, int64, error) {
			return "", 0, sentinel
		},
		notFoundErr: sentinel,
	})
	_, rpcErr := reg.Dispatch(context.Background(), "sendReaction", json.RawMessage(`{"message_id":"GHOST","reaction":"👍"}`))
	if rpcErr == nil {
		t.Fatalf("expected rpc error")
	}
	if rpcErr.Code != rpc.ErrCodeInvalidParams {
		t.Errorf("code = %d, want ErrCodeInvalidParams", rpcErr.Code)
	}
	if rpcErr.Message != "unknown message_id" {
		t.Errorf("message = %q, want 'unknown message_id'", rpcErr.Message)
	}
}

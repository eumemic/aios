package handler

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

	"aios.dev/connectors/whatsapp/daemon/internal/rpc"
)

type stubGroups struct {
	list   func(ctx context.Context) ([]GroupSummary, error)
	create func(ctx context.Context, name string, participants []string) (*GroupSummary, error)
	rename func(ctx context.Context, jid, name string) error
}

func (s *stubGroups) ListGroups(ctx context.Context) ([]GroupSummary, error) {
	return s.list(ctx)
}
func (s *stubGroups) CreateGroup(ctx context.Context, name string, p []string) (*GroupSummary, error) {
	return s.create(ctx, name, p)
}
func (s *stubGroups) RenameGroup(ctx context.Context, jid, name string) error {
	return s.rename(ctx, jid, name)
}

func TestListGroupsWrapsEmptyAsObject(t *testing.T) {
	// Empty list must serialize as ``{"groups": []}`` rather than
	// top-level null — JSON-RPC requires a non-null result for
	// success and the Python tool wrapper assumes a dict.
	reg := NewRegistry()
	RegisterGroups(reg, &stubGroups{
		list: func(context.Context) ([]GroupSummary, error) { return nil, nil },
	})
	result, rpcErr := reg.Dispatch(context.Background(), "listGroups", json.RawMessage(`{}`))
	if rpcErr != nil {
		t.Fatalf("unexpected rpc error: %+v", rpcErr)
	}
	wrapped, ok := result.(map[string]any)
	if !ok {
		t.Fatalf("result = %T, want map", result)
	}
	if _, ok := wrapped["groups"]; !ok {
		t.Errorf("result missing 'groups' key: %v", wrapped)
	}
}

func TestCreateGroupForwardsArgs(t *testing.T) {
	var seenName string
	var seenParts []string
	reg := NewRegistry()
	RegisterGroups(reg, &stubGroups{
		create: func(_ context.Context, name string, parts []string) (*GroupSummary, error) {
			seenName = name
			seenParts = parts
			return &GroupSummary{JID: "g@g.us", Name: name}, nil
		},
	})
	params := json.RawMessage(`{"name":"Test","participants":["a@s.whatsapp.net","b@s.whatsapp.net"]}`)
	_, rpcErr := reg.Dispatch(context.Background(), "createGroup", params)
	if rpcErr != nil {
		t.Fatalf("unexpected rpc error: %+v", rpcErr)
	}
	if seenName != "Test" || len(seenParts) != 2 || seenParts[1] != "b@s.whatsapp.net" {
		t.Errorf("seen = (%q, %v)", seenName, seenParts)
	}
}

func TestCreateGroupRejectsMissingName(t *testing.T) {
	reg := NewRegistry()
	RegisterGroups(reg, &stubGroups{
		create: func(context.Context, string, []string) (*GroupSummary, error) {
			t.Fatalf("create should not be called when name missing")
			return nil, nil
		},
	})
	_, rpcErr := reg.Dispatch(context.Background(), "createGroup", json.RawMessage(`{"participants":[]}`))
	if rpcErr == nil || rpcErr.Code != rpc.ErrCodeInvalidParams {
		t.Errorf("expected ErrCodeInvalidParams, got %+v", rpcErr)
	}
}

func TestRenameGroupRejectsMissingFields(t *testing.T) {
	reg := NewRegistry()
	RegisterGroups(reg, &stubGroups{
		rename: func(context.Context, string, string) error {
			t.Fatalf("rename should not be called")
			return nil
		},
	})
	cases := []string{
		`{"name":"x"}`,
		`{"jid":"g@g.us"}`,
		`{}`,
	}
	for _, c := range cases {
		_, rpcErr := reg.Dispatch(context.Background(), "renameGroup", json.RawMessage(c))
		if rpcErr == nil || rpcErr.Code != rpc.ErrCodeInvalidParams {
			t.Errorf("expected ErrCodeInvalidParams for %s, got %+v", c, rpcErr)
		}
	}
}

func TestGroupErrorMapsToServerError(t *testing.T) {
	reg := NewRegistry()
	RegisterGroups(reg, &stubGroups{
		rename: func(context.Context, string, string) error { return errors.New("server rejected") },
	})
	_, rpcErr := reg.Dispatch(context.Background(), "renameGroup", json.RawMessage(`{"jid":"g@g.us","name":"x"}`))
	if rpcErr == nil || rpcErr.Code != rpc.ErrCodeServerError {
		t.Errorf("expected ErrCodeServerError, got %+v", rpcErr)
	}
}

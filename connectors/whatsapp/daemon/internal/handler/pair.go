package handler

import (
	"context"
	"encoding/json"

	"aios.dev/connectors/whatsapp/daemon/internal/rpc"
)

// Pairer is the daemon-internal interface for the pairing surface.
// Taking an interface (rather than a *wameow.Client) keeps the handler
// decoupled from whatsmeow types so tests can stub it.
type Pairer interface {
	StartPairing(ctx context.Context) (string, error)
	ConfirmPairing(ctx context.Context) (PairingOutcome, error)
	Unpair(ctx context.Context) error
}

// PairingOutcome mirrors wameow.PairingOutcome at the handler layer so
// this package doesn't import wameow.  The wire shape is determined
// by the json tags on confirmPairingResult below.
type PairingOutcome struct {
	Status   string
	JID      string
	PushName string
	Reason   string
}

type startPairingResult struct {
	Code string `json:"code"`
}

type confirmPairingResult struct {
	Status   string `json:"status"`
	JID      string `json:"jid,omitempty"`
	PushName string `json:"push_name,omitempty"`
	Reason   string `json:"reason,omitempty"`
}

// RegisterPairing wires the pairing RPC methods into reg.
func RegisterPairing(reg *Registry, p Pairer) {
	reg.Register("startPairing", func(ctx context.Context, _ json.RawMessage) (any, *rpc.Error) {
		code, err := p.StartPairing(ctx)
		if err != nil {
			return nil, &rpc.Error{Code: rpc.ErrCodeServerError, Message: err.Error()}
		}
		return startPairingResult{Code: code}, nil
	})
	reg.Register("confirmPairing", func(ctx context.Context, _ json.RawMessage) (any, *rpc.Error) {
		outcome, err := p.ConfirmPairing(ctx)
		if err != nil {
			return nil, &rpc.Error{Code: rpc.ErrCodeServerError, Message: err.Error()}
		}
		return confirmPairingResult{
			Status:   outcome.Status,
			JID:      outcome.JID,
			PushName: outcome.PushName,
			Reason:   outcome.Reason,
		}, nil
	})
	reg.Register("unpair", func(ctx context.Context, _ json.RawMessage) (any, *rpc.Error) {
		if err := p.Unpair(ctx); err != nil {
			return nil, &rpc.Error{Code: rpc.ErrCodeServerError, Message: err.Error()}
		}
		return map[string]string{"status": "ok"}, nil
	})
}

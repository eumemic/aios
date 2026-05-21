package wameow

import (
	"context"
	"errors"
	"fmt"
	"sync"

	"go.mau.fi/whatsmeow"
)

// PairingOutcome is the terminal state of a pairing attempt.  Status is
// always set; JID + PushName are populated on success; Reason on error.
type PairingOutcome struct {
	Status   string
	JID      string
	PushName string
	Reason   string
}

// pairing tracks one in-flight QR pairing attempt.  Reset when the
// caller starts a new attempt after a terminal outcome.
type pairing struct {
	mu         sync.Mutex
	inProgress bool
	done       chan struct{}
	outcome    PairingOutcome
}

// StartPairing initiates a QR pairing session.  Returns the first QR
// code; the operator scans it within the whatsmeow channel's lifetime
// (~100 s total across multiple refreshes).  Subsequent QR refreshes
// are drained but not surfaced — the operator scans the first code in
// the ~20 s before it rotates.
//
// Errors: "device already paired", "pairing already in progress",
// or whatever whatsmeow returns from GetQRChannel / Connect.
func (c *Client) StartPairing(ctx context.Context) (string, error) {
	c.pair.mu.Lock()
	if c.hasPairedDevice() {
		c.pair.mu.Unlock()
		return "", errors.New("device already paired")
	}
	if c.pair.inProgress {
		c.pair.mu.Unlock()
		return "", errors.New("pairing already in progress")
	}
	c.pair.inProgress = true
	c.pair.done = make(chan struct{})
	c.pair.outcome = PairingOutcome{}
	c.pair.mu.Unlock()

	qrChan, err := c.wa.GetQRChannel(ctx)
	if err != nil {
		c.completePair(PairingOutcome{Status: "error", Reason: err.Error()})
		return "", fmt.Errorf("get qr channel: %w", err)
	}
	if err := c.wa.Connect(); err != nil {
		c.completePair(PairingOutcome{Status: "error", Reason: err.Error()})
		return "", fmt.Errorf("connect: %w", err)
	}

	select {
	case item, ok := <-qrChan:
		if !ok {
			c.completePair(PairingOutcome{Status: "error", Reason: "qr channel closed"})
			return "", errors.New("qr channel closed before first code")
		}
		if item.Event != "code" {
			c.completeFromQRItem(item)
			return "", fmt.Errorf("pairing failed before first code: %s", item.Event)
		}
		go c.drainQRChannel(qrChan)
		return item.Code, nil
	case <-ctx.Done():
		c.completePair(PairingOutcome{Status: "error", Reason: ctx.Err().Error()})
		return "", ctx.Err()
	}
}

// ConfirmPairing blocks until the in-flight pairing terminates (or
// ctx cancels).  If pairing has already terminated, returns the cached
// outcome immediately.  Errors if no pairing has ever been started.
func (c *Client) ConfirmPairing(ctx context.Context) (PairingOutcome, error) {
	c.pair.mu.Lock()
	if !c.pair.inProgress {
		outcome := c.pair.outcome
		c.pair.mu.Unlock()
		if outcome.Status == "" {
			return PairingOutcome{}, errors.New("no pairing in progress")
		}
		return outcome, nil
	}
	done := c.pair.done
	c.pair.mu.Unlock()

	select {
	case <-done:
		c.pair.mu.Lock()
		outcome := c.pair.outcome
		c.pair.mu.Unlock()
		return outcome, nil
	case <-ctx.Done():
		return PairingOutcome{}, ctx.Err()
	}
}

// Unpair calls whatsmeow's Logout, which un-links the device on the
// server and deletes the local store.  After Unpair, hasPairedDevice
// becomes false and StartPairing can be called fresh.
func (c *Client) Unpair(ctx context.Context) error {
	if !c.hasPairedDevice() {
		return errors.New("device not paired")
	}
	return c.wa.Logout(ctx)
}

// drainQRChannel consumes the QR channel after StartPairing has
// returned the first code.  Refresh "code" events are silently
// dropped (this PR scopes to single-scan pairing); the first non-code
// item terminates the pairing.
func (c *Client) drainQRChannel(qrChan <-chan whatsmeow.QRChannelItem) {
	for item := range qrChan {
		if item.Event == "code" {
			continue
		}
		if item.Event == "error" {
			reason := "unknown"
			if item.Error != nil {
				reason = item.Error.Error()
			}
			c.completePair(PairingOutcome{Status: "error", Reason: reason})
			return
		}
		c.completeFromQRItem(item)
		return
	}
	c.completePair(PairingOutcome{Status: "error", Reason: "qr channel closed without terminal"})
}

func (c *Client) completeFromQRItem(item whatsmeow.QRChannelItem) {
	var outcome PairingOutcome
	switch item.Event {
	case "success":
		outcome = PairingOutcome{Status: "success", PushName: c.wa.Store.PushName}
		if id := c.wa.Store.ID; id != nil {
			outcome.JID = id.String()
		}
	case "timeout":
		outcome = PairingOutcome{Status: "timeout"}
	default:
		outcome = PairingOutcome{Status: "error", Reason: item.Event}
	}
	c.completePair(outcome)
}

func (c *Client) completePair(outcome PairingOutcome) {
	c.pair.mu.Lock()
	defer c.pair.mu.Unlock()
	if !c.pair.inProgress {
		return
	}
	c.pair.inProgress = false
	c.pair.outcome = outcome
	close(c.pair.done)
}

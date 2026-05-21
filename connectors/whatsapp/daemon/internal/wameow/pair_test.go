package wameow

import (
	"context"
	"errors"
	"testing"
	"time"

	"go.mau.fi/whatsmeow"
)

func TestCompletePairTransitionsState(t *testing.T) {
	c := &Client{}
	c.pair.inProgress = true
	c.pair.done = make(chan struct{})

	c.completePair(PairingOutcome{Status: "success", JID: "alice@s.whatsapp.net"})

	if c.pair.inProgress {
		t.Error("expected inProgress = false after completePair")
	}
	if c.pair.outcome.Status != "success" {
		t.Errorf("outcome.Status = %q, want 'success'", c.pair.outcome.Status)
	}
	select {
	case <-c.pair.done:
	default:
		t.Error("done channel was not closed")
	}
}

func TestCompletePairIsIdempotent(t *testing.T) {
	c := &Client{}
	c.pair.inProgress = true
	c.pair.done = make(chan struct{})

	c.completePair(PairingOutcome{Status: "success"})
	// Second call must be a no-op (must not re-close the channel).
	c.completePair(PairingOutcome{Status: "timeout"})

	if c.pair.outcome.Status != "success" {
		t.Errorf("outcome was overwritten by second completePair: %q", c.pair.outcome.Status)
	}
}

func TestConfirmPairingErrorsWhenNoPairing(t *testing.T) {
	c := &Client{}
	_, err := c.ConfirmPairing(context.Background())
	if err == nil {
		t.Error("expected error when no pairing has been started")
	}
}

func TestConfirmPairingReturnsCachedOutcomeAfterTerminated(t *testing.T) {
	c := &Client{}
	// Simulate a previously-terminated pairing.
	c.pair.outcome = PairingOutcome{Status: "timeout"}

	outcome, err := c.ConfirmPairing(context.Background())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if outcome.Status != "timeout" {
		t.Errorf("outcome.Status = %q, want 'timeout'", outcome.Status)
	}
}

func TestConfirmPairingBlocksUntilCompletePair(t *testing.T) {
	c := &Client{}
	c.pair.inProgress = true
	c.pair.done = make(chan struct{})

	go func() {
		// Slight delay so ConfirmPairing reaches the select before we fire.
		time.Sleep(50 * time.Millisecond)
		c.completePair(PairingOutcome{Status: "success", JID: "x@y"})
	}()

	outcome, err := c.ConfirmPairing(context.Background())
	if err != nil {
		t.Fatalf("ConfirmPairing returned error: %v", err)
	}
	if outcome.JID != "x@y" {
		t.Errorf("outcome.JID = %q, want 'x@y'", outcome.JID)
	}
}

func TestConfirmPairingCancelsOnContext(t *testing.T) {
	c := &Client{}
	c.pair.inProgress = true
	c.pair.done = make(chan struct{})

	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		time.Sleep(20 * time.Millisecond)
		cancel()
	}()

	_, err := c.ConfirmPairing(ctx)
	if !errors.Is(err, context.Canceled) {
		t.Errorf("expected context.Canceled, got %v", err)
	}
}

func TestDrainQRChannelEndsOnTimeout(t *testing.T) {
	c := &Client{}
	c.pair.inProgress = true
	c.pair.done = make(chan struct{})

	ch := make(chan whatsmeow.QRChannelItem, 3)
	ch <- whatsmeow.QRChannelItem{Event: "code", Code: "first"}
	ch <- whatsmeow.QRChannelItem{Event: "code", Code: "refresh-ignored"}
	ch <- whatsmeow.QRChannelTimeout
	close(ch)

	c.drainQRChannel(ch)

	if c.pair.outcome.Status != "timeout" {
		t.Errorf("outcome.Status = %q, want 'timeout'", c.pair.outcome.Status)
	}
}

func TestDrainQRChannelSurfacesErrorReason(t *testing.T) {
	c := &Client{}
	c.pair.inProgress = true
	c.pair.done = make(chan struct{})

	ch := make(chan whatsmeow.QRChannelItem, 1)
	ch <- whatsmeow.QRChannelItem{Event: "error", Error: errors.New("boom")}
	close(ch)

	c.drainQRChannel(ch)

	if c.pair.outcome.Status != "error" {
		t.Errorf("outcome.Status = %q, want 'error'", c.pair.outcome.Status)
	}
	if c.pair.outcome.Reason != "boom" {
		t.Errorf("outcome.Reason = %q, want 'boom'", c.pair.outcome.Reason)
	}
}

func TestDrainQRChannelClosedWithoutTerminal(t *testing.T) {
	c := &Client{}
	c.pair.inProgress = true
	c.pair.done = make(chan struct{})

	ch := make(chan whatsmeow.QRChannelItem)
	close(ch)

	c.drainQRChannel(ch)

	if c.pair.outcome.Status != "error" {
		t.Errorf("outcome.Status = %q, want 'error'", c.pair.outcome.Status)
	}
}

package wameow

import (
	"context"
	"errors"
	"fmt"
	"sync"

	"go.mau.fi/whatsmeow"
)

// PairingOutcome is the terminal state of a pairing attempt.  Status
// is always one of "success" / "timeout" / "error".  JID is populated
// on success; Reason on error/timeout.  PushName is intentionally not
// set here — whatsmeow doesn't populate Store.PushName at PairSuccess
// time, so reading it would always be "" and mislead the caller.  A
// later PR can fill PushName from a subsequent Connected event.
type PairingOutcome struct {
	Status   string
	JID      string
	PushName string
	Reason   string
}

// pairAttempt holds one attempt's terminal state behind a done channel
// that's closed exactly once.  outcome is set under c.pair.mu BEFORE
// close(done); the close synchronizes the write so readers waking on
// <-done observe the final outcome without re-acquiring the lock.
//
// Per-attempt allocation lets ConfirmPairing snapshot the attempt it's
// waiting on; a concurrent StartPairing reset re-binds c.pair.attempt
// to a NEW *pairAttempt but doesn't mutate the snapshot, so the
// waiter still sees its original attempt's outcome.
type pairAttempt struct {
	done    chan struct{}
	outcome PairingOutcome

	// code is the QR string currently displayed to the operator;
	// rotationSeq increments each time whatsmeow rotates it (~every
	// 20s).  Both are written under c.pair.mu by recordQRCode — the
	// first code at StartPairing time (seq 0) and each subsequent
	// "code" refresh in drainQRChannel.  GetPairingCode snapshots
	// them under the same lock so a polling operator can re-render on
	// seq change.  done being closed means the attempt terminated and
	// no live code remains.
	code        string
	rotationSeq int
}

// pairing tracks the current or most recent pairing attempt.  attempt
// is nil before the first StartPairing and after Unpair forgets the
// cached outcome — both states mean ConfirmPairing should error.
type pairing struct {
	mu         sync.Mutex
	inProgress bool
	attempt    *pairAttempt
}

// StartPairing initiates a QR pairing session.  Returns the first QR
// code; subsequent refreshes are consumed but not surfaced (single-
// scan v1 scope).  The operator scans within whatsmeow's QR window.
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
	attempt := &pairAttempt{done: make(chan struct{})}
	c.pair.inProgress = true
	c.pair.attempt = attempt
	c.pair.mu.Unlock()

	wa := c.wa.Load()
	qrChan, err := wa.GetQRChannel(ctx)
	if err != nil {
		c.completePair(PairingOutcome{Status: "error", Reason: err.Error()})
		return "", fmt.Errorf("get qr channel: %w", err)
	}
	if err := wa.Connect(); err != nil {
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
		c.recordQRCode(attempt, item.Code, 0)
		go c.drainQRChannel(qrChan)
		return item.Code, nil
	case <-ctx.Done():
		c.completePair(PairingOutcome{Status: "error", Reason: ctx.Err().Error()})
		// Drain the QR channel so whatsmeow's emitter doesn't block
		// on a buffer-full send to an un-read channel.  The drain
		// goroutine's completePair calls will be no-ops since the
		// outcome is already set.
		go c.drainQRChannel(qrChan)
		return "", ctx.Err()
	}
}

// ConfirmPairing blocks until the in-flight pairing terminates (or
// ctx cancels), then returns the terminal outcome.  Snapshotting the
// attempt under the lock ensures a concurrent StartPairing reset
// can't replace the outcome THIS caller is waiting on.  Errors if no
// pairing has ever been started or Unpair has forgotten the cache.
func (c *Client) ConfirmPairing(ctx context.Context) (PairingOutcome, error) {
	c.pair.mu.Lock()
	attempt := c.pair.attempt
	c.pair.mu.Unlock()
	if attempt == nil {
		return PairingOutcome{}, errors.New("no pairing in progress")
	}
	select {
	case <-attempt.done:
		// outcome was written under c.pair.mu before close(done);
		// the channel close synchronizes the write.
		return attempt.outcome, nil
	case <-ctx.Done():
		return PairingOutcome{}, ctx.Err()
	}
}

// GetPairingCode returns the QR code currently displayed for the
// in-flight attempt plus its rotationSeq.  Operators poll this every
// few seconds and re-render when rotationSeq changes — surfacing the
// rotated codes whatsmeow emits ~every 20s rather than just the first.
//
// Errors if no attempt has been started, if Unpair forgot the cache,
// or if the attempt has already terminated (done closed): a terminated
// attempt has no live code, and returning a stale one would invite the
// operator to scan a dead QR.  The snapshot is taken under c.pair.mu so
// a concurrent recordQRRefresh can't tear the {code, seq} pair.
func (c *Client) GetPairingCode(ctx context.Context) (string, int, error) {
	c.pair.mu.Lock()
	defer c.pair.mu.Unlock()
	if c.pair.attempt == nil {
		return "", 0, errors.New("no pairing in progress")
	}
	if !c.pair.inProgress {
		return "", 0, errors.New("pairing attempt already terminated")
	}
	return c.pair.attempt.code, c.pair.attempt.rotationSeq, nil
}

// recordQRCode stores the QR code + rotationSeq for attempt under
// c.pair.mu.  Used for the first code (seq 0) at StartPairing time;
// recordQRRefresh handles subsequent rotations.  Guarded against a
// concurrent terminal: if the attempt is no longer current (Unpair or
// completePair already swapped/cleared it), the write is dropped so a
// late code can't resurrect a finished attempt's live-code state.
func (c *Client) recordQRCode(attempt *pairAttempt, code string, seq int) {
	c.pair.mu.Lock()
	defer c.pair.mu.Unlock()
	if c.pair.attempt != attempt {
		return
	}
	attempt.code = code
	attempt.rotationSeq = seq
}

// recordQRRefresh stores a rotated QR code on the current attempt,
// incrementing rotationSeq.  No-op if no attempt is in progress (a late
// refresh arriving after the attempt terminated): a terminated attempt
// has no live code to update.
func (c *Client) recordQRRefresh(code string) {
	c.pair.mu.Lock()
	defer c.pair.mu.Unlock()
	if !c.pair.inProgress || c.pair.attempt == nil {
		return
	}
	c.pair.attempt.code = code
	c.pair.attempt.rotationSeq++
}

// Unpair unlinks the device server-side, deletes the local store, and
// forgets any cached pairing attempt so the next ConfirmPairing
// errors "no pairing in progress" instead of returning a stale
// success.
//
// whatsmeow's Logout requires a connected client and short-circuits
// local cleanup on send failure; since PairSuccess triggers an
// immediate auto-Disconnect, an unpair-right-after-pair would
// otherwise wedge the operator with hasPairedDevice=true and no
// in-process recovery path.  Fall back to forced Disconnect +
// Store.Delete on Logout failure, per whatsmeow's documented offline
// cleanup recipe.
func (c *Client) Unpair(ctx context.Context) error {
	if !c.hasPairedDevice() {
		return errors.New("device not paired")
	}
	wa := c.wa.Load()
	err := wa.Logout(ctx)
	if err != nil {
		c.log.Warn("wameow.logout_failed_forcing_local_cleanup", "err", err)
		wa.Disconnect()
		if delErr := wa.Store.Delete(ctx); delErr != nil {
			return fmt.Errorf("logout failed (%w); local store delete also failed: %v", err, delErr)
		}
	}
	c.resetPairState()
	// whatsmeow's Logout (and its Store.Delete fallback) marks the
	// device as permanently Deleted; subsequent ops on this Client
	// return ErrDeviceDeleted.  Swap in a fresh Client so the same
	// daemon process can immediately serve a new StartPairing.
	c.replaceWhatsmeowClient()
	return nil
}

// replaceWhatsmeowClient builds a fresh whatsmeow.Client around a new
// Device in the existing sqlstore container, rewires the event handler
// to the same Notifier-bound Client receiver, and atomically swaps c.wa.
// Concurrent reads see either the old (deleted) Client or the new one —
// never a torn pointer.
//
// Also truncates the MessageStore: post-unpair, the old MessageKey
// rows are unreachable by the new client (different device identity,
// different whatsmeow Signal sessions).  Leaving them would let a
// subsequent React/Edit/Revoke Lookup-succeed and then either fail
// at the protocol layer or — worse — misdeliver, since BuildEdit/
// BuildRevoke would emit an envelope referencing an msgID created
// under the old identity.  A truncate failure logs but doesn't
// abort the replace: a stale row that the new client can't
// authenticate is at worst a confusing "edit refused" later, which
// is observable; aborting here would leave the daemon with a dead
// Client and no recovery path.
func (c *Client) replaceWhatsmeowClient() {
	newDevice := c.store.NewDevice()
	newWa := whatsmeow.NewClient(newDevice, newWaLogger(c.log, "client"))
	newWa.AddEventHandler(c.handleEvent)
	c.wa.Store(newWa)
	if err := c.msgs.Truncate(context.Background()); err != nil {
		c.log.Warn("wameow.msgstore_truncate_failed", "err", err)
	}
}

// resetPairState forgets any cached pairing attempt.  Called after
// Unpair so the next ConfirmPairing reports "no pairing in progress"
// rather than returning a stale success outcome.  If an attempt is
// still in progress (narrow race where Store.ID was set by
// whatsmeow's handlePair before QRChannelSuccess reached
// drainQRChannel), synthesize a terminal outcome so any waiting
// ConfirmPairing unblocks instead of hanging on a never-closed done.
func (c *Client) resetPairState() {
	c.pair.mu.Lock()
	defer c.pair.mu.Unlock()
	if c.pair.inProgress && c.pair.attempt != nil {
		c.pair.inProgress = false
		c.pair.attempt.outcome = PairingOutcome{Status: "error", Reason: "unpaired during pairing"}
		close(c.pair.attempt.done)
	}
	c.pair.attempt = nil
}

// drainQRChannel consumes the QR channel after StartPairing returned
// the first code.  "code" refreshes and the documented-non-terminal
// "scanned-without-multidevice" event keep the loop alive; only
// explicit terminal events trip completePair and return.
func (c *Client) drainQRChannel(qrChan <-chan whatsmeow.QRChannelItem) {
	for item := range qrChan {
		switch item.Event {
		case "code":
			// QR refresh: whatsmeow rotates the code ~every 20s.
			// Retain it (incrementing rotationSeq) so a polling
			// operator can re-render the live code mid-attempt
			// instead of being stuck on the first ~20s window.
			c.recordQRRefresh(item.Code)
			continue
		case "scanned-without-multidevice":
			// Per whatsmeow this is non-terminal: the QR session
			// stays alive so the user can rescan from a multi-device-
			// enabled WhatsApp client.
			c.log.Warn("wameow.qr.scanned_without_multidevice")
			continue
		case "success", "timeout":
			c.completeFromQRItem(item)
			return
		case "error":
			reason := "unknown"
			if item.Error != nil {
				reason = item.Error.Error()
			}
			c.completePair(PairingOutcome{Status: "error", Reason: reason})
			return
		default:
			// err-* events (err-unexpected-state, err-client-outdated,
			// …) all terminate as errors.  Surface the raw event name
			// as the reason.
			c.completeFromQRItem(item)
			return
		}
	}
	c.completePair(PairingOutcome{Status: "error", Reason: "qr channel closed without terminal"})
}

func (c *Client) completeFromQRItem(item whatsmeow.QRChannelItem) {
	var outcome PairingOutcome
	switch item.Event {
	case "success":
		outcome = PairingOutcome{Status: "success"}
		if id := c.wa.Load().Store.ID; id != nil {
			outcome.JID = id.String()
		}
		// Intentionally not reading Store.PushName — whatsmeow doesn't
		// populate it at PairSuccess time.
	case "timeout":
		outcome = PairingOutcome{Status: "timeout"}
	default:
		outcome = PairingOutcome{Status: "error", Reason: item.Event}
	}
	c.completePair(outcome)
}

// completePair atomically sets the current attempt's outcome and
// closes its done channel.  Idempotent: re-entries (e.g. a late
// terminal from drainQRChannel after Unpair already nilled the
// attempt) are no-ops.
func (c *Client) completePair(outcome PairingOutcome) {
	c.pair.mu.Lock()
	defer c.pair.mu.Unlock()
	if !c.pair.inProgress || c.pair.attempt == nil {
		return
	}
	c.pair.inProgress = false
	c.pair.attempt.outcome = outcome
	close(c.pair.attempt.done)
}

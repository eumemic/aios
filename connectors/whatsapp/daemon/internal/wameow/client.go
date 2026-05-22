// Package wameow wraps the whatsmeow client + sqlstore lifecycle and
// translates whatsmeow events into wire-protocol notifications via the
// daemon's RPC broadcast surface.  The handler-facing API is
// SendMessage (matches handler.SendMessageFn); event delivery flows
// the other way via Notifier.Broadcast.
package wameow

import (
	"context"
	"fmt"
	"log/slog"
	"path/filepath"
	"sync/atomic"

	"go.mau.fi/whatsmeow"
	"go.mau.fi/whatsmeow/store/sqlstore"
	"go.mau.fi/whatsmeow/types"

	// modernc.org/sqlite is a pure-Go SQLite driver so the daemon
	// compiles without CGo.  Registered under the driver name "sqlite".
	_ "modernc.org/sqlite"
)

// Client wraps a whatsmeow.Client + its persistent sqlstore.  One per
// paired phone; created in NewClient, owns its lifecycle until Close.
//
// wa is held behind an atomic.Pointer so Unpair can swap in a fresh
// whatsmeow.Client (built from a new Device) after a successful Logout
// without racing concurrent reads from RPC handlers.  whatsmeow marks
// the Device as permanently Deleted on Logout and every subsequent
// operation on it errors with ErrDeviceDeleted ("invalid use of
// deleted device") — the swap is what lets a re-pair work in the same
// daemon process.
type Client struct {
	wa       atomic.Pointer[whatsmeow.Client]
	store    *sqlstore.Container
	msgs     *MessageStore
	mediaDir string
	notify   Notifier
	log      *slog.Logger
	pair     pairing
	// lifetimeCtx is the daemon-lifetime context (set by NewClient
	// from main's signal-notify ctx).  Used as the background
	// context for in-event-handler ops like media download and
	// inbound msgstore writes, so they cancel on daemon shutdown
	// instead of running against a torn-down store.
	lifetimeCtx context.Context
}

// NewClient opens the sqlstore at <storeDir>/store.db, picks the first
// device (or creates one), and wires the event handler to Notifier.
// Does NOT initiate a WhatsApp connection — call Connect() for that.
func NewClient(
	ctx context.Context,
	storeDir string,
	notify Notifier,
	log *slog.Logger,
) (*Client, error) {
	dbPath := filepath.Join(storeDir, "store.db")
	dsn := fmt.Sprintf("file:%s?_pragma=foreign_keys(1)", dbPath)
	container, err := sqlstore.New(ctx, "sqlite", dsn, newWaLogger(log, "db"))
	if err != nil {
		return nil, fmt.Errorf("open sqlstore at %q: %w", dbPath, err)
	}
	device, err := container.GetFirstDevice(ctx)
	if err != nil {
		_ = container.Close()
		return nil, fmt.Errorf("get first device: %w", err)
	}
	msgs, err := openMessageStore(storeDir)
	if err != nil {
		_ = container.Close()
		return nil, err
	}
	wa := whatsmeow.NewClient(device, newWaLogger(log, "client"))
	c := &Client{
		store:       container,
		msgs:        msgs,
		mediaDir:    filepath.Join(storeDir, "media"),
		notify:      notify,
		log:         log,
		lifetimeCtx: ctx,
	}
	c.wa.Store(wa)
	wa.AddEventHandler(c.handleEvent)
	return c, nil
}

func (c *Client) hasPairedDevice() bool {
	return c.wa.Load().Store.ID != nil
}

// Connect attaches to WhatsApp if the device is paired.  Unpaired is
// a no-op so the daemon keeps serving RPC until the pairing flow
// provisions a device.
func (c *Client) Connect(ctx context.Context) error {
	if !c.hasPairedDevice() {
		c.log.Warn("wameow.no_paired_device")
		return nil
	}
	if err := c.wa.Load().Connect(); err != nil {
		return fmt.Errorf("whatsmeow connect: %w", err)
	}
	return nil
}

// recordOutbound stamps an outbound message into the message store
// so the model can later react to / edit / revoke its own sends by
// id.  Best-effort: a store failure here is logged but doesn't fail
// the send (the send already went through on the wire; failing the
// RPC would push the model toward a retry that produces a duplicate
// peer-visible message).  A later react/edit/delete on this id
// returns ErrMessageNotFound — operators see the put_failed warning
// and can diagnose; the model retries with the user's help if
// needed.
//
// Takes the `wa` that performed the send rather than re-loading
// c.wa.  A concurrent unpair could otherwise swap c.wa between
// sendOne returning and recordOutbound reading Store.ID, stamping
// the row with the NEW identity instead of the one that actually
// sent — breaking subsequent edit/revoke routing.
func (c *Client) recordOutbound(
	ctx context.Context,
	wa *whatsmeow.Client,
	msgID string,
	chatJID types.JID,
) {
	ourID := wa.Store.ID
	if ourID == nil {
		// Shouldn't happen: wa is the client that just successfully
		// sent, so its Store.ID was populated.  Defensive: skip the
		// record rather than panic.
		return
	}
	if err := c.msgs.Put(ctx, msgID, chatJID.String(), ourID.String(), true); err != nil {
		c.log.Warn("wameow.msgstore_put_failed", "id", msgID, "err", err)
	}
}

// Close disconnects from WhatsApp and closes the sqlstore.
func (c *Client) Close() {
	c.wa.Load().Disconnect()
	if err := c.msgs.Close(); err != nil {
		c.log.Warn("wameow.msgstore_close_error", "err", err)
	}
	if err := c.store.Close(); err != nil {
		c.log.Warn("wameow.store_close_error", "err", err)
	}
}

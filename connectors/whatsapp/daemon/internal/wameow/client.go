// Package wameow wraps the whatsmeow client + sqlstore lifecycle and
// translates whatsmeow events into wire-protocol notifications via the
// daemon's RPC broadcast surface.  The handler-facing API is
// SendMessage (matches handler.SendMessageFn); event delivery flows
// the other way via Notifier.Broadcast.
package wameow

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"path/filepath"
	"sync/atomic"

	"go.mau.fi/whatsmeow"
	"go.mau.fi/whatsmeow/proto/waE2E"
	"go.mau.fi/whatsmeow/store/sqlstore"
	"go.mau.fi/whatsmeow/types"
	"google.golang.org/protobuf/proto"

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
	wa     atomic.Pointer[whatsmeow.Client]
	store  *sqlstore.Container
	notify Notifier
	log    *slog.Logger
	pair   pairing
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
	wa := whatsmeow.NewClient(device, newWaLogger(log, "client"))
	c := &Client{store: container, notify: notify, log: log}
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

// SendMessage matches handler.SendMessageFn.
func (c *Client) SendMessage(ctx context.Context, jidStr, text string) (string, int64, error) {
	wa := c.wa.Load()
	if !wa.IsConnected() {
		return "", 0, errors.New("whatsmeow: not connected")
	}
	jid, err := types.ParseJID(jidStr)
	if err != nil {
		return "", 0, fmt.Errorf("invalid JID %q: %w", jidStr, err)
	}
	msg := &waE2E.Message{Conversation: proto.String(text)}
	resp, err := wa.SendMessage(ctx, jid, msg)
	if err != nil {
		return "", 0, err
	}
	return string(resp.ID), resp.Timestamp.UnixMilli(), nil
}

// Close disconnects from WhatsApp and closes the sqlstore.
func (c *Client) Close() {
	c.wa.Load().Disconnect()
	if err := c.store.Close(); err != nil {
		c.log.Warn("wameow.store_close_error", "err", err)
	}
}

package wameow

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"path/filepath"

	// modernc.org/sqlite is already registered by client.go's import.
	_ "modernc.org/sqlite"
)

// MessageStore is a sqlite-backed index of every WhatsApp message the
// daemon has seen — inbound or outbound — keyed by message_id and
// carrying the MessageKey context (chat_jid, sender_jid, from_me) that
// whatsmeow's Build{Reaction,Edit,Revoke} require.
//
// WhatsApp's protocol routes message-targeted operations by a 4-tuple
// MessageKey rather than the bare message_id, and its servers expose
// no "look up message by id" affordance to clients.  We own the
// mapping locally because there's no upstream we can defer to.
//
// Lives in a sibling file (messages.db) to whatsmeow's store.db so
// schema migrations stay independent.
type MessageStore struct {
	db *sql.DB
}

// ErrMessageNotFound is returned by Lookup when no row matches the id.
// React/edit/revoke handlers surface it back to the caller as
// "unknown message_id" so the model retries with a different target
// instead of silently no-oping.
var ErrMessageNotFound = errors.New("message not found")

func openMessageStore(storeDir string) (*MessageStore, error) {
	dbPath := filepath.Join(storeDir, "messages.db")
	dsn := fmt.Sprintf("file:%s?_pragma=foreign_keys(1)&_pragma=journal_mode(WAL)", dbPath)
	return openMessageStoreDSN(dsn)
}

// openMessageStoreDSN is the DSN-form constructor used by tests that
// want an in-memory store ("file::memory:?cache=shared").  Production
// callers go through openMessageStore which computes the DSN from a
// store directory.
func openMessageStoreDSN(dsn string) (*MessageStore, error) {
	db, err := sql.Open("sqlite", dsn)
	if err != nil {
		return nil, fmt.Errorf("open messages db %q: %w", dsn, err)
	}
	if _, err := db.Exec(`
		CREATE TABLE IF NOT EXISTS messages (
			id TEXT PRIMARY KEY,
			chat_jid TEXT NOT NULL,
			sender_jid TEXT NOT NULL,
			from_me INTEGER NOT NULL CHECK (from_me IN (0, 1)),
			created_at INTEGER NOT NULL DEFAULT (CAST(strftime('%s', 'now') * 1000 AS INTEGER))
		)
	`); err != nil {
		_ = db.Close()
		return nil, fmt.Errorf("create messages table: %w", err)
	}
	return &MessageStore{db: db}, nil
}

func (s *MessageStore) Close() error {
	return s.db.Close()
}

// Put records a message's MessageKey.  Idempotent on conflict: a
// duplicate id is a no-op so retries and re-deliveries don't fail.
func (s *MessageStore) Put(ctx context.Context, id, chatJID, senderJID string, fromMe bool) error {
	fromMeInt := 0
	if fromMe {
		fromMeInt = 1
	}
	_, err := s.db.ExecContext(ctx, `
		INSERT INTO messages (id, chat_jid, sender_jid, from_me)
		VALUES (?, ?, ?, ?)
		ON CONFLICT(id) DO NOTHING
	`, id, chatJID, senderJID, fromMeInt)
	if err != nil {
		return fmt.Errorf("insert message %s: %w", id, err)
	}
	return nil
}

// Lookup returns the MessageKey for id, or ErrMessageNotFound.
func (s *MessageStore) Lookup(ctx context.Context, id string) (chatJID, senderJID string, fromMe bool, err error) {
	var fromMeInt int
	row := s.db.QueryRowContext(ctx, `
		SELECT chat_jid, sender_jid, from_me FROM messages WHERE id = ?
	`, id)
	if scanErr := row.Scan(&chatJID, &senderJID, &fromMeInt); scanErr != nil {
		if errors.Is(scanErr, sql.ErrNoRows) {
			return "", "", false, ErrMessageNotFound
		}
		return "", "", false, fmt.Errorf("lookup message %s: %w", id, scanErr)
	}
	return chatJID, senderJID, fromMeInt == 1, nil
}

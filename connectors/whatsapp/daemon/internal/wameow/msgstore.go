package wameow

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"path/filepath"
	"strings"

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

// ErrNotOwnMessage is returned by Edit/Revoke when the target msgID
// exists in the store but was sent by a peer (from_me=false).
// WhatsApp only allows editing or revoking your own outbound
// messages, so this is a precondition refusal — not a server error.
// The handler layer maps it to ErrCodeInvalidParams so the model
// distinguishes "wrong target" from "infrastructure failure".
var ErrNotOwnMessage = errors.New("message not sent by us")

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
			sent_at INTEGER NOT NULL DEFAULT 0,
			text TEXT NOT NULL DEFAULT '',
			created_at INTEGER NOT NULL DEFAULT (CAST(strftime('%s', 'now') * 1000 AS INTEGER))
		)
	`); err != nil {
		_ = db.Close()
		return nil, fmt.Errorf("create messages table: %w", err)
	}
	// Forward-migrate older stores that pre-date the sent_at + text
	// columns.  Bare ALTER TABLE ADD COLUMN on a column that already
	// exists raises "duplicate column name" — swallow that specific
	// error so the migration is idempotent on fresh AND pre-existing
	// databases.
	for _, alter := range []string{
		`ALTER TABLE messages ADD COLUMN sent_at INTEGER NOT NULL DEFAULT 0`,
		`ALTER TABLE messages ADD COLUMN text TEXT NOT NULL DEFAULT ''`,
	} {
		if _, err := db.Exec(alter); err != nil && !strings.Contains(err.Error(), "duplicate column name") {
			_ = db.Close()
			return nil, fmt.Errorf("alter messages: %w", err)
		}
	}
	return &MessageStore{db: db}, nil
}

func (s *MessageStore) Close() error {
	return s.db.Close()
}

// Put records a message's MessageKey.  Idempotent on conflict: a
// duplicate id is a no-op so retries and re-deliveries don't fail.
//
// ``sentAtMs`` is the message's wall-clock send time (whatsmeow's
// ``MessageInfo.Timestamp``), used by Edit/Revoke window guards.
// ``text`` is the message body, used to build the QuotedMessage stub
// when the model issues a reply with ``quoted_message_id``.
func (s *MessageStore) Put(ctx context.Context, id, chatJID, senderJID string, fromMe bool, sentAtMs int64, text string) error {
	fromMeInt := 0
	if fromMe {
		fromMeInt = 1
	}
	_, err := s.db.ExecContext(ctx, `
		INSERT INTO messages (id, chat_jid, sender_jid, from_me, sent_at, text)
		VALUES (?, ?, ?, ?, ?, ?)
		ON CONFLICT(id) DO NOTHING
	`, id, chatJID, senderJID, fromMeInt, sentAtMs, text)
	if err != nil {
		return fmt.Errorf("insert message %s: %w", id, err)
	}
	return nil
}

// MessageRecord is the row returned by Lookup.
type MessageRecord struct {
	ChatJID   string
	SenderJID string
	FromMe    bool
	SentAtMs  int64
	Text      string
}

// Lookup returns the MessageRecord for id, or ErrMessageNotFound.
func (s *MessageStore) Lookup(ctx context.Context, id string) (MessageRecord, error) {
	var (
		rec       MessageRecord
		fromMeInt int
	)
	row := s.db.QueryRowContext(ctx, `
		SELECT chat_jid, sender_jid, from_me, sent_at, text FROM messages WHERE id = ?
	`, id)
	if scanErr := row.Scan(&rec.ChatJID, &rec.SenderJID, &fromMeInt, &rec.SentAtMs, &rec.Text); scanErr != nil {
		if errors.Is(scanErr, sql.ErrNoRows) {
			return MessageRecord{}, ErrMessageNotFound
		}
		return MessageRecord{}, fmt.Errorf("lookup message %s: %w", id, scanErr)
	}
	rec.FromMe = fromMeInt == 1
	return rec, nil
}

// Truncate empties the messages table.  Called after Unpair so the
// next pairing session doesn't carry stale MessageKey rows from the
// previous device identity — those rows would Lookup-succeed but
// reference whatsmeow Signal sessions the new device can't address.
func (s *MessageStore) Truncate(ctx context.Context) error {
	if _, err := s.db.ExecContext(ctx, `DELETE FROM messages`); err != nil {
		return fmt.Errorf("truncate messages: %w", err)
	}
	return nil
}

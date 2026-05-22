package wameow

import (
	"context"
	"errors"
	"sync"
	"testing"
)

// newTestMessageStore returns an in-memory MessageStore wired with a
// shared-cache DSN so concurrent test connections see the same data.
// The store is closed at test cleanup.
//
// uniqueInMemoryDSN guarantees parallel tests don't collide on the
// same in-memory database via sqlite's "shared cache" name namespace.
func newTestMessageStore(t *testing.T) *MessageStore {
	t.Helper()
	s, err := openMessageStoreDSN(uniqueInMemoryDSN())
	if err != nil {
		t.Fatalf("openMessageStoreDSN: %v", err)
	}
	t.Cleanup(func() { _ = s.Close() })
	return s
}

var (
	dsnMu  sync.Mutex
	dsnSeq int
)

func uniqueInMemoryDSN() string {
	dsnMu.Lock()
	defer dsnMu.Unlock()
	dsnSeq++
	return "file:msgstore_test_" + itoa(dsnSeq) + "?mode=memory&cache=shared"
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var buf [20]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	return string(buf[i:])
}

func TestMessageStorePutLookup(t *testing.T) {
	ctx := context.Background()
	s := newTestMessageStore(t)

	if err := s.Put(ctx, "MSG1", "chat@s.whatsapp.net", "sender@s.whatsapp.net", false); err != nil {
		t.Fatalf("Put: %v", err)
	}

	chat, sender, fromMe, err := s.Lookup(ctx, "MSG1")
	if err != nil {
		t.Fatalf("Lookup: %v", err)
	}
	if chat != "chat@s.whatsapp.net" || sender != "sender@s.whatsapp.net" || fromMe {
		t.Errorf("Lookup got (%q, %q, %t); want (chat@..., sender@..., false)", chat, sender, fromMe)
	}
}

func TestMessageStoreLookupNotFound(t *testing.T) {
	s := newTestMessageStore(t)
	_, _, _, err := s.Lookup(context.Background(), "DOES-NOT-EXIST")
	if !errors.Is(err, ErrMessageNotFound) {
		t.Errorf("Lookup miss returned %v, want ErrMessageNotFound", err)
	}
}

func TestMessageStorePutIsIdempotent(t *testing.T) {
	// Duplicate inserts must not error: whatsmeow can re-deliver
	// messages on reconnect, and the outbound-then-echo cycle hits
	// Put twice for the same id (once at send, once on the inbound
	// echo we don't ignore).
	ctx := context.Background()
	s := newTestMessageStore(t)
	if err := s.Put(ctx, "MSG1", "chat", "us", true); err != nil {
		t.Fatalf("first Put: %v", err)
	}
	// The conflict path must NOT overwrite the existing row's
	// from_me — re-delivery from the same sender shouldn't downgrade
	// "ours" to "theirs".  Try a "different" payload and verify the
	// first one wins.
	if err := s.Put(ctx, "MSG1", "chat2", "them", false); err != nil {
		t.Fatalf("second Put: %v", err)
	}
	chat, sender, fromMe, err := s.Lookup(ctx, "MSG1")
	if err != nil {
		t.Fatalf("Lookup: %v", err)
	}
	if chat != "chat" || sender != "us" || !fromMe {
		t.Errorf("idempotent Put: row was overwritten — got (%q, %q, %t)", chat, sender, fromMe)
	}
}

package wameow

import "testing"

func TestMarkInboundUnreadAppendsPerChat(t *testing.T) {
	c := &Client{log: discardLogger()}
	c.markInboundUnread("chat1", "msg1", "alice@s.whatsapp.net")
	c.markInboundUnread("chat1", "msg2", "alice@s.whatsapp.net")
	c.markInboundUnread("chat2", "msg3", "bob@s.whatsapp.net")

	got := c.drainUnread("chat1")
	if len(got) != 2 || got[0].id != "msg1" || got[1].id != "msg2" {
		t.Errorf("chat1 unread = %v, want [msg1 msg2]", got)
	}
	got = c.drainUnread("chat2")
	if len(got) != 1 || got[0].id != "msg3" {
		t.Errorf("chat2 unread = %v, want [msg3]", got)
	}
}

func TestDrainUnreadIsOneShot(t *testing.T) {
	// After draining, the queue must be empty so a second send to the
	// same chat doesn't re-mark the same messages as read.
	c := &Client{log: discardLogger()}
	c.markInboundUnread("chat1", "msg1", "alice@s.whatsapp.net")

	_ = c.drainUnread("chat1")
	again := c.drainUnread("chat1")
	if len(again) != 0 {
		t.Errorf("second drain returned %v, want empty", again)
	}
}

func TestDrainUnreadOnUnknownChatIsNil(t *testing.T) {
	c := &Client{log: discardLogger()}
	if got := c.drainUnread("never-touched"); got != nil {
		t.Errorf("drain of untouched chat = %v, want nil", got)
	}
}

func TestDrainUnreadBeforeAnyMarkIsSafe(t *testing.T) {
	// The unread map is lazy-initialized; drainUnread before any
	// markInboundUnread must not panic on the nil map.
	c := &Client{log: discardLogger()}
	if got := c.drainUnread("any"); got != nil {
		t.Errorf("drain on nil-map client = %v, want nil", got)
	}
}

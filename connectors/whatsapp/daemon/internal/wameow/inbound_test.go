package wameow

import (
	"context"
	"log/slog"
	"testing"
	"time"

	"go.mau.fi/whatsmeow/proto/waE2E"
	"go.mau.fi/whatsmeow/types"
	"go.mau.fi/whatsmeow/types/events"
	"google.golang.org/protobuf/proto"
)

func TestChatTypeFromJID(t *testing.T) {
	cases := []struct {
		jid  types.JID
		want string
	}{
		{types.NewJID("15551234567", types.DefaultUserServer), "dm"},
		{types.NewJID("15551234567", types.HiddenUserServer), "dm"},
		{types.NewJID("111222333", types.GroupServer), "group"},
		{types.NewJID("12345", types.BroadcastServer), "broadcast"},
		{types.NewJID("99999", types.NewsletterServer), "newsletter"},
		{types.NewJID("bot", types.BotServer), "unknown"},
	}
	for _, tc := range cases {
		got := chatTypeFromJID(tc.jid)
		if got != tc.want {
			t.Errorf("chatTypeFromJID(%s) = %q, want %q", tc.jid, got, tc.want)
		}
	}
}

func TestExtractTextFromConversation(t *testing.T) {
	msg := &waE2E.Message{Conversation: proto.String("hello world")}
	if got := extractText(msg); got != "hello world" {
		t.Errorf("extractText conversation = %q, want 'hello world'", got)
	}
}

func TestExtractTextFromExtendedTextMessage(t *testing.T) {
	msg := &waE2E.Message{
		ExtendedTextMessage: &waE2E.ExtendedTextMessage{
			Text: proto.String("formatted *bold*"),
		},
	}
	if got := extractText(msg); got != "formatted *bold*" {
		t.Errorf("extractText extended = %q, want 'formatted *bold*'", got)
	}
}

func TestExtractTextReturnsEmptyForUnsupportedShape(t *testing.T) {
	// Attachment-without-caption: empty text is the contract.  Sticker
	// and audio messages don't carry a caption surface, so they
	// always return empty here.
	msg := &waE2E.Message{ImageMessage: &waE2E.ImageMessage{}}
	if got := extractText(msg); got != "" {
		t.Errorf("extractText image-no-caption = %q, want empty", got)
	}
	msg = &waE2E.Message{AudioMessage: &waE2E.AudioMessage{}}
	if got := extractText(msg); got != "" {
		t.Errorf("extractText audio = %q, want empty", got)
	}
}

func TestExtractTextFromImageCaption(t *testing.T) {
	// Pre-fix this returned "" — extractText didn't reach into
	// ImageMessage.Caption, so an image with a caption surfaced to
	// the model as a caption-less attachment with text="".
	msg := &waE2E.Message{
		ImageMessage: &waE2E.ImageMessage{
			Caption: proto.String("look at this — meeting moved to 3pm"),
		},
	}
	if got := extractText(msg); got != "look at this — meeting moved to 3pm" {
		t.Errorf("extractText image-caption = %q, want the caption", got)
	}
}

func TestExtractTextFromVideoCaption(t *testing.T) {
	msg := &waE2E.Message{
		VideoMessage: &waE2E.VideoMessage{Caption: proto.String("clip from yesterday")},
	}
	if got := extractText(msg); got != "clip from yesterday" {
		t.Errorf("extractText video-caption = %q, want the caption", got)
	}
}

func TestExtractTextFromDocumentCaption(t *testing.T) {
	msg := &waE2E.Message{
		DocumentMessage: &waE2E.DocumentMessage{Caption: proto.String("Q3 numbers")},
	}
	if got := extractText(msg); got != "Q3 numbers" {
		t.Errorf("extractText document-caption = %q, want the caption", got)
	}
}

func TestSanitizeFilenameNeutralizesDodgyChars(t *testing.T) {
	cases := []struct{ in, want string }{
		{"normal.pdf", "normal.pdf"},
		{"with/slash.pdf", "with_slash.pdf"},
		{`with\back.pdf`, "with_back.pdf"},
		{".hidden", "hidden"},
		{"with\x00null.pdf", "with_null.pdf"},
		{"with\nnewline.pdf", "with_newline.pdf"},
		{"with\ttab.pdf", "with_tab.pdf"},
		{"\x7fdel.pdf", "_del.pdf"},
		{"", "unnamed"},
		{"...", "unnamed"},
	}
	for _, tc := range cases {
		if got := sanitizeFilename(tc.in); got != tc.want {
			t.Errorf("sanitizeFilename(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
}

func TestRecordInboundEnqueuesUnreadForPeer(t *testing.T) {
	// Peer-sent message (is_self=false) on a DM chat enqueues an
	// unread entry so the next outbound to that chat marks it read.
	c := &Client{
		log:         discardLogger(),
		msgs:        newTestMessageStore(t),
		lifetimeCtx: context.Background(),
	}
	peer := types.NewJID("15553334444", types.DefaultUserServer)
	c.recordInbound(&events.Message{
		Info: types.MessageInfo{
			MessageSource: types.MessageSource{Chat: peer, Sender: peer, IsFromMe: false},
			ID:            "INBOUND-1",
		},
		Message: &waE2E.Message{Conversation: proto.String("hi")},
	})

	got := c.drainUnread(peer.String())
	if len(got) != 1 || got[0].id != "INBOUND-1" || got[0].sender != peer.String() {
		t.Errorf("expected one unread entry, got %v", got)
	}
}

func TestRecordInboundSkipsUnreadForReactionOnly(t *testing.T) {
	// Pre-fix: reaction-only inbounds entered the unread queue,
	// causing the next outbound to MarkRead a reaction id which
	// whatsmeow rejects with a noisy mark_read_failed warning.
	// Post-fix: reaction envelopes are tracked in msgstore but not
	// in the unread queue.
	c := &Client{
		log:         discardLogger(),
		msgs:        newTestMessageStore(t),
		lifetimeCtx: context.Background(),
	}
	peer := types.NewJID("15553334444", types.DefaultUserServer)
	c.recordInbound(&events.Message{
		Info: types.MessageInfo{
			MessageSource: types.MessageSource{Chat: peer, Sender: peer, IsFromMe: false},
			ID:            "REACTION-1",
		},
		Message: &waE2E.Message{ReactionMessage: &waE2E.ReactionMessage{}},
	})

	if got := c.drainUnread(peer.String()); got != nil {
		t.Errorf("reaction-only inbound enqueued unread: %v", got)
	}
}

func TestRecordInboundSkipsUnreadForOwnEchoes(t *testing.T) {
	// Our own outbound messages echo back through *events.Message
	// with is_self=true.  Those aren't "unread" by us; the unread
	// queue should ignore them so a self-echo doesn't trigger a
	// MarkRead on the bot's own send.
	c := &Client{
		log:         discardLogger(),
		msgs:        newTestMessageStore(t),
		lifetimeCtx: context.Background(),
	}
	peer := types.NewJID("15553334444", types.DefaultUserServer)
	c.recordInbound(&events.Message{
		Info: types.MessageInfo{
			MessageSource: types.MessageSource{Chat: peer, Sender: peer, IsFromMe: true},
			ID:            "OUR-ECHO",
		},
		Message: &waE2E.Message{Conversation: proto.String("hi")},
	})

	if got := c.drainUnread(peer.String()); got != nil {
		t.Errorf("self-echo unexpectedly enqueued: %v", got)
	}
}

func TestIsPureProtocolMessage(t *testing.T) {
	// Real user-facing message — even with a ProtocolMessage rider —
	// must NOT be filtered out of msgstore writes.
	withContent := &waE2E.Message{
		Conversation:    proto.String("hello"),
		ProtocolMessage: &waE2E.ProtocolMessage{},
	}
	if isPureProtocolMessage(withContent) {
		t.Error("message with Conversation should not be treated as pure protocol")
	}
	// Pure protocol envelope (history sync, key share, edit-only) —
	// recordInbound should skip these to avoid msgstore growing
	// with rows the model can never target.
	pure := &waE2E.Message{ProtocolMessage: &waE2E.ProtocolMessage{}}
	if !isPureProtocolMessage(pure) {
		t.Error("ProtocolMessage-only envelope should be treated as pure protocol")
	}
	// Reactions are user-facing, even though they ride alone too.
	reaction := &waE2E.Message{ReactionMessage: &waE2E.ReactionMessage{}}
	if isPureProtocolMessage(reaction) {
		t.Error("ReactionMessage should not be treated as pure protocol")
	}
}

func TestExtractTextPrefersConversationOverImageCaption(t *testing.T) {
	// Defensive: if both fields are set (shouldn't happen in
	// practice), Conversation wins — that's the dominant path and
	// what existing tests assume.
	msg := &waE2E.Message{
		Conversation: proto.String("outer text"),
		ImageMessage: &waE2E.ImageMessage{Caption: proto.String("ignored caption")},
	}
	if got := extractText(msg); got != "outer text" {
		t.Errorf("extractText preference = %q, want 'outer text'", got)
	}
}

func TestTranslateMessageDM(t *testing.T) {
	sender := types.NewJID("15553334444", types.DefaultUserServer)
	e := &events.Message{
		Info: types.MessageInfo{
			MessageSource: types.MessageSource{
				Chat:     sender,
				Sender:   sender,
				IsFromMe: false,
			},
			ID:        "3EB0BB36C97D4F8C29A4",
			PushName:  "Alice",
			Timestamp: time.UnixMilli(1700000000000).UTC(),
		},
		Message: &waE2E.Message{Conversation: proto.String("hello bot")},
	}
	got := translateMessage(e)
	if got == nil {
		t.Fatalf("translateMessage returned nil for valid DM")
	}
	if got["id"] != "3EB0BB36C97D4F8C29A4" {
		t.Errorf("id = %v, want '3EB0BB36C97D4F8C29A4'", got["id"])
	}
	if got["timestamp_ms"].(int64) != 1700000000000 {
		t.Errorf("timestamp_ms = %v, want 1700000000000", got["timestamp_ms"])
	}
	if got["from_jid"] != sender.String() {
		t.Errorf("from_jid = %v, want %q", got["from_jid"], sender.String())
	}
	if got["chat_type"] != "dm" {
		t.Errorf("chat_type = %v, want 'dm'", got["chat_type"])
	}
	if got["is_self"] != false {
		t.Errorf("is_self = %v, want false", got["is_self"])
	}
	if got["text"] != "hello bot" {
		t.Errorf("text = %v, want 'hello bot'", got["text"])
	}
	if _, hasChatName := got["chat_name"]; hasChatName {
		t.Errorf("chat_name should be omitted until pairing carries roster info; got %v", got["chat_name"])
	}
}

func TestTranslateMessageGroupSelfEcho(t *testing.T) {
	self := types.NewJID("15551112222", types.DefaultUserServer)
	group := types.NewJID("111222333", types.GroupServer)
	e := &events.Message{
		Info: types.MessageInfo{
			MessageSource: types.MessageSource{
				Chat:     group,
				Sender:   self,
				IsFromMe: true,
				IsGroup:  true,
			},
			ID:        "3EB0OWN",
			PushName:  "Bot",
			Timestamp: time.UnixMilli(1700000001000).UTC(),
		},
		Message: &waE2E.Message{Conversation: proto.String("echo of our send")},
	}
	got := translateMessage(e)
	if got == nil {
		t.Fatalf("translateMessage returned nil")
	}
	if got["chat_type"] != "group" {
		t.Errorf("chat_type = %v, want 'group'", got["chat_type"])
	}
	if got["chat_jid"] != group.String() {
		t.Errorf("chat_jid = %v, want %q", got["chat_jid"], group.String())
	}
	if got["is_self"] != true {
		t.Errorf("is_self = %v, want true (so parse.py drops as self-echo)", got["is_self"])
	}
}

func TestTranslateMessageDropsEmptyEnvelope(t *testing.T) {
	if got := translateMessage(&events.Message{}); got != nil {
		t.Errorf("translateMessage(empty) = %v, want nil", got)
	}
}

// captureNotifier records every Broadcast call for assertions.
type captureNotifier struct {
	calls []notifyCall
}

type notifyCall struct {
	Method string
	Params any
}

func (n *captureNotifier) Broadcast(method string, params any) {
	n.calls = append(n.calls, notifyCall{Method: method, Params: params})
}

func TestHandleEventDispatch(t *testing.T) {
	notify := &captureNotifier{}
	c := &Client{notify: notify, log: discardLogger(), msgs: newTestMessageStore(t), lifetimeCtx: context.Background()}

	sender := types.NewJID("15553334444", types.DefaultUserServer)
	c.handleEvent(&events.Message{
		Info: types.MessageInfo{
			MessageSource: types.MessageSource{Chat: sender, Sender: sender},
			ID:            "ID1",
			Timestamp:     time.UnixMilli(1700000000000).UTC(),
		},
		Message: &waE2E.Message{Conversation: proto.String("hi")},
	})

	c.handleEvent(&events.Connected{})

	// Empty Message envelope → dropped at translate boundary, no broadcast.
	c.handleEvent(&events.Message{})

	if len(notify.calls) != 2 {
		t.Fatalf("got %d broadcasts, want 2 (message + connectionState)", len(notify.calls))
	}
	if notify.calls[0].Method != "message" {
		t.Errorf("call 0 method = %q, want 'message'", notify.calls[0].Method)
	}
	if notify.calls[1].Method != "connectionState" {
		t.Errorf("call 1 method = %q, want 'connectionState'", notify.calls[1].Method)
	}
	params, ok := notify.calls[1].Params.(map[string]any)
	if !ok || params["state"] != "connected" {
		t.Errorf("connectionState params = %#v, want {state:connected}", notify.calls[1].Params)
	}
}

func TestHandleEventLoggedOutOmitsReasonWhenStreamError(t *testing.T) {
	notify := &captureNotifier{}
	c := &Client{notify: notify, log: discardLogger(), msgs: newTestMessageStore(t), lifetimeCtx: context.Background()}

	// OnConnect == false (stream:error path): Reason is the zero value
	// and must not be surfaced to the wire as misleading data.
	c.handleEvent(&events.LoggedOut{OnConnect: false})

	if len(notify.calls) != 1 {
		t.Fatalf("got %d broadcasts, want 1", len(notify.calls))
	}
	if notify.calls[0].Method != "loggedOut" {
		t.Errorf("method = %q, want 'loggedOut'", notify.calls[0].Method)
	}
	params, ok := notify.calls[0].Params.(map[string]any)
	if !ok {
		t.Fatalf("params not a map: %T", notify.calls[0].Params)
	}
	if params["on_connect"] != false {
		t.Errorf("on_connect = %v, want false", params["on_connect"])
	}
	if _, hasReason := params["reason"]; hasReason {
		t.Errorf("reason should be omitted when OnConnect=false; got %v", params["reason"])
	}
}

func discardLogger() *slog.Logger {
	// slog.DiscardHandler short-circuits in Enabled() so the Sprintf
	// gate in slogWaLogger never fires either.
	return slog.New(slog.DiscardHandler)
}

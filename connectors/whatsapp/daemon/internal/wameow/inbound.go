package wameow

import (
	"go.mau.fi/whatsmeow/proto/waE2E"
	"go.mau.fi/whatsmeow/types"
	"go.mau.fi/whatsmeow/types/events"
)

// Notifier is the broadcast sink for daemon-initiated notifications.
// Server in internal/rpc satisfies this via its Broadcast method.
type Notifier interface {
	Broadcast(method string, params any)
}

// handleEvent routes whatsmeow events to wire-protocol notifications.
// Events not listed here are intentionally ignored at the wire boundary
// (receipts, presence, history sync, etc.) — handle in a later PR if
// the model needs them.
func (c *Client) handleEvent(evt any) {
	switch e := evt.(type) {
	case *events.Message:
		if params := translateMessage(e); params != nil {
			c.notify.Broadcast("message", params)
		}
	case *events.Connected:
		c.notify.Broadcast("connectionState", connectionState{State: "connected"})
	case *events.Disconnected:
		c.notify.Broadcast("connectionState", connectionState{State: "disconnected"})
	case *events.LoggedOut:
		c.log.Warn("wameow.logged_out", "on_connect", e.OnConnect, "reason", e.Reason.String())
		c.notify.Broadcast("loggedOut", loggedOut{Reason: e.Reason.String(), OnConnect: e.OnConnect})
	}
}

type connectionState struct {
	State string `json:"state"`
}

type loggedOut struct {
	Reason    string `json:"reason"`
	OnConnect bool   `json:"on_connect"`
}

// translateMessage normalizes a whatsmeow *events.Message into the
// wire-protocol shape Python's parse.py expects.  Returns nil to drop
// at this layer (currently only when the message envelope itself is
// missing; parse.py applies the policy drops on text-empty, is-self,
// broadcast, newsletter).
func translateMessage(e *events.Message) map[string]any {
	if e == nil || e.Message == nil {
		return nil
	}
	return map[string]any{
		"id":             string(e.Info.ID),
		"timestamp_ms":   e.Info.Timestamp.UnixMilli(),
		"from_jid":       e.Info.Sender.String(),
		"from_push_name": e.Info.PushName,
		"chat_jid":       e.Info.Chat.String(),
		"chat_type":      chatTypeFromJID(e.Info.Chat),
		"chat_name":      nil,
		"is_self":        e.Info.IsFromMe,
		"text":           extractText(e.Message),
	}
}

// extractText pulls the user-visible text out of the whatsmeow message
// envelope.  Plain texts arrive as Conversation; texts with formatting,
// mentions, or quote-context arrive wrapped in ExtendedTextMessage.
// Media captions are NOT extracted here — PR 5 handles attachments
// (caption ships alongside the media blob, not as a free text).
func extractText(msg *waE2E.Message) string {
	if conv := msg.GetConversation(); conv != "" {
		return conv
	}
	if ext := msg.GetExtendedTextMessage(); ext != nil {
		return ext.GetText()
	}
	return ""
}

// chatTypeFromJID maps a WhatsApp JID's server suffix to our wire
// vocabulary.  Unknown servers (msgr / interop / hosted / bot / etc.)
// fall through as "unknown" — Python's parse.py drops anything not
// "dm" or "group".
func chatTypeFromJID(j types.JID) string {
	switch j.Server {
	case types.DefaultUserServer, types.HiddenUserServer:
		return "dm"
	case types.GroupServer:
		return "group"
	case types.BroadcastServer:
		return "broadcast"
	case types.NewsletterServer:
		return "newsletter"
	default:
		return "unknown"
	}
}

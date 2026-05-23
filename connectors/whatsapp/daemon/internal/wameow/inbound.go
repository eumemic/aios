package wameow

import (
	"go.mau.fi/whatsmeow/proto/waE2E"
	"go.mau.fi/whatsmeow/types"
	"go.mau.fi/whatsmeow/types/events"
)

// Notifier is the broadcast sink for daemon-initiated notifications.
type Notifier interface {
	Broadcast(method string, params any)
}

func (c *Client) handleEvent(evt any) {
	switch e := evt.(type) {
	case *events.Message:
		if params := translateMessage(e); params != nil {
			c.notify.Broadcast("message", params)
		}
	case *events.Connected:
		c.notify.Broadcast("connectionState", map[string]any{"state": "connected"})
	case *events.Disconnected:
		c.notify.Broadcast("connectionState", map[string]any{"state": "disconnected"})
	case *events.LoggedOut:
		c.log.Warn("wameow.logged_out", "on_connect", e.OnConnect, "reason", e.Reason.String())
		// Per whatsmeow contract, Reason is only populated when
		// OnConnect == true (connect failure); the stream:error path
		// has no reason code.
		params := map[string]any{"on_connect": e.OnConnect}
		if e.OnConnect {
			params["reason"] = e.Reason.String()
		}
		c.notify.Broadcast("loggedOut", params)
	}
}

// translateMessage normalizes a whatsmeow *events.Message into the
// wire-protocol shape Python's parse.py expects.  Returns nil to drop
// at this layer (currently only when the inner message envelope is
// missing; parse.py applies the policy drops on text-empty, is-self,
// broadcast, newsletter).
func translateMessage(e *events.Message) map[string]any {
	if e.Message == nil {
		return nil
	}
	return map[string]any{
		"id":             string(e.Info.ID),
		"timestamp_ms":   e.Info.Timestamp.UnixMilli(),
		"from_jid":       e.Info.Sender.String(),
		"from_push_name": e.Info.PushName,
		"chat_jid":       e.Info.Chat.String(),
		"chat_type":      chatTypeFromJID(e.Info.Chat),
		"is_self":        e.Info.IsFromMe,
		"text":           extractText(e.Message),
	}
}

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
// vocabulary.  Servers downstream doesn't currently route ("unknown")
// fall through; parse.py drops anything that isn't "dm" or "group".
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

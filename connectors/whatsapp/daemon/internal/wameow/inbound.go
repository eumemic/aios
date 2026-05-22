package wameow

import (
	"context"

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
		if e.Message != nil {
			c.recordInbound(e)
		}
		if params := c.translateMessageWithMedia(e); params != nil {
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

// translateMessageWithMedia layers extracted attachments, sticker
// emoji, and reaction metadata on top of the base translation.  Media
// download is done inline (in whatsmeow's event-handler goroutine) —
// the broadcast of THIS message waits for the download, but other
// events on the same connection continue in parallel since whatsmeow's
// dispatcher fans out per-event.  A download failure logs but doesn't
// drop the message: the text/caption alone is still useful.
func (c *Client) translateMessageWithMedia(e *events.Message) map[string]any {
	params := translateMessage(e)
	if params == nil {
		return nil
	}
	if attachment, err := c.extractAndDownloadMedia(
		context.Background(), c.wa.Load(), string(e.Info.ID), e.Message, c.mediaDir,
	); err != nil {
		c.log.Warn("wameow.media_download_failed", "id", e.Info.ID, "err", err)
	} else if attachment != nil {
		params["attachments"] = []*MediaAttachment{attachment}
	}
	if emoji := extractStickerEmoji(e.Message); emoji != "" {
		params["sticker_emoji"] = emoji
	}
	if rxn := extractReaction(e.Message); rxn != nil {
		params["reaction"] = rxn
	}
	if info := extractProtocolInfo(e.Message); info != nil {
		switch {
		case info.edited:
			params["edit"] = map[string]any{"target_message_id": info.targetID}
			// Edits ship a fresh body in ProtocolMessage.EditedMessage;
			// the outer Conversation/ExtendedTextMessage are empty, so
			// override the base text with the inner content.
			if info.newText != "" {
				params["text"] = info.newText
			}
		case info.revoked:
			params["revoke"] = map[string]any{"target_message_id": info.targetID}
		}
	}
	return params
}

// extractProtocolInfo decodes the ProtocolMessage that whatsmeow uses
// for in-band edits and revokes.  Returns nil when this isn't a
// protocol message (the common case), or for ProtocolMessage types
// the daemon doesn't surface yet (history sync, app-state sync, etc).
type protocolInfo struct {
	edited   bool
	revoked  bool
	targetID string
	newText  string // populated for edits only
}

func extractProtocolInfo(m *waE2E.Message) *protocolInfo {
	if m == nil || m.ProtocolMessage == nil {
		return nil
	}
	pm := m.ProtocolMessage
	targetID := ""
	if key := pm.GetKey(); key != nil {
		targetID = key.GetID()
	}
	switch pm.GetType() {
	case waE2E.ProtocolMessage_MESSAGE_EDIT:
		info := &protocolInfo{edited: true, targetID: targetID}
		if edited := pm.GetEditedMessage(); edited != nil {
			info.newText = extractText(edited)
		}
		return info
	case waE2E.ProtocolMessage_REVOKE:
		return &protocolInfo{revoked: true, targetID: targetID}
	}
	return nil
}

// extractReaction surfaces a peer's reaction to a message as a
// metadata block — emoji + the target message_id the reaction
// applies to.  Empty emoji means the peer removed an earlier
// reaction; the model needs to see that explicitly so it can update
// any "they reacted with X" state.
func extractReaction(m *waE2E.Message) map[string]any {
	if m == nil || m.ReactionMessage == nil {
		return nil
	}
	rxn := m.ReactionMessage
	out := map[string]any{
		"emoji": rxn.GetText(),
	}
	if key := rxn.GetKey(); key != nil {
		out["target_message_id"] = key.GetID()
	}
	return out
}

// recordInbound stamps a received message into the message store so
// the model can later react to / edit / revoke it by id.  Best-effort:
// a store failure is logged but doesn't block the broadcast.
//
// Use a background context — handleEvent doesn't get one from whatsmeow,
// and the put is tiny.
func (c *Client) recordInbound(e *events.Message) {
	err := c.msgs.Put(
		context.Background(),
		string(e.Info.ID),
		e.Info.Chat.String(),
		e.Info.Sender.String(),
		e.Info.IsFromMe,
	)
	if err != nil {
		c.log.Warn("wameow.msgstore_put_failed", "id", e.Info.ID, "err", err)
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

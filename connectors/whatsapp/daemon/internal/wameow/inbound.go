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
		if e.Message != nil {
			c.recordInbound(e)
		}
		if params := c.translateMessageWithMedia(e); params != nil {
			c.notify.Broadcast("message", params)
		}
	case *events.UndecryptableMessage:
		// Peer sent us a message we couldn't decrypt — usually a stale
		// Signal session from before a re-pair, or a key-rotation
		// race.  whatsmeow has already requested a retry on its end.
		// Surface this loudly so operators see the failure mode
		// rather than wondering why a known-good chat went silent.
		c.log.Warn("wameow.undecryptable_message", "from_jid", e.Info.Sender.String(), "id", string(e.Info.ID))
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
		// whatsmeow's LoggedOut handler internally marks the device as
		// Deleted (same path as our operator-initiated Unpair).  Swap
		// in a fresh Client so a subsequent StartPairing on this same
		// daemon process works without a restart — symmetric with the
		// Unpair side at pair.go:155.
		c.replaceWhatsmeowClient()
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
	params := map[string]any{
		"id":             string(e.Info.ID),
		"timestamp_ms":   e.Info.Timestamp.UnixMilli(),
		"from_jid":       e.Info.Sender.String(),
		"from_push_name": e.Info.PushName,
		"chat_jid":       e.Info.Chat.String(),
		"chat_type":      chatTypeFromJID(e.Info.Chat),
		"is_self":        e.Info.IsFromMe,
		"text":           extractText(e.Message),
	}
	if mentioned := extractMentions(e.Message); len(mentioned) > 0 {
		params["mentioned_jids"] = mentioned
	}
	return params
}

// extractMentions returns the list of JIDs the peer's message
// addresses by @-tag.  Pulled from whatever submessage carries a
// ContextInfo (text, image caption, video caption, etc.).
func extractMentions(m *waE2E.Message) []string {
	ctx := messageContextInfo(m)
	if ctx == nil {
		return nil
	}
	return ctx.GetMentionedJID()
}

// extractQuotedMessageID returns the StanzaID of the message the peer is
// quoting/replying to, or "" when this isn't a reply.  Source is the same
// ContextInfo as :func:`extractMentions`: WhatsApp threads the quoted-
// message id on the new envelope so peers can render the "↪ Replying to"
// bubble without re-sending the original content.
func extractQuotedMessageID(m *waE2E.Message) string {
	ctx := messageContextInfo(m)
	if ctx == nil {
		return ""
	}
	return ctx.GetStanzaID()
}

// messageContextInfo walks the submessage variants that can carry a
// ContextInfo (text, captioned media, audio) and returns the populated
// one, or nil when the envelope has none.  Centralises the per-variant
// switch shared by extractMentions and extractQuotedMessageID.
func messageContextInfo(m *waE2E.Message) *waE2E.ContextInfo {
	if m == nil {
		return nil
	}
	switch {
	case m.ExtendedTextMessage != nil:
		return m.ExtendedTextMessage.GetContextInfo()
	case m.ImageMessage != nil:
		return m.ImageMessage.GetContextInfo()
	case m.VideoMessage != nil:
		return m.VideoMessage.GetContextInfo()
	case m.DocumentMessage != nil:
		return m.DocumentMessage.GetContextInfo()
	case m.AudioMessage != nil:
		return m.AudioMessage.GetContextInfo()
	}
	return nil
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
		c.lifetimeCtx, c.wa.Load(), string(e.Info.ID), e.Message, c.mediaDir,
	); err != nil {
		c.log.Warn("wameow.media_download_failed", "id", e.Info.ID, "err", err)
	} else if attachment != nil {
		params["attachments"] = []*MediaAttachment{attachment}
	}
	// Stickers: always emit when StickerMessage is present, even when
	// the emoji label is empty.  Custom stickers (made via WhatsApp's
	// sticker maker) frequently carry no emoji tag — without an explicit
	// signal here the parse.py "no signal" drop would silently swallow
	// the message and the model would never know the peer sent a sticker.
	if e.Message != nil && e.Message.StickerMessage != nil {
		params["sticker_emoji"] = extractStickerEmoji(e.Message)
	}
	if rxn := extractReaction(e.Message); rxn != nil {
		params["reaction"] = rxn
	}
	if edit := c.extractEditInfo(e); edit != nil {
		params["edit"] = map[string]any{"target_message_id": edit.targetID}
		// Edits ship a fresh body either in
		// ProtocolMessage.EditedMessage (legacy) or in the decrypted
		// SecretEncryptedMessage payload (newer encrypted edits) —
		// either way the outer Conversation/ExtendedTextMessage is
		// empty, so override the base text with the inner content.
		if edit.newText != "" {
			params["text"] = edit.newText
		}
	} else if target := extractRevokeTargetID(e.Message); target != "" {
		params["revoke"] = map[string]any{"target_message_id": target}
	}
	if quoted := extractQuotedMessageID(e.Message); quoted != "" {
		params["quoted_message_id"] = quoted
	}
	return params
}

// editInfo carries the target message id + new body text of a peer's
// edit, regardless of which on-the-wire envelope shape the edit used.
type editInfo struct {
	targetID string
	newText  string // may be empty if the edit's encrypted payload couldn't be decoded
}

// extractEditInfo decodes a peer's edit envelope.  WhatsApp delivers
// edits over TWO distinct wire shapes:
//
//   - LEGACY: ProtocolMessage{Type: MESSAGE_EDIT, EditedMessage: …}
//     with the new body in plaintext inside the protobuf.
//
//   - CURRENT: SecretEncryptedMessage{SecretEncType: MESSAGE_EDIT,
//     TargetMessageKey: …, EncPayload: …} — the new body is encrypted
//     and must be decrypted via :func:`whatsmeow.Client.DecryptSecretEncryptedMessage`
//     against the recipient-derived msg-secret key.
//
// Returns nil when the envelope is neither shape.  When the
// SecretEncryptedMessage decrypt fails (key rotation race, stale
// session, etc.), the returned editInfo still carries the targetID so
// the model knows an edit happened on a known message even if it can't
// see the new content.
func (c *Client) extractEditInfo(evt *events.Message) *editInfo {
	if evt == nil || evt.Message == nil {
		return nil
	}
	msg := evt.Message
	// Legacy: ProtocolMessage envelope.
	if pm := msg.GetProtocolMessage(); pm != nil && pm.GetType() == waE2E.ProtocolMessage_MESSAGE_EDIT {
		// Same rationale as the SecretEncryptedMessage branch below:
		// an edit envelope without a target id is meaningless to the
		// model, so drop it rather than emit an orphan signal.
		key := pm.GetKey()
		if key == nil || key.GetID() == "" {
			c.log.Warn("wameow.legacy_edit_missing_target", "id", string(evt.Info.ID))
			return nil
		}
		out := &editInfo{targetID: key.GetID()}
		if edited := pm.GetEditedMessage(); edited != nil {
			out.newText = extractText(edited)
		}
		return out
	}
	// Current: SecretEncryptedMessage envelope.
	if enc := msg.GetSecretEncryptedMessage(); enc != nil && enc.GetSecretEncType() == waE2E.SecretEncryptedMessage_MESSAGE_EDIT {
		// A malformed envelope without TargetMessageKey would leave us
		// emitting an edit signal with no target the model can match
		// against any prior message_id — return nil so the inbound
		// flows the no-signal drop path instead of confusing the
		// model with an orphan edit.
		key := enc.GetTargetMessageKey()
		if key == nil || key.GetID() == "" {
			c.log.Warn("wameow.secret_edit_missing_target", "id", string(evt.Info.ID))
			return nil
		}
		out := &editInfo{targetID: key.GetID()}
		decrypted, err := c.wa.Load().DecryptSecretEncryptedMessage(c.lifetimeCtx, evt)
		if err != nil {
			c.log.Warn(
				"wameow.decrypt_secret_edit_failed",
				"id", string(evt.Info.ID),
				"target_id", out.targetID,
				"err", err,
			)
			return out
		}
		// The decrypted plaintext is itself wrapped as a
		// ProtocolMessage{Type: MESSAGE_EDIT, EditedMessage: {…}} —
		// same shape as the legacy path's outer envelope, just one
		// level deeper.  Unwrap to reach the new body; fall back to
		// the top-level Message for forward-compatibility if a future
		// whatsmeow version flattens the structure.
		if pm := decrypted.GetProtocolMessage(); pm != nil && pm.GetType() == waE2E.ProtocolMessage_MESSAGE_EDIT {
			if edited := pm.GetEditedMessage(); edited != nil {
				out.newText = extractText(edited)
			}
		} else {
			out.newText = extractText(decrypted)
		}
		return out
	}
	return nil
}

// extractRevokeTargetID returns the id of the message the peer is
// revoking ("Delete for everyone"), or "" when this envelope isn't a
// revoke.  Revokes are always delivered via the legacy ProtocolMessage
// envelope (whatsmeow's SecretEncryptedMessage path doesn't carry a
// MESSAGE_REVOKE secret-enc-type as of v0.0.0-20260516…).
func extractRevokeTargetID(m *waE2E.Message) string {
	if m == nil || m.ProtocolMessage == nil {
		return ""
	}
	pm := m.ProtocolMessage
	if pm.GetType() != waE2E.ProtocolMessage_REVOKE {
		return ""
	}
	if key := pm.GetKey(); key != nil {
		return key.GetID()
	}
	return ""
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
// Filters out non-DM/group chat types (broadcasts, newsletters,
// status updates) and pure ProtocolMessage envelopes (history sync,
// app-state sync) — those ids are never user-visible and would just
// grow the store with rows the model can never legitimately target.
//
// Uses the daemon-lifetime context so a shutdown-during-write
// cancels cleanly rather than hitting a torn-down *sql.DB.
func (c *Client) recordInbound(e *events.Message) {
	if !c.isUserVisibleChat(e.Info.Chat) || isPureProtocolMessage(e.Message) {
		return
	}
	err := c.msgs.Put(
		c.lifetimeCtx,
		string(e.Info.ID),
		e.Info.Chat.String(),
		e.Info.Sender.String(),
		e.Info.IsFromMe,
		e.Info.Timestamp.UnixMilli(),
		extractText(e.Message),
	)
	if err != nil {
		c.log.Warn("wameow.msgstore_put_failed", "id", e.Info.ID, "err", err)
	}
	// Track unread peer messages so the next outbound to this chat
	// implicitly marks them read.  is_self echoes of our own sends
	// are not "unread" by us — skip.  Reaction envelopes are
	// acknowledgements, not unread messages: MarkRead'ing a
	// reaction id is the wrong semantic (whatsmeow surfaces the
	// rejection as a noisy mark_read_failed warning), so skip
	// those too.
	if !e.Info.IsFromMe && !isReactionOnly(e.Message) {
		c.markInboundUnread(
			e.Info.Chat.String(),
			string(e.Info.ID),
			e.Info.Sender.String(),
		)
	}
}

// isReactionOnly reports whether the message is exclusively a
// peer-sent reaction (ReactionMessage set, no user-facing text or
// media).  Reactions ride through recordInbound for msgstore
// tracking but must NOT enter the read-receipt queue.
func isReactionOnly(m *waE2E.Message) bool {
	if m == nil || m.ReactionMessage == nil {
		return false
	}
	return m.Conversation == nil &&
		m.ExtendedTextMessage == nil &&
		m.ImageMessage == nil &&
		m.VideoMessage == nil &&
		m.AudioMessage == nil &&
		m.DocumentMessage == nil &&
		m.StickerMessage == nil
}

// isUserVisibleChat reports whether the chat type can carry messages
// the model legitimately interacts with.  DMs and groups qualify;
// broadcasts/newsletters/status/server-status JIDs do not.
func (c *Client) isUserVisibleChat(j types.JID) bool {
	switch j.Server {
	case types.DefaultUserServer, types.HiddenUserServer, types.GroupServer:
		return true
	}
	return false
}

// isPureProtocolMessage reports whether the message body is ONLY a
// ProtocolMessage (history sync, app-state sync, edit/revoke
// envelopes, etc.) with no user-facing content.  Edit-envelope ids
// are intentionally excluded from the messages.db too: the EDIT
// envelope's id is distinct from the original target's id and the
// model only needs the target_message_id (rendered in metadata).
func isPureProtocolMessage(m *waE2E.Message) bool {
	if m == nil || m.ProtocolMessage == nil {
		return false
	}
	// If ANY user-facing content field is also set, treat as a real
	// message that incidentally has a ProtocolMessage rider.
	return m.Conversation == nil &&
		m.ExtendedTextMessage == nil &&
		m.ImageMessage == nil &&
		m.VideoMessage == nil &&
		m.AudioMessage == nil &&
		m.DocumentMessage == nil &&
		m.StickerMessage == nil &&
		m.ReactionMessage == nil
}

// extractText returns the user-visible text body of a message,
// covering both plain text (Conversation / ExtendedTextMessage) and
// the Caption fields on Image/Video/Document attachments.  Sticker
// and audio messages have no caption surface.
//
// Pre-fix this missed media captions entirely, so an inbound image
// "Look at this — meeting moved to 3pm" arrived at the model as a
// caption-less image (text="").
func extractText(msg *waE2E.Message) string {
	if conv := msg.GetConversation(); conv != "" {
		return conv
	}
	if ext := msg.GetExtendedTextMessage(); ext != nil {
		if t := ext.GetText(); t != "" {
			return t
		}
	}
	if img := msg.GetImageMessage(); img != nil {
		if c := img.GetCaption(); c != "" {
			return c
		}
	}
	if vid := msg.GetVideoMessage(); vid != nil {
		if c := vid.GetCaption(); c != "" {
			return c
		}
	}
	if doc := msg.GetDocumentMessage(); doc != nil {
		if c := doc.GetCaption(); c != "" {
			return c
		}
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

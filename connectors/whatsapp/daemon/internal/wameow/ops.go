package wameow

import (
	"context"
	"errors"
	"fmt"

	"go.mau.fi/whatsmeow/types"
	"google.golang.org/protobuf/proto"
)

// IsNotFoundErr reports whether err is ErrMessageNotFound (the
// daemon-side "I have no MessageKey for that id" signal).  Used by
// the handler layer to map this case to ErrCodeInvalidParams so the
// caller can distinguish it from a generic server failure.
func (c *Client) IsNotFoundErr(err error) bool {
	return errors.Is(err, ErrMessageNotFound)
}

// IsNotOwnMessageErr reports whether err is ErrNotOwnMessage (the
// "target message exists but was sent by a peer, not by us"
// precondition refusal for Edit/Revoke).  Same handler treatment as
// IsNotFoundErr: maps to ErrCodeInvalidParams so the model sees a
// clear "wrong target" rather than a generic server error.
func (c *Client) IsNotOwnMessageErr(err error) bool {
	return errors.Is(err, ErrNotOwnMessage)
}

// React sends (or clears) a reaction to a message we've previously
// seen.  Passing an empty reaction string clears any prior reaction
// from us on that message — that's whatsmeow's documented contract
// and matches the WhatsApp client UI ("remove reaction").
//
// Returns the reaction message's own id + timestamp so the caller can
// surface delivery confirmation.  The reaction is itself a sent
// message; its id goes into the message store too, so the model can
// later edit/revoke its own reactions if it wants.
func (c *Client) React(ctx context.Context, msgID, emoji string) (string, int64, error) {
	wa := c.wa.Load()
	if !wa.IsConnected() {
		return "", 0, errors.New("whatsmeow: not connected")
	}
	chatJIDStr, senderJIDStr, fromMe, err := c.msgs.Lookup(ctx, msgID)
	if err != nil {
		return "", 0, err
	}
	chat, err := types.ParseJID(chatJIDStr)
	if err != nil {
		return "", 0, fmt.Errorf("invalid chat jid %q: %w", chatJIDStr, err)
	}
	sender, err := types.ParseJID(senderJIDStr)
	if err != nil {
		return "", 0, fmt.Errorf("invalid sender jid %q: %w", senderJIDStr, err)
	}
	reaction := wa.BuildReaction(chat, sender, types.MessageID(msgID), emoji)
	// BuildReaction internally sets FromMe via the Key — but it reads
	// the fromMe flag off the SenderKeyDistributionMessage path, so
	// we override the inner Key explicitly when reacting to our own
	// messages to keep MessageKey unambiguous.
	if fromMe && reaction.ReactionMessage != nil && reaction.ReactionMessage.Key != nil {
		reaction.ReactionMessage.Key.FromMe = proto.Bool(true)
	}
	resp, err := wa.SendMessage(ctx, chat, reaction)
	if err != nil {
		return "", 0, err
	}
	c.recordOutbound(ctx, wa, string(resp.ID), chat)
	// Reactions are bot-initiated outbound activity to the chat;
	// the prior peer messages should be marked read just like a
	// text reply via sendOne would do.
	c.flushReadReceipts(ctx, wa, chat)
	return string(resp.ID), resp.Timestamp.UnixMilli(), nil
}

// Edit replaces the text of a previously-sent message.  WhatsApp only
// allows editing your own outbound messages and only within ~15
// minutes of the original send; whatsmeow surfaces the server's
// rejection if either window is exceeded.
//
// ``mentionedJIDs`` rides on the new content's ContextInfo so an edit
// that introduces or rewrites an @-mention renders as a pill on the
// peer's WhatsApp UI — without this, edits silently strip the
// MentionedJID list even when the model embeds ``@<E.164>`` in the
// new text.
func (c *Client) Edit(ctx context.Context, msgID, newText string, mentionedJIDs []string) (string, int64, error) {
	if newText == "" {
		// An empty edit blanks the peer's view of the message — that
		// almost certainly isn't the model's intent; reject explicitly
		// rather than silently push a blank bubble.  The model can
		// call whatsapp_delete_message if it wants the message gone.
		return "", 0, errors.New("edit refused: new text is empty (use whatsapp_delete_message to revoke instead)")
	}
	wa := c.wa.Load()
	if !wa.IsConnected() {
		return "", 0, errors.New("whatsmeow: not connected")
	}
	chatJIDStr, _, fromMe, err := c.msgs.Lookup(ctx, msgID)
	if err != nil {
		return "", 0, err
	}
	if !fromMe {
		return "", 0, fmt.Errorf("edit %s: %w", msgID, ErrNotOwnMessage)
	}
	chat, err := types.ParseJID(chatJIDStr)
	if err != nil {
		return "", 0, fmt.Errorf("invalid chat jid %q: %w", chatJIDStr, err)
	}
	newContent := buildTextMessage(newText, mentionedJIDs)
	edit := wa.BuildEdit(chat, types.MessageID(msgID), newContent)
	resp, err := wa.SendMessage(ctx, chat, edit)
	if err != nil {
		return "", 0, err
	}
	c.recordOutbound(ctx, wa, string(resp.ID), chat)
	c.flushReadReceipts(ctx, wa, chat)
	return string(resp.ID), resp.Timestamp.UnixMilli(), nil
}

// Revoke deletes a message ("delete for everyone").  Like Edit, only
// works on our own outbound messages — and whatsmeow's BuildRevoke
// expects that constraint upstream, so we enforce it here rather than
// letting the server reject with an opaque error.
func (c *Client) Revoke(ctx context.Context, msgID string) (string, int64, error) {
	wa := c.wa.Load()
	if !wa.IsConnected() {
		return "", 0, errors.New("whatsmeow: not connected")
	}
	chatJIDStr, senderJIDStr, fromMe, err := c.msgs.Lookup(ctx, msgID)
	if err != nil {
		return "", 0, err
	}
	if !fromMe {
		return "", 0, fmt.Errorf("revoke %s: %w", msgID, ErrNotOwnMessage)
	}
	chat, err := types.ParseJID(chatJIDStr)
	if err != nil {
		return "", 0, fmt.Errorf("invalid chat jid %q: %w", chatJIDStr, err)
	}
	sender, err := types.ParseJID(senderJIDStr)
	if err != nil {
		return "", 0, fmt.Errorf("invalid sender jid %q: %w", senderJIDStr, err)
	}
	revoke := wa.BuildRevoke(chat, sender, types.MessageID(msgID))
	resp, err := wa.SendMessage(ctx, chat, revoke)
	if err != nil {
		return "", 0, err
	}
	c.recordOutbound(ctx, wa, string(resp.ID), chat)
	c.flushReadReceipts(ctx, wa, chat)
	return string(resp.ID), resp.Timestamp.UnixMilli(), nil
}

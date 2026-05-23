package wameow

import (
	"context"
	"errors"
	"fmt"
	"time"

	"go.mau.fi/whatsmeow/types"
	"google.golang.org/protobuf/proto"
)

// EditWindowMs is the maximum age in milliseconds at which WhatsApp's
// servers will still apply an edit.  whatsmeow does NOT surface the
// server's rejection of an out-of-window edit (the server accepts the
// envelope and silently drops the change), so the daemon enforces the
// window client-side: an edit attempt past this age returns
// ErrEditWindowExpired before BuildEdit/SendMessage even runs.
//
// The official WhatsApp limit is 15 minutes; we leave 30 s of slack
// for clock skew between the daemon host and WhatsApp's edge.
const EditWindowMs = 15*60*1000 - 30*1000

// ErrEditWindowExpired is returned by :func:`Client.Edit` when the
// target message is older than :const:`EditWindowMs`.  Surfaced to the
// model as a clear "outside edit window" rather than a misleading
// success — pre-fix the tool returned a stale envelope id and the
// model believed the edit had landed.
var ErrEditWindowExpired = errors.New("edit refused: outside 15-minute window")

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
	rec, err := c.msgs.Lookup(ctx, msgID)
	if err != nil {
		return "", 0, err
	}
	chat, err := types.ParseJID(rec.ChatJID)
	if err != nil {
		return "", 0, fmt.Errorf("invalid chat jid %q: %w", rec.ChatJID, err)
	}
	sender, err := types.ParseJID(rec.SenderJID)
	if err != nil {
		return "", 0, fmt.Errorf("invalid sender jid %q: %w", rec.SenderJID, err)
	}
	reaction := wa.BuildReaction(chat, sender, types.MessageID(msgID), emoji)
	// BuildReaction internally sets FromMe via the Key — but it reads
	// the fromMe flag off the SenderKeyDistributionMessage path, so
	// we override the inner Key explicitly when reacting to our own
	// messages to keep MessageKey unambiguous.
	if rec.FromMe && reaction.ReactionMessage != nil && reaction.ReactionMessage.Key != nil {
		reaction.ReactionMessage.Key.FromMe = proto.Bool(true)
	}
	resp, err := wa.SendMessage(ctx, chat, reaction)
	if err != nil {
		return "", 0, err
	}
	c.recordOutbound(ctx, wa, string(resp.ID), chat, resp.Timestamp.UnixMilli(), emoji)
	// Reactions are bot-initiated outbound activity to the chat;
	// the prior peer messages should be marked read just like a
	// text reply via sendOne would do.
	c.flushReadReceipts(ctx, wa, chat)
	return string(resp.ID), resp.Timestamp.UnixMilli(), nil
}

// Edit replaces the text of a previously-sent message.  WhatsApp only
// allows editing your own outbound messages and only within ~15
// minutes of the original send.  Despite an earlier comment here
// claiming otherwise, whatsmeow does NOT surface the server's silent
// drop of an out-of-window edit — the protocol-message envelope is
// acknowledged at the wire level but the recipient's WhatsApp client
// never renders the change.  We therefore enforce the window
// client-side using the ``sent_at`` we stamped at recordOutbound time:
// an attempt past :const:`EditWindowMs` returns :var:`ErrEditWindowExpired`
// before BuildEdit/SendMessage runs.
//
// ``mentionedJIDs`` rides on the new content's ContextInfo so an edit
// that introduces or rewrites an @-mention renders as a pill on the
// peer's WhatsApp UI — without this, edits silently strip the
// MentionedJID list even when the model embeds ``@<E.164>`` in the
// new text.
//
// Returns the ORIGINAL target ``msgID`` (not the edit-envelope's id) so
// the model can verify which message its edit applied to.  Pre-fix the
// daemon returned the envelope id, which has no meaning to a model
// that only knows the target.
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
	rec, err := c.msgs.Lookup(ctx, msgID)
	if err != nil {
		return "", 0, err
	}
	if !rec.FromMe {
		return "", 0, fmt.Errorf("edit %s: %w", msgID, ErrNotOwnMessage)
	}
	if elapsed := time.Now().UnixMilli() - rec.SentAtMs; rec.SentAtMs > 0 && elapsed > EditWindowMs {
		return "", 0, fmt.Errorf("edit %s: %w (elapsed=%ds)", msgID, ErrEditWindowExpired, elapsed/1000)
	}
	chat, err := types.ParseJID(rec.ChatJID)
	if err != nil {
		return "", 0, fmt.Errorf("invalid chat jid %q: %w", rec.ChatJID, err)
	}
	// Edits don't carry their own quote pointer — the edit replaces
	// the body of an existing message; the original message keeps
	// whatever quote pointer it already had.
	newContent := buildTextMessage(newText, mentionedJIDs, nil)
	edit := wa.BuildEdit(chat, types.MessageID(msgID), newContent)
	resp, err := wa.SendMessage(ctx, chat, edit)
	if err != nil {
		return "", 0, err
	}
	c.recordOutbound(ctx, wa, string(resp.ID), chat, resp.Timestamp.UnixMilli(), newText)
	c.flushReadReceipts(ctx, wa, chat)
	return msgID, resp.Timestamp.UnixMilli(), nil
}

// Revoke deletes a message ("delete for everyone").  Like Edit, only
// works on our own outbound messages — and whatsmeow's BuildRevoke
// expects that constraint upstream, so we enforce it here rather than
// letting the server reject with an opaque error.
//
// Returns the ORIGINAL target ``msgID`` (not the revoke-envelope id)
// so the model can confirm which message was revoked — symmetric with
// :func:`Client.Edit`.  WhatsApp's revoke window is wider than the
// edit window (~2 days) and is left unguarded here; if smoke surfaces
// silently-failing revokes past that, add a guard mirroring
// :var:`ErrEditWindowExpired`.
func (c *Client) Revoke(ctx context.Context, msgID string) (string, int64, error) {
	wa := c.wa.Load()
	if !wa.IsConnected() {
		return "", 0, errors.New("whatsmeow: not connected")
	}
	rec, err := c.msgs.Lookup(ctx, msgID)
	if err != nil {
		return "", 0, err
	}
	if !rec.FromMe {
		return "", 0, fmt.Errorf("revoke %s: %w", msgID, ErrNotOwnMessage)
	}
	chat, err := types.ParseJID(rec.ChatJID)
	if err != nil {
		return "", 0, fmt.Errorf("invalid chat jid %q: %w", rec.ChatJID, err)
	}
	sender, err := types.ParseJID(rec.SenderJID)
	if err != nil {
		return "", 0, fmt.Errorf("invalid sender jid %q: %w", rec.SenderJID, err)
	}
	revoke := wa.BuildRevoke(chat, sender, types.MessageID(msgID))
	resp, err := wa.SendMessage(ctx, chat, revoke)
	if err != nil {
		return "", 0, err
	}
	c.recordOutbound(ctx, wa, string(resp.ID), chat, resp.Timestamp.UnixMilli(), "")
	c.flushReadReceipts(ctx, wa, chat)
	return msgID, resp.Timestamp.UnixMilli(), nil
}

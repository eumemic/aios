package wameow

import (
	"context"
	"errors"
	"fmt"
	"os"
	"strings"

	"go.mau.fi/whatsmeow"
	"go.mau.fi/whatsmeow/proto/waE2E"
	"go.mau.fi/whatsmeow/types"
	"google.golang.org/protobuf/proto"
)

// Attachment is a host-path-referenced media payload bound for one of
// whatsmeow's typed messages.  Kind is derived from Mimetype at send
// time; Filename appears on the wire only for DocumentMessage but the
// daemon carries it unconditionally so the inference layer can decide
// based on full context.
type Attachment struct {
	Path     string
	Mimetype string
	Filename string
}

// SendMessage dispatches text and/or N media attachments.  This is
// the single outbound entry point — handler/send.go's SendMessageFn
// wires to it.
//
// Empty attachments: a single text-only Conversation message.
//
// One or more attachments: WhatsApp has no media-group equivalent,
// so each attachment becomes its own message.  Caption (= text)
// rides on the first attachment only; subsequent attachments arrive
// caption-less.  Exception: WhatsApp's AudioMessage proto has no
// Caption field — when text is non-empty AND the first attachment is
// audio, the text is sent as a separate Conversation message first
// so it isn't silently dropped on the wire.
//
// Returns the FULL slice of delivered message_ids (in send order)
// plus the FIRST send's timestamp.  All-or-nothing semantics are
// impossible at the WhatsApp protocol level (no atomic batch send);
// a mid-loop failure returns a *PartialSendError carrying the ids
// that DID make it onto the wire, so the model can still
// react/edit/revoke each delivered attachment by id.
func (c *Client) SendMessage(
	ctx context.Context,
	jidStr, text string,
	attachments []Attachment,
	mentionedJIDs []string,
) ([]string, int64, error) {
	wa, jid, err := c.prepareSend(jidStr)
	if err != nil {
		return nil, 0, err
	}
	if len(attachments) == 0 {
		msg := buildTextMessage(text, mentionedJIDs)
		id, ts, sendErr := c.sendOne(ctx, wa, jid, msg)
		if sendErr != nil {
			return nil, 0, sendErr
		}
		return []string{id}, ts, nil
	}

	var firstTS int64
	delivered := make([]string, 0, len(attachments)+1)
	captionForFirstAttachment := text
	if text != "" && classify(attachments[0].Mimetype) == attachKindAudio {
		// Audio carries no caption surface.  Send the text as its
		// own Conversation message FIRST so it isn't silently
		// dropped, then send the audio caption-less.
		textMsg := buildTextMessage(text, mentionedJIDs)
		id, ts, sendErr := c.sendOne(ctx, wa, jid, textMsg)
		if sendErr != nil {
			return nil, 0, fmt.Errorf("send accompanying text for audio: %w", sendErr)
		}
		delivered = append(delivered, id)
		firstTS = ts
		captionForFirstAttachment = ""
	}

	for i, att := range attachments {
		caption := ""
		var captionMentions []string
		if i == 0 {
			caption = captionForFirstAttachment
			captionMentions = mentionedJIDs
		}
		msg, buildErr := c.buildAttachmentMessage(ctx, wa, att, caption, captionMentions)
		if buildErr != nil {
			return delivered, firstTS, &PartialSendError{
				Cause: buildErr, DeliveredIDs: append([]string{}, delivered...),
				FailedIndex: i, FailedFilename: att.Filename,
			}
		}
		id, ts, sendErr := c.sendOne(ctx, wa, jid, msg)
		if sendErr != nil {
			return delivered, firstTS, &PartialSendError{
				Cause: sendErr, DeliveredIDs: append([]string{}, delivered...),
				FailedIndex: i, FailedFilename: att.Filename,
			}
		}
		delivered = append(delivered, id)
		if firstTS == 0 {
			firstTS = ts
		}
	}
	return delivered, firstTS, nil
}

// PartialSendError wraps a mid-loop multi-attachment failure with the
// ids that DID make it onto the wire before the loop aborted.  The
// handler layer surfaces these ids via rpc.Error.Data so the model
// can reference the delivered attachments by id (for react/edit/
// revoke) instead of treating the partial delivery as a complete
// loss.
type PartialSendError struct {
	Cause          error
	DeliveredIDs   []string
	FailedIndex    int
	FailedFilename string
}

func (e *PartialSendError) Error() string {
	if len(e.DeliveredIDs) > 0 {
		return fmt.Sprintf(
			"attachment %d (%s): %v (delivered ids before failure: %v)",
			e.FailedIndex, e.FailedFilename, e.Cause, e.DeliveredIDs,
		)
	}
	return fmt.Sprintf("attachment %d (%s): %v", e.FailedIndex, e.FailedFilename, e.Cause)
}

func (e *PartialSendError) Unwrap() error { return e.Cause }

// Partial fulfils the duck-typed interface that handler/send.go
// expects via errors.As, keeping the handler package decoupled from
// the wameow concrete type.
func (e *PartialSendError) Partial() (delivered []string, failedIndex int, failedFilename string) {
	return e.DeliveredIDs, e.FailedIndex, e.FailedFilename
}

func (c *Client) prepareSend(jidStr string) (*whatsmeow.Client, types.JID, error) {
	wa := c.wa.Load()
	if !wa.IsConnected() {
		return nil, types.EmptyJID, errors.New("whatsmeow: not connected")
	}
	jid, err := types.ParseJID(jidStr)
	if err != nil {
		return nil, types.EmptyJID, fmt.Errorf("invalid JID %q: %w", jidStr, err)
	}
	return wa, jid, nil
}

func (c *Client) sendOne(ctx context.Context, wa *whatsmeow.Client, jid types.JID, msg *waE2E.Message) (string, int64, error) {
	resp, err := wa.SendMessage(ctx, jid, msg)
	if err != nil {
		return "", 0, err
	}
	c.recordOutbound(ctx, wa, string(resp.ID), jid)
	return string(resp.ID), resp.Timestamp.UnixMilli(), nil
}

// buildTextMessage shapes a text-only outbound message, attaching a
// ContextInfo.MentionedJID list when the model has @-tagged anyone.
// WhatsApp clients render the mentions as pills if the JIDs resolve
// to chat participants; non-participants fall through as plain text.
func buildTextMessage(text string, mentionedJIDs []string) *waE2E.Message {
	if len(mentionedJIDs) == 0 {
		return &waE2E.Message{Conversation: proto.String(text)}
	}
	return &waE2E.Message{
		ExtendedTextMessage: &waE2E.ExtendedTextMessage{
			Text: proto.String(text),
			ContextInfo: &waE2E.ContextInfo{
				MentionedJID: mentionedJIDs,
			},
		},
	}
}

func (c *Client) buildAttachmentMessage(
	ctx context.Context,
	wa *whatsmeow.Client,
	att Attachment,
	caption string,
	captionMentions []string,
) (*waE2E.Message, error) {
	data, err := os.ReadFile(att.Path)
	if err != nil {
		return nil, fmt.Errorf("read attachment: %w", err)
	}
	kind := classify(att.Mimetype)
	uploaded, err := wa.Upload(ctx, data, mediaTypeForKind(kind))
	if err != nil {
		return nil, fmt.Errorf("upload: %w", err)
	}
	size := uint64(len(data))
	mime := att.Mimetype
	ctxInfo := maybeMentionContextInfo(captionMentions)

	switch kind {
	case attachKindImage:
		return &waE2E.Message{
			ImageMessage: &waE2E.ImageMessage{
				URL:           proto.String(uploaded.URL),
				DirectPath:    proto.String(uploaded.DirectPath),
				Mimetype:      proto.String(mime),
				FileEncSHA256: uploaded.FileEncSHA256,
				FileSHA256:    uploaded.FileSHA256,
				FileLength:    proto.Uint64(size),
				MediaKey:      uploaded.MediaKey,
				Caption:       maybeString(caption),
				ContextInfo:   ctxInfo,
			},
		}, nil
	case attachKindVideo:
		return &waE2E.Message{
			VideoMessage: &waE2E.VideoMessage{
				URL:           proto.String(uploaded.URL),
				DirectPath:    proto.String(uploaded.DirectPath),
				Mimetype:      proto.String(mime),
				FileEncSHA256: uploaded.FileEncSHA256,
				FileSHA256:    uploaded.FileSHA256,
				FileLength:    proto.Uint64(size),
				MediaKey:      uploaded.MediaKey,
				Caption:       maybeString(caption),
				ContextInfo:   ctxInfo,
			},
		}, nil
	case attachKindAudio:
		return &waE2E.Message{
			AudioMessage: &waE2E.AudioMessage{
				URL:           proto.String(uploaded.URL),
				DirectPath:    proto.String(uploaded.DirectPath),
				Mimetype:      proto.String(mime),
				FileEncSHA256: uploaded.FileEncSHA256,
				FileSHA256:    uploaded.FileSHA256,
				FileLength:    proto.Uint64(size),
				MediaKey:      uploaded.MediaKey,
				ContextInfo:   ctxInfo,
			},
		}, nil
	default:
		// Documents carry a filename on the wire; everything else
		// folds back into this default arm too (unknown mimetype).
		return &waE2E.Message{
			DocumentMessage: &waE2E.DocumentMessage{
				URL:           proto.String(uploaded.URL),
				DirectPath:    proto.String(uploaded.DirectPath),
				Mimetype:      proto.String(mime),
				FileEncSHA256: uploaded.FileEncSHA256,
				FileSHA256:    uploaded.FileSHA256,
				FileLength:    proto.Uint64(size),
				MediaKey:      uploaded.MediaKey,
				FileName:      proto.String(att.Filename),
				Caption:       maybeString(caption),
				ContextInfo:   ctxInfo,
			},
		}, nil
	}
}

// maybeMentionContextInfo builds a ContextInfo populated only with
// MentionedJID, or nil when no mentions need to ride on this
// submessage.  Returning nil keeps unrelated ContextInfo fields off
// the wire.
func maybeMentionContextInfo(mentionedJIDs []string) *waE2E.ContextInfo {
	if len(mentionedJIDs) == 0 {
		return nil
	}
	return &waE2E.ContextInfo{MentionedJID: mentionedJIDs}
}

type attachKind int

const (
	attachKindDocument attachKind = iota
	attachKindImage
	attachKindVideo
	attachKindAudio
)

func classify(mimetype string) attachKind {
	switch {
	case strings.HasPrefix(mimetype, "image/"):
		return attachKindImage
	case strings.HasPrefix(mimetype, "video/"):
		return attachKindVideo
	case strings.HasPrefix(mimetype, "audio/"):
		return attachKindAudio
	default:
		return attachKindDocument
	}
}

func mediaTypeForKind(k attachKind) whatsmeow.MediaType {
	switch k {
	case attachKindImage:
		return whatsmeow.MediaImage
	case attachKindVideo:
		return whatsmeow.MediaVideo
	case attachKindAudio:
		return whatsmeow.MediaAudio
	default:
		return whatsmeow.MediaDocument
	}
}

func maybeString(s string) *string {
	if s == "" {
		return nil
	}
	return proto.String(s)
}

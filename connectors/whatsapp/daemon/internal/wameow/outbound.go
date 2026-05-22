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
// caption-less.  Returns the FIRST send's id+timestamp — for the
// caption-bearing path that's the most useful confirmation handle
// for the model.  All-or-nothing semantics are impossible at the
// WhatsApp protocol level (no atomic batch send); a mid-loop
// failure leaves the already-sent attachments delivered and
// surfaces an error that names which attachment failed.
func (c *Client) SendMessage(
	ctx context.Context,
	jidStr, text string,
	attachments []Attachment,
) (string, int64, error) {
	wa, jid, err := c.prepareSend(jidStr)
	if err != nil {
		return "", 0, err
	}
	if len(attachments) == 0 {
		msg := &waE2E.Message{Conversation: proto.String(text)}
		return c.sendOne(ctx, wa, jid, msg)
	}

	var firstID string
	var firstTS int64
	for i, att := range attachments {
		caption := ""
		if i == 0 {
			caption = text
		}
		msg, buildErr := c.buildAttachmentMessage(ctx, wa, att, caption)
		if buildErr != nil {
			return firstID, firstTS, attachmentSendError(i, att.Filename, firstID, buildErr)
		}
		id, ts, sendErr := c.sendOne(ctx, wa, jid, msg)
		if sendErr != nil {
			return firstID, firstTS, attachmentSendError(i, att.Filename, firstID, sendErr)
		}
		if i == 0 {
			firstID, firstTS = id, ts
		}
	}
	return firstID, firstTS, nil
}

func attachmentSendError(index int, filename, deliveredFirstID string, cause error) error {
	if deliveredFirstID != "" {
		return fmt.Errorf("attachment %d (%s): %w (first attachment %s already delivered)", index, filename, cause, deliveredFirstID)
	}
	return fmt.Errorf("attachment %d (%s): %w", index, filename, cause)
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
	c.recordOutbound(ctx, string(resp.ID), jid)
	return string(resp.ID), resp.Timestamp.UnixMilli(), nil
}

func (c *Client) buildAttachmentMessage(
	ctx context.Context,
	wa *whatsmeow.Client,
	att Attachment,
	caption string,
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
			},
		}, nil
	}
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

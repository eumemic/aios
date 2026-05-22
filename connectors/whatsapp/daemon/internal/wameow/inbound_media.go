package wameow

import (
	"context"
	"fmt"
	"mime"
	"os"
	"path/filepath"
	"strings"

	"go.mau.fi/whatsmeow"
	"go.mau.fi/whatsmeow/proto/waE2E"
)

// MediaAttachment is the inbound media bundle the daemon surfaces to
// Python in the ``message`` notification's ``attachments`` array.
// Filename + mimetype reflect what WhatsApp delivered (so the model
// sees the originator's labelling); Path is where the daemon wrote
// the decrypted bytes under <store_dir>/media/.
type MediaAttachment struct {
	Path     string `json:"path"`
	Mimetype string `json:"mimetype"`
	Filename string `json:"filename"`
}

// extractAndDownloadMedia inspects e.Message for a media submessage,
// downloads + decrypts via whatsmeow, and writes the bytes to a path
// the Python side can read.  Returns nil + nil when the message has
// no media (text-only, reaction, edit, etc.); the caller treats that
// as "no attachments to surface".
//
// Sticker messages are intentionally handled by extractStickerEmoji
// rather than downloaded — the emoji label carries the model-useful
// signal at near-zero cost, where the 100-200 KB .webp does not.
func (c *Client) extractAndDownloadMedia(
	ctx context.Context,
	wa *whatsmeow.Client,
	msgID string,
	m *waE2E.Message,
	mediaDir string,
) (*MediaAttachment, error) {
	if m == nil {
		return nil, nil
	}
	var (
		downloadable whatsmeow.DownloadableMessage
		mimetype     string
		filename     string
	)
	switch {
	case m.ImageMessage != nil:
		downloadable = m.ImageMessage
		mimetype = m.ImageMessage.GetMimetype()
	case m.VideoMessage != nil:
		downloadable = m.VideoMessage
		mimetype = m.VideoMessage.GetMimetype()
	case m.AudioMessage != nil:
		downloadable = m.AudioMessage
		mimetype = m.AudioMessage.GetMimetype()
	case m.DocumentMessage != nil:
		downloadable = m.DocumentMessage
		mimetype = m.DocumentMessage.GetMimetype()
		filename = m.DocumentMessage.GetFileName()
	default:
		return nil, nil
	}
	if mimetype == "" {
		mimetype = "application/octet-stream"
	}

	data, err := wa.Download(ctx, downloadable)
	if err != nil {
		return nil, fmt.Errorf("download media for msg %s: %w", msgID, err)
	}

	if err := os.MkdirAll(mediaDir, 0o700); err != nil {
		return nil, fmt.Errorf("mkdir media dir: %w", err)
	}
	// Build the on-disk filename.  Peer-supplied filenames get
	// sandwiched with msgID so two inbounds with the same
	// DocumentMessage.FileName don't clobber each other.  When no
	// peer-supplied name exists, the daemon-built name already
	// includes msgID, so don't double-prefix.
	var diskName string
	if filename == "" {
		diskName = msgID + extensionForMimetype(mimetype)
		filename = diskName // surface this back to the model too
	} else {
		diskName = msgID + "_" + sanitizeFilename(filename)
	}
	diskPath := filepath.Join(mediaDir, diskName)
	// 0o600: media files often contain sensitive content (photos,
	// documents) — match whatsmeow's upstream store.db perms, not
	// the world-readable 0o644 default.
	if err := os.WriteFile(diskPath, data, 0o600); err != nil {
		return nil, fmt.Errorf("write media to %s: %w", diskPath, err)
	}
	return &MediaAttachment{
		Path:     diskPath,
		Mimetype: mimetype,
		Filename: filename,
	}, nil
}

// extractStickerEmoji returns the sticker's emoji label, or "" when
// the message isn't a sticker or carries no emoji.  Per the locked
// design, sticker bytes themselves are intentionally NOT downloaded:
// the emoji is the model-relevant signal (a literal label from the
// WhatsApp sticker picker), and vision tokens on a 512×512 .webp
// would be high-cost for low-signal.
func extractStickerEmoji(m *waE2E.Message) string {
	if m == nil || m.StickerMessage == nil {
		return ""
	}
	return m.StickerMessage.GetEmojis()
}

// sanitizeFilename neutralizes peer-supplied filenames so we can't be
// tricked into writing outside mediaDir, to a hidden file, or to a
// path whose name confuses downstream tooling.
//
// Rules:
// * Path separators (/, \) → "_" (prevent directory traversal).
// * Leading "." stripped (prevent hidden-file write).
// * Control bytes (< 0x20) and NUL → "_" (prevent log/path
//   misparsing; some downstream consumers truncate at NUL).
// * Empty result → "unnamed".
func sanitizeFilename(name string) string {
	var out strings.Builder
	for _, r := range name {
		switch {
		case r == '/' || r == '\\':
			out.WriteByte('_')
		case r < 0x20 || r == 0x7f:
			// C0 controls + DEL
			out.WriteByte('_')
		default:
			out.WriteRune(r)
		}
	}
	cleaned := strings.TrimLeft(out.String(), ".")
	if cleaned == "" {
		return "unnamed"
	}
	return cleaned
}

// extensionForMimetype returns ".jpg" / ".mp4" / etc. for the
// common WhatsApp media mimetypes.  Falls back to ".bin" so the
// disk file always has SOMETHING and the model can still spot the
// path in metadata.
func extensionForMimetype(mimetype string) string {
	// Strip any "; charset=utf-8"-style parameter; mime.ExtensionsByType
	// is parameter-aware but its output ordering changes between
	// platforms, so we hardcode the canonical extension for the
	// types WhatsApp actually emits.
	if i := strings.Index(mimetype, ";"); i != -1 {
		mimetype = strings.TrimSpace(mimetype[:i])
	}
	switch mimetype {
	case "image/jpeg":
		return ".jpg"
	case "image/png":
		return ".png"
	case "image/webp":
		return ".webp"
	case "image/gif":
		return ".gif"
	case "video/mp4":
		return ".mp4"
	case "audio/ogg":
		return ".ogg"
	case "audio/mpeg":
		return ".mp3"
	case "audio/mp4":
		return ".m4a"
	case "application/pdf":
		return ".pdf"
	}
	// Last resort: ask Go's mime package and pick the first ext.
	if exts, err := mime.ExtensionsByType(mimetype); err == nil && len(exts) > 0 {
		return exts[0]
	}
	return ".bin"
}


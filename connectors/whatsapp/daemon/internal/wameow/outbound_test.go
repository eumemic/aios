package wameow

import "testing"

func TestClassifyAudio(t *testing.T) {
	tests := []struct {
		mime string
		want attachKind
	}{
		{"audio/ogg", attachKindAudio},
		{"audio/mpeg", attachKindAudio},
		{"audio/mp4", attachKindAudio},
		{"audio/ogg; codecs=opus", attachKindAudio},
		{"video/mp4", attachKindVideo},
		{"image/jpeg", attachKindImage},
		{"application/pdf", attachKindDocument},
		{"", attachKindDocument}, // empty mime falls through
	}
	for _, tt := range tests {
		if got := classify(tt.mime); got != tt.want {
			t.Errorf("classify(%q) = %v, want %v", tt.mime, got, tt.want)
		}
	}
}

func TestMaybeStringElidesEmpty(t *testing.T) {
	// Empty captions should become nil pointers so the protobuf
	// doesn't carry a Caption field at all (a present-but-empty
	// Caption renders to peers as a literal empty bubble below the
	// media).
	if got := maybeString(""); got != nil {
		t.Errorf("maybeString('') = %v, want nil", got)
	}
	v := maybeString("hi")
	if v == nil || *v != "hi" {
		t.Errorf("maybeString('hi') = %v, want pointer to 'hi'", v)
	}
}

// The full SendMessage flow can't be unit-tested without a live
// whatsmeow.Client (Upload + SendMessage hit network).  The
// audio-caption-routing branch is covered by the live smoke; this
// test pins the classify() decision that drives the branch so a
// future refactor of the mimetype prefix logic doesn't silently
// route audio through the caption-carrying path.
func TestAudioMimePrefixClassification(t *testing.T) {
	// Any mimetype starting with "audio/" must classify as audio,
	// since SendMessage's audio-text-routing check uses classify()
	// to decide whether to pre-send the text as a Conversation
	// message.  Missing this would silently drop captions for any
	// "audio/foo" subtype WhatsApp's clients negotiate.
	for _, mime := range []string{"audio/ogg", "audio/wav", "audio/x-m4a", "audio/aac"} {
		if got := classify(mime); got != attachKindAudio {
			t.Errorf("classify(%q) = %v, want attachKindAudio (caption-routing depends on this)", mime, got)
		}
	}
}

package wameow

import (
	"context"
	"errors"
	"fmt"

	"go.mau.fi/whatsmeow"
	"go.mau.fi/whatsmeow/types"
)

// GroupSummary is the daemon-internal flattening of whatsmeow's
// rich ``types.GroupInfo`` to a wire-friendly shape Python's tools
// can surface to the model.
type GroupSummary struct {
	JID          string                 `json:"jid"`
	Name         string                 `json:"name"`
	Topic        string                 `json:"topic,omitempty"`
	Participants []GroupParticipantInfo `json:"participants"`
}

// GroupParticipantInfo carries the model-relevant fields of a
// whatsmeow participant: who they are and whether they're an admin.
// Other details (LID vs phone JIDs, anonymous display names) stay
// inside whatsmeow's types.
type GroupParticipantInfo struct {
	JID     string `json:"jid"`
	IsAdmin bool   `json:"is_admin"`
}

// ListGroups returns every group the bot is currently a member of.
// Empty slice when not in any groups.
func (c *Client) ListGroups(ctx context.Context) ([]GroupSummary, error) {
	wa := c.wa.Load()
	if !wa.IsConnected() {
		return nil, errors.New("whatsmeow: not connected")
	}
	groups, err := wa.GetJoinedGroups(ctx)
	if err != nil {
		return nil, fmt.Errorf("get joined groups: %w", err)
	}
	out := make([]GroupSummary, 0, len(groups))
	for _, g := range groups {
		out = append(out, summarizeGroup(g))
	}
	return out, nil
}

// CreateGroup creates a new group with the given name and
// participants (full WhatsApp JIDs).  The bot is added implicitly
// by WhatsApp's server.  Returns the resulting group's summary.
func (c *Client) CreateGroup(ctx context.Context, name string, participantJIDs []string) (*GroupSummary, error) {
	wa := c.wa.Load()
	if !wa.IsConnected() {
		return nil, errors.New("whatsmeow: not connected")
	}
	if name == "" {
		return nil, errors.New("create_group: name required")
	}
	parts := make([]types.JID, 0, len(participantJIDs))
	for _, s := range participantJIDs {
		jid, err := types.ParseJID(s)
		if err != nil {
			return nil, fmt.Errorf("invalid participant jid %q: %w", s, err)
		}
		parts = append(parts, jid)
	}
	info, err := wa.CreateGroup(ctx, whatsmeow.ReqCreateGroup{
		Name:         name,
		Participants: parts,
	})
	if err != nil {
		return nil, fmt.Errorf("create group: %w", err)
	}
	summary := summarizeGroup(info)
	return &summary, nil
}

// RenameGroup changes a group's display name.  The bot must be an
// admin; whatsmeow surfaces the server's rejection if not.
func (c *Client) RenameGroup(ctx context.Context, groupJIDStr, newName string) error {
	wa := c.wa.Load()
	if !wa.IsConnected() {
		return errors.New("whatsmeow: not connected")
	}
	if newName == "" {
		return errors.New("rename_group: name required")
	}
	jid, err := types.ParseJID(groupJIDStr)
	if err != nil {
		return fmt.Errorf("invalid group jid %q: %w", groupJIDStr, err)
	}
	if err := wa.SetGroupName(ctx, jid, newName); err != nil {
		return fmt.Errorf("set group name: %w", err)
	}
	return nil
}

func summarizeGroup(g *types.GroupInfo) GroupSummary {
	parts := make([]GroupParticipantInfo, 0, len(g.Participants))
	for _, p := range g.Participants {
		parts = append(parts, GroupParticipantInfo{
			JID:     p.JID.String(),
			IsAdmin: p.IsAdmin || p.IsSuperAdmin,
		})
	}
	return GroupSummary{
		JID:          g.JID.String(),
		Name:         g.Name,
		Topic:        g.Topic,
		Participants: parts,
	}
}

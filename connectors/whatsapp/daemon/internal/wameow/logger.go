package wameow

import (
	"fmt"
	"log/slog"

	waLog "go.mau.fi/whatsmeow/util/log"
)

// slogWaLogger adapts whatsmeow's waLog.Logger interface to slog so
// the daemon emits one consistent log stream rather than two formats
// from whatsmeow's stdout helper and slog mixed.
type slogWaLogger struct {
	log    *slog.Logger
	module string
}

// newWaLogger wraps a slog.Logger so it can be handed to
// whatsmeow.NewClient / sqlstore.New.
func newWaLogger(log *slog.Logger, module string) waLog.Logger {
	return &slogWaLogger{log: log, module: module}
}

func (l *slogWaLogger) Warnf(msg string, args ...any) {
	l.log.Warn(fmt.Sprintf(msg, args...), "wa", l.module)
}

func (l *slogWaLogger) Errorf(msg string, args ...any) {
	l.log.Error(fmt.Sprintf(msg, args...), "wa", l.module)
}

func (l *slogWaLogger) Infof(msg string, args ...any) {
	l.log.Info(fmt.Sprintf(msg, args...), "wa", l.module)
}

func (l *slogWaLogger) Debugf(msg string, args ...any) {
	l.log.Debug(fmt.Sprintf(msg, args...), "wa", l.module)
}

func (l *slogWaLogger) Sub(module string) waLog.Logger {
	return newWaLogger(l.log, l.module+"."+module)
}

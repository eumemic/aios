package wameow

import (
	"context"
	"fmt"
	"log/slog"

	waLog "go.mau.fi/whatsmeow/util/log"
)

// slogWaLogger adapts whatsmeow's waLog.Logger interface to slog so
// the daemon emits one consistent log stream.  Each call gates the
// fmt.Sprintf on the slog level being enabled — whatsmeow logs heavily
// at Debug during history sync, and the format-then-discard pattern
// would otherwise burn CPU for filtered-out lines.
type slogWaLogger struct {
	log    *slog.Logger
	module string
}

func newWaLogger(log *slog.Logger, module string) waLog.Logger {
	return &slogWaLogger{log: log, module: module}
}

func (l *slogWaLogger) emit(level slog.Level, msg string, args ...any) {
	if !l.log.Enabled(context.Background(), level) {
		return
	}
	l.log.Log(context.Background(), level, fmt.Sprintf(msg, args...), "wa", l.module)
}

func (l *slogWaLogger) Warnf(msg string, args ...any)  { l.emit(slog.LevelWarn, msg, args...) }
func (l *slogWaLogger) Errorf(msg string, args ...any) { l.emit(slog.LevelError, msg, args...) }
func (l *slogWaLogger) Infof(msg string, args ...any)  { l.emit(slog.LevelInfo, msg, args...) }
func (l *slogWaLogger) Debugf(msg string, args ...any) { l.emit(slog.LevelDebug, msg, args...) }

func (l *slogWaLogger) Sub(module string) waLog.Logger {
	return newWaLogger(l.log, l.module+"."+module)
}

// Command whatsapp-daemon speaks line-delimited JSON-RPC 2.0 over a
// loopback TCP port to the Python aios_whatsapp connector, which
// spawns it as a subprocess and tears it down on shutdown.  It owns
// the whatsmeow WhatsApp client + sqlstore; the Python side owns aios
// integration (session routing, attachment marshalling, tool dispatch).
package main

import (
	"context"
	"flag"
	"fmt"
	"log/slog"
	"os"
	"os/signal"
	"syscall"

	"aios.dev/connectors/whatsapp/daemon/internal/handler"
	"aios.dev/connectors/whatsapp/daemon/internal/rpc"
	"aios.dev/connectors/whatsapp/daemon/internal/wameow"
)

// Version is stamped at build time via -ldflags="-X main.Version=...".
// Defaults to "dev" so a local `go run` round-trip surfaces an obvious
// non-release tag in the `version` RPC response.
var Version = "dev"

const daemonName = "whatsapp-daemon"

func main() {
	listen := flag.String("listen", "127.0.0.1:7584", "TCP address to listen on (host:port)")
	storeDir := flag.String("store-dir", "", "directory holding whatsmeow's sqlstore + media cache (required)")
	logLevel := flag.String("log-level", "info", "log level: debug, info, warn, error")
	showVersion := flag.Bool("version", false, "print version and exit")
	flag.Parse()

	if *showVersion {
		fmt.Printf("%s %s\n", daemonName, Version)
		return
	}

	var level slog.Level
	if err := level.UnmarshalText([]byte(*logLevel)); err != nil {
		fmt.Fprintf(os.Stderr, "invalid -log-level=%q: %v\n", *logLevel, err)
		os.Exit(2)
	}
	logger := slog.New(slog.NewJSONHandler(os.Stderr, &slog.HandlerOptions{Level: level}))
	slog.SetDefault(logger)

	if *storeDir == "" {
		logger.Error("daemon.config.invalid", "reason", "-store-dir is required")
		os.Exit(2)
	}

	ctx, cancel := signal.NotifyContext(context.Background(), syscall.SIGTERM, syscall.SIGINT)
	defer cancel()

	reg := handler.NewRegistry()
	handler.RegisterLifecycle(reg, daemonName, Version)

	srv := rpc.NewServer(*listen, reg)

	client, err := wameow.NewClient(ctx, *storeDir, srv, logger.With("component", "wameow"))
	if err != nil {
		logger.Error("wameow.init_failed", "err", err)
		os.Exit(1)
	}
	defer client.Close()

	handler.RegisterSend(reg, client.SendMessage)
	handler.RegisterPairing(reg, &clientPairAdapter{client: client})
	handler.RegisterMessageOps(reg, client)

	// Connect runs in parallel with srv.Run so the listener binds (and
	// `version` RPC starts answering) while the WhatsApp handshake is
	// still in flight.
	go func() {
		if err := client.Connect(ctx); err != nil {
			logger.Warn("wameow.connect_failed", "err", err)
		}
	}()

	if err := srv.Run(ctx); err != nil {
		logger.Error("daemon.exit.error", "err", err)
		os.Exit(1)
	}
	logger.Info("daemon.exit.ok")
}

// clientPairAdapter bridges wameow.Client to handler.Pairer.  The
// translation step is a wameow.PairingOutcome → handler.PairingOutcome
// memcopy; the wameow package can't import handler (would be a
// dependency cycle once handler grows wameow-typed methods) so the
// adapter sits in main.
type clientPairAdapter struct {
	client *wameow.Client
}

func (a *clientPairAdapter) StartPairing(ctx context.Context) (string, error) {
	return a.client.StartPairing(ctx)
}

func (a *clientPairAdapter) ConfirmPairing(ctx context.Context) (handler.PairingOutcome, error) {
	outcome, err := a.client.ConfirmPairing(ctx)
	if err != nil {
		return handler.PairingOutcome{}, err
	}
	return handler.PairingOutcome{
		Status:   outcome.Status,
		JID:      outcome.JID,
		PushName: outcome.PushName,
		Reason:   outcome.Reason,
	}, nil
}

func (a *clientPairAdapter) Unpair(ctx context.Context) error {
	return a.client.Unpair(ctx)
}

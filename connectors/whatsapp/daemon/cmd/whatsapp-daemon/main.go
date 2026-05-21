// Command whatsapp-daemon is the Go-side half of the aios WhatsApp
// connector. It speaks line-delimited JSON-RPC 2.0 over a loopback TCP
// port to the Python aios_whatsapp connector, which spawns it as a
// subprocess and tears it down on shutdown.
//
// This binary is the long-running counterpart to signal-cli for the
// Signal connector — same shape, different protocol. It owns the
// whatsmeow WhatsApp client + sqlstore; the Python side owns aios
// integration (session routing, attachment marshalling, tool dispatch).
//
// PR-1 scope: lifecycle + the `version` RPC only. Subsequent PRs add
// pairing, send/receive, media, reactions, edits, groups.
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"syscall"

	"aios.dev/connectors/whatsapp/daemon/internal/handler"
	"aios.dev/connectors/whatsapp/daemon/internal/rpc"
)

// Version is stamped at build time via -ldflags="-X main.Version=...".
// Defaults to "dev" so a local `go run` round-trip surfaces an obvious
// non-release tag in the `version` RPC response.
var Version = "dev"

const daemonName = "whatsapp-daemon"

func main() {
	listen := flag.String("listen", "127.0.0.1:7584", "TCP address to listen on (host:port)")
	storeDir := flag.String("store-dir", "", "directory holding whatsmeow's sqlstore + media cache (required)")
	logLevel := flag.String("log-level", "info", "log level: debug, info, warn, error (reserved; PR-1 logs everything)")
	showVersion := flag.Bool("version", false, "print version and exit")
	flag.Parse()

	if *showVersion {
		fmt.Printf("%s %s\n", daemonName, Version)
		return
	}

	// -log-level is reserved for future per-level filtering once the
	// daemon grows enough event types to warrant it. Accepted now so
	// the Python side's spawn args are stable across PR boundaries.
	_ = *logLevel

	if *storeDir == "" {
		log.Println("daemon.config.invalid reason=-store-dir is required")
		os.Exit(2)
	}

	ctx, cancel := signal.NotifyContext(context.Background(), syscall.SIGTERM, syscall.SIGINT)
	defer cancel()

	reg := handler.NewRegistry()
	handler.RegisterLifecycle(reg, daemonName, Version)

	srv := rpc.NewServer(*listen, reg)
	if err := srv.Run(ctx); err != nil {
		log.Printf("daemon.exit.error err=%v", err)
		os.Exit(1)
	}
	log.Println("daemon.exit.ok")
}

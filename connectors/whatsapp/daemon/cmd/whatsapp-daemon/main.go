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
	"log"
	"os"
	"os/signal"
	"syscall"
	"time"

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
	showVersion := flag.Bool("version", false, "print version and exit")
	flag.Parse()

	if *showVersion {
		fmt.Printf("%s %s\n", daemonName, Version)
		return
	}

	if *storeDir == "" {
		log.Println("daemon.config.invalid reason=-store-dir is required")
		os.Exit(2)
	}

	ctx, cancel := signal.NotifyContext(context.Background(), syscall.SIGTERM, syscall.SIGINT)
	defer cancel()

	reg := handler.NewRegistry()
	handler.RegisterLifecycle(reg, daemonName, Version)
	// stubSend is replaced with a whatsmeow-backed closure once the
	// real WhatsApp client integration lands; until then the daemon
	// answers sendMessage with deterministic fake data so the wire
	// boundary can be exercised without an actual pairing.
	handler.RegisterSend(reg, stubSend)

	srv := rpc.NewServer(*listen, reg)
	if err := srv.Run(ctx); err != nil {
		log.Printf("daemon.exit.error err=%v", err)
		os.Exit(1)
	}
	log.Println("daemon.exit.ok")
}

func stubSend(_ context.Context, jid, text string) (string, int64, error) {
	log.Printf("daemon.send.stub jid=%s text_len=%d", jid, len(text))
	return "STUB-" + jid, time.Now().UnixMilli(), nil
}

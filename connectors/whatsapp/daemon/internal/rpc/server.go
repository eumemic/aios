// Package rpc implements a minimal JSON-RPC 2.0 server speaking
// line-delimited JSON over TCP.  One frame = one line.  Requests carry
// an "id"; notifications (server → client) omit it.
//
// Subscription model: only connections that have called the
// special-cased "subscribe" RPC receive daemon-initiated notifications
// via Server.Broadcast.  Fresh client connections (the Python
// RpcClient pattern) never subscribe, so they cannot read a stray
// notification in place of their RPC response.
package rpc

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net"
	"sync"
	"time"
)

// Handler dispatches a parsed JSON-RPC request to a registered method.
// Implementors return either a result (which must be JSON-marshallable)
// or an *Error; never both.
type Handler interface {
	Dispatch(ctx context.Context, method string, params json.RawMessage) (any, *Error)
}

// Error is the JSON-RPC 2.0 error object.  Code follows the spec for
// the pre-defined ranges; application errors use Server-error range
// (-32000 to -32099).
type Error struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Data    any    `json:"data,omitempty"`
}

// Standard JSON-RPC 2.0 error codes plus the server-error range we use
// for application errors.  Referenced by handler implementations.
const (
	ErrCodeMethodNotFound = -32601
	ErrCodeInvalidParams  = -32602
	ErrCodeServerError    = -32000
)

// Server accepts TCP connections and serves JSON-RPC.  Use Broadcast
// to push daemon-initiated notifications to all subscribed connections.
type Server struct {
	addr    string
	handler Handler

	subsMu sync.Mutex
	subs   []*subscriber

	// ready is closed once Run has bound the listener, so callers (and
	// tests in particular) can wait deterministically rather than
	// polling the port for connectability.
	readyOnce sync.Once
	ready     chan struct{}
}

// subscriber is one connection that has called "subscribe".  The
// per-connection mutex serialises Broadcast vs response writes so two
// goroutines never interleave bytes on the same socket.
type subscriber struct {
	writer *bufio.Writer
	mu     sync.Mutex
}

func (s *subscriber) writeLine(line []byte) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, err := s.writer.Write(line); err != nil {
		return err
	}
	if err := s.writer.WriteByte('\n'); err != nil {
		return err
	}
	return s.writer.Flush()
}

// NewServer builds an unstarted Server.  Call Run() to bind + accept.
func NewServer(addr string, h Handler) *Server {
	return &Server{addr: addr, handler: h, ready: make(chan struct{})}
}

// Ready returns a channel that's closed once Run has bound the listener.
// Useful for tests that want to wait deterministically instead of
// polling the port for connectability.
func (s *Server) Ready() <-chan struct{} {
	return s.ready
}

// shutdownPollInterval is how often a serve loop checks ctx.Err() while
// blocked on an idle read.  Small enough that SIGTERM-to-Go-daemon
// returns promptly; large enough not to burn CPU.
const shutdownPollInterval = 500 * time.Millisecond

// Run binds the configured address and accepts connections until ctx
// is cancelled.
func (s *Server) Run(ctx context.Context) error {
	var lc net.ListenConfig
	ln, err := lc.Listen(ctx, "tcp", s.addr)
	if err != nil {
		return fmt.Errorf("listen %q: %w", s.addr, err)
	}
	s.readyOnce.Do(func() { close(s.ready) })
	slog.Info("rpc.listening", "addr", ln.Addr().String())

	go func() {
		<-ctx.Done()
		_ = ln.Close()
	}()

	var wg sync.WaitGroup
	for {
		conn, err := ln.Accept()
		if err != nil {
			if ctx.Err() != nil {
				wg.Wait()
				return nil
			}
			return fmt.Errorf("accept: %w", err)
		}
		wg.Add(1)
		go func() {
			defer wg.Done()
			s.serve(ctx, conn)
		}()
	}
}

// Broadcast sends a JSON-RPC notification frame to every subscribed
// connection.  Best-effort: a write failure on one connection is
// logged but does not block others, and the failing connection will
// be removed from the subscriber list when its serve loop returns.
func (s *Server) Broadcast(method string, params any) {
	frame, err := json.Marshal(struct {
		JSONRPC string `json:"jsonrpc"`
		Method  string `json:"method"`
		Params  any    `json:"params,omitempty"`
	}{JSONRPC: "2.0", Method: method, Params: params})
	if err != nil {
		slog.Error("rpc.broadcast.marshal_failed", "method", method, "err", err)
		return
	}

	s.subsMu.Lock()
	targets := make([]*subscriber, len(s.subs))
	copy(targets, s.subs)
	s.subsMu.Unlock()

	for _, sub := range targets {
		if werr := sub.writeLine(frame); werr != nil {
			slog.Warn("rpc.broadcast.write_error", "method", method, "err", werr)
		}
	}
}

func (s *Server) addSubscriber(sub *subscriber) {
	s.subsMu.Lock()
	defer s.subsMu.Unlock()
	s.subs = append(s.subs, sub)
}

func (s *Server) removeSubscriber(sub *subscriber) {
	s.subsMu.Lock()
	defer s.subsMu.Unlock()
	for i, x := range s.subs {
		if x == sub {
			s.subs = append(s.subs[:i], s.subs[i+1:]...)
			return
		}
	}
}

// request is the canonical JSON-RPC 2.0 request frame.  Missing "id"
// means notification; we ignore client-initiated notifications.
type request struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      json.RawMessage `json:"id,omitempty"`
	Method  string          `json:"method"`
	Params  json.RawMessage `json:"params,omitempty"`
}

// successResponse and errorResponse are split structs so a nil result
// from an ack-only method marshals to spec-compliant `"result": null`
// rather than being omitted via omitempty.
type successResponse struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      json.RawMessage `json:"id"`
	Result  any             `json:"result"`
}

type errorResponse struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      json.RawMessage `json:"id"`
	Error   *Error          `json:"error"`
}

// subscribeAck is the immutable shape returned to a subscribe request;
// hoisted to package scope so each handshake reuses the same map
// instead of allocating per call.
var subscribeAck = map[string]string{"status": "subscribed"}

func (s *Server) serve(ctx context.Context, conn net.Conn) {
	defer func() { _ = conn.Close() }()
	remote := conn.RemoteAddr().String()

	sub := &subscriber{writer: bufio.NewWriter(conn)}
	subscribed := false
	defer func() {
		if subscribed {
			s.removeSubscriber(sub)
		}
	}()

	reader := bufio.NewReader(conn)
	for {
		if ctx.Err() != nil {
			return
		}
		_ = conn.SetReadDeadline(time.Now().Add(shutdownPollInterval))
		line, err := reader.ReadBytes('\n')
		if err != nil {
			var netErr net.Error
			if errors.As(err, &netErr) && netErr.Timeout() {
				continue
			}
			if !errors.Is(err, io.EOF) {
				slog.Warn("rpc.conn.read_error", "remote", remote, "err", err)
			}
			return
		}

		var req request
		if err := json.Unmarshal(line, &req); err != nil {
			slog.Warn("rpc.frame.bad_json", "remote", remote, "err", err)
			return
		}

		if len(req.ID) == 0 {
			// Client-initiated notification: ignore.
			continue
		}

		var out []byte
		if req.Method == "subscribe" {
			if !subscribed {
				s.addSubscriber(sub)
				subscribed = true
			}
			out, err = json.Marshal(successResponse{
				JSONRPC: "2.0",
				ID:      req.ID,
				Result:  subscribeAck,
			})
		} else {
			result, rpcErr := s.handler.Dispatch(ctx, req.Method, req.Params)
			if rpcErr != nil {
				out, err = json.Marshal(errorResponse{JSONRPC: "2.0", ID: req.ID, Error: rpcErr})
			} else {
				out, err = json.Marshal(successResponse{JSONRPC: "2.0", ID: req.ID, Result: result})
			}
		}
		if err != nil {
			slog.Error("rpc.response.marshal_failed", "method", req.Method, "err", err)
			return
		}
		if werr := sub.writeLine(out); werr != nil {
			return
		}
	}
}

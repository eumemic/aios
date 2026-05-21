// Package rpc implements a minimal JSON-RPC 2.0 server speaking
// line-delimited JSON over TCP. One frame = one line. Requests carry
// an "id"; notifications (server → client) omit it.
//
// Mirrors the wire shape Python's aios_whatsapp.rpc.RpcClient /
// RpcListener expect, which in turn mirrors what signal-cli's daemon
// produces. Keeping the wire shape identical to Signal means
// aios_whatsapp.rpc is a near-verbatim copy of aios_signal.rpc and
// the test patterns transfer.
package rpc

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net"
	"sync"
)

// Handler dispatches a parsed JSON-RPC request to a registered method.
// Implementors return either a result (which must be JSON-marshallable)
// or an *Error; never both.
type Handler interface {
	Dispatch(ctx context.Context, method string, params json.RawMessage) (any, *Error)
}

// Error is the JSON-RPC 2.0 error object. Code follows the spec's
// conventions where useful (e.g. -32601 = method not found) but the
// daemon is free to use custom positive codes for application errors.
type Error struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Data    any    `json:"data,omitempty"`
}

// Server accepts TCP connections, parses one JSON-RPC frame per line,
// dispatches via the supplied Handler, and writes one response per
// request. Notifications (frames without an "id") from the client are
// ignored — the Python side does not send them.
type Server struct {
	addr    string
	handler Handler

	// Set by Run() once the listener binds, so callers can ask for the
	// resolved address (useful for tests that pass :0 for an
	// OS-assigned port).
	mu      sync.Mutex
	boundAt net.Addr
}

// NewServer builds an unstarted Server. Call Run() to bind + accept.
func NewServer(addr string, h Handler) *Server {
	return &Server{addr: addr, handler: h}
}

// Addr returns the resolved listen address once Run() has bound. Nil
// before the bind completes.
func (s *Server) Addr() net.Addr {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.boundAt
}

// Run binds the configured address and accepts connections until ctx
// is cancelled. Cancellation triggers a graceful shutdown: the
// listener closes, in-flight connections drain (their reads will
// observe EOF on the next iteration), and Run returns nil.
func (s *Server) Run(ctx context.Context) error {
	var lc net.ListenConfig
	ln, err := lc.Listen(ctx, "tcp", s.addr)
	if err != nil {
		return fmt.Errorf("listen %q: %w", s.addr, err)
	}
	s.mu.Lock()
	s.boundAt = ln.Addr()
	s.mu.Unlock()
	log.Printf("rpc.listening addr=%s", ln.Addr().String())

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

// request is the canonical JSON-RPC 2.0 request frame we accept.
// Unknown fields are tolerated; missing "id" means notification (we
// ignore client-initiated notifications).
type request struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      json.RawMessage `json:"id,omitempty"`
	Method  string          `json:"method"`
	Params  json.RawMessage `json:"params,omitempty"`
}

// response mirrors the response shape. Exactly one of Result / Error
// is populated; the other is omitted via omitempty.
type response struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      json.RawMessage `json:"id"`
	Result  any             `json:"result,omitempty"`
	Error   *Error          `json:"error,omitempty"`
}

func (s *Server) serve(ctx context.Context, conn net.Conn) {
	defer func() { _ = conn.Close() }()
	remote := conn.RemoteAddr().String()

	reader := bufio.NewReader(conn)
	writer := bufio.NewWriter(conn)
	for {
		if ctx.Err() != nil {
			return
		}
		line, err := reader.ReadBytes('\n')
		if err != nil {
			if !errors.Is(err, io.EOF) {
				log.Printf("rpc.conn.read_error remote=%s err=%v", remote, err)
			}
			return
		}

		var req request
		if err := json.Unmarshal(line, &req); err != nil {
			// Malformed frame: close the connection. Don't try to
			// recover — line framing is broken at this point.
			log.Printf("rpc.frame.bad_json remote=%s err=%v", remote, err)
			return
		}

		if len(req.ID) == 0 {
			// Notification from the client side: ignore. The Python
			// side talks request/response only; notifications flow
			// the other direction.
			continue
		}

		result, rpcErr := s.handler.Dispatch(ctx, req.Method, req.Params)
		resp := response{JSONRPC: "2.0", ID: req.ID}
		if rpcErr != nil {
			resp.Error = rpcErr
		} else {
			resp.Result = result
		}

		out, err := json.Marshal(resp)
		if err != nil {
			log.Printf("rpc.response.marshal_failed method=%s err=%v", req.Method, err)
			return
		}
		if _, err := writer.Write(out); err != nil {
			return
		}
		if err := writer.WriteByte('\n'); err != nil {
			return
		}
		if err := writer.Flush(); err != nil {
			return
		}
	}
}

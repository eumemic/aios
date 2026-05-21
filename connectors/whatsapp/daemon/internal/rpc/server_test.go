package rpc

import (
	"bufio"
	"context"
	"encoding/json"
	"net"
	"strconv"
	"sync"
	"testing"
	"time"
)

// stubHandler routes every method to a single closure; tests pass
// whatever shape they need.
type stubHandler struct {
	dispatch func(method string, params json.RawMessage) (any, *Error)
}

func (h *stubHandler) Dispatch(_ context.Context, method string, params json.RawMessage) (any, *Error) {
	return h.dispatch(method, params)
}

// runServer brings up a Server on a chosen ephemeral port and returns
// the bound address.  The caller cancels ctx to shut it down.
func runServer(t *testing.T, h Handler) (string, context.CancelFunc) {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	addr := ln.Addr().String()
	_ = ln.Close()

	srv := NewServer(addr, h)
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() {
		_ = srv.Run(ctx)
		close(done)
	}()

	// Wait for the server to bind by probing the port.
	deadline := time.Now().Add(2 * time.Second)
	for {
		c, err := net.DialTimeout("tcp", addr, 100*time.Millisecond)
		if err == nil {
			_ = c.Close()
			break
		}
		if time.Now().After(deadline) {
			cancel()
			t.Fatalf("server never bound %s: %v", addr, err)
		}
		time.Sleep(20 * time.Millisecond)
	}

	t.Cleanup(func() {
		cancel()
		<-done
	})

	// Expose the server for Broadcast tests via a side channel.
	servers[addr] = srv
	return addr, cancel
}

var servers = map[string]*Server{}

// rpcRequest writes a single JSON-RPC request and reads one response line.
func rpcRequest(t *testing.T, conn net.Conn, method string, id int) map[string]any {
	t.Helper()
	req := map[string]any{"jsonrpc": "2.0", "method": method, "id": id}
	out, _ := json.Marshal(req)
	if _, err := conn.Write(append(out, '\n')); err != nil {
		t.Fatalf("write request: %v", err)
	}
	reader := bufio.NewReader(conn)
	line, err := reader.ReadBytes('\n')
	if err != nil {
		t.Fatalf("read response: %v", err)
	}
	var resp map[string]any
	if err := json.Unmarshal(line, &resp); err != nil {
		t.Fatalf("unmarshal response: %v", err)
	}
	return resp
}

func TestServerSubscribeReturnsAck(t *testing.T) {
	addr, _ := runServer(t, &stubHandler{
		dispatch: func(string, json.RawMessage) (any, *Error) {
			return nil, &Error{Code: -32601, Message: "method not found"}
		},
	})

	conn, err := net.Dial("tcp", addr)
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	defer conn.Close()

	resp := rpcRequest(t, conn, "subscribe", 1)
	if resp["error"] != nil {
		t.Fatalf("subscribe returned error: %v", resp["error"])
	}
	result, ok := resp["result"].(map[string]any)
	if !ok {
		t.Fatalf("subscribe result not a map: %T %v", resp["result"], resp["result"])
	}
	if result["status"] != "subscribed" {
		t.Fatalf("subscribe result status = %v, want 'subscribed'", result["status"])
	}
}

func TestServerBroadcastReachesSubscribers(t *testing.T) {
	addr, _ := runServer(t, &stubHandler{
		dispatch: func(string, json.RawMessage) (any, *Error) {
			return nil, &Error{Code: -32601, Message: "method not found"}
		},
	})

	// Open a subscriber connection and subscribe.
	subConn, err := net.Dial("tcp", addr)
	if err != nil {
		t.Fatalf("dial subscriber: %v", err)
	}
	defer subConn.Close()
	_ = rpcRequest(t, subConn, "subscribe", 1)

	// Broadcast a notification from the server side.
	srv := servers[addr]
	go func() {
		// Small delay so the subscriber is parked on Read before we broadcast.
		time.Sleep(50 * time.Millisecond)
		srv.Broadcast("ping", map[string]string{"hello": "world"})
	}()

	subConn.SetReadDeadline(time.Now().Add(2 * time.Second))
	subReader := bufio.NewReader(subConn)
	line, err := subReader.ReadBytes('\n')
	if err != nil {
		t.Fatalf("subscriber read: %v", err)
	}
	var notif map[string]any
	if err := json.Unmarshal(line, &notif); err != nil {
		t.Fatalf("unmarshal notification: %v", err)
	}
	if notif["method"] != "ping" {
		t.Fatalf("notification method = %v, want 'ping'", notif["method"])
	}
	params, ok := notif["params"].(map[string]any)
	if !ok || params["hello"] != "world" {
		t.Fatalf("notification params = %v, want {hello: world}", notif["params"])
	}
	// Notifications must not carry an id.
	if _, hasID := notif["id"]; hasID {
		t.Fatalf("notification contains id: %v", notif)
	}
}

func TestServerBroadcastSkipsNonSubscribers(t *testing.T) {
	addr, _ := runServer(t, &stubHandler{
		dispatch: func(method string, _ json.RawMessage) (any, *Error) {
			return map[string]string{"echo": method}, nil
		},
	})

	// Connection that calls a non-subscribe method.
	clientConn, err := net.Dial("tcp", addr)
	if err != nil {
		t.Fatalf("dial client: %v", err)
	}
	defer clientConn.Close()

	resp := rpcRequest(t, clientConn, "ping", 1)
	if resp["error"] != nil {
		t.Fatalf("ping returned error: %v", resp["error"])
	}

	// Broadcast — this connection did NOT subscribe so should not see it.
	srv := servers[addr]
	srv.Broadcast("event", map[string]string{"k": "v"})

	clientConn.SetReadDeadline(time.Now().Add(200 * time.Millisecond))
	reader := bufio.NewReader(clientConn)
	if _, err := reader.ReadBytes('\n'); err == nil {
		t.Fatalf("non-subscriber received a notification it shouldn't have")
	}
}

func TestServerCleansUpSubscriberOnDisconnect(t *testing.T) {
	addr, _ := runServer(t, &stubHandler{
		dispatch: func(string, json.RawMessage) (any, *Error) {
			return nil, &Error{Code: -32601, Message: "method not found"}
		},
	})

	srv := servers[addr]

	// Connect + subscribe N parallel connections then close them all.
	const n = 5
	conns := make([]net.Conn, 0, n)
	for i := 0; i < n; i++ {
		c, err := net.Dial("tcp", addr)
		if err != nil {
			t.Fatalf("dial %d: %v", i, err)
		}
		_ = rpcRequest(t, c, "subscribe", i+1)
		conns = append(conns, c)
	}

	srv.subsMu.Lock()
	if got := len(srv.subs); got != n {
		srv.subsMu.Unlock()
		t.Fatalf("subscribers after subscribe = %d, want %d", got, n)
	}
	srv.subsMu.Unlock()

	var wg sync.WaitGroup
	for _, c := range conns {
		wg.Add(1)
		go func(c net.Conn) {
			defer wg.Done()
			c.Close()
		}(c)
	}
	wg.Wait()

	// The serve loop notices the closed read deadline + EOF and unwinds —
	// give it a moment to remove itself.
	deadline := time.Now().Add(2 * time.Second)
	for {
		srv.subsMu.Lock()
		got := len(srv.subs)
		srv.subsMu.Unlock()
		if got == 0 {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("subscribers after close = %d, want 0", got)
		}
		time.Sleep(20 * time.Millisecond)
	}
}

// Suppress unused import warning when the test file is otherwise
// compiled but skipped via build tags.
var _ = strconv.Itoa

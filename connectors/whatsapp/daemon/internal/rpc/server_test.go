package rpc

import (
	"bufio"
	"context"
	"encoding/json"
	"net"
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

// runServer brings up a Server on an OS-assigned ephemeral port and
// returns it.  The caller waits on srv.Ready() before connecting.
// Cleanup is registered with t.Cleanup so the server is cancelled and
// drained at the end of the test.
func runServer(t *testing.T, h Handler) (srv *Server, addr string) {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	addr = ln.Addr().String()
	_ = ln.Close()

	srv = NewServer(addr, h)
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() {
		_ = srv.Run(ctx)
		close(done)
	}()

	select {
	case <-srv.Ready():
	case <-time.After(2 * time.Second):
		cancel()
		t.Fatalf("server never bound %s", addr)
	}

	t.Cleanup(func() {
		cancel()
		<-done
	})
	return srv, addr
}

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
	_, addr := runServer(t, &stubHandler{
		dispatch: func(string, json.RawMessage) (any, *Error) {
			return nil, &Error{Code: ErrCodeMethodNotFound, Message: "method not found"}
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
	srv, addr := runServer(t, &stubHandler{
		dispatch: func(string, json.RawMessage) (any, *Error) {
			return nil, &Error{Code: ErrCodeMethodNotFound, Message: "method not found"}
		},
	})

	subConn, err := net.Dial("tcp", addr)
	if err != nil {
		t.Fatalf("dial subscriber: %v", err)
	}
	defer subConn.Close()
	// Subscribe handshake is synchronous: rpcRequest returns only after
	// the server has added the connection to its subscribers list and
	// written the ack.  No sleep needed before broadcasting.
	_ = rpcRequest(t, subConn, "subscribe", 1)
	srv.Broadcast("ping", map[string]string{"hello": "world"})

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
	if _, hasID := notif["id"]; hasID {
		t.Fatalf("notification contains id: %v", notif)
	}
}

func TestServerBroadcastSkipsNonSubscribers(t *testing.T) {
	srv, addr := runServer(t, &stubHandler{
		dispatch: func(method string, _ json.RawMessage) (any, *Error) {
			return map[string]string{"echo": method}, nil
		},
	})

	clientConn, err := net.Dial("tcp", addr)
	if err != nil {
		t.Fatalf("dial client: %v", err)
	}
	defer clientConn.Close()

	resp := rpcRequest(t, clientConn, "ping", 1)
	if resp["error"] != nil {
		t.Fatalf("ping returned error: %v", resp["error"])
	}

	srv.Broadcast("event", map[string]string{"k": "v"})

	clientConn.SetReadDeadline(time.Now().Add(200 * time.Millisecond))
	reader := bufio.NewReader(clientConn)
	if _, err := reader.ReadBytes('\n'); err == nil {
		t.Fatalf("non-subscriber received a notification it shouldn't have")
	}
}

func TestServerCleansUpSubscriberOnDisconnect(t *testing.T) {
	srv, addr := runServer(t, &stubHandler{
		dispatch: func(string, json.RawMessage) (any, *Error) {
			return nil, &Error{Code: ErrCodeMethodNotFound, Message: "method not found"}
		},
	})

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

	if got := subscriberCount(srv); got != n {
		t.Fatalf("subscribers after subscribe = %d, want %d", got, n)
	}

	var wg sync.WaitGroup
	for _, c := range conns {
		wg.Add(1)
		go func(c net.Conn) {
			defer wg.Done()
			c.Close()
		}(c)
	}
	wg.Wait()

	deadline := time.Now().Add(2 * time.Second)
	for {
		if subscriberCount(srv) == 0 {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("subscribers after close = %d, want 0", subscriberCount(srv))
		}
		time.Sleep(20 * time.Millisecond)
	}
}

func subscriberCount(srv *Server) int {
	srv.subsMu.Lock()
	defer srv.subsMu.Unlock()
	return len(srv.subs)
}

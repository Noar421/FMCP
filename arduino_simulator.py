import socket
import time
from config import ArduinoConfig

LISTEN_ADDR = ("0.0.0.0", 32000)
FORWARD_ADDR = ("127.0.0.1", 32001)

def udp_nonblocking_echo_server():
    # Create UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(LISTEN_ADDR)
    sock.setblocking(False)  # Non-blocking mode

    print(f"Listening on {LISTEN_ADDR[0]}:{LISTEN_ADDR[1]} (non-blocking)...")
    print(f"Echoing packets to {FORWARD_ADDR[0]}:{FORWARD_ADDR[1]}")

    while True:
        try:
            data, addr = sock.recvfrom(4096)
            print(f"Received {len(data)} bytes from {addr}: {data!r}")

            # Decode packet
            data_str = data.decode("utf-8")
            command, relay, delay = data_str.split(",")
            resp = ""
            if relay not in ["1","2","3","4","5","6","7","8"]:
                resp = "ERR,BAD_RELAY_NUM"
            elif command not in ["SET", "RESET", "STATUS"]:
                resp = "ERR,UNKNOWN_CMD"
            else:
                resp = f"OK,{command},{relay},{delay}"
            # Convert resp to bytes
            resp_packet = resp.encode("utf-8")

            # Forward data to 127.0.0.1:32001
            sock.sendto(resp_packet, FORWARD_ADDR)
            print(f"Responded {resp} ({len(resp_packet)} bytes to {FORWARD_ADDR})")
            
        except socket.timeout:
            time.sleep(0.01)
        except BlockingIOError:
            # No data ready to read; just sleep briefly to avoid busy waiting
            time.sleep(0.01)
        except KeyboardInterrupt:
            print("\nServer stopped.")
            break
        except Exception as e:
            print(f"Error: {e}")

    sock.close()

if __name__ == "__main__":
    udp_nonblocking_echo_server()

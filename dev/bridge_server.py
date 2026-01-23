# dev/bridge_server.py
import socket
import struct
import jax
import jax.numpy as jnp
import numpy as np
import pickle
from model import create_network

# --- CONFIG ---
HOST = '0.0.0.0'
PORT = 5000
# Grid shape matches your config: 12 rows, 20 cols
GRID_SHAPE = (12, 20, 1) 
EXPECTED_BYTES = 12 * 20 * 4  # 240 floats * 4 bytes each = 960 bytes

print("Loading JAX Model...")
model = create_network()

try:
    with open("snake_params.pkl", "rb") as f:
        params = pickle.load(f)
    print("Model Weights Loaded.")
except FileNotFoundError:
    print("CRITICAL ERROR: snake_params.pkl not found!")
    exit(1)

# JIT Compile the inference function for max speed
@jax.jit
def infer(grid):
    # grid: (12, 20, 1)
    # Add batch dim -> (1, 12, 20, 1)
    batched = jnp.expand_dims(grid, axis=0)
    logits, _ = model.apply(params, batched)
    return jnp.argmax(logits[0])

def start_server():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # Disable Nagle's algorithm (Waiting for data to buffer) -> Send IMMEDIATELY
    server_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    
    server_socket.bind((HOST, PORT))
    server_socket.listen(1)
    
    print(f"🚀 TCP Turbo Server listening on {HOST}:{PORT}")
    
    while True:
        print("Waiting for client connection...")
        conn, addr = server_socket.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f"Connected by {addr}")
        
        try:
            while True:
                # 1. Receive Raw Data
                # We expect exactly 960 bytes (the grid)
                data = b''
                while len(data) < EXPECTED_BYTES:
                    packet = conn.recv(EXPECTED_BYTES - len(data))
                    if not packet:
                        raise ConnectionResetError
                    data += packet
                
                # 2. Decode (Bytes -> Numpy)
                # 'f' is float32
                grid_flat = np.frombuffer(data, dtype=np.float32)
                grid = grid_flat.reshape(GRID_SHAPE)
                
                # 3. Inference
                action_idx = int(infer(grid))
                
                # 4. Reply (1 Byte)
                # We send back a single integer byte (0, 1, 2, 3)
                conn.sendall(action_idx.to_bytes(1, 'big'))
                
        except (ConnectionResetError, BrokenPipeError):
            print("Client disconnected.")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            conn.close()

if __name__ == '__main__':
    start_server()
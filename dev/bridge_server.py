# dev/bridge_server.py
# RUN THIS INSIDE WSL (Where JAX/GPU works)
# pip install flask jax jaxlib flax numpy

from flask import Flask, request, jsonify
import jax
import jax.numpy as jnp
import numpy as np
import pickle
from model import create_network

app = Flask(__name__)

# --- LOAD MODEL ---
print("Loading Model in WSL...")
model = create_network()

# Load the parameters (ensure snake_params.pkl is in the same folder in WSL)
with open("snake_params.pkl", "rb") as f:
    params = pickle.load(f)

print("Model Loaded. GPU Enabled." if jax.default_backend() == "gpu" else "Model Loaded (CPU Mode).")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Receive Grid from Windows
        data = request.json
        # Expecting a 2D list (e.g., 10x10)
        grid_list = data.get("grid") 
        
        # 2. Convert to JAX Input Format
        # Model expects: (Batch, Height, Width, Channels) -> (1, 10, 10, 1)
        # The capture script sends 1s (Snake) and -1s (Food).
        grid_np = np.array(grid_list, dtype=np.float32)
        
        # Reshape to (1, H, W, 1)
        grid_jax = jnp.array(grid_np).reshape(1, grid_np.shape[0], grid_np.shape[1], 1)
        
        # 3. Inference
        logits, _ = model.apply(params, grid_jax)
        action = int(jnp.argmax(logits[0]))
        
        # 4. Return Action
        # 0: Up, 1: Right, 2: Down, 3: Left
        return jsonify({"action": action})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Host 0.0.0.0 allows connections from the Windows host
    app.run(host='0.0.0.0', port=5000)
import jax
import jax.numpy as jnp
from snake_env import reset, step, GRID_SIZE
from model import create_network
import numpy as np
from PIL import Image
import pickle
import time

# --- SETUP ---
# We need to make sure we use the same architecture as training
model = create_network()

def run_simulation(params):
    # 1. Init one environment
    current_time_seed = int(time.time() * 1000)
    key = jax.random.PRNGKey(current_time_seed)

    print(f"Playing Game Seed: {current_time_seed}")
    state = reset(key)
    
    frames = []
    
    # Run for 1000 steps max
    for _ in range(1000):
        # Save the grid for the GIF
        frames.append(state.grid)
        
        # --- FIX: USE THE GRID, NOT GPS ---
        # The model now expects (Batch, 10, 10, 2)
        # We take the grid from the state
        obs = state.grid.astype(jnp.float32)
        
        # Add batch dim: (10, 10, 2) -> (1, 10, 10, 2)
        logits, _ = model.apply(params, obs[None, ...])
        
        # DEBUG: Print logits to ensure they aren't all identical
        # print(f"Logits: {logits[0]}") 

        action = jnp.argmax(logits[0])
        
        # 3. Step
        state, reward, done = step(state, action)
        
        if done:
            print("Snake died!")
            break
            
    return frames
    
def save_gif(frames, filename="replay.gif"):
    print(f"Saving {len(frames)} frames...")
    pil_images = []
    
    for grid in frames:
        # grid is (10, 10, 2)
        grid = np.array(grid)
        
        # 1. Extract Channels
        snake_channel = grid[..., 0] # Values from 0.0 to 1.0
        food_channel = grid[..., 1]  # Values 0.0 or 1.0
        
        # 2. Create RGB Canvas
        # Shape (10, 10, 3)
        img = np.zeros((GRID_SIZE, GRID_SIZE, 3), dtype=np.uint8)
        
        # 3. Render Snake (Green Gradient)
        # We simply multiply the float value (0.0 to 1.0) by 255.
        # This automatically makes the head bright and the tail dark.
        snake_intensity = (snake_channel * 255).astype(np.uint8)
        img[..., 1] = snake_intensity  # Set Green Channel
        
        # 4. Render Food (Red)
        # Food is binary, so we just set it to 255 where active.
        img[food_channel > 0.5] = [255, 0, 0] # Red
        
        # 5. Optional: Highlight the Head Explicitly (Blue pixel?)
        # If you want the head to really pop, you can add this:
        # head_mask = snake_channel >= 0.99
        # img[head_mask] = [0, 255, 255] # Cyan Head
        
        # Upscale for visibility
        pil_img = Image.fromarray(img).resize((300, 300), resample=Image.NEAREST)
        pil_images.append(pil_img)
        
    pil_images[0].save(filename, save_all=True, append_images=pil_images[1:], duration=100, loop=0)
    print(f"Saved to {filename}")

if __name__ == "__main__":
    # 1. Load the trained weights
    print("Loading weights...")
    with open("snake_params.pkl", "rb") as f:
        params = pickle.load(f)
        
    # 2. Run
    frames = run_simulation(params)
    
    # 3. Save
    save_gif(frames, "snake_replay.gif")
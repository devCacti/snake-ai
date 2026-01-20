import jax
import jax.numpy as jnp
from snake_env import reset, step, GRID_SIZE
from model import create_network
import numpy as np
from PIL import Image, ImageDraw, ImageFont  # <--- Added ImageDraw
import pickle
import time

# --- HYPERPARAMETERS ---
MAX_STEPS = 5000

# --- SETUP ---
model = create_network()

# --- THE "GOD MODE" SIMULATOR ---
@jax.jit
def run_game_scan(params, rng_key):
    init_state = reset(rng_key)
    
    def scan_step(state, _):
        obs = state.grid.astype(jnp.float32)
        logits, _ = model.apply(params, obs[None, ...])
        action = jnp.argmax(logits[0])
        next_state, reward, done = step(state, action) #type: ignore
        return next_state, (state.grid, done)

    # Run for a guaranteed long time
    final_state, (all_frames, all_dones) = jax.lax.scan(
        scan_step, init_state, None, length=MAX_STEPS
    )
    
    return all_frames, all_dones

def save_gif(frames, filename="replay.gif"):
    print(f"Transferring {len(frames)} frames to CPU...")
    frames_np = np.array(frames) # Shape (N, 10, 10, 1)
    
    print("Rendering GIF with values...")
    pil_images = []
    
    # Calculate cell size for the upscaled image (300 / 10 = 30px per block)
    upscale_size = 300
    cell_size = upscale_size // GRID_SIZE
    
    # Optional: Load a font (default is fine, but this is cleaner)
    try:
        # Try to load a standard font if available, else default
        font = ImageFont.load_default()
    except:
        font = None

    for grid in frames_np:
        # grid shape is (10, 10, 1)
        layer = grid[..., 0] 
        
        img = np.zeros((GRID_SIZE, GRID_SIZE, 3), dtype=np.uint8)
        
        # 1. RENDER SNAKE (Positive Values)
        snake_vals = np.maximum(layer, 0.0)
        snake_intensity = (snake_vals * 255).astype(np.uint8)
        img[..., 1] = snake_intensity 
        
        # 2. RENDER FOOD (Negative Values)
        is_food = layer < -0.1
        img[is_food] = [255, 0, 0] # Set Red
        
        # 3. UPSCALE
        pil_img = Image.fromarray(img).resize((upscale_size, upscale_size), resample=Image.NEAREST) #type: ignore
        
        # 4. DRAW NUMBERS
        draw = ImageDraw.Draw(pil_img)
        
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                val = layer[y, x]
                
                # Only draw if not 0.0 (Empty) to keep it clean
                if abs(val) > 0.001:
                    # Format: 2 decimal places (e.g., 0.99, -1.0)
                    text = f"{val:.2f}"
                    
                    # Center the text in the 30x30 cell
                    # (Simple centering logic)
                    pos_x = x * cell_size + 2
                    pos_y = y * cell_size + 10
                    
                    # Draw White Text
                    draw.text((pos_x, pos_y), text, fill=(255, 255, 255), font=font)

        pil_images.append(pil_img)
        
    pil_images[0].save(filename, save_all=True, append_images=pil_images[1:], duration=100, loop=0) # Slower duration (100ms) to read numbers
    print(f"Saved to {filename}")

if __name__ == "__main__":
    print("Loading weights...")
    with open("snake_params.pkl", "rb") as f:
        params = pickle.load(f)

    current_time_seed = int(time.time() * 1000)
    key = jax.random.PRNGKey(current_time_seed)
    print(f"Playing Game Seed: {current_time_seed}")
    
    t0 = time.time()
    # This will now return 5000 frames
    all_frames, all_dones = run_game_scan(params, key) 
    
    dones_np = np.array(all_dones)
    t1 = time.time()
    
    # Logic to trim the empty tail of the video
    if np.any(dones_np):
        death_idx = np.argmax(dones_np)
        print(f"Game ended at step {death_idx + 1}!")
        frames_to_save = all_frames[:death_idx+1]
    else:
        print(f"Snake survived all {MAX_STEPS} steps (Infinite Loop?)")
        frames_to_save = all_frames

    save_gif(frames_to_save, "snake_replay.gif")
    t2 = time.time()
    
    print(f"\n=== SPEED REPORT ===")
    print(f"Simulation (GPU): {t1 - t0:.4f}s")
    print(f"GIF Creation:     {t2 - t1:.4f}s")
    print(f"Total Time:       {t2 - t0:.4f}s")
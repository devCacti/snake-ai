import jax
import jax.numpy as jnp
# --- UPDATED IMPORT: Fetch X and Y dimensions separately ---
from snake_env import reset, step, GRID_SIZE_X, GRID_SIZE_Y #type: ignore
from model import create_network
import numpy as np
from PIL import Image, ImageDraw, ImageFont
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
    frames_np = np.array(frames) # Shape (N, H, W, 1)
    
    print("Rendering GIF with values...")
    pil_images = []
    
    # --- UPDATED SCALING LOGIC ---
    # Instead of a fixed 300px square, we use a fixed pixel size per cell
    CELL_PIXELS = 30
    upscale_w = GRID_SIZE_X * CELL_PIXELS
    upscale_h = GRID_SIZE_Y * CELL_PIXELS
    
    try:
        font = ImageFont.load_default()
    except:
        font = None

    for grid in frames_np:
        # grid shape is (Height, Width, 1)
        layer = grid[..., 0] 
        
        # Create image with correct aspect ratio
        img = np.zeros((GRID_SIZE_Y, GRID_SIZE_X, 3), dtype=np.uint8)
        
        # 1. RENDER SNAKE (Positive Values)
        snake_vals = np.maximum(layer, 0.0)
        snake_intensity = (snake_vals * 255).astype(np.uint8)
        img[..., 1] = snake_intensity 
        
        # 2. RENDER FOOD (Negative Values)
        is_food = layer < -0.1
        img[is_food] = [255, 0, 0] # Set Red
        
        # 3. UPSCALE
        pil_img = Image.fromarray(img).resize((upscale_w, upscale_h), resample=Image.NEAREST) #type: ignore
        
        # 4. DRAW NUMBERS
        draw = ImageDraw.Draw(pil_img)
        
        for y in range(GRID_SIZE_Y):
            for x in range(GRID_SIZE_X):
                val = layer[y, x]
                
                if abs(val) > 0.001:
                    text = f"{val:.2f}"
                    
                    # Center text relative to the specific cell
                    pos_x = x * CELL_PIXELS + 2
                    pos_y = y * CELL_PIXELS + 10
                    
                    draw.text((pos_x, pos_y), text, fill=(255, 255, 255), font=font)

        pil_images.append(pil_img)
        
    pil_images[0].save(filename, save_all=True, append_images=pil_images[1:], duration=100, loop=0)
    print(f"Saved to {filename} ({upscale_w}x{upscale_h})")

if __name__ == "__main__":
    print("Loading weights...")
    with open("snake_params.pkl", "rb") as f:
        params = pickle.load(f)

    current_time_seed = int(time.time() * 1000)
    key = jax.random.PRNGKey(current_time_seed)
    print(f"Playing Game Seed: {current_time_seed}")
    
    t0 = time.time()
    all_frames, all_dones = run_game_scan(params, key) 
    
    dones_np = np.array(all_dones)
    t1 = time.time()
    
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
import jax
import jax.numpy as jnp
from typing import NamedTuple

# --- CONFIGURATION ---
GRID_SIZE_X = 20
GRID_SIZE_Y = 12
MAX_LEN = GRID_SIZE_X * GRID_SIZE_Y

# --- THE "GOD TIER" STATE ---
class State(NamedTuple):
    # VISUAL: The Grid seen by the Neural Network
    # Shape: (12, 20, 1) -> Channel 0: Snake (+1) & Food (-1)
    grid: jnp.ndarray 

    # PHYSICS: The Ring Buffer tracking body coordinates
    # Shape: (240, 2)
    body_buffer: jnp.ndarray
    
    # PHYSICS: Pointers for the Ring Buffer
    head_idx: jnp.int32 # type: ignore
    tail_idx: jnp.int32 # type: ignore
    length: jnp.int32 # type: ignore
    
    # TRACKING: Head & Food positions for fast access (Y, X)
    head_pos: jnp.ndarray 
    food_pos: jnp.ndarray
    
    # GAME STATUS
    key: jnp.ndarray
    done: jnp.bool_  # type: ignore # True if crashed
    step_count: jnp.int32 # type: ignore # To prevent infinite loops

# --- 1. THE RESET FUNCTION ---
def reset(key):
    k1, k2, k3 = jax.random.split(key, 3)
    
    # A. Setup Positions
    # We use array-based min/max to handle rectangular bounds (Y, X)
    # Y range: [2, 10], X range: [2, 18]
    min_bound = jnp.array([2, 2])
    max_bound = jnp.array([GRID_SIZE_Y - 2, GRID_SIZE_X - 2])
    
    head_pos = jax.random.randint(k1, shape=(2,), minval=min_bound, maxval=max_bound)
    
    # B. Setup Buffer
    body_buffer = jnp.zeros((MAX_LEN, 2), dtype=jnp.int32)
    body_buffer = body_buffer.at[0].set(head_pos)
    
    # C. Setup Grid
    # Shape is (Height, Width, Channels) -> (12, 20, 1)
    grid = jnp.zeros((GRID_SIZE_Y, GRID_SIZE_X, 1), dtype=jnp.float32)
    grid = grid.at[head_pos[0], head_pos[1], 0].set(1.0)
    
    # D. Setup Food
    # Bounds for food: [0,0] to [12, 20]
    food_max = jnp.array([GRID_SIZE_Y, GRID_SIZE_X])
    food_pos = jax.random.randint(k2, shape=(2,), minval=0, maxval=food_max)
    
    # Paint food on Channel 0
    grid = grid.at[food_pos[0], food_pos[1], 0].set(-1.0)

    return State(
        grid=grid,
        body_buffer=body_buffer,
        head_idx=jnp.int32(0),
        tail_idx=jnp.int32(0),
        length=jnp.int32(1),
        head_pos=head_pos,
        food_pos=food_pos,
        key=k3,
        done=jnp.bool_(False),
        step_count=jnp.int32(0)
    )

# --- 2. THE STEP FUNCTION (With Auto-Reset) ---
def step(state: State, action: int):
    # --- 1. PREVENT 180 TURNS (The "Neck Check") ---
    neck_idx = (state.head_idx - 1) % MAX_LEN
    neck_pos = state.body_buffer[neck_idx]
    
    # Action maps to (dy, dx)
    moves = jnp.array([[-1, 0], [0, 1], [1, 0], [0, -1]]) # Up, Right, Down, Left
    delta = moves[action]
    proposed_new_head = state.head_pos + delta
    
    is_reverse = jnp.all(proposed_new_head == neck_pos)
    current_momentum = state.head_pos - neck_pos
    
    final_delta = jnp.where(is_reverse, current_momentum, delta)
    final_new_head = state.head_pos + final_delta
    
    # --- 2. COLLISION CHECKS ---
    # Check Y bounds (0 to 11) and X bounds (0 to 19) separately
    y, x = final_new_head[0], final_new_head[1]
    hit_wall = (y < 0) | (y >= GRID_SIZE_Y) | (x < 0) | (x >= GRID_SIZE_X)
    
    # Clip for safe indexing (even if dead)
    max_vals = jnp.array([GRID_SIZE_Y - 1, GRID_SIZE_X - 1])
    safe_pos = jnp.clip(final_new_head, 0, max_vals)
    
    # Check Self-Collision
    hit_self = state.grid[safe_pos[0], safe_pos[1], 0] > 0.0
    
    done = hit_wall | hit_self
    
    # --- 3. FOOD CHECK ---
    hit_food = jnp.all(final_new_head == state.food_pos)
    
    # --- 4. PHYSICS UPDATE ---
    new_head_idx = (state.head_idx + 1) % MAX_LEN
    new_body_buffer = state.body_buffer.at[new_head_idx].set(final_new_head)
    new_tail_idx = jnp.where(hit_food, state.tail_idx, (state.tail_idx + 1) % MAX_LEN)
    new_length = jnp.where(hit_food, state.length + 1, state.length)

    # --- 5. VISUAL UPDATE ---
    new_grid = state.grid
    
    # A. Decay
    DECAY = 0.005
    snake_channel = new_grid[..., 0]
    decayed_snake = jnp.where(snake_channel > 0, snake_channel - DECAY, 0.0)
    new_grid = new_grid.at[..., 0].set(decayed_snake)
    decayed_snake = jnp.maximum(decayed_snake, 0.0)
    
    # B. Clear Tail
    tail_pos = state.body_buffer[state.tail_idx]
    new_grid = new_grid.at[tail_pos[0], tail_pos[1], 0].set(
        jnp.where(hit_food, new_grid[tail_pos[0], tail_pos[1], 0], 0.0)
    )
    
    new_grid = new_grid.at[state.food_pos[0], state.food_pos[1], 0].set(0.0)
    
    # C. Paint New Head
    new_grid = new_grid.at[safe_pos[0], safe_pos[1], 0].set(1.0)
    
    # D. Spawn New Food (If Ate)
    key, subkey = jax.random.split(state.key)
    
    flat_body = new_grid[..., 0].ravel()
    candidate_logits = jnp.where(flat_body > 0, -1e9, 1.0)
    new_food_idx = jax.random.categorical(subkey, candidate_logits)
    
    # Convert 1D index -> 2D (Y, X)
    # Integer division by WIDTH gives Row (Y)
    new_food_y = new_food_idx // GRID_SIZE_X 
    new_food_x = new_food_idx % GRID_SIZE_X
    smart_food_pos = jnp.array([new_food_y, new_food_x])

    new_food_pos = jnp.where(hit_food, smart_food_pos, state.food_pos)
    new_grid = new_grid.at[new_food_pos[0], new_food_pos[1], 0].set(-1.0)

    # --- 6. REWARDS ---
    reward = -0.01 
    reward = jnp.where(done, -1.0, reward)
    
    food_value = 1.0 
    reward = jnp.where(hit_food, food_value, reward)
    
    board_area = GRID_SIZE_X * GRID_SIZE_Y
    is_win = (state.length >= board_area - 1)
    done = done | is_win
    reward = jnp.where(is_win, 10.0, reward)
    
    next_state = State(
        grid=new_grid,
        body_buffer=new_body_buffer,
        head_idx=new_head_idx,
        tail_idx=new_tail_idx,
        length=new_length,
        head_pos=final_new_head,
        food_pos=new_food_pos,
        key=key,
        done=done,
        step_count=state.step_count + 1
    )
    
    # --- 7. AUTO-RESET ---
    reset_key, _ = jax.random.split(key)
    fresh_state = reset(reset_key)
    
    final_state = jax.tree_util.tree_map(
        lambda x, y: jnp.where(done, x, y), 
        fresh_state, 
        next_state
    )
    
    return final_state, reward, done

step_batch = jax.vmap(step, in_axes=(0, 0))

if __name__ == "__main__":
    reset_fast = jax.jit(reset)
    step_fast = jax.jit(step)
    
    key = jax.random.PRNGKey(0)
    state = reset_fast(key)
    
    print(f"Grid Shape: {state.grid.shape}")
    print("Initial Head:", state.head_pos)
    print("Initial Food:", state.food_pos)
    
    state, reward, done = step_fast(state, 1)
    print("New Head:", state.head_pos)
import jax
import jax.numpy as jnp
from typing import NamedTuple

# --- CONFIGURATION ---
GRID_SIZE = 10
MAX_LEN = GRID_SIZE * GRID_SIZE

# --- THE "GOD TIER" STATE ---
class State(NamedTuple):
    # VISUAL: The Grid seen by the Neural Network
    # Shape: (10, 10, 2) -> Channel 0: Snake (0 or 1), Channel 1: Food (0 or 1)
    grid: jnp.ndarray 

    # PHYSICS: The Ring Buffer tracking body coordinates
    # Shape: (100, 2)
    body_buffer: jnp.ndarray
    
    # PHYSICS: Pointers for the Ring Buffer
    head_idx: jnp.int32 # type: ignore
    tail_idx: jnp.int32 # type: ignore
    length: jnp.int32 # type: ignore
    
    # TRACKING: Head & Food positions for fast access
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
    # Start head somewhat centrally to avoid instant death
    head_pos = jax.random.randint(k1, shape=(2,), minval=2, maxval=GRID_SIZE-2)
    
    # B. Setup Buffer
    # Create empty buffer
    body_buffer = jnp.zeros((MAX_LEN, 2), dtype=jnp.int32)
    # Set the first slot (index 0) to be the head
    body_buffer = body_buffer.at[0].set(head_pos)
    
    # C. Setup Grid
    # Channel 0 (Snake), Channel 1 (Food)
    grid = jnp.zeros((GRID_SIZE, GRID_SIZE, 1), dtype=jnp.float32)
    # Paint the head on Channel 0
    grid = grid.at[head_pos[0], head_pos[1], 0].set(1.0)
    
    # D. Setup Food
    # Simple logic: random pos. (Might overlap head, but rarely. We fix this in Step)
    food_pos = jax.random.randint(k2, shape=(2,), minval=0, maxval=GRID_SIZE)
    # Paint food on Channel 1
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
# snake_env.py

def step(state: State, action: int):
    # --- 1. PREVENT 180 TURNS (The "Neck Check") ---
    # We find where the "Neck" is (the segment before the head).
    # Since head_idx points to the CURRENT head, head_idx - 1 is the NECK.
    neck_idx = (state.head_idx - 1) % MAX_LEN
    neck_pos = state.body_buffer[neck_idx]
    
    # Calculate the move based on the action
    moves = jnp.array([[-1, 0], [0, 1], [1, 0], [0, -1]]) # Up, Right, Down, Left
    delta = moves[action]
    proposed_new_head = state.head_pos + delta
    
    # Check if we are trying to reverse (Proposed Move lands exactly on Neck)
    # Note: We need a special check for Length=1 (no neck), but our logic handles it 
    # because if Length=1, neck_pos is same as head, so is_reverse is False.
    is_reverse = jnp.all(proposed_new_head == neck_pos)
    
    # If it is a reverse move, we IGNORE the action and keep current momentum.
    # Current Momentum = Head - Neck
    current_momentum = state.head_pos - neck_pos
    
    # If Length is 1, current_momentum is [0,0], so we accept the proposed move.
    # If Length > 1 and is_reverse, we use momentum.
    # Otherwise, we use the proposed move.
    
    # Robust logic:
    # If is_reverse is True, it implies Length > 1.
    final_delta = jnp.where(is_reverse, current_momentum, delta)
    final_new_head = state.head_pos + final_delta
    
    # --- 2. COLLISION CHECKS (Using the sanitized move) ---
    hit_wall = (final_new_head < 0) | (final_new_head >= GRID_SIZE)
    hit_wall = jnp.any(hit_wall)
    
    safe_pos = jnp.clip(final_new_head, 0, GRID_SIZE-1)
    
    # Check Self-Collision
    # We check > 0.0 because of the gradient body.
    # CRITICAL: The Neck is currently on the grid. We must ensure we don't 'collide' 
    # with the neck if we are just moving away from it.
    # However, since we already did the Neck Check above, `final_new_head` 
    # is guaranteed NOT to be the neck. So any non-zero pixel is a valid collision.
    hit_self = state.grid[safe_pos[0], safe_pos[1], 0] > 0.0
    
    done = hit_wall | hit_self
    
    # --- 3. FOOD CHECK ---
    hit_food = jnp.all(final_new_head == state.food_pos)
    
    # --- 4. PHYSICS UPDATE ---
    new_head_idx = (state.head_idx + 1) % MAX_LEN
    new_body_buffer = state.body_buffer.at[new_head_idx].set(final_new_head)
    new_tail_idx = jnp.where(hit_food, state.tail_idx, (state.tail_idx + 1) % MAX_LEN)
    new_length = jnp.where(hit_food, state.length + 1, state.length)

    # --- 5. VISUAL UPDATE (Gradient Body) ---
    new_grid = state.grid
    
    # A. Decay the old snake body (Subtract DECAY value)
    # This creates the "Time Gradient" (Head=1.0, Tail=Small)
    """DECAY VALUE EXPLANATION:
    We want the tail to approximately reach 0.0 when the snake is at max length.
    """
    DECAY = 0.005
    snake_channel = new_grid[..., 0]
    decayed_snake = jnp.where(snake_channel > 0, snake_channel - DECAY, 0.0)
    new_grid = new_grid.at[..., 0].set(decayed_snake)
    decayed_snake = jnp.maximum(decayed_snake, 0.0)
    
    # B. Clear the tail (if we didn't eat)
    tail_pos = state.body_buffer[state.tail_idx]

    # If we ate, we keep the tail (snake grew). If not, clear it.
    new_grid = new_grid.at[tail_pos[0], tail_pos[1], 0].set(
        jnp.where(hit_food, new_grid[tail_pos[0], tail_pos[1], 0], 0.0)
    )
    
    new_grid = new_grid.at[state.food_pos[0], state.food_pos[1], 0].set(0.0)
    
    # C. Paint the NEW Head (Always 1.0)
    new_grid = new_grid.at[safe_pos[0], safe_pos[1], 0].set(1.0)
    
    # 2. Generate a new position ONLY if we ate
    key, subkey = jax.random.split(state.key)
    
    # --- SMART SPAWN LOGIC START ---
    # We flatten the grid to 1D (size 100) to choose an index.
    # We look at Channel 0 (Snake Body). Any pixel > 0 is occupied.
    flat_body = new_grid[..., 0].ravel()
    
    # Create logits: 
    # If occupied (> 0), logit is -1e9 (Impossible).
    # If empty (== 0), logit is 1.0 (Possible).
    candidate_logits = jnp.where(flat_body > 0, -1e9, 1.0)
    
    # Sample a valid index from the empty spots
    # jax.random.categorical is robust and efficient here.
    new_food_idx = jax.random.categorical(subkey, candidate_logits)
    
    # Convert 1D index back to (y, x)
    new_food_y = new_food_idx // GRID_SIZE
    new_food_x = new_food_idx % GRID_SIZE
    smart_food_pos = jnp.array([new_food_y, new_food_x])
    # --- SMART SPAWN LOGIC END ---

    # Only apply the new position if we actually hit the food.
    # Otherwise, keep the old position.
    new_food_pos = jnp.where(hit_food, smart_food_pos, state.food_pos)
    
    # Paint the food on Channel 1
    new_grid = new_grid.at[new_food_pos[0], new_food_pos[1], 0].set(-1.0)

    # --- 6. REWARDS (PRO VERSION) ---
    
    # 1. Base Time Penalty (The "Tax" for living)
    # Encourages efficiency.
    reward = -0.01 
    
    # 2. Death Penalty
    # If done (and not a win), we punish heavily.
    # Note: We will handle the "Win" case below so we don't punish winning.
    reward = jnp.where(done, -1.0, reward)
    
    # 3. Scaled Food Reward (The "Gourmet")
    # Base 1.0 + (0.01 * length). 
    # Example: Length 10 -> +1.1 | Length 90 -> +1.9
    food_value = 1.0 # + (state.length * 0.01)
    reward = jnp.where(hit_food, food_value, reward)
    
    # 4. The "Grand Slam" (Win Condition)
    # If the snake fills the board (length >= 100), give massive payout.
    # We must override the 'done' penalty (-1.0) because winning also sets done=True!
    board_area = GRID_SIZE * GRID_SIZE
    is_win = (state.length >= board_area -1)

    done = done | is_win
    
    # Override: If we won, the reward is +10.0 (not -1.0)
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

# --- TEST IT ---
if __name__ == "__main__":
    # Compile the reset function for speed
    reset_fast = jax.jit(reset)
    step_fast = jax.jit(step)
    
    # Create 1 Game to test logic
    key = jax.random.PRNGKey(0)
    state = reset_fast(key)
    
    print("Initial Head:", state.head_pos)
    print("Initial Food:", state.food_pos)
    
    # Move Right (Action 1)
    state, reward, done = step_fast(state, 1)
    print("New Head:", state.head_pos)
    print("Grid Value at Head:", state.grid[state.head_pos[0], state.head_pos[1], 0])
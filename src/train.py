import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from snake_env import reset, step_batch as step, GRID_SIZE
from model import create_network
import time

# --- HYPERPARAMETERS ---
TOTAL_STEPS = 500_000_000 
NUM_ENVS = 4096
STEPS_PER_EPOCH = 128
LEARNING_RATE = 2.5e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
ENTROPY_COEF = 0.01
CRITIC_COEF = 0.5

NUM_UPDATES = TOTAL_STEPS // NUM_ENVS // STEPS_PER_EPOCH

class TrainState(TrainState):
    key: jax.Array

# --- HELPER: Turn State into GPS Coordinates ---
# --- HELPER: Relative GPS Coordinates ---
# train.py

# Remove the old GPS math. 
# We just want the raw grid state (Snake Body + Food).
def get_obs(state):
    # Returns shape (Batch, 10, 10, 2)
    return state.grid.astype(jnp.float32)


def create_train_state(rng, learning_rate):
    model = create_network()
    
    # OLD: dummy_input = jnp.zeros((1, 4))
    # NEW: Dummy input matches the Grid Shape (Batch, 10, 10, 2)
    dummy_input = jnp.zeros((1, 10, 10, 2)) 
    
    params = model.init(rng, dummy_input)
    tx = optax.adam(learning_rate)
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx, key=rng)

# --- ROLLOUT ---
def rollout(train_state, env_state):
    def step_fn(carry, _):
        t_state, e_state = carry
        key, action_key = jax.random.split(t_state.key)
        
        # 1. Get GPS Observation
        obs = get_obs(e_state)
        
        # 2. Run Model
        logits, value = t_state.apply_fn(t_state.params, obs)
        action = jax.random.categorical(action_key, logits)
        log_prob = jax.nn.log_softmax(logits)[jnp.arange(NUM_ENVS), action]
        
        # 3. Step
        next_e_state, reward, done = step(e_state, action)
        
        # --- CRITICAL FIX: Time Penalty ---
        # Penalize existing (-0.01) so it wants to eat FAST.
        # If it just dances up/down, it will lose points.
        reward = reward - 0.01
        
        t_state = t_state.replace(key=key)
        transition = (obs, action, reward, done, value, log_prob)
        return (t_state, next_e_state), transition

    (train_state, last_env_state), transitions = jax.lax.scan(
        step_fn, (train_state, env_state), None, length=STEPS_PER_EPOCH
    )
    
    # Get last value using last observation
    last_obs = get_obs(last_env_state)
    _, last_val = train_state.apply_fn(train_state.params, last_obs)
    
    return train_state, last_env_state, transitions, last_val

# --- GAE & UPDATE (Standard) ---
def calculate_gae(transitions, last_val):
    _, _, rewards, dones, values, _ = transitions
    last_val = last_val.squeeze()
    values = values.squeeze()
    
    def gae_scan(last_gae_and_val, item):
        reward, done, value = item
        last_gae, next_val = last_gae_and_val
        delta = reward + GAMMA * next_val * (1 - done) - value
        gae = delta + GAMMA * GAE_LAMBDA * (1 - done) * last_gae
        return (gae, value), gae

    _, advantages = jax.lax.scan(
        gae_scan, (jnp.zeros_like(last_val), last_val), 
        (rewards, dones, values), reverse=True
    )
    targets = advantages + values
    return advantages, targets

def update_step(train_state, batch):
    obs, actions, advantages, targets, old_log_probs = batch
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    def loss_fn(params):
        logits, values = train_state.apply_fn(params, obs)
        value_loss = jnp.mean(jnp.square(values.squeeze() - targets))
        log_probs = jax.nn.log_softmax(logits)
        new_log_probs = jnp.sum(log_probs * jax.nn.one_hot(actions, 4), axis=-1)
        ratio = jnp.exp(new_log_probs - old_log_probs)
        clip_ratio = jnp.clip(ratio, 0.8, 1.2)
        actor_loss = -jnp.mean(jnp.minimum(ratio * advantages, clip_ratio * advantages))
        probs = jax.nn.softmax(logits)
        entropy = -jnp.sum(probs * log_probs, axis=-1).mean()
        return actor_loss + CRITIC_COEF * value_loss - ENTROPY_COEF * entropy, (value_loss, actor_loss, entropy)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, metrics), grads = grad_fn(train_state.params)
    return train_state.apply_gradients(grads=grads), metrics

# --- MAIN ---
def run_training():
    print(f"Devices: {jax.devices()}")
    rng = jax.random.PRNGKey(42)
    rng, init_rng = jax.random.split(rng)
    
    # Initialize the Trainer
    train_state = create_train_state(init_rng, LEARNING_RATE)
    
    # Initialize the Environments
    print("Initializing Environments...")
    rng, env_rng = jax.random.split(rng)
    env_keys = jax.random.split(env_rng, NUM_ENVS)
    env_state = jax.vmap(reset)(env_keys)
    
    # Define the Training Epoch
    def train_epoch(carry, _):
        t_state, e_state = carry
        
        # 1. Rollout (Play Game)
        t_state, e_state, transitions, last_val = rollout(t_state, e_state)
        obs, actions, rewards, dones, values, old_log_probs = transitions
        
        # 2. Calculate Advantage (Math)
        advantages, targets = calculate_gae(transitions, last_val)
        
        # 3. Flatten Data for Training
        def flatten(x): return x.reshape((STEPS_PER_EPOCH * NUM_ENVS, -1))
        
        # --- FIX THIS SECTION ---
        batch = (
            obs.reshape((STEPS_PER_EPOCH * NUM_ENVS, 10, 10, 2)), 
            
            flatten(actions).squeeze(),
            flatten(advantages).squeeze(),
            flatten(targets).squeeze(),
            flatten(old_log_probs).squeeze()
        )
        
        # 4. Update Weights
        t_state, metrics = update_step(t_state, batch)
        
        return (t_state, e_state), (metrics, jnp.mean(rewards))

    # Compile the loop
    print("Compiling...")
    train_scan = jax.jit(lambda t, e: jax.lax.scan(train_epoch, (t, e), None, length=NUM_UPDATES))
    
    # Run the training
    print(f"Training Started for {TOTAL_STEPS:,.0f} steps...")
    start = time.time()
    
    (final_state, _), (metrics, rewards) = train_scan(train_state, env_state)
    
    # IMPORTANT: Force JAX to finish all math before stopping the clock
    rewards.block_until_ready()
    end = time.time()
    
    # --- STATISTICS ---
    duration = end - start
    steps_per_sec = TOTAL_STEPS / duration
    
    print(f"Done in {duration:.2f}s")
    print(f"Steps/Sec: {steps_per_sec:,.0f}")  # <--- HERE IS YOUR STATUS
    print(f"Final Reward: {rewards[-1]:.3f}")
    
    return final_state



if __name__ == "__main__":
    state = run_training()
    import pickle
    with open("snake_params.pkl", "wb") as f:
        pickle.dump(state.params, f)
    print("Saved model.")
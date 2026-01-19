import jax
import jax.numpy as jnp
from flax import linen as nn

class ActorCritic(nn.Module):
    @nn.compact
    def __call__(self, x):
        # Input: (Batch, 10, 10, 2)

        # --- 1. SPATIAL PROCESSING (The Eyes) ---
        # Layer 1: 32 Filters. Finds edges, corners, food.
        # Kernel (3,3) means it looks at immediate neighbors.
        x = nn.Conv(features=64, kernel_size=(3, 3), padding='SAME', dtype=jnp.float32)(x)
        x = nn.relu(x)

        x = nn.Conv(features=128, kernel_size=(3, 3), padding='SAME', dtype=jnp.float32)(x)
        x = nn.relu(x)

        # Layer 2: 64 Filters. Finds patterns like "Head pointing at Wall".
        # We double the features because combinations are more complex than raw pixels.x = nn.Conv(features=64, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.relu(x) 
        x = x.reshape((x.shape[0], -1)) 
        
        # Layer 1: The "Eye" (Wide)
        # 512 neurons to capture every possible detail of the 200-pixel input.
        x = nn.Dense(features=512)(x)
        x = nn.relu(x)

        # Outputs
        actor_logits = nn.Dense(features=4)(x)
        critic_value = nn.Dense(features=1)(x)

        return actor_logits, critic_value

def create_network():
    return ActorCritic()
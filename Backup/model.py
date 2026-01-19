import jax
import jax.numpy as jnp
from flax import linen as nn

class ActorCritic(nn.Module):
    @nn.compact
    def __call__(self, x):
        
        # Input: (Batch, 10, 10, 2)
        x = x.reshape((x.shape[0], -1)) 
        
        # Layer 1: The "Eye" (Wide)
        # 512 neurons to capture every possible detail of the 200-pixel input.
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        
        # Layer 2: The "Brain" (Medium)
        # 256 neurons to combine features (e.g., "Wall + Food").
        x = nn.Dense(features=128)(x)
        x = nn.relu(x)

        # Layer 3: The "Reflex" (Condensed)
        # 128 neurons to finalize the decision.
        x = nn.Dense(features=64)(x)
        x = nn.relu(x)

        # Outputs
        actor_logits = nn.Dense(features=4)(x)
        critic_value = nn.Dense(features=1)(x)

        return actor_logits, critic_value

def create_network():
    return ActorCritic()
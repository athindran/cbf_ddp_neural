import numpy as np
import jax
from jax import numpy as jnp
from simulators import BasePolicy
from typing import Dict, Optional

class LinearPolicy(BasePolicy):
    def __init__(
        self, config, gain_matrix, target_indices, endpoint_indices
    ) -> None:
        #super().__init__(config)
        self.gain_matrix = gain_matrix
        self.target_indices = target_indices
        self.endpoint_indices = endpoint_indices

    def get_action(
        self, obs: jax.Array
    ) -> jax.Array:
        def true_fun(args):
            control_init, zero_first_control = args
            return control_init

        def false_fun(args):
            control_init, zero_first_control = args
            return zero_first_control

        endpoint = obs[self.endpoint_indices]
        target = obs[self.target_indices]
        error = (target - endpoint)
        control_init = -self.gain_matrix @ (target - endpoint)
        cond_near_endpoint = jnp.linalg.norm(error) < 0.2
        
        zero_first_control = control_init.at[1].set(0.0)
        control = jax.lax.cond(cond_near_endpoint, true_fun, false_fun, (control_init, zero_first_control,))

        print(control)

        return control, {'id': 'linear_policy'}
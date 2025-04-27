from functools import partial
import jax
from jax import numpy as jnp
from jax import Array
from brax import envs
from brax.envs.base import State
from abc import ABC

class WrappedBraxEnv(ABC):
    def __init__(self, env_name, backend) -> None:
        self.env = envs.get_environment(env_name=env_name,
                           backend=backend)
        self.dim_x = 8
        self.dim_u = 2
    
    @partial(jax.jit, static_argnames='self')
    def step(self, state, action) -> jax.Array:
        return self.env.step(state, action)
    
    @partial(jax.jit, static_argnames='self')
    def get_generalized_coordinates(self, state) -> jax.Array:
        return jnp.concatenate([state.pipeline_state.q, state.pipeline_state.qd], axis=-1) 

    @partial(jax.jit, static_argnames=['self'])
    def step_generalized_coordinates(self, state, action):
        new_state = self.env.step(state, action)
        new_generalized_coordinates = self.get_generalized_coordinates(new_state)
        return new_generalized_coordinates
    
    @partial(jax.jit, static_argnames=['self'])
    def get_generalized_coordinates_grad(self, state, action):
        state_grad = jax.jacobian(self.step_generalized_coordinates, argnums=0)(state, action).pipeline_state
        action_grad = jax.jacobian(self.step_generalized_coordinates, argnums=1)(state, action)
        return jnp.concatenate([state_grad.q, state_grad.qd], axis=-1), action_grad

    @partial(jax.jit, static_argnames=['self'])
    def get_obs_grad(self, pipeline_state):
        state_grad = jax.jacobian(self._get_obs, argnums=0)(pipeline_state)
        return jnp.concatenate([state_grad.q, state_grad.qd], axis=-1)

    @partial(jax.jit, static_argnames='self')    
    def reset(self, rng) -> jax.Array:
        return self.env.reset(rng)
    
    @partial(jax.jit, static_argnames='self')
    def _get_obs(self, pipeline_state: State) -> jax.Array:
        return self.env._get_obs(pipeline_state)

    @partial(jax.jit, static_argnames='self')
    def get_fingertip(self, generalized_coordinates: jax.Array) -> jax.Array:
        fingertip = jnp.zeros((2,))
        fingertip = fingertip.at[0].set(0.1*jnp.cos(generalized_coordinates[0]) + 0.11*jnp.cos(generalized_coordinates[0] + generalized_coordinates[1]))
        fingertip = fingertip.at[1].set(0.1*jnp.sin(generalized_coordinates[0]) + 0.11*jnp.sin(generalized_coordinates[0] + generalized_coordinates[1]))
        return fingertip


from functools import partial
import jax
from jax import numpy as jnp
from jax import Array
from typing import Tuple
from brax import envs
from brax.envs.base import State
from abc import ABC

env_name = 'reacher'  # @param ['ant', 'halfcheetah', 'hopper', 'humanoid', 'humanoidstandup', 'inverted_pendulum', 'inverted_double_pendulum', 'pusher', 'reacher', 'walker2d']
backend = 'generalized'  # @param ['generalized', 'positional', 'spring']

env = envs.get_environment(env_name=env_name,
                           backend=backend)

class WrappedBraxEnv(ABC):
    def __init__(self, env_name, backend) -> None:
        self.env = envs.get_environment(env_name=env_name,
                           backend=backend)
    
    @partial(jax.jit, static_argnames='self')
    def step(self, state, action) -> jax.Array:
        return self.env.step(state, action)
    
    @partial(jax.jit, static_argnames='self')
    def get_trimmed_state(self, state) -> jax.Array:
        return jnp.concatenate([state.pipeline_state.q, state.pipeline_state.qd], axis=-1) 

    @partial(jax.jit, static_argnames=['self'])
    def step_obs(self, state, action):
        new_state = self.env.step(state, action)
        new_trimmed_state = self.get_trimmed_state(new_state)
        return new_trimmed_state
    
    @partial(jax.jit, static_argnames=['self'])
    def get_trimmed_state_grad(self, state, action):
        state_grad = jax.jacobian(self.step_obs, argnums=0)(state, action).pipeline_state
        actor_grad = jax.jacobian(self.step_obs, argnums=1)(state, action)
        return jnp.concatenate([state_grad.q, state_grad.qd], axis=-1), actor_grad

    @partial(jax.jit, static_argnames='self')    
    def reset(self, rng) -> jax.Array:
        return self.env.reset(rng)
    
    @partial(jax.jit, static_argnames='self')
    def _get_obs(self, pipeline_state: State) -> jax.Array:
        return self.env._get_obs(pipeline_state)


from functools import partial
import jax
from jax import numpy as jnp
from jax import Array
from brax import envs
from brax.envs.base import State
from abc import ABC
from matplotlib import pyplot as plt
import numpy as np
import os


class WrappedBraxEnv(ABC):
    def __init__(self, env_name, backend) -> None:
        self.env = envs.get_environment(env_name=env_name,
                           backend=backend)
        if env_name=='reacher':
            self.dim_x = 8
            self.dim_u = 2
        else:
            # Raise not implemented error.
            pass
        self.env_name = env_name
    
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

    def plot_states_and_controls(self, states, ctrls, policy_type, save_folder):
        if(self.env_name == 'reacher'):
            fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True)
            axes[0, 0].plot(states[:, 0])
            axes[0, 0].set_ylabel('q0')
            axes[0, 1].plot(states[:, 1])
            axes[0, 1].set_ylabel('q1')
            axes[1, 0].plot(states[:, 4])
            axes[1, 0].set_ylabel('q0d')
            axes[1, 1].plot(states[:, 5])
            axes[1, 1].set_ylabel('q1d')
            axes[1, 0].set_xlabel('Timesteps')
            axes[1, 1].set_xlabel('Timesteps')
            fig.suptitle(f'Policy: {policy_type}, Environment: {self.env_name}', fontsize=18)
            fig.savefig(os.path.join(save_folder, f'{policy_type}_states.png'))

            fig, axes = plt.subplots(1, 2, figsize=(9, 4))
            axes[0].plot(ctrls[:, 0])
            axes[0].set_ylabel('Action 0')
            axes[0].set_xlabel('Timesteps')
            axes[1].plot(ctrls[:, 1])
            axes[1].set_ylabel('Action 1')
            axes[1].set_xlabel('Timesteps')
            fig.suptitle(f'Policy: {policy_type}, Environment: {self.env_name}', fontsize=18)
            fig.savefig(os.path.join(save_folder, f'{policy_type}_actions.png'))


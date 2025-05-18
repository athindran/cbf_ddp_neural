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
            self.dim_x = 4
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
        return jnp.concatenate([state.pipeline_state.q[0:2], state.pipeline_state.qd[0:2]], axis=-1) 

    @partial(jax.jit, static_argnames=['self'])
    def step_generalized_coordinates(self, state, action):
        new_state = self.env.step(state, action)
        new_generalized_coordinates = self.get_generalized_coordinates(new_state)
        return new_generalized_coordinates
    
    @partial(jax.jit, static_argnames=['self'])
    def get_generalized_coordinates_grad(self, state, action):
        state_grad = jax.jacobian(self.step_generalized_coordinates, argnums=0)(state, action).pipeline_state
        action_grad = jax.jacobian(self.step_generalized_coordinates, argnums=1)(state, action)
        return jnp.concatenate([state_grad.q[..., 0:2], state_grad.qd[..., 0:2]], axis=-1), action_grad

    @partial(jax.jit, static_argnames=['self'])
    def get_obs_grad(self, pipeline_state):
        state_grad = jax.jacobian(self._get_obs, argnums=0)(pipeline_state)
        return jnp.concatenate([state_grad.q[..., 0:2], state_grad.qd[..., 0:2]], axis=-1)

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

    def plot_states_and_controls(self, save_dict, save_folder):
        states = save_dict['gc_states']
        ctrls = save_dict['actions']
        control_cycle_times = save_dict['process_times']
        values = save_dict['values']
        policy_type = save_dict['policy_type']
        is_filter_active = save_dict['filter_active']
        nsteps = states.shape[0]
        range_space = np.arange(0, nsteps)

        if(self.env_name == 'reacher'):
            fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True)
            axes[0, 0].plot(states[:, 0])
            axes[0, 0].set_ylabel('q0')
            axes[0, 1].plot(states[:, 1])
            axes[0, 1].set_ylabel('q1')
            axes[1, 0].plot(states[:, 2])
            axes[1, 0].set_ylabel('q0d')
            axes[1, 1].plot(states[:, 3])
            axes[1, 1].set_ylabel('q1d')
            axes[1, 0].set_xlabel('Timesteps')
            axes[1, 1].set_xlabel('Timesteps')
            fig.suptitle(f'Policy: {policy_type}, Environment: {self.env_name}', fontsize=14)
            fig.savefig(os.path.join(save_folder, f'{policy_type}_states.png'))
            plt.close()

            fig, axes = plt.subplots(1, 2, figsize=(9, 4))
            axes[0].plot(ctrls[:, 0])
            axes[0].set_ylabel('Action 0')
            axes[0].set_xlabel('Timesteps')
            axes[1].plot(ctrls[:, 1])
            axes[1].set_ylabel('Action 1')
            axes[1].set_xlabel('Timesteps')
            fig.suptitle(f'Policy: {policy_type}, Environment: {self.env_name}', fontsize=14)
            axes[0].fill_between(range_space, -0.25, 0.25,
                                     where=is_filter_active[0:nsteps], color='b', alpha=0.35)
            axes[1].fill_between(range_space, -0.25, 0.25, 
                                    where=is_filter_active[0:nsteps], color='b', alpha=0.35)
 
            fig.savefig(os.path.join(save_folder, f'{policy_type}_actions.png'))
            plt.close()

            fig = plt.figure(figsize=(5.5, 3.5))
            ax = plt.gca()
            ax.plot(control_cycle_times)
            ax.set_ylabel('Cycle time(s)')
            ax.set_xlabel('Timesteps')
            fig.suptitle(f'Policy: {policy_type}, Environment: {self.env_name}', fontsize=14)
            fig.savefig(os.path.join(save_folder, f'{policy_type}_process_times.png'))
            plt.close()

            fig = plt.figure(figsize=(5.5, 3.5))
            ax = plt.gca()
            ax.plot(values)
            ax.set_ylabel('Reachability value')
            ax.set_xlabel('Timesteps')
            ax.fill_between(range_space, -1.0, 200.0, 
                                where=is_filter_active[0:nsteps], color='b', alpha=0.35)
            ax.set_ylim([-1.0, values.max()])
            fig.suptitle(f'Policy: {policy_type}, Environment: {self.env_name}', fontsize=14)
            fig.savefig(os.path.join(save_folder, f'{policy_type}_values.png'))
            plt.close()


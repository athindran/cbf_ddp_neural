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
import math


class WrappedBraxEnv(ABC):
    def __init__(self, env_name, backend) -> None:
        self.env = envs.get_environment(env_name=env_name,
                           backend=backend)
        if env_name=='reacher':
            self.dim_x = 4
            self.dim_u = 2
            self.dim_q_states = 2
            self.dim_qd_states = 2
            self.action_limits = np.array([[-1., -1.], [1., 1.]])
        elif env_name=='ant':
            self.dim_x = 29
            self.dim_u = 8
            self.dim_q_states = 15
            self.dim_qd_states = 14
            self.action_limits = np.array([[-1., -1., -1., -1., -1., -1., -1., -1.], [1., 1., 1., 1., 1., 1., 1., 1.]])
        else:
            # Raise not implemented error.
            pass
        self.env_name = env_name
    
    @partial(jax.jit, static_argnames='self')
    def step(self, state, action) -> jax.Array:
        return self.env.step(state, action)
    
    @partial(jax.jit, static_argnames='self')
    def get_generalized_coordinates(self, state) -> jax.Array:
        return jnp.concatenate([state.pipeline_state.qpos[0:self.dim_q_states], state.pipeline_state.qvel[0:self.dim_qd_states]], axis=-1) 

    @partial(jax.jit, static_argnames=['self'])
    def step_generalized_coordinates(self, state, qpos, qvel, action):
        pipeline_state_qqd = state.pipeline_state.tree_replace({'qpos': qpos, 'qvel': qvel})
        state_qqd = state.tree_replace({'pipeline_state': pipeline_state_qqd})
        new_state = self.env.step(state_qqd, action)
        new_generalized_coordinates = self.get_generalized_coordinates(new_state)
        return new_generalized_coordinates
    
    @partial(jax.jit, static_argnames=['self'])
    def get_generalized_coordinates_grad(self, state, action):
        qpos = jnp.array(state.pipeline_state.qpos)
        qvel = jnp.array(state.pipeline_state.qvel)
        q_grad = jax.jacfwd(self.step_generalized_coordinates, argnums=1)(state, qpos, qvel,  action)
        qd_grad = jax.jacfwd(self.step_generalized_coordinates, argnums=2)(state, qpos, qvel, action)
        action_grad = jax.jacfwd(self.step_generalized_coordinates, argnums=3)(state, qpos, qvel, action)
        return jnp.concatenate([q_grad[..., 0:self.dim_q_states], qd_grad[..., 0:self.dim_qd_states]], axis=-1), action_grad

    @partial(jax.jit, static_argnames=['self'])
    def get_obs_grad(self, pipeline_state):
        state_grad = jax.jacobian(self._get_obs, argnums=0)(pipeline_state)
        return jnp.concatenate([state_grad.q[..., 0:self.dim_q_states], state_grad.qd[..., 0:self.dim_qd_states]], axis=-1)

    @partial(jax.jit, static_argnames='self')    
    def reset(self, rng) -> jax.Array:
        return self.env.reset(rng)
    
    @partial(jax.jit, static_argnames='self')
    def _get_obs(self, pipeline_state: State) -> jax.Array:
        return self.env._get_obs(pipeline_state)

    def plot_states_and_controls(self, save_dict, save_folder):
        states = save_dict['gc_states']
        ctrls = save_dict['actions']
        control_cycle_times = save_dict['process_times']
        values = save_dict['values']
        policy_type = save_dict['policy_type']
        is_filter_active = save_dict['filter_active']
        nsteps = states.shape[0]
        range_space = np.arange(0, nsteps)

        rowdict = {'ant': 4, 'reacher': 2}
        nrows = rowdict[self.env_name]
        ncols = math.ceil(self.dim_q_states/nrows)
        figsize = {'ant': (22, 16), 'reacher': (9, 4)}
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize[self.env_name], sharex=True)
        axes = axes.ravel()
        for idx in range(self.dim_q_states):
            axes[idx].plot(states[:, idx])
            axes[idx].set_ylabel(f'q {idx}')
            axes[idx].set_xlabel('Timesteps')
            min_r = states[:, idx].min()
            max_r = states[:, idx].max()
            axes[idx].fill_between(range_space, 1.2*min_r, 1.2*max_r,
                                    where=is_filter_active[0:nsteps], color='b', alpha=0.35)

        fig.suptitle(f'Policy: {policy_type}, Environment: {self.env_name}', fontsize=14)
        fig.savefig(os.path.join(save_folder, f'{policy_type}_q_states.png'))
        plt.close()

        nrows = rowdict[self.env_name]
        ncols = math.ceil(self.dim_qd_states/nrows)
        figsize = {'ant': (22, 16), 'reacher': (9, 4)}
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize[self.env_name], sharex=True)
        axes = axes.ravel()
        for idx in range(self.dim_qd_states):
            axes[idx].plot(states[:, self.dim_q_states + idx])
            axes[idx].set_ylabel(f'qd {idx}')
            axes[idx].set_xlabel('Timesteps')
            min_r = states[:, self.dim_q_states + idx].min()
            max_r = states[:, self.dim_q_states + idx].max()
            axes[idx].fill_between(range_space, 1.2*min_r, 1.2*max_r,
                                    where=is_filter_active[0:nsteps], color='b', alpha=0.35)

        fig.suptitle(f'Policy: {policy_type}, Environment: {self.env_name}', fontsize=14)
        fig.savefig(os.path.join(save_folder, f'{policy_type}_qd_states.png'))
        plt.close()

        rowdict = {'ant': 2, 'reacher': 1}
        figsize = {'ant': (18, 9), 'reacher': (9, 4)}
        nrows = rowdict[self.env_name]
        ncols = math.ceil(self.dim_u/nrows)

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize[self.env_name])
        axes = axes.ravel()
        for idx in range(self.dim_u):
            axes[idx].plot(ctrls[:, idx])
            axes[idx].set_ylabel(f'Action {idx}')
            axes[idx].set_xlabel('Timesteps')
            axes[idx].set_ylim([-1.0, 1.0])
            axes[idx].fill_between(range_space, -1.0, 1.0,
                                    where=is_filter_active[0:nsteps], color='b', alpha=0.35)

        fig.suptitle(f'Policy: {policy_type}, Environment: {self.env_name}', fontsize=14)
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
        ax.fill_between(range_space, values.min(), values.max()*2.0, 
                            where=is_filter_active[0:nsteps], color='b', alpha=0.35)
        ax.set_ylim([values.min(), values.max()*2.0])
        fig.suptitle(f'Policy: {policy_type}, Environment: {self.env_name}', fontsize=14)
        fig.savefig(os.path.join(save_folder, f'{policy_type}_values.png'), bbox_inches='tight')
        plt.close()



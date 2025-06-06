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
            self.dim_qdd_states = 0
            self.action_limits = np.array([[-1., -1.], [1., 1.]])
            self.dt = 0.02
        elif env_name=='ant':
            self.dim_x = 29
            self.dim_u = 8
            self.dim_q_states = 15
            self.dim_qd_states = 14
            self.dim_qdd_states = 0
            self.action_limits = np.array([[-1., -1., -1., -1., -1., -1., -1., -1.], [1., 1., 1., 1., 1., 1., 1., 1.]])
            self.dt = 0.05
        else:
            # Raise not implemented error.
            pass
        self.env_name = env_name
    
    @partial(jax.jit, static_argnames='self')
    def step(self, state, action) -> jax.Array:
        return self.env.step(state, action)
    
    @partial(jax.jit, static_argnames='self')
    def get_generalized_coordinates(self, state) -> jax.Array:
        return jnp.concatenate([state.pipeline_state.q[0:self.dim_q_states], state.pipeline_state.qd[0:self.dim_qd_states], state.pipeline_state.qdd[0:self.dim_qdd_states]], axis=-1) 

    @partial(jax.jit, static_argnames=['self'])
    def step_generalized_coordinates(self, state, action):
        new_state = self.env.step(state, action)
        new_generalized_coordinates = self.get_generalized_coordinates(new_state)
        return new_generalized_coordinates
    
    @partial(jax.jit, static_argnames=['self'])
    def get_generalized_coordinates_grad(self, state, action):
        state_grad = jax.jacobian(self.step_generalized_coordinates, argnums=0)(state, action).pipeline_state
        action_grad = jax.jacobian(self.step_generalized_coordinates, argnums=1)(state, action)
        return jnp.concatenate([state_grad.q[..., 0:self.dim_q_states], state_grad.qd[..., 0:self.dim_qd_states], state_grad.qdd[..., 0:self.dim_qdd_states]], axis=-1), action_grad
    
    @partial(jax.jit, static_argnames=['self'])
    def get_generalized_coordinates_hess(self, state, action):
        # unused currently as too slow to extract hessian without vmap.
        hess_xx, hess_ux = jax.jacfwd(self.get_generalized_coordinates_grad, argnums=0)(state, action)
        _, hess_uu = jax.jacfwd(self.get_generalized_coordinates_grad, argnums=1)(state, action)

        f_xx = jnp.concatenate([hess_xx.pipeline_state.q[..., 0:self.dim_q_states], 
                                hess_xx.pipeline_state.qd[..., 0:self.dim_qd_states],
                                hess_xx.pipeline_state.qdd[..., 0:self.dim_qdd_states]], axis=-1)

        f_ux = jnp.concatenate([hess_ux.pipeline_state.q[..., 0:self.dim_q_states], 
                                hess_ux.pipeline_state.qd[..., 0:self.dim_qd_states],
                                hess_ux.pipeline_state.qdd[..., 0:self.dim_qdd_states]], axis=-1)

        return f_xx, f_ux, hess_uu

    @partial(jax.jit, static_argnames=['self'])
    def get_batched_generalized_coordinates_grad(self, nominal_states, nominal_actions):
        jac = jax.jit(
            jax.vmap(
                self.get_generalized_coordinates_grad, in_axes=(
                    -1, -1), out_axes=(
                    2, 2)))
        return jac(nominal_states, nominal_actions)

    @partial(jax.jit, static_argnames=['self'])
    def get_obs_grad(self, pipeline_state):
        state_grad = jax.jacobian(self._get_obs, argnums=0)(pipeline_state)
        return jnp.concatenate([state_grad.q[..., 0:self.dim_q_states], state_grad.qd[..., 0:self.dim_qd_states], state_grad.qdd[..., 0:self.dim_qdd_states]], axis=-1)

    @partial(jax.jit, static_argnames='self')    
    def reset(self, rng) -> jax.Array:
        return self.env.reset(rng)
    
    @partial(jax.jit, static_argnames='self')
    def _get_obs(self, pipeline_state: State) -> jax.Array:
        return self.env._get_obs(pipeline_state)

    def test_gc_rollout(self):
        # FAILS - to provide information about MJX
        rng = jax.random.PRNGKey(seed=0)
        test_state = self.reset(rng=rng)

        for _ in range(100):
            action = np.random.rand(self.dim_u)
            new_state = self.step(test_state, action)
            print(test_state.pipeline_state.qpos, test_state.pipeline_state.qvel)
            new_gc_state = self.step_generalized_coordinates(test_state, test_state.pipeline_state.qpos, test_state.pipeline_state.qvel, action)

            np.testing.assert_allclose(new_state.pipeline_state.qpos, new_gc_state[0:self.dim_q_states], atol=1e-4, rtol=1e-4)
            np.testing.assert_allclose(new_state.pipeline_state.qvel, new_gc_state[self.dim_q_states:], atol=1e-4, rtol=1e-4)

            test_state = new_state

        print("Unit test passed")

        return True

    def plot_states_and_controls(self, save_dict, save_folder):
        states = save_dict['gc_states']
        ctrls = save_dict['actions']
        control_cycle_times = save_dict['process_times']
        values = save_dict['values']
        policy_type = save_dict['policy_type']
        is_filter_active = save_dict['filter_active']
        is_filter_fail = save_dict['filter_failed']
        nsteps = states.shape[0]
        range_space = np.arange(0, nsteps) * self.dt

        rowdict = {'ant': 4, 'reacher': 2}
        nrows = rowdict[self.env_name]
        ncols = math.ceil(self.dim_q_states/nrows)
        figsize = {'ant': (35, 16), 'reacher': (9, 4)}
        fontsize = {'ant': 14, 'reacher': 10}
        linewidths = {'ant': 1.5, 'reacher': 2.5}
        legend_fontsize = fontsize[self.env_name]
        linewidth = linewidths[self.env_name]

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize[self.env_name], sharex=True)
        axes = axes.ravel()
        for idx in range(self.dim_q_states):
            axes[idx].plot(range_space, states[:, idx], linewidth=linewidth)
            axes[idx].set_ylabel(f'q {idx}', fontsize=legend_fontsize)
            axes[idx].set_xlabel('Time (s)', fontsize=legend_fontsize)
            min_r = states[:, idx].min()
            max_r = states[:, idx].max()
            axes[idx].fill_between(range_space, 1.2*min_r, 1.2*max_r,
                                    where=is_filter_active[0:nsteps], color='b', alpha=0.15)
            axes[idx].yaxis.set_label_coords(-0.04, 0.5)
            axes[idx].xaxis.set_label_coords(0.5, -0.04)
            axes[idx].set_xticks(ticks=[0, round(self.dt*nsteps, 2)], labels=[0, round(self.dt*nsteps, 2)], fontsize=legend_fontsize)
            axes[idx].set_yticks(ticks=[round(states[:, idx].min(), 2), round(states[:, idx].max(), 2)], 
            labels=[round(states[:, idx].min(), 2), round(states[:, idx].max(), 2)], 
            fontsize=legend_fontsize)


        fig.suptitle(f'Policy: {policy_type}, Environment: {self.env_name}', fontsize=legend_fontsize)
        fig.savefig(os.path.join(save_folder, f'{policy_type}_q_states.png'), bbox_inches='tight')
        plt.close()

        nrows = rowdict[self.env_name]
        ncols = math.ceil(self.dim_qd_states/nrows)
        figsize = {'ant': (35, 16), 'reacher': (9, 4)}
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize[self.env_name], sharex=True)
        axes = axes.ravel()
        for idx in range(self.dim_qd_states):
            axes[idx].plot(range_space, states[:, self.dim_q_states + idx], linewidth=linewidth)
            axes[idx].set_ylabel(f'qd {idx}', fontsize=legend_fontsize)
            axes[idx].set_xlabel('Time (s)', fontsize=legend_fontsize)
            min_r = states[:, self.dim_q_states + idx].min()
            max_r = states[:, self.dim_q_states + idx].max()
            axes[idx].fill_between(range_space, 1.2*min_r, 1.2*max_r,
                                    where=is_filter_active[0:nsteps], color='b', alpha=0.15)
            axes[idx].yaxis.set_label_coords(-0.04, 0.5)
            axes[idx].xaxis.set_label_coords(0.5, -0.04)
            axes[idx].set_xticks(ticks=[0, round(self.dt*nsteps, 2)], labels=[0, round(self.dt*nsteps, 2)], fontsize=legend_fontsize)
            axes[idx].set_yticks(ticks=[round(states[:, self.dim_q_states + idx].min(), 2), round(states[:, self.dim_q_states + idx].max(), 2)], 
                        labels=[round(states[:, self.dim_q_states + idx].min(), 2), round(states[:, self.dim_q_states + idx].max(), 2)], 
                        fontsize=legend_fontsize)

        fig.suptitle(f'Policy: {policy_type}, Environment: {self.env_name}', fontsize=legend_fontsize)
        fig.savefig(os.path.join(save_folder, f'{policy_type}_qd_states.png'), bbox_inches='tight')
        plt.close()

        rowdict = {'ant': 2, 'reacher': 1}
        figsize = {'ant': (35, 9), 'reacher': (9, 4)}
        nrows = rowdict[self.env_name]
        ncols = math.ceil(self.dim_u/nrows)

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize[self.env_name])
        axes = axes.ravel()
        for idx in range(self.dim_u):
            axes[idx].plot(range_space, ctrls[:, idx], linewidth=linewidth)
            axes[idx].set_ylim([-1.0, 1.0])
            axes[idx].fill_between(range_space, -1.0, 1.0,
                                    where=is_filter_active[0:nsteps], color='b', alpha=0.15)
            axes[idx].fill_between(range_space, -1.0, 1.0,
                            where=is_filter_fail[0:nsteps], color='r', alpha=0.15)
            axes[idx].yaxis.set_label_coords(-0.04, 0.5)
            axes[idx].xaxis.set_label_coords(0.5, -0.04)
            axes[idx].set_xticks(ticks=[0, round(self.dt*nsteps, 2)], labels=[0, round(self.dt*nsteps, 2)], fontsize=legend_fontsize)
            axes[idx].set_yticks(ticks=[-1.0, 1.0], 
                        labels=[-1.0, 1.0], 
                        fontsize=legend_fontsize)
            axes[idx].set_xlabel('Time (s)', 
                        fontsize=legend_fontsize)
            axes[idx].set_ylabel('Control ' + str(idx), 
                        fontsize=legend_fontsize)


        fig.suptitle(f'Policy: {policy_type}, Environment: {self.env_name}', fontsize=legend_fontsize)
        fig.savefig(os.path.join(save_folder, f'{policy_type}_actions.png'), bbox_inches='tight')
        plt.close()

        fig = plt.figure(figsize=(7.5, 3.5))
        ax = plt.gca()
        ax.plot(range_space, control_cycle_times, linewidth=linewidth)
        ax.yaxis.set_label_coords(-0.04, 0.5)
        ax.xaxis.set_label_coords(0.5, -0.04)
        ax.set_xticks(ticks=[0, round(self.dt*nsteps, 2)], labels=[0, round(self.dt*nsteps, 2)], fontsize=legend_fontsize)
        ax.set_yticks(ticks=[round(control_cycle_times.min(), 4), round(control_cycle_times.max(), 4)], 
                        labels=[round(control_cycle_times.min(), 4), round(control_cycle_times.max(), 4)], 
                        fontsize=legend_fontsize)
        ax.set_ylim([round(control_cycle_times.min(), 4), round(control_cycle_times.max(), 4)])
        ax.set_xlabel('Time (s)', 
                        fontsize=legend_fontsize)
        ax.set_ylabel('Control cycle time (s)', 
                        fontsize=legend_fontsize)
        fig.suptitle(f'Policy: {policy_type}, Environment: {self.env_name}', fontsize=legend_fontsize)
        fig.savefig(os.path.join(save_folder, f'{policy_type}_process_times.png'), bbox_inches='tight')
        plt.close()

        fig = plt.figure(figsize=(9.5, 3.5))
        ax = plt.gca()
        ax.plot(range_space, values, linewidth=linewidth)
        ax.plot(range_space, np.zeros_like(range_space), 'k--')
        ax.fill_between(range_space, min(values.min(), 0.0), values.max()*2.0, 
                            where=is_filter_active[0:nsteps], color='b', alpha=0.15)
        ax.set_ylim([min(round(values.min(), 2), 0.0), round(values.max()*1.2, 2)])
        ax.yaxis.set_label_coords(-0.04, 0.5)
        ax.xaxis.set_label_coords(0.5, -0.04)
        ax.set_xticks(ticks=[0, round(self.dt*nsteps, 2)], labels=[0, round(self.dt*nsteps, 2)], fontsize=legend_fontsize)
        ax.set_yticks(ticks=[min(round(values.min(), 2), 0.0), round(values.max()*1.2, 2)], 
                        labels=[min(round(values.min(), 2), 0.0), round(values.max()*1.2, 2)], 
                        fontsize=legend_fontsize)
        ax.set_xlabel('Time (s)', 
                        fontsize=legend_fontsize)
        ax.set_ylabel('Value function', 
                        fontsize=legend_fontsize)
        fig.suptitle(f'Policy: {policy_type}, Environment: {self.env_name}', fontsize=legend_fontsize)
        fig.savefig(os.path.join(save_folder, f'{policy_type}_values.png'), bbox_inches='tight')
        plt.close()


class WrappedMJXEnv(WrappedBraxEnv):

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

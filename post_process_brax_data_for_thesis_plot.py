import numpy as np
import time
import jax

from matplotlib import pyplot as plt
from brax_utils import ( WrappedBraxEnv, 
      WrappedMJXEnv, 
      ReacherRegularizedGoalCost, 
      iLQRBrax, 
      iLQRBraxReachability, 
      iLQRBraxSafetyFilter,
      ReacherReachabilityMargin,
      AntReachabilityMargin, )


def get_brax_env(env_name, backend):
    if backend=='mjx':
        return WrappedMJXEnv(env_name, backend)
    else:
        return WrappedBraxEnv(env_name, backend)


def make_barkour_reachability_plot():
    barkour_cbfddp_save_folder = './brax_videos/barkour/seed_0/ilqr_filter_with_neural_policy_Reachability_save_data.npy'
    barkour_lrddp_save_folder = './brax_videos/barkour/seed_0/lr_filter_with_neural_policy_Reachability_save_data.npy'
    barkour_neural_save_folder = './brax_videos/barkour/seed_0/neural_Reachability_save_data.npy'

    barkour_cbfddp_data = np.load(barkour_cbfddp_save_folder, allow_pickle=True)
    barkour_lrddp_data = np.load(barkour_lrddp_save_folder, allow_pickle=True)
    barkour_neural_data = np.load(barkour_neural_save_folder, allow_pickle=True)
    barkour_cbfddp_data = barkour_cbfddp_data.ravel()[0]
    barkour_lrddp_data = barkour_lrddp_data.ravel()[0]
    barkour_neural_data = barkour_neural_data.ravel()[0]

    fig = plt.figure(layout='constrained', figsize=(5.5, 4.5))
    title_string = 'Barkour 36D'
    fig.suptitle(title_string, fontsize=14)
    subfigs = fig.subfigures(1, 2, wspace=0.05, width_ratios=[0.8, 1.2])

    brax_env = get_brax_env('barkour', backend='mjx')
    rng = jax.random.PRNGKey(seed=0)
    state = brax_env.reset(rng=rng)

    barkour_cbfddp_controls = barkour_cbfddp_data['actions']
    barkour_cbfddp_states = barkour_cbfddp_data['gc_states']
    barkour_cbfddp_control_cycle_times = barkour_cbfddp_data['process_times']
    barkour_cbfddp_values = barkour_cbfddp_data['values']
    barkour_cbfddp_policy_type = barkour_cbfddp_data['policy_type']
    barkour_cbfddp_is_filter_active = barkour_cbfddp_data['filter_active']
    barkour_cbfddp_is_filter_fail = barkour_cbfddp_data['filter_failed']

    barkour_lrddp_controls = barkour_lrddp_data['actions']
    barkour_lrddp_states = barkour_lrddp_data['gc_states']
    barkour_lrddp_control_cycle_times = barkour_lrddp_data['process_times']
    barkour_lrddp_values = barkour_lrddp_data['values']
    barkour_lrddp_policy_type = barkour_lrddp_data['policy_type']
    barkour_lrddp_is_filter_active = barkour_lrddp_data['filter_active']
    barkour_lrddp_is_filter_fail = barkour_lrddp_data['filter_failed']

    barkour_neural_controls = barkour_neural_data['actions']
    barkour_neural_states = barkour_neural_data['gc_states']
    barkour_neural_control_cycle_times = barkour_neural_data['process_times']
    barkour_neural_values = barkour_neural_data['values']
    barkour_neural_policy_type = barkour_neural_data['policy_type']
    barkour_neural_is_filter_active = barkour_neural_data['filter_active']
    barkour_neural_is_filter_fail = barkour_neural_data['filter_failed']

    nsteps = barkour_cbfddp_states.shape[0]
    range_space = np.arange(0, nsteps) * brax_env.dt

    img_default = brax_env.env.render(state.pipeline_state, camera='default')
    img_track = brax_env.env.render(state.pipeline_state, camera='track')
    legend_fontsize = 10

    subfigs_col1 = subfigs[0].subfigures(2, 1, wspace=0.05, height_ratios=[1, 1])
    ax = subfigs_col1[0].subplots(1, 1)
    ax.imshow(img_default)
    ax.set_xticks(ticks=[], labels=[], fontsize=legend_fontsize)
    ax.set_yticks(ticks=[], labels=[], fontsize=legend_fontsize)

    ax = subfigs_col1[1].subplots(1, 1)
    ax.imshow(img_track)
    ax.set_xticks(ticks=[], labels=[], fontsize=legend_fontsize)
    ax.set_yticks(ticks=[], labels=[], fontsize=legend_fontsize)
    # ax.set_xticks(ticks=[0, 320], labels=[0, 320], fontsize=legend_fontsize)
    # ax.set_yticks(ticks=[0, 220], labels=[0, 220], fontsize=legend_fontsize)

    subfigs_col2 = subfigs[1].subfigures(2, 1, wspace=0.05, height_ratios=[1, 1])
    ax = subfigs_col2[0].subplots(1, 1)
    ax.plot(range_space, barkour_cbfddp_values, label='CBFDDP (HM)', color='b')
    ax.plot(range_space, barkour_lrddp_values, label='LRDDP (HM)', color='r')
    ax.plot(range_space, barkour_neural_values, label='Neural', color='k', alpha=0.6)
    ax.set_xticks(ticks=[0, nsteps*brax_env.dt], labels=[0, nsteps], fontsize=legend_fontsize)
    ax.set_yticks(ticks=[0.0, 6.5], labels=[0.0, 6.5], fontsize=legend_fontsize)
    ax.fill_between(range_space, 0.0, 6.5,
                            where=barkour_cbfddp_is_filter_active[0:nsteps], color='b', alpha=0.15)
    ax.fill_between(range_space, 0.0, 6.5,
                    where=barkour_cbfddp_is_filter_fail[0:nsteps], color='r', alpha=0.15)
    ax.set_ylim([-0.5, 6.5])
    ax.yaxis.set_label_coords(-0.04, 0.5)
    ax.xaxis.set_label_coords(0.5, -0.04)
    ax.set_ylabel('Value function')
    ax.set_xlabel('Time (s)')
    ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(0.05, 1.43), ncol=1, framealpha=0.0)
    
    ax = subfigs_col2[1].subplots(1, 1)
    ax.plot(range_space, barkour_cbfddp_control_cycle_times, label='CBFDDP (HM)', color='b')
    ax.plot(range_space, barkour_lrddp_control_cycle_times, label='LRDDP (HM)', color='r')
    ax.set_xticks(ticks=[0, nsteps*brax_env.dt], labels=[0, nsteps], fontsize=legend_fontsize)
    ax.set_yticks(ticks=[0.0, 3.0], labels=[0.0, 3.0], fontsize=legend_fontsize)
    ax.fill_between(range_space, 0.0, 3.0,
                            where=barkour_cbfddp_is_filter_active[0:nsteps], color='b', alpha=0.15)
    ax.fill_between(range_space, 0.0, 3.0,
                    where=barkour_cbfddp_is_filter_fail[0:nsteps], color='r', alpha=0.15)
    ax.yaxis.set_label_coords(-0.04, 0.5)
    ax.xaxis.set_label_coords(0.5, -0.04)
    ax.set_ylim([0.0, 3.0])
    ax.set_ylabel('Filter time $(s)$')
    ax.set_xlabel('Time (s)')

    plt.savefig('./plots_summary/barkour_summary.png')


def make_reacher_plot():
    reacher_cbfddp_save_folder = './brax_videos/reacher/seed_0/ilqr_filter_with_neural_policy_Reachability_save_data.npy'
    reacher_lrddp_save_folder = './brax_videos/reacher/seed_0/lr_filter_with_neural_policy_Reachability_save_data.npy'
    reacher_neural_save_folder = './brax_videos/reacher/seed_0/neural_save_data.npy'

    reacher_cbfddp_data = np.load(reacher_cbfddp_save_folder, allow_pickle=True)
    reacher_lrddp_data = np.load(reacher_lrddp_save_folder, allow_pickle=True)
    reacher_neural_data = np.load(reacher_neural_save_folder, allow_pickle=True)
    reacher_cbfddp_data = reacher_cbfddp_data.ravel()[0]
    reacher_lrddp_data = reacher_lrddp_data.ravel()[0]
    reacher_neural_data = reacher_neural_data.ravel()[0]

    fig = plt.figure(layout='constrained', figsize=(5.5, 4.5))
    title_string = 'Reacher 4D'
    fig.suptitle(title_string, fontsize=14)
    subfigs = fig.subfigures(1, 2, wspace=0.05, width_ratios=[1.3, 1])

    brax_env = get_brax_env('reacher', backend='mjx')
    rng = jax.random.PRNGKey(seed=0)
    state = brax_env.reset(rng=rng)

    reacher_cbfddp_controls = reacher_cbfddp_data['actions']
    reacher_cbfddp_states = reacher_cbfddp_data['gc_states']
    reacher_cbfddp_control_cycle_times = reacher_cbfddp_data['process_times']
    reacher_cbfddp_values = reacher_cbfddp_data['values']
    reacher_cbfddp_policy_type = reacher_cbfddp_data['policy_type']
    reacher_cbfddp_is_filter_active = reacher_cbfddp_data['filter_active']
    reacher_cbfddp_is_filter_fail = reacher_cbfddp_data['filter_failed']

    reacher_lrddp_controls = reacher_lrddp_data['actions']
    reacher_lrddp_states = reacher_lrddp_data['gc_states']
    reacher_lrddp_control_cycle_times = reacher_lrddp_data['process_times']
    reacher_lrddp_values = reacher_lrddp_data['values']
    reacher_lrddp_policy_type = reacher_lrddp_data['policy_type']
    reacher_lrddp_is_filter_active = reacher_lrddp_data['filter_active']
    reacher_lrddp_is_filter_fail = reacher_lrddp_data['filter_failed']

    reacher_neural_controls = reacher_neural_data['actions']
    reacher_neural_states = reacher_neural_data['gc_states']
    reacher_neural_control_cycle_times = reacher_neural_data['process_times']
    reacher_neural_values = reacher_neural_data['values']
    reacher_neural_policy_type = reacher_neural_data['policy_type']
    reacher_neural_is_filter_active = reacher_neural_data['filter_active']
    reacher_neural_is_filter_fail = reacher_neural_data['filter_failed']

    nsteps = reacher_cbfddp_states.shape[0]
    range_space = np.arange(0, nsteps) * brax_env.dt

    img = brax_env.env.render(state.pipeline_state, camera=None)

    subfigs_col1 = subfigs[0].subfigures(3, 1, wspace=0.05, height_ratios=[1, 1, 1])
    ax = subfigs_col1[0].subplots(1, 1)
    ax.imshow(img)
    legend_fontsize = 8.2
    ax.set_xticks(ticks=[], labels=[], fontsize=legend_fontsize)
    ax.set_yticks(ticks=[], labels=[], fontsize=legend_fontsize)

    ax = subfigs_col1[1].subplots(1, 1)
    ax.plot(range_space, reacher_lrddp_values, label='LRDDP (HM)', color='r')
    ax.plot(range_space, reacher_neural_values, label='Neural', color='k', alpha=0.6)
    ax.plot(range_space, reacher_cbfddp_values, label='CBFDDP (HM)', color='b')
    ax.set_xticks(ticks=[0, nsteps*brax_env.dt], labels=[0, nsteps], fontsize=legend_fontsize)
    ax.set_yticks(ticks=[0.0, 30.0], labels=[0.0, 30.0], fontsize=legend_fontsize)
    ax.fill_between(range_space, 0.0, 31.0,
                            where=reacher_cbfddp_is_filter_active[0:nsteps], color='b', alpha=0.15)
    ax.fill_between(range_space, 0.0, 31.0,
                    where=reacher_cbfddp_is_filter_fail[0:nsteps], color='r', alpha=0.15)
    ax.yaxis.set_label_coords(-0.04, 0.5)
    ax.xaxis.set_label_coords(0.5, -0.04)
    ax.set_ylim([-1.0, 30.0])
    ax.set_ylabel('Value function', 
                        fontsize=legend_fontsize)
    ax.set_xlabel('Time (s)', 
                        fontsize=legend_fontsize)

    ax = subfigs_col1[2].subplots(1, 1)
    # ax.plot(range_space, reacher_lrddp_control_cycle_times, label='LRDDP (HM)', color='r')
    ax.plot(range_space, reacher_cbfddp_control_cycle_times, label='CBFDDP (HM)', color='b')
    ax.set_xticks(ticks=[0, round(brax_env.dt*nsteps, 2)], labels=[0, round(brax_env.dt*nsteps, 2)], fontsize=legend_fontsize)
    ax.set_yticks(ticks=[round(reacher_cbfddp_control_cycle_times.min(), 2), round(reacher_cbfddp_control_cycle_times.max(), 2)], 
                    labels=[round(reacher_cbfddp_control_cycle_times.min(), 2), round(reacher_cbfddp_control_cycle_times.max(), 2)], 
                    fontsize=legend_fontsize)
    ax.set_ylim([round(reacher_cbfddp_control_cycle_times.min(), 2), round(reacher_cbfddp_control_cycle_times.max(), 2)])
    ax.yaxis.set_label_coords(-0.04, 0.5)
    ax.xaxis.set_label_coords(0.5, -0.04)
    ax.set_xlabel('Time $(s)$', 
                        fontsize=legend_fontsize)
    ax.set_ylabel('Filter time $(s)$', 
                        fontsize=legend_fontsize)

    col1_axes = subfigs[1].subplots(2, 1)
    col1_axes[0].plot(range_space, reacher_lrddp_controls[:, 0], color='r', label='LRDDP (HM)', alpha=0.6)
    col1_axes[1].plot(range_space, reacher_lrddp_controls[:, 1], color='r')
    col1_axes[0].plot(range_space, reacher_neural_controls[:, 0], color='k', label='Neural', alpha=0.6)
    col1_axes[1].plot(range_space, reacher_neural_controls[:, 1], color='k', alpha=0.6)
    col1_axes[0].plot(range_space, reacher_cbfddp_controls[:, 0], color='b', label='CBFDDP (HM)')
    col1_axes[1].plot(range_space, reacher_cbfddp_controls[:, 1], color='b')
    col1_axes[1].set_xlabel('Time (s)', 
                        fontsize=legend_fontsize)
    col1_axes[0].set_ylabel('Torque 0 ($Nm$)', 
                        fontsize=legend_fontsize)
    col1_axes[1].set_ylabel('Torque 1 ($Nm$)', 
                        fontsize=legend_fontsize)
    col1_axes[0].set_xticks(ticks=[0, nsteps*brax_env.dt], labels=[0, nsteps], fontsize=legend_fontsize)
    col1_axes[0].set_yticks(ticks=[-0.3, 0.3], labels=[-0.3, 0.3], fontsize=legend_fontsize)
    col1_axes[1].set_xticks(ticks=[0, nsteps*brax_env.dt], labels=[0, nsteps], fontsize=legend_fontsize)
    col1_axes[1].set_yticks(ticks=[-0.3, 0.3], labels=[-0.3, 0.3], fontsize=legend_fontsize)
    col1_axes[0].yaxis.set_label_coords(-0.04, 0.5)
    col1_axes[0].xaxis.set_label_coords(0.5, -0.04)
    col1_axes[1].yaxis.set_label_coords(-0.04, 0.5)
    col1_axes[1].xaxis.set_label_coords(0.5, -0.04)
    col1_axes[0].fill_between(range_space, -1.0, 1.0,
                            where=reacher_cbfddp_is_filter_active[0:nsteps], color='b', alpha=0.15)
    col1_axes[0].fill_between(range_space, -1.0, 1.0,
                    where=reacher_cbfddp_is_filter_fail[0:nsteps], color='r', alpha=0.15)
    col1_axes[0].set_ylim([-0.3, 0.3])
    col1_axes[1].fill_between(range_space, -1.0, 1.0,
                            where=reacher_cbfddp_is_filter_active[0:nsteps], color='b', alpha=0.15)
    col1_axes[1].fill_between(range_space, -1.0, 1.0,
                    where=reacher_cbfddp_is_filter_fail[0:nsteps], color='r', alpha=0.15)
    col1_axes[1].set_ylim([-0.3, 0.3])
    col1_axes[0].legend(fontsize=9, loc='upper left', bbox_to_anchor=(-0.05, 1.48), ncol=1, framealpha=0)

    plt.savefig('./plots_summary/reacher_summary.png')

make_reacher_plot()
make_barkour_reachability_plot()
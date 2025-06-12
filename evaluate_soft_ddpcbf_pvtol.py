import jax
from shutil import copyfile
import argparse
import imageio
import numpy as np
import copy
from typing import Dict
import os
import sys

from simulators import(
    load_config,
    Pvtol6DEnv,
    Pvtol6DCost, 
    PvtolReachAvoid6DMargin,
    PrintLogger)

from summary.utils import make_animation_plots, plot_run_summary, make_pvtol_comparison_report

sys.path.append(".")

os.environ["CUDA_VISIBLE_DEVICES"] = " "


jax.config.update('jax_platform_name', 'cpu')

def main(config_file, filter_type, is_task_ilqr):
    ## ------------------------------------- Warmup fields ------------------------------------------ ##
    config = load_config(config_file)
    config_env = config['environment']
    config_agent = config['agent']
    config_agent.is_task_ilqr = is_task_ilqr

    config_solver = config['solver']
    config_solver.LINE_SEARCH = 'baseline'

    # Hacks to get information everywhere.
    config_solver.is_task_ilqr = is_task_ilqr
    config_solver.FILTER_TYPE = filter_type
    config_agent.FILTER_TYPE = filter_type

    plot_tag = config_env.tag
    if config_agent.is_task_ilqr:
        plot_tag += '_ilqrtask_'
    else:
        plot_tag += '_naivetask_'

    config_cost = config['cost']

    # May not be needed ATM.
    config_cost.N = config_solver.N
    config_cost.WIDTH_RIGHT = config_env.WIDTH_RIGHT
    config_cost.WIDTH_LEFT = config_env.WIDTH_LEFT
    config_cost.HEIGHT_BOTTOM = config_env.HEIGHT_BOTTOM
    config_cost.HEIGHT_TOP = config_env.HEIGHT_TOP

    env = Pvtol6DEnv(config_env, config_agent, config_cost)
    x_cur = np.array(
        getattr(
            config_solver, "INIT_STATE", [
                3., 2., 0.5, 0., 0., 0.]))
    env.reset(x_cur)

    # region: Constructs placeholder and initializes iLQR
    config_ilqr_cost = copy.deepcopy(config_cost)

    policy_type = None
    cost = None
    config_solver.COST_TYPE = config_cost.COST_TYPE
    if config_cost.COST_TYPE == "Reachavoid":
        if config_solver.FILTER_TYPE == "none":
            policy_type = "iLQRReachAvoid"
            cost = PvtolReachAvoid6DMargin(
                config_ilqr_cost, copy.deepcopy(env.agent.dyn), filter_type=filter_type)
            task_cost = Pvtol6DCost(
                config_ilqr_cost, copy.deepcopy(
                    env.agent.dyn))
            env.cost = cost  # ! hacky
        else:
            policy_type = "iLQRSafetyFilter"
            task_cost = Pvtol6DCost(
                config_ilqr_cost, copy.deepcopy(
                    env.agent.dyn))
            cost = PvtolReachAvoid6DMargin(
                config_ilqr_cost, copy.deepcopy(env.agent.dyn), filter_type=filter_type)
            env.cost = cost  # ! hacky
    # Not supported
    elif config_cost.COST_TYPE == "Reachability":
        if config_solver.FILTER_TYPE == "none":
            policy_type = "iLQRReachability"
            cost = PvtolReachAvoid6DMargin(
                config_ilqr_cost, copy.deepcopy(env.agent.dyn), filter_type=filter_type)
            env.cost = cost  # ! hacky
        else:
            policy_type = "iLQRSafetyFilter"
            task_cost = Pvtol6DCost(
                config_ilqr_cost, copy.deepcopy(
                    env.agent.dyn))
            cost = PvtolReachAvoid6DMargin(
                config_ilqr_cost, copy.deepcopy(env.agent.dyn), filter_type=filter_type)
            env.cost = cost

    env.agent.init_policy(
        policy_type=policy_type,
        config=config_solver,
        cost=cost,
        task_cost=task_cost)
    max_iter_receding = config_solver.MAX_ITER_RECEDING

    # region: Runs iLQR
    # Warms up jit
    env.agent.get_action(obs=x_cur, state=x_cur, warmup=True)
    #env.report()
    ## ------------------------------------ Evaluation starts -------------------------------------------
    # Callback after each timestep for plotting and summarizing evaluation
    def rollout_step_callback(
            env,
            state_history,
            action_history,
            plan_history,
            step_history,
            *args,
            **kwargs):
        solver_info = plan_history[-1]
        states = np.array(state_history).T  # last one is the next state.
        curr_state = states[-1]
        make_animation_plots(
            env,
            state_history,
            solver_info,
            kwargs['safety_plan'],
            config_solver,
            fig_prog_folder)

        if config_solver.FILTER_TYPE == "none":
            print(
                "[{}]: solver returns status {}, cost {:.1e}, and uses {:.3f}.".format(
                    states.shape[1] - 1,
                    solver_info['status'],
                    solver_info['Vopt'],
                    solver_info['t_process']),
                end=' -> ')
        else:
            print(
                "[{}]: solver returns status {}, margin {:.1e}, future margin {:.1e}, and uses {:.3f}.".format(
                    states.shape[1] - 1,
                    solver_info['status'],
                    solver_info['marginopt'],
                    solver_info['marginopt_next'],
                    solver_info['process_time']))
    
    # Callback after episode for plotting and summarizing evaluation
    def rollout_episode_callback(
            env,
            state_history,
            action_history,
            plan_history,
            step_history,
            *args,
            **kwargs):
        plot_run_summary(
            config_agent.DYN,
            env,
            state_history,
            action_history,
            config_solver,
            config_agent,
            fig_folder,
            **kwargs)
        save_dict = {
            'states': state_history,
            'actions': action_history,
            "values": kwargs["value_history"],
            "process_times": kwargs["process_time_history"],
            "barrier_indices": kwargs["barrier_filter_indices"],
            "complete_indices": kwargs["complete_filter_indices"],
            'deviation_history': kwargs['deviation_history'],
            'safety_metrics': kwargs['safety_metric_history'],
            'safe_opt_history': kwargs['safe_opt_history'],
            'task_ctrl_history': kwargs['task_ctrl_history']}
        np.save(os.path.join(fig_folder, "save_data.npy"), save_dict)

        solver_info = plan_history[-1]
        if config_solver.FILTER_TYPE != "none":
            print(
                "\n\n --> Barrier filtering performed at {:.3f} steps.".format(
                    solver_info['barrier_filter_steps']))
            print(
                "\n\n --> Complete filtering performed at {:.3f} steps.".format(
                    solver_info['filter_steps']))

    # Run sim
    end_criterion = "failure"
    out_folder = config_solver.OUT_FOLDER

    # Naive task is not compatible with current configuration.
    if not config_agent.is_task_ilqr:
        out_folder = os.path.join(out_folder, "naivetask")

    current_out_folder = os.path.join(out_folder, filter_type)
    config_solver.OUT_FOLDER = current_out_folder

    fig_folder = os.path.join(current_out_folder, "figure")
    fig_prog_folder = os.path.join(fig_folder, "progress")
    os.makedirs(fig_prog_folder, exist_ok=True)
    copyfile(
        config_file,
        os.path.join(
            current_out_folder,
            'config.yaml'))
    sys.stdout = PrintLogger(
        os.path.join(
            config_solver.OUT_FOLDER,
            'log.txt'))
    sys.stderr = PrintLogger(
        os.path.join(
            config_solver.OUT_FOLDER,
            'log.txt'))

    # config_current_cost = copy.deepcopy(config_ilqr_cost)
    # if 'LR' in filter_type:
    #     config_current_cost.W_1 = 1e-4
    #     config_current_cost.W_2 = 1e-4

    # Warmup again
    # cost = PvtolReachAvoid6DMargin(
    #             config_current_cost, copy.deepcopy(
    #                 env.agent.dyn), filter_type=filter_type)
    # env.cost = cost
    # env.agent.init_policy(
    #     policy_type=policy_type,
    #     config=config_solver,
    #     cost=cost,
    #     task_cost=task_cost)
    #env.agent.policy.get_action(obs=x_cur, state=x_cur, warmup=True)

    nominal_states, result, traj_info = env.simulate_one_trajectory(
        T_rollout=max_iter_receding, end_criterion=end_criterion,
        reset_kwargs=dict(state=x_cur),
        rollout_step_callback=rollout_step_callback,
        rollout_episode_callback=rollout_episode_callback,
    )

    print("result:", result)
    print(traj_info['step_history'][-1]["done_type"])
    constraints: Dict = traj_info['step_history'][-1]['constraints']
    for k, v in constraints.items():
        print(f"{k}: {v[0, 1]:.1e}")

    # endregion

    # region: Visualizes
    gif_path = os.path.join(fig_folder, 'rollout.gif')
    frame_skip = getattr(config_solver, "FRAME_SKIP", 10)
    with imageio.get_writer(gif_path, mode='I') as writer:
        for i in range(len(nominal_states) - 1):
            if frame_skip != 1 and (i + 1) % frame_skip != 0:
                continue
            filename = os.path.join(
                fig_prog_folder, str(i + 1) + ".png")
            image = imageio.imread(filename)
            writer.append_data(image)

    return out_folder, plot_tag, config_agent

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-cf",
        "--config_file",
        help="Config file path",
        type=str,
        default=os.path.join(
            "./simulators/test_config_yamls",
            "test_config.yaml"))

    parser.set_defaults(naive_task=False)

    args = parser.parse_args()
    is_task_ilqr = False
    out_folder, plot_tag, config_agent = main(args.config_file, filter_type='SoftCBF', is_task_ilqr=is_task_ilqr)
    out_folder, plot_tag, config_agent = main(args.config_file, filter_type='CBF', is_task_ilqr=is_task_ilqr)

    make_pvtol_comparison_report(
        out_folder,
        plot_folder='./plots_summary/',
        tag=plot_tag,
        dt=config_agent.DT,
        filters=['SoftCBF', 'CBF'])


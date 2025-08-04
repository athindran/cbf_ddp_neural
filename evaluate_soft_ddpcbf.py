from summary.utils import(
    make_animation_plots,
    make_bicycle_comparison_report,
    plot_run_summary)
from simulators import(
    load_config,
    CarSingle5DEnv,
    BicycleReachAvoid5DMargin,
    PrintLogger,
    Bicycle5DCost)
import jax
from shutil import copyfile
import argparse
import imageio
import numpy as np
import copy
from typing import Dict
import os
import sys
sys.path.append(".")

os.environ["CUDA_VISIBLE_DEVICES"] = " "


jax.config.update('jax_platform_name', 'cpu')


def main(config_file, road_boundary, filter_type, is_task_ilqr, line_search):
    # Callback after each timestep for plotting and summarizing evaluation
    def rollout_step_callback(
            env: CarSingle5DEnv,
            state_history,
            obs_history,
            action_history,
            plan_history,
            step_history,
            *args,
            **kwargs):
        solver_info = plan_history[-1]
        states = np.asarray(state_history).T  # last one is the next state.
        action_history = np.asarray(action_history)
        make_animation_plots(
            env,
            obs_history,
            action_history,
            solver_info,
            kwargs['safety_plan'],
            config_solver,
            config_agent,
            np.asarray(kwargs['barrier_filter_indices']),
            np.asarray(kwargs['complete_filter_indices']),
            fig_prog_folder)

        # if config_solver.FILTER_TYPE == "none":
        #     print(
        #         "[{}]: solver returns status {}, cost {:.1e}, and uses {:.3f}.".format(
        #             states.shape[1] - 1,
        #             solver_info['status'],
        #             solver_info['Vopt'],
        #             solver_info['t_process']),
        #         end=' -> ')
        # else:
        #     print(
        #         "[{}]: solver returns status {}, margin {:.1e}, future margin {:.1e}, and uses {:.3f}.".format(
        #             states.shape[1] - 1,
        #             solver_info['status'],
        #             solver_info['marginopt'],
        #             solver_info['marginopt_next'],
        #             solver_info['process_time']))
    
    # Callback after episode for plotting and summarizing evaluation
    def rollout_episode_callback(
            env,
            state_history,
            obs_history,
            action_history,
            plan_history,
            step_history,
            *args,
            **kwargs):
        plot_run_summary(
            dyn_id,
            env,
            obs_history,
            action_history,
            config_solver,
            config_agent,
            fig_folder,
            **kwargs)
        save_dict = {
            'states': state_history,
            'obses': obs_history,
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


    ## ------------------------------------- Warmup fields ------------------------------------------ ##
    config = load_config(config_file)
    config_env = config['environment']
    config_agent = config['agent']
    config_solver = config['solver']
    config_solver.LINE_SEARCH = line_search
    config_agent.is_task_ilqr = is_task_ilqr
    config_solver.FILTER_TYPE = filter_type
    config_agent.FILTER_TYPE = filter_type

    # use only ILQR for comparison to CBF.
    if filter_type == 'CBF':
        config_solver.ORDER ='ILQR'

    config_cost = config['cost']
    dyn_id = config_agent.DYN
    plot_tag = config_env.tag
    if is_task_ilqr:
        plot_tag += '_ilqrtask_'
    else:
        plot_tag += '_naivetask_'

    # Provide common fields to cost
    config_cost.N = config_solver.N
    config_cost.V_MIN = config_agent.V_MIN
    config_cost.DELTA_MIN = config_agent.DELTA_MIN
    config_cost.V_MAX = config_agent.V_MAX
    config_cost.DELTA_MAX = config_agent.DELTA_MAX

    config_cost.TRACK_WIDTH_RIGHT = road_boundary
    config_cost.TRACK_WIDTH_LEFT = road_boundary
    config_env.TRACK_WIDTH_RIGHT = road_boundary
    config_env.TRACK_WIDTH_LEFT = road_boundary

    env = CarSingle5DEnv(config_env, config_agent, config_cost)
    x_cur = np.array(
        getattr(
            config_solver, "INIT_STATE", [
                0., 0., 0.5, 0., 0.]))
    env.reset(x_cur)

    # region: Constructs placeholder and initializes iLQR
    config_ilqr_cost = copy.deepcopy(config_cost)

    policy_type = None
    cost = None
    config_solver.COST_TYPE = config_cost.COST_TYPE
    if config_cost.COST_TYPE == "Reachavoid":
        if config_solver.FILTER_TYPE == "none":
            policy_type = "iLQRReachAvoid"
            cost = BicycleReachAvoid5DMargin(
                config_ilqr_cost, copy.deepcopy(env.agent.dyn), filter_type)
            env.cost = cost  # ! hacky
        else:
            policy_type = "iLQRSafetyFilter"
            task_cost = Bicycle5DCost(
                config_ilqr_cost, copy.deepcopy(
                    env.agent.dyn))
            cost = BicycleReachAvoid5DMargin(
                config_ilqr_cost, copy.deepcopy(env.agent.dyn), filter_type)
            evaluation_cost = BicycleReachAvoid5DMargin(
                config_ilqr_cost, copy.deepcopy(env.agent.dyn), 'SoftCBF')
            env.cost = cost  # ! hacky
    # Not supported
    elif config_cost.COST_TYPE == "Reachability":
        if config_solver.FILTER_TYPE == "none":
            policy_type = "iLQRReachability"
            cost = BicycleReachAvoid5DMargin(
                config_ilqr_cost, copy.deepcopy(env.agent.dyn), filter_type)
            env.cost = cost  # ! hacky
        else:
            policy_type = "iLQRSafetyFilter"
            task_cost = Bicycle5DCost(
                config_ilqr_cost, copy.deepcopy(
                    env.agent.dyn))
            cost = BicycleReachAvoid5DMargin(
                config_ilqr_cost, copy.deepcopy(env.agent.dyn), filter_type)
            evaluation_cost = BicycleReachAvoid5DMargin(
                config_ilqr_cost, copy.deepcopy(env.agent.dyn), 'SoftCBF')
            env.cost = cost

    env.agent.init_policy(
        policy_type=policy_type,
        config=config_solver,
        cost=cost,
        evaluation_cost=evaluation_cost,
        task_cost=task_cost)
    max_iter_receding = config_solver.MAX_ITER_RECEDING

    # region: Runs iLQR
    # Warms up jit
    env.agent.get_action(obs=x_cur, state=x_cur, warmup=True)
    env.report()
    ## ------------------------------------ Evaluation starts -------------------------------------------
    end_criterion = "failure"

    out_folder = config_solver.OUT_FOLDER

    if not config_agent.is_task_ilqr:
        out_folder = os.path.join(out_folder, "naivetask")

    yaw_constraint = None
    print(f"Road boundary: {road_boundary}")
    print(f"Line search: {line_search}" )
    print(f"Filter type: {filter_type}" )

    # if yaw_constraint is not None:
    #     current_out_folder = os.path.join(out_folder, "road_boundary=" + str(road_boundary) +
    #                                       ", yaw=" +
    #                                       str(round(yaw_constraint, 2)))
    # else:
    current_out_folder = os.path.join(
        out_folder,
        "road_boundary=" +
        str(road_boundary))
    current_out_folder = os.path.join(current_out_folder, filter_type)
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
    #     config_current_cost.W_ACCEL = 1e-4
    #     config_current_cost.W_OMEGA = 1e-4
    # config_current_cost.TRACK_WIDTH_RIGHT = road_boundary
    # config_current_cost.TRACK_WIDTH_LEFT = road_boundary
    # env.visual_extent[2] = -road_boundary
    # env.visual_extent[3] = road_boundary
    # cost = BicycleReachAvoid5DMargin(
    #     config_current_cost, copy.deepcopy(
    #         env.agent.dyn), filter_type=filter_type)
    # env.cost = cost
    # env.agent.init_policy(
    #     policy_type=policy_type,
    #     config=config_solver,
    #     cost=cost,
    #     task_cost=task_cost)
    # Warms up jit again
    env.agent.get_action(obs=x_cur, state=x_cur, warmup=True)

    should_animate = False
    nominal_states, result, traj_info = env.simulate_one_trajectory(
        T_rollout=max_iter_receding, end_criterion=end_criterion,
        reset_kwargs=dict(state=x_cur),
        rollout_step_callback=rollout_step_callback,
        rollout_episode_callback=rollout_episode_callback,
        advanced_animate=should_animate,
    )

    print("result:", result)
    print(traj_info['step_history'][-1]["done_type"])
    constraints: Dict = traj_info['step_history'][-1]['constraints']
    for k, v in constraints.items():
        print(f"{k}: {v[0, 1]:.1e}")

    # endregion

    if should_animate:
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
                #Image(open(gif_path, 'rb').read(), width=400)
        # endregion

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

    parser.add_argument(
        "-rb", "--road_boundary", help="Choose road width", type=float,
        default=2.0
    )

    parser.add_argument(
        "-ls", "--line_search", help="Choose line search", type=str,
    )

    parser.add_argument('--naive_task', dest='naive_task', action='store_true')
    parser.add_argument(
        '--no-naive_task',
        dest='naive_task',
        action='store_false')
    parser.set_defaults(naive_task=False)

    args = parser.parse_args()

    filters=['CBF', 'SoftCBF']
    
    out_folder, plot_tag, config_agent = None, None, None
    for filter_type in filters:
        out_folder, plot_tag, config_agent = main(args.config_file, args.road_boundary, filter_type=filter_type, is_task_ilqr=(not args.naive_task),         
                                                    line_search=args.line_search)

    make_bicycle_comparison_report(
        out_folder,
        plot_folder='./plots_summary_' + args.line_search + '/',
        tag=plot_tag + "_" + str(args.road_boundary,),
        road_boundary=args.road_boundary,
        dt=config_agent.DT,
        filters=filters)


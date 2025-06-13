from simulators import(
    load_config,
    CarSingle5DEnv,
    BicycleReachAvoid5DMargin)
import jax
import argparse
import imageio
import numpy as np
import copy
import time
from matplotlib import pyplot as plt
import os
import sys
sys.path.append(".")

os.environ["CUDA_VISIBLE_DEVICES"] = " "


jax.config.update('jax_platform_name', 'cpu')


def main(config_file, road_boundary, filter_type):
    ## ------------------------------------- Warmup fields ------------------------------------------ ##
    config = load_config(config_file)
    config_env = config['environment']
    config_agent = config['agent']
    config_solver = config['solver']
    config_solver.FILTER_TYPE = filter_type
    config_agent.FILTER_TYPE = filter_type
    config_cost = config['cost']
    dyn_id = config_agent.DYN

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
    plot_tag = config_env.tag + '-' + str(filter_type)

    env = CarSingle5DEnv(config_env, config_agent, config_cost)
    x_cur = np.array(getattr(
            config_solver, "INIT_STATE", [
                0., 0., 0.5, 0., 0.]))
    env.reset(x_cur)

    # region: Constructs placeholder and initializes iLQR
    config_ilqr_cost = copy.deepcopy(config_cost)

    policy_type = None
    cost = None
    config_solver.COST_TYPE = config_cost.COST_TYPE
    if config_cost.COST_TYPE == "Reachavoid":
        policy_type = "iLQRReachAvoid"
        cost = BicycleReachAvoid5DMargin(
            config_ilqr_cost, copy.deepcopy(env.agent.dyn), filter_type)
        env.cost = cost  # ! hacky
    # Not supported
    elif config_cost.COST_TYPE == "Reachability":
        policy_type = "iLQRReachability"
        cost = BicycleReachAvoid5DMargin(
            config_ilqr_cost, copy.deepcopy(env.agent.dyn), filter_type)
        env.cost = cost  # ! hacky

    env.agent.init_policy(
        policy_type=policy_type,
        config=config_solver,
        cost=cost,
        task_cost=None)
    max_iter_receding = config_solver.MAX_ITER_RECEDING

    # region: Runs iLQR
    # Warms up jit
    env.agent.get_action(obs=x_cur, state=x_cur, warmup=True)
    env.report()

    fig = plt.figure(figsize=(config_solver.FIG_SIZE_X, config_solver.FIG_SIZE_Y*2))
    ax = plt.gca()
    ax.axis(env.visual_extent)
    ax.set_aspect('equal')
    env.render_obs(ax=ax, c='k')

    runtimes = []

    dim_0_samples = 30
    dim_1_samples = 10

    boot_controls = None
    for x0_idx in range(dim_0_samples):
        for x1_idx in range(dim_1_samples):
            x0_frac = x0_idx/float(dim_0_samples)
            x1_frac = x1_idx/float(dim_1_samples)
            delta = np.array([x0_frac*config_env.TRACK_LEN, x1_frac*road_boundary])
            x_upd = np.array(x_cur)
            x_upd[0] += delta[0]
            x_upd[1] += delta[1]
            start_time = time.time()
            _, solver_info = env.agent.get_action(obs=x_upd, state=x_upd, controls=boot_controls)
            end_time = time.time() - start_time

            runtimes.append(end_time)
            #boot_controls = solver_info['controls']
            if solver_info['Vopt'] > 0:
                ax.plot(solver_info['states'][0], solver_info['states'][1], color='g', alpha=0.8)
                ax.scatter(solver_info['states'][0, 0], solver_info['states'][1, 0], color='k', s=12, alpha=0.5)
            # else:
            #     ax.plot(solver_info['states'][0], solver_info['states'][1], color='r', alpha=0.8)
            #     ax.scatter(solver_info['states'][0, 0], solver_info['states'][1, 0], color='k', s=12, alpha=0.5)

    boot_controls = None
    for x0_idx in range(dim_0_samples):
        for x1_idx in range(dim_1_samples):
            x0_frac = x0_idx/float(dim_0_samples)
            x1_frac = x1_idx/float(dim_1_samples)
            delta = np.array([x0_frac*config_env.TRACK_LEN, x1_frac*road_boundary])
            x_upd = np.array(x_cur)
            x_upd[0] += delta[0]
            x_upd[1] -= delta[1]
            start_time = time.time()
            _, solver_info = env.agent.get_action(obs=x_upd, state=x_upd, controls=boot_controls)
            end_time = time.time() - start_time

            runtimes.append(end_time)
            #boot_controls = solver_info['controls']
            if solver_info['Vopt'] > 0:
                ax.plot(solver_info['states'][0], solver_info['states'][1], color='g', alpha=0.8)
                ax.scatter(solver_info['states'][0, 0], solver_info['states'][1, 0], color='k', s=12, alpha=0.5)

    plt.savefig(os.path.join('./contour_plots', plot_tag+'.png'))

    runtimes = np.array(runtimes)
    print(f"Mean solver time: {np.mean(runtimes)}")
    print(f"Max solver time: {np.max(runtimes)}")
    print(f"Min solver time: {np.min(runtimes)}")


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

    parser.add_argument('--naive_task', dest='naive_task', action='store_true')
    parser.add_argument(
        '--no-naive_task',
        dest='naive_task',
        action='store_false')
    parser.set_defaults(naive_task=False)

    args = parser.parse_args()

    filters=['SoftCBF', 'CBF']
    
    out_folder, plot_tag, config_agent = None, None, None
    for filter_type in filters:
        main(args.config_file, args.road_boundary, filter_type=filter_type)


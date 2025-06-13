from simulators import(
    load_config,
    CarSingle5DEnv,
    BicycleReachAvoid5DMargin)
import jax
import argparse
import imageio
import numpy as np
from jax import numpy as jnp
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
    x_cur = np.array([3.0, 0., 2.0, 0.1])
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
    status = 0
    tol = 1e-5
    min_alpha = 1e-12
    line_search = 'armijo'
    fig_anim_folder = os.path.join('./line_search_plots', line_search)
    os.makedirs(fig_anim_folder, exist_ok=True)

    controls = np.random.rand(env.agent.policy.dim_u, env.agent.policy.N)
    controls = jnp.array(controls)

    # Rolls out the nominal trajectory and gets the initial cost.
    states, controls = env.agent.policy.rollout_nominal(
        jnp.array(x_cur), controls
    )

    failure_margins = env.agent.policy.cost.constraint.get_mapped_margin(
        states, controls
    )

    # Target cost derivatives are manually computed for more well-behaved backpropagation
    target_margins = env.agent.policy.cost.get_mapped_target_margin(states, controls)
    # target_margins, c_x_t, c_xx_t, c_u_t, c_uu_t = env.cost.get_mapped_target_margin_with_derivative(
    #     states, controls)

    is_inside_target = (target_margins[0] > 0)
    ctrl_costs = env.agent.policy.cost.ctrl_cost.get_mapped_margin(states, controls)
    critical, reachavoid_margin = env.agent.policy.get_critical_points(
        failure_margins, target_margins)

    J = (reachavoid_margin + jnp.sum(ctrl_costs)).astype(float)

    converged = False
    time0 = time.time()

    for idx in range(env.agent.policy.max_iter):
        # We need cost derivatives from 0 to N-1, but we only need dynamics
        c_x, c_u, c_xx, c_uu, c_ux = env.agent.policy.cost.get_derivatives(
            states, controls
        )

        c_x_t, c_u_t, c_xx_t, c_uu_t, c_ux_t = env.agent.policy.cost.get_derivatives_target(
            states, controls
        )

        fx, fu = env.agent.policy.dyn.get_jacobian(states[:, :-1], controls[:, :-1])
        
        fxx, fuu, fux = env.agent.policy.dyn.get_hessian(states[:, :-1], controls[:, :-1])
        V_x, V_xx, constant_term_loop, k_open_loop, K_closed_loop, _, _, Q_u = env.agent.policy.backward_pass_ddp(
            c_x=c_x, c_u=c_u, c_xx=c_xx, c_uu=c_uu, c_ux=c_ux,
            c_x_t=c_x_t, c_u_t=c_u_t, c_xx_t=c_xx_t, c_uu_t=c_uu_t, c_ux_t=c_ux_t, fx=fx, fu=fu,
            fxx=fxx, fuu=fuu, fux=fux,
            critical=critical
        )

        # Choose the best alpha scaling using appropriate line search methods
        if line_search == 'baseline':
            alpha_chosen = env.agent.policy.baseline_line_search( states, controls, K_closed_loop, k_open_loop, J)
        elif line_search == 'armijo':
            alpha_chosen = env.agent.policy.armijo_line_search( states=states, controls=controls, Ks1=K_closed_loop, ks1=k_open_loop, critical=critical, J=J, Q_u=Q_u)
        elif line_search == 'trust_region_constant_margin':
            alpha_chosen = env.agent.policy.trust_region_search_constant_margin( states=states, controls=controls, Ks1=K_closed_loop, ks1=k_open_loop, critical=critical, J=J, Q_u=Q_u)
        elif line_search == 'trust_region_tune_margin':
            alpha_chosen = env.agent.policy.trust_region_search_tune_margin( states=states, controls=controls, Ks1=K_closed_loop, ks1=k_open_loop, critical=critical, J=J,  
                c_x=c_x, c_xx=c_xx, Q_u=Q_u)
        # alpha_chosen = env.trust_region_search_conservative(states=states, controls=controls, Ks1=K_closed_loop, ks1=k_open_loop, critical=critical,
        #                                                      J=J, c_x=c_x, c_xx=c_xx)

        states, controls, J_new, critical, failure_margins, target_margins, reachavoid_margin = env.agent.policy.forward_pass(states, controls, K_closed_loop, k_open_loop, alpha_chosen)
        
        if reachavoid_margin>0:
            color='g'
        else:
            color='r'

        fig = plt.figure(figsize=(config_solver.FIG_SIZE_X, config_solver.FIG_SIZE_Y*2))
        ax = plt.gca()
        ax.axis(env.visual_extent)
        ax.set_aspect('equal')
        env.render_obs(ax=ax, c='k')
        # target_circle = plt.Circle(
        #             [env.cost.constraint.target_constraint.center[0], env.cost.constraint.target_constraint.center[1]], 
        #             env.cost.constraint.target_constraint.radius, alpha=0.4, color='b')
        # ax.add_patch(target_circle)
        ax.plot(states[0], states[1], color=color, alpha=0.8)
        ax.scatter(states[0, 0], states[1, 0], color='k', s=12, alpha=0.5)
        plt.savefig(os.path.join(fig_anim_folder, f'{idx}.png'))
        plt.close()
        
        # states, controls, J_new, critical, failure_margins, target_margins, reachavoid_margin, c_x_t, c_xx_t, c_u_t, c_uu_t = env.forward_pass(
        #     states, controls, K_closed_loop, k_open_loop, alpha_chosen)
        if (np.abs((J - J_new) / J) < tol):  # Small improvement.
            status = 1
            if J_new > 0:
                converged = True

        J = J_new

        if alpha_chosen < min_alpha:
            status = 2
            break
        # Terminates early if the objective improvement is negligible.
        if converged:
            break

    t_process = time.time() - time0
    print(f"Reachavoid solver took {t_process} seconds with status {status}")
    states = np.asarray(states)
    controls = np.asarray(controls)
    K_closed_loop = np.asarray(K_closed_loop)
    k_open_loop = np.asarray(k_open_loop)
    solver_info = dict(
        states=states, controls=controls, reinit_controls=controls, t_process=t_process,
        status=status, Vopt=J, marginopt=reachavoid_margin,
        grad_x=V_x, grad_xx=V_xx, B0=fu[:, :, 0], critical=critical,
        is_inside_target=is_inside_target, K_closed_loop=K_closed_loop, k_open_loop=k_open_loop,
        constant_term=np.float64(constant_term_loop[0])
    )
    gif_path = os.path.join(fig_anim_folder, 'convergence.gif')
    frame_skip = 1
    with imageio.get_writer(gif_path, mode='I') as writer:
        for i in range(idx):
            filename = os.path.join(
                fig_anim_folder, f'{i}.png')
            image = imageio.imread(filename)
            writer.append_data(image)
    
    # env.report()

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


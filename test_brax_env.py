import jax
import copy
import jax.numpy as jnp
import mediapy as media
import sys
import functools
import os
import numpy as np
import time

from brax.training.agents.ppo import train as ppo
from brax.io import model
from brax import envs
from brax_utils import ( WrappedBraxEnv, 
      ReacherRegularizedGoalCost, 
      iLQRBrax, 
      iLQRBraxReachability, 
      iLQRBraxSafetyFilter,
      ReacherReachabilityMargin,
      AntReachabilityMargin, )

from simulators import load_config
sys.path.append(".")

# RL trained policy
def get_neural_policy(env_name, backend):
    train_fn = {
      'reacher': functools.partial(ppo.train, num_timesteps=1, num_evals=1, reward_scaling=5, episode_length=1000, normalize_observations=True, action_repeat=4, unroll_length=50, num_minibatches=1, num_updates_per_batch=8, discounting=0.95, learning_rate=3e-4, entropy_cost=1e-3, num_envs=1, batch_size=256, max_devices_per_host=8, seed=1),
      'ant': functools.partial(ppo.train, num_timesteps=0, num_evals=1, reward_scaling=10, episode_length=1000, normalize_observations=True, action_repeat=1, unroll_length=5, num_minibatches=32, num_updates_per_batch=4, discounting=0.97, learning_rate=3e-4, entropy_cost=1e-2, num_envs=4096, batch_size=2048, seed=1),
    }[env_name]
    brax_env = WrappedBraxEnv(env_name, backend)
    make_inference_fn, params, _ = train_fn(environment=brax_env.env)
    params = model.load_params(f'./brax_utils/trained_models/{env_name}_params')
    inference_fn = make_inference_fn(params)
    jit_inference_fn = jax.jit(inference_fn)
    return jit_inference_fn

def warmup_jit_with_task_policy_rollout(rng, state, brax_env, task_policy, safety_filter):
    """
    Warmup safety filter by running a few time along the task policy rollout. 
    This is a substitute for the rejection sampling that was used in the bicycle dynamics.
    """
    act_rng, rng = jax.random.split(rng)
    task_ctrl, _ = task_policy(state.obs, act_rng)
    safety_filter.get_action(obs=state, state=state, task_ctrl=task_ctrl, prev_ctrl = np.zeros((brax_env.dim_u, )), warmup=True)

    # Substitute for rejection sampling
    for _ in range(10):
      task_ctrl, _ = task_policy(state.obs, act_rng)
      safety_filter.get_action(obs=state, state=state, task_ctrl=task_ctrl, prev_ctrl = np.zeros((brax_env.dim_u, )), warmup=True)
      state = brax_env.step(state, task_ctrl)


def main(seed: int, env_name='reacher', policy_type="neural"):
    # @param ['ant', 'halfcheetah', 'hopper', 'humanoid', 'humanoidstandup', 'inverted_pendulum', 'inverted_double_pendulum', 'pusher', 'reacher', 'walker2d']
    backend = 'generalized'  # @param ['generalized', 'positional', 'spring']
    brax_env = WrappedBraxEnv(env_name, backend)
    rng = jax.random.PRNGKey(seed=seed)
    state = brax_env.reset(rng=rng)
    save_folder = f"./brax_videos/{env_name}/seed_{seed}"
    if not os.path.exists(save_folder):
      os.makedirs(save_folder, exist_ok=True)

    if policy_type=="neural":
      policy = get_neural_policy(env_name, backend)
      # Warmup
      act_rng, rng = jax.random.split(rng)
      task_ctrl, _ = policy(state.obs, act_rng)

      # Load safety filter for logging value function
      config = load_config(f'./brax_utils/configs/{env_name}.yaml')
      config_solver = config['solver']
      config_cost = config['cost']
      config_cost.N = config_solver.N
      reachability_cost = None
      if env_name=="reacher":
        reachability_cost = ReacherReachabilityMargin(config=config_cost, env=WrappedBraxEnv(env_name, backend))
      elif env_name=="ant":
        reachability_cost = AntReachabilityMargin(config=config_cost, env=WrappedBraxEnv(env_name, backend))
      else:
        raise NotImplementedError("Other environments not implemented.")

      safe_policy = iLQRBraxReachability(id=env_name, brax_env=WrappedBraxEnv(env_name, backend), cost=reachability_cost, config=config_solver)
      # Warmup
      safe_policy.get_action(state, controls=None)
      T = config_solver.MAX_ITER_RECEDING
    elif policy_type=="ilqr":
      assert env_name=="reacher"
      config = load_config(f'./brax_utils/configs/{env_name}.yaml')
      config_solver = config['solver']
      cost = ReacherRegularizedGoalCost(center=brax_env._get_obs(state.pipeline_state)[4:6], 
                                          env=WrappedBraxEnv(env_name, backend), ctrl_cost_matrix=-3*jnp.eye(2))
      policy = iLQRBrax(id=env_name, brax_env=WrappedBraxEnv(env_name, backend), cost=cost, config=config_solver)
      # Warmup
      policy.get_action(state, controls=None)
      T = config_solver.MAX_ITER_RECEDING
    elif policy_type=="ilqr_reachability":
      config = load_config(f'./brax_utils/configs/{env_name}.yaml')
      config_solver = config['solver']
      config_cost = config['cost']
      config_cost.N = config_solver.N

      reachability_cost = None
      if env_name=="reacher":
        reachability_cost = ReacherReachabilityMargin(config=config_cost, env=WrappedBraxEnv(env_name, backend))
      elif env_name=="ant":
        reachability_cost = AntReachabilityMargin(config=config_cost, env=WrappedBraxEnv(env_name, backend))
      else:
        raise NotImplementedError("Other environments not implemented.")

      policy = iLQRBraxReachability(id=env_name, brax_env=WrappedBraxEnv(env_name, backend), cost=reachability_cost, config=config_solver)
      # Warmup
      policy.get_action(state, controls=None)
      T = config_solver.MAX_ITER_RECEDING
    elif policy_type=="ilqr_filter_with_neural_policy":
      config = load_config(f'./brax_utils/configs/{env_name}.yaml')
      config_solver = config['solver']
      config_cost = config['cost']
      config_cost.N = config_solver.N
      
      if env_name=="reacher":
        reachability_cost = ReacherReachabilityMargin(config=config_cost, env=WrappedBraxEnv(env_name, backend))
      elif env_name=="ant":
        # NOTE: DOES NOT WORK
        reachability_cost = AntReachabilityMargin(config=config_cost, env=WrappedBraxEnv(env_name, backend))
      else:
        raise NotImplementedError("Other environments not implemented.")
      
      #safe_policy = iLQRBraxReachability(id=env_name, brax_env=WrappedBraxEnv(env_name, backend), cost=reachability_cost, config=config_solver)
      task_policy = get_neural_policy(env_name, backend)
      brax_envs = [WrappedBraxEnv(env_name, backend), WrappedBraxEnv(env_name, backend), WrappedBraxEnv(env_name, backend), WrappedBraxEnv(env_name, backend)]
      safety_filter =  iLQRBraxSafetyFilter(id=env_name, brax_envs=brax_envs, cost=reachability_cost, config=config_solver)

      # Warmup
      warmup_jit_with_task_policy_rollout(rng, state, brax_env, task_policy, safety_filter)
      act_rng, rng = jax.random.split(rng)
      task_ctrl, _ = task_policy(state.obs, act_rng)
      safety_filter.get_action(obs=state, state=state, task_ctrl=task_ctrl, prev_ctrl = np.zeros((brax_env.dim_u, )), warmup=True)
      prev_sol = None
      prev_ctrl = np.zeros((brax_env.dim_u, ))
      T = config_solver.MAX_ITER_RECEDING
    elif policy_type=="ilqr_filter_with_ilqr_policy":
      assert env_name=="reacher"
      config = load_config(f'./brax_utils/configs/{env_name}.yaml')
      config_solver = config['solver']
      config_cost = config['cost']
      config_cost.N = config_solver.N
      reachability_cost = ReacherReachabilityMargin(config=config_cost, env=WrappedBraxEnv(env_name, backend))
      #safe_policy = iLQRBraxReachability(id=env_name, brax_env=WrappedBraxEnv(env_name, backend), cost=reachability_cost, config=config_solver)
      cost = ReacherRegularizedGoalCost(center=brax_env._get_obs(state.pipeline_state)[4:6], 
                                          env=WrappedBraxEnv(env_name, backend), ctrl_cost_matrix=-3*jnp.eye(2))
      task_policy = iLQRBrax(id=env_name, brax_env=WrappedBraxEnv(env_name, backend), cost=cost, config=config_solver)
      brax_envs = [WrappedBraxEnv(env_name, backend), WrappedBraxEnv(env_name, backend), WrappedBraxEnv(env_name, backend), WrappedBraxEnv(env_name, backend)]
      safety_filter =  iLQRBraxSafetyFilter(id=env_name, brax_envs=brax_envs, cost=reachability_cost, config=config_solver)
      # Warmup
      act_rng, rng = jax.random.split(rng)
      task_ctrl, _ = task_policy.get_action(state, controls=None)
      safety_filter.get_action(obs=state, state=state, task_ctrl=task_ctrl, prev_ctrl = np.zeros((brax_env.dim_u, )), warmup=True)
      prev_sol = None
      prev_ctrl = np.zeros((brax_env.dim_u, ))
      T = config_solver.MAX_ITER_RECEDING


    rollout = []
    controls_init = None   
    controls_init_task = None 
    actions_to_sys = np.zeros((T, brax_env.dim_u))
    gc_states_sys = np.zeros((T, brax_env.dim_x))
    values_sys = np.zeros((T,))
    filter_active = np.full_like(values_sys, False)
    filter_failed = np.full_like(values_sys, False)
    control_cycle_times = np.zeros((T, ))
    for idx in range(T):
      print(f"Starting time {idx}")
      rollout.append(state.pipeline_state)
      if policy_type=="neural":
        # act_rng, rng = jax.random.split(rng)
        time0 = time.time()
        act, _ = policy(state.obs, act_rng)
        control_cycle_times[idx] = time.time() - time0

        # Run safety filter to get value function
        _, solver_dict = safe_policy.get_action(state, controls=controls_init)
        values_sys[idx] = solver_dict['marginopt']
        controls_init = jnp.asarray(solver_dict['reinit_controls'])
      elif policy_type=="ilqr":
        time0 = time.time()
        act, solver_dict = policy.get_action(state, controls=controls_init)
        control_cycle_times[idx] = time.time() - time0
        controls_init = jnp.asarray(solver_dict['controls'])
      elif policy_type=="ilqr_reachability":
        time0 = time.time()
        act, solver_dict = policy.get_action(state, controls=controls_init)
        control_cycle_times[idx] = time.time() - time0
        controls_init = jnp.asarray(solver_dict['reinit_controls'])
      elif policy_type=="ilqr_filter_with_neural_policy":
        time0 = time.time()
        task_ctrl, _ = task_policy(state.obs, act_rng)
        act, solver_dict = safety_filter.get_action(obs=state, state=state, task_ctrl=task_ctrl, prev_sol=prev_sol, prev_ctrl=prev_ctrl)
        control_cycle_times[idx] = time.time() - time0
        prev_sol = copy.deepcopy(solver_dict)
        controls_init = jnp.asarray(solver_dict['reinit_controls'])
        # act_rng, rng = jax.random.split(rng)
        values_sys[idx] = solver_dict['marginopt']
        filter_active[idx] = solver_dict['mark_barrier_filter']
        filter_failed[idx] = solver_dict['mark_complete_filter']
      elif policy_type=="ilqr_filter_with_ilqr_policy":
        time0 = time.time()
        task_ctrl, solver_dict_task = task_policy.get_action(state, controls=controls_init_task)
        controls_init_task = jnp.asarray(solver_dict_task['controls'])
        act, solver_dict = safety_filter.get_action(obs=state, state=state, task_ctrl=task_ctrl, prev_sol=prev_sol, prev_ctrl=prev_ctrl)
        prev_sol = copy.deepcopy(solver_dict)
        # act_rng, rng = jax.random.split(rng)
        control_cycle_times[idx] = time.time() - time0
        controls_init = jnp.asarray(solver_dict['reinit_controls'])
        values_sys[idx] = solver_dict['marginopt']  
        filter_active[idx] = solver_dict['mark_barrier_filter']
        filter_failed[idx] = solver_dict['mark_complete_filter']
        #print(f"value: {solver_dict['marginopt']}")
        #print(f"Gc coord: {brax_env.get_generalized_coordinates(state)}")

      state = brax_env.step(state, act)
      prev_ctrl = np.asarray( act )
      actions_to_sys[idx] = np.asarray(act)
      gc_states_sys[idx] = np.asarray(brax_env.get_generalized_coordinates(state))
      print(f"Completed time {idx} with {control_cycle_times[idx]}s  control time")

      #print("action", act)
      # gc_from_state_grad, gc_from_action_grad = brax_env.get_generalized_coordinates_grad(state, act)
      # obs_from_gc_grad = brax_env.get_obs_grad(state.pipeline_state)
      # print(f"Gradients: {gc_from_state_grad}, {obs_from_gc_grad}, {gc_from_action_grad}")
      #obs = brax_env._get_obs(state.pipeline_state)
      #fingertip = brax_env.get_fingertip(brax_env.get_generalized_coordinates(state))
      #current_state_cost = cost.get_stage_margin(brax_env.get_generalized_coordinates(state), act)
      #print(state.pipeline_state)
      #print(f"obs: {obs}")
      #print(f"fingertip: {fingertip}, extracted fingertip: {obs[8:10] + obs[4:6]}")
      #print(f"State cost: {current_state_cost}")

    # Log results for inspection.
    render_every = 2
    camera = 'track' if env_name == 'ant' else None
    media.write_video(os.path.join(save_folder, f'{policy_type}_policy.mp4'),
        brax_env.env.render(rollout[::render_every], camera=camera),
        fps=1.0 / brax_env.env.dt / render_every)
    save_dict = {'policy_type': policy_type,  'gc_states': gc_states_sys, 
                  'actions': actions_to_sys, 'process_times': control_cycle_times,
                  'values': values_sys, 'filter_active': filter_active, 'filter_failed': filter_failed}
    brax_env.plot_states_and_controls(save_dict, save_folder)
    np.save(os.path.join(save_folder, f'{policy_type}_save_data.npy'), save_dict)

if __name__ == "__main__":
    for seed in range(5):
      for policy_type in ["neural", "ilqr_filter_with_neural_policy"]:
        print(seed, policy_type)
        env_name = 'reacher'
        main(seed, env_name=env_name, policy_type=policy_type)


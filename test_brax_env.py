import jax
import copy
import jax.numpy as jnp
import mediapy as media
import sys
import functools
import os
import numpy as np

from brax.training.agents.ppo import train as ppo
from brax.io import model
from brax import envs
from brax_utils import WrappedBraxEnv, ReacherRegularizedGoalCost, iLQRBrax

from simulators import load_config
sys.path.append(".")

# RL trained policy
def get_neural_policy(env_name, backend):
    train_fn = {
      'reacher': functools.partial(ppo.train, num_timesteps=1, num_evals=1, reward_scaling=5, episode_length=1000, normalize_observations=True, action_repeat=4, unroll_length=50, num_minibatches=1, num_updates_per_batch=8, discounting=0.95, learning_rate=3e-4, entropy_cost=1e-3, num_envs=1, batch_size=256, max_devices_per_host=8, seed=1),
    }[env_name]
    brax_env = WrappedBraxEnv(env_name, backend)
    make_inference_fn, params, _ = train_fn(environment=brax_env.env)
    eval_env = envs.create(env_name=env_name, backend=backend)
    params = model.load_params(f'./brax_utils/trained_models/{env_name}_params')
    inference_fn = make_inference_fn(params)
    jit_inference_fn = jax.jit(inference_fn)
    return jit_inference_fn


def main(seed: int, policy_type="neural"):
    env_name = 'reacher'  # @param ['ant', 'halfcheetah', 'hopper', 'humanoid', 'humanoidstandup', 'inverted_pendulum', 'inverted_double_pendulum', 'pusher', 'reacher', 'walker2d']
    backend = 'generalized'  # @param ['generalized', 'positional', 'spring']
    brax_env = WrappedBraxEnv(env_name, backend)
    rng = jax.random.PRNGKey(seed=seed)
    state = brax_env.reset(rng=rng)
    #print(f"State: {state}")
    save_folder = f"./brax_videos/reacher/seed_{seed}"
    if not os.path.exists(save_folder):
      os.makedirs(save_folder, exist_ok=True)

    if policy_type=="neural":
      policy = get_neural_policy(env_name, backend)
    elif policy_type=="ilqr":
      config = load_config('./brax_utils/configs/reacher.yaml')
      config_solver = config['solver']
      cost = ReacherRegularizedGoalCost(dim_u=2, dim_x=8, center=brax_env._get_obs(state.pipeline_state)[4:6], 
                                          env=WrappedBraxEnv(env_name, backend), ctrl_cost_matrix=-3*jnp.eye(2))
      policy = iLQRBrax(id=env_name, brax_env=WrappedBraxEnv(env_name, backend), cost=cost, config=config_solver)

    rollout = []
    controls_init = None
    T = 100
    actions_to_sys = np.zeros((T, brax_env.dim_u))
    gc_states_sys = np.zeros((T, brax_env.dim_x))
    for idx in range(T):
      rollout.append(state.pipeline_state)
      if policy_type=="neural":
        act_rng, rng = jax.random.split(rng)
        act, _ = policy(state.obs, act_rng)
      elif policy_type=="ilqr":
        act, solver_dict = policy.get_action(state, controls=controls_init)
        controls_init = jnp.array(solver_dict['controls'])
      state = brax_env.step(state, act)
      actions_to_sys[idx] = np.array(act)
      gc_states_sys[idx] = np.array(brax_env.get_generalized_coordinates(state))

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
    media.write_video(os.path.join(save_folder, f'{policy_type}_policy.mp4'),
        brax_env.env.render(rollout[::render_every]),
        fps=1.0 / brax_env.env.dt / render_every)
    brax_env.plot_states_and_controls(gc_states_sys, actions_to_sys, policy_type, save_folder)
    save_dict = {'gc_states': gc_states_sys, 'actions': actions_to_sys}
    np.save(os.path.join(save_folder, f'{policy_type}_save_data.npy'), save_dict)

if __name__ == "__main__":
    for seed in range(5):
      for policy_type in ["neural", "ilqr"]:
        main(seed, policy_type=policy_type)


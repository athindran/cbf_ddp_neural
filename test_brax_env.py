import jax
import copy
import jax.numpy as jnp
import mediapy as media
import sys
import functools

from brax.training.agents.ppo import train as ppo
from brax.io import model
from brax import envs
from brax_utils import WrappedBraxEnv, LinearPolicy, ReacherRegularizedGoalCost, iLQRBrax

from simulators import load_config

# RL trained policy
def get_neural_policy():
    make_inference_fn, params, _ = train_fn(environment=brax_env.env)
    eval_env = envs.create(env_name=env_name, backend=backend)
    params = model.load_params(f'./brax_utils/trained_models/{env_name}_params')
    inference_fn = make_inference_fn(params)
    jit_inference_fn = jax.jit(inference_fn)
    return jit_inference_fn

sys.path.append(".")

env_name = 'reacher'  # @param ['ant', 'halfcheetah', 'hopper', 'humanoid', 'humanoidstandup', 'inverted_pendulum', 'inverted_double_pendulum', 'pusher', 'reacher', 'walker2d']
backend = 'generalized'  # @param ['generalized', 'positional', 'spring']
brax_env = WrappedBraxEnv(env_name, backend)
rng = jax.random.PRNGKey(seed=2)
state = brax_env.reset(rng=rng)
#print(f"State: {state}")

train_fn = {
  'reacher': functools.partial(ppo.train, num_timesteps=1, num_evals=1, reward_scaling=5, episode_length=1000, normalize_observations=True, action_repeat=4, unroll_length=50, num_minibatches=1, num_updates_per_batch=8, discounting=0.95, learning_rate=3e-4, entropy_cost=1e-3, num_envs=1, batch_size=256, max_devices_per_host=8, seed=1),
}[env_name]

config = load_config('./brax_utils/configs/reacher.yaml')
config_solver = config['solver']
cost = ReacherRegularizedGoalCost(dim_u=2, dim_x=8, center=brax_env._get_obs(state.pipeline_state)[4:6], 
                                    goal_radius=0.05, env=WrappedBraxEnv(env_name, backend), ctrl_cost_matrix=-10*jnp.eye(2))
ilqr_policy = iLQRBrax(id=env_name, brax_env=WrappedBraxEnv(env_name, backend), cost=cost, config=config_solver)

#policy = get_neural_policy()

rollout = []
controls_init = None
for _ in range(60):
  rollout.append(state.pipeline_state)
  #act_rng, rng = jax.random.split(rng)
  #act, _ = policy(state.obs, act_rng)
  act, solver_dict = ilqr_policy.get_action(state, controls=controls_init)
  print("ILQR action", act)
  controls_init = jnp.array(solver_dict['controls'])
  state = brax_env.step(state, act)
  # gc_from_state_grad, gc_from_action_grad = brax_env.get_generalized_coordinates_grad(state, act)
  # obs_from_gc_grad = brax_env.get_obs_grad(state.pipeline_state)
  # print(f"Gradients: {gc_from_state_grad}, {obs_from_gc_grad}, {gc_from_action_grad}")
  obs = brax_env._get_obs(state.pipeline_state)
  current_state_cost = cost.get_stage_margin(brax_env.get_generalized_coordinates(state), act)
  fingertip = brax_env.get_fingertip(brax_env.get_generalized_coordinates(state))
  #print(state.pipeline_state)
  #print(f"obs: {obs}")
  print(f"fingertip: {fingertip}, extracted fingertip: {obs[8:10] + obs[4:6]}")
  print(f"State cost: {current_state_cost}")

render_every = 2
media.write_video(f'./brax_videos/reacher_linear_policy.mp4',
    brax_env.env.render(rollout[::render_every]),
    fps=1.0 / brax_env.env.dt / render_every)

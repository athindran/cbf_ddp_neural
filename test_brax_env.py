from simulators.brax.wrapper_env import WrappedBraxEnv
from simulators.policy.linear_policy import LinearPolicy
from simulators.policy.ilqr_policy import iLQR
from simulators.config.utils import load_config
from simulators.costs.reacher_margin import ReacherRegularizedGoalCost
import jax
import jax.numpy as jnp
import mediapy as media
import sys
sys.path.append(".")

env_name = 'reacher'  # @param ['ant', 'halfcheetah', 'hopper', 'humanoid', 'humanoidstandup', 'inverted_pendulum', 'inverted_double_pendulum', 'pusher', 'reacher', 'walker2d']
backend = 'generalized'  # @param ['generalized', 'positional', 'spring']
brax_env = WrappedBraxEnv(env_name, backend)
#linear_policy = LinearPolicy(config={}, gain_matrix=jnp.eye(2), target_indices=jnp.array([4, 5]), endpoint_indices=jnp.array([8, 9]))
rng = jax.random.PRNGKey(seed=1)
state = brax_env.reset(rng=rng)
print("State", state)
reacher_cost = ReacherRegularizedGoalCost(dim_u=brax_env.env.action_size, dim_x=8, goal_spec=state.obs[8:10], ctrl_cost_matrix=jnp.diag(jnp.array([0.1, 0.1])))

config_file = f'./test_configs/brax/reacher.yaml'
config = load_config(config_file)
config_solver = config['solver']
linear_policy = LinearPolicy(config={}, gain_matrix=jnp.eye(2), target_indices=jnp.array([4, 5]), endpoint_indices=jnp.array([8, 9]))
rollout = []
for _ in range(300):
  rollout.append(state.pipeline_state)
  act_rng, rng = jax.random.split(rng)
  act, _ = linear_policy.get_action(state.obs)
  state = brax_env.step(state, act)
  state_grad, action_grad = brax_env.get_trimmed_state_grad(state, act)

render_every = 2
media.write_video(f'/Users/athindranrameshkumar/Documents/Code/CBF_DDP_softmax/videos/reacher_linear_policy.mp4',
    brax_env.env.render(rollout[::render_every]),
    fps=1.0 / brax_env.env.dt / render_every)

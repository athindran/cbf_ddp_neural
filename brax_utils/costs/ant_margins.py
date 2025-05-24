from functools import partial

from brax_utils import WrappedBraxEnv
from jax import Array
import jax.numpy as jnp
import jax

from simulators import BaseMargin, CircleObsMargin, QuadraticControlCost
from simulators import LowerHalfMargin, UpperHalfMargin


class AntConstraintCost(BaseMargin):
    def __init__(self, config, env: WrappedBraxEnv):
        super().__init__()
        self.dim_x = env.dim_x
        self.dim_u = env.dim_u
        self.dim_q_states = env.dim_q_states
        self.dim_qd_states = env.dim_qd_states
        self.max_velocity = config.MAX_VELOCITY

    @partial(jax.jit, static_argnames='self')
    def get_stage_margin(
        self, state: Array, ctrl: Array
    ) -> Array:
        """
        Args:
            state (Array, vector shape)
            ctrl (Array, vector shape)

        Returns:
            Array: scalar.
        """
        cost = jnp.inf

        for idx in range(self.dim_q_states, self.dim_q_states + self.dim_qd_states):
            cost = jnp.minimum(cost, self.max_velocity[idx - self.dim_q_states]**2 - state[idx]**2)

        return cost

    @partial(jax.jit, static_argnames='self')
    def get_target_stage_margin(
        self, state: Array, ctrl: Array
    ) -> Array:
        """
        Args:
            state (Array, vector shape)
            ctrl (Array, vector shape)

        Returns:
            Array: scalar.
        """
        return self.stage_margin(state, ctrl)
    

class AntReachabilityMargin(BaseMargin):

    def __init__(self, config, env: WrappedBraxEnv, filter_type: str ='CBF'):
        super().__init__()
        self.constraint = AntConstraintCost(config, env)
        R = jnp.diag(jnp.array(config.W_ctrl))
        self.ctrl_cost = QuadraticControlCost(R=R, r=jnp.zeros(env.dim_u))
        self.constraint.ctrl_cost = QuadraticControlCost(
            R=R, r=jnp.zeros(env.dim_u))
        self.N = config.N

    @partial(jax.jit, static_argnames='self')
    def get_stage_margin(
        self, state: Array, ctrl: Array
    ) -> Array:
        """

        Args:
            state (Array, vector shape)
            ctrl (Array, vector shape)

        Returns:
            Array: scalar.
        """
        state_cost = self.constraint.get_stage_margin(
            state, ctrl
        )
        ctrl_cost = self.ctrl_cost.get_stage_margin(state, ctrl)

        return state_cost + ctrl_cost

    @partial(jax.jit, static_argnames='self')
    def get_target_stage_margin(
        self, state: Array, ctrl: Array
    ) -> Array:
        """

        Args:
            state (Array, vector shape)
            ctrl (Array, vector shape)

        Returns:
            Array: scalar.
        """
        target_cost = self.constraint.get_target_stage_margin(
            state, ctrl
        )
        ctrl_cost = self.ctrl_cost.get_stage_margin(state, ctrl)

        return target_cost + ctrl_cost
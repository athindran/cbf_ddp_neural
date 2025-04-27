from functools import partial

from brax_utils import WrappedBraxEnv
from jax import Array
import jax.numpy as jnp
import jax

from simulators import BaseMargin, CircleObsMargin, QuadraticControlCost

class ReacherGoalCost(BaseMargin):
    def __init__(self, dim_u: int, center: Array, goal_radius: float, env: WrappedBraxEnv):
        super().__init__()
        self.dim_u = dim_u
        self.goal_radius = goal_radius
        self.center = center
        #self.goal_reaching_cost = CircleObsMargin(circle_spec=jnp.array([center[0], center[1], goal_radius]), buffer=0.0)
        self.env = env

    @partial(jax.jit, static_argnames='self')
    def get_stage_margin(
        self, state: Array, ctrl: Array
    ) -> Array:
        """

        Args:
            state aka generalized coordinates (Array, vector shape)
            ctrl (Array, vector shape)

        Returns:
            Array: scalar.
        """
        fingertip = jnp.zeros((2,))
        fingertip = fingertip.at[0].set(0.1*jnp.cos(state[0]) + 0.11*jnp.cos(state[0] + state[1]))
        fingertip = fingertip.at[1].set(0.1*jnp.sin(state[0]) + 0.11*jnp.sin(state[0] + state[1]))
        cost = (self.center[0] - fingertip[0])**2 + (self.center[1] - fingertip[1])**2

        return cost

    @partial(jax.jit, static_argnames='self')
    def get_target_stage_margin(
        self, state, ctrl
    ) -> Array:
        """

        Args:
            state aka generalized coordinates (Array, vector shape)
            ctrl (Array, vector shape)

        Returns:
            Array: scalar.
        """
        return self.get_state_margin(state, ctrl)


class ReacherRegularizedGoalCost(BaseMargin):
    def __init__(self, dim_u: int, dim_x: int, center: Array, goal_radius: float, ctrl_cost_matrix: jnp.ndarray, env: WrappedBraxEnv):
        super().__init__()
        self.dim_x = dim_x
        self.dim_u = dim_u
        self.goal_cost = ReacherGoalCost(dim_u, center, goal_radius, env)
        self.ctrl_cost = QuadraticControlCost(R = ctrl_cost_matrix, r = jnp.zeros(dim_u))

    @partial(jax.jit, static_argnames='self')
    def get_stage_margin(
        self, state: Array, ctrl: Array
    ) -> Array:
        """

        Args:
            state aka generalized coordinates (Array, vector shape)
            ctrl (Array, vector shape)

        Returns:
            Array: scalar.
        """
        state_cost = self.goal_cost.get_stage_margin(
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
            state aka generalized coordinates (Array, vector shape)
            ctrl (Array, vector shape)

        Returns:
            Array: scalar.
        """
        target_cost = self.goal_cost.get_target_stage_margin(
            state, ctrl
        )
        ctrl_cost = self.ctrl_cost.get_stage_margin(state, ctrl)

        return target_cost + ctrl_cost

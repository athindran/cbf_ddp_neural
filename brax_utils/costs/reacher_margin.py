from functools import partial

from brax_utils import WrappedBraxEnv
from jax import Array
import jax.numpy as jnp
import jax

from simulators import BaseMargin, CircleObsMargin, QuadraticControlCost

class ReacherGoalCost(BaseMargin):
    def __init__(self, dim_u: int, center: Array, env: WrappedBraxEnv):
        super().__init__()
        self.dim_u = dim_u
        self.center = center        
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
        fingertip = self.env.get_fingertip(state)
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
    def __init__(self, dim_u: int, dim_x: int, center: Array, ctrl_cost_matrix: jnp.ndarray, env: WrappedBraxEnv):
        super().__init__()
        self.dim_x = dim_x
        self.dim_u = dim_u
        self.goal_cost = ReacherGoalCost(dim_u, center, env)
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

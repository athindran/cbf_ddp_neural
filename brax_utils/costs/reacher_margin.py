from functools import partial

from brax_utils import WrappedBraxEnv
from jax import Array
import jax.numpy as jnp
import jax

from simulators import BaseMargin, CircleObsMargin, QuadraticControlCost
from simulators import LowerHalfMargin, UpperHalfMargin

class ReacherGoalCost(BaseMargin):
    def __init__(self, center: Array):
        super().__init__()
        self.center = center        

    @partial(jax.jit, static_argnames='self')
    def get_fingertip(self, generalized_coordinates: jax.Array) -> jax.Array:
        fingertip = jnp.zeros((2,))
        fingertip = fingertip.at[0].set(0.1*jnp.cos(generalized_coordinates[0]) + 0.11*jnp.cos(generalized_coordinates[0] + generalized_coordinates[1]))
        fingertip = fingertip.at[1].set(0.1*jnp.sin(generalized_coordinates[0]) + 0.11*jnp.sin(generalized_coordinates[0] + generalized_coordinates[1]))
        return fingertip

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
        fingertip = self.get_fingertip(state)
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
    def __init__(self, center: Array, ctrl_cost_matrix: jnp.ndarray, env: WrappedBraxEnv):
        super().__init__()
        self.dim_x = env.dim_x
        self.dim_u = env.dim_u
        self.goal_cost = ReacherGoalCost(center)
        self.ctrl_cost = QuadraticControlCost(R = ctrl_cost_matrix, r = jnp.zeros(self.dim_u))

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

class ReacherAngularVelocityConstraintCost(BaseMargin):
    def __init__(self, config, env: WrappedBraxEnv):
        super().__init__()
        self.dim_x = env.dim_x
        self.dim_u = env.dim_u
        self.max_angular_velocity = config.MAX_ANGULAR_VELOCITY

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
        cost = jnp.minimum(cost, self.max_angular_velocity[0]**2 - state[2]**2)
        cost = jnp.minimum(cost, self.max_angular_velocity[1]**2 - state[3]**2)

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

class ReacherReachabilityMargin(BaseMargin):

    def __init__(self, config, env: WrappedBraxEnv, filter_type: str ='CBF'):
        super().__init__()
        self.constraint = ReacherAngularVelocityConstraintCost(config, env)
        R = jnp.array([[config.W_1, 0.0], [0.0, config.W_2]])
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

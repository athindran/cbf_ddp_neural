import numpy as np
import jax
import jax.numpy as jnp
from jax import Array as DeviceArray
from functools import partial

from .base_margin import BaseMargin


class CircleObsMargin(BaseMargin):
    """
    We want s[i] < lb[i] or s[i] > ub[i].
    """

    def __init__(
        self, circle_spec=np.ndarray,
        buffer: float = 0.
    ):
        super().__init__()
        self.center = jnp.array(circle_spec[0:2])
        self.radius = circle_spec[2]
        self.buffer = buffer

    @partial(jax.jit, static_argnames='self')
    def get_stage_margin(
        self, state: DeviceArray, ctrl: DeviceArray
    ) -> DeviceArray:
        # signed distance to the box, positive inside
        circ_distance = jnp.sqrt(
            (state[0] - self.center[0])**2 + (state[1] - self.center[1])**2) - self.radius
        return circ_distance - self.buffer

    @partial(jax.jit, static_argnames='self')
    def get_target_stage_margin(
        self, state: DeviceArray, ctrl: DeviceArray
    ) -> DeviceArray:
        return self.get_stage_margin(state, ctrl)


class BoxObsMargin(BaseMargin):

    def __init__(
        self, box_spec: np.ndarray, buffer: float = 0.
    ):
        """
        Args:
            box_spec (np.ndarray): [x, y, heading, half_length, half_width], spec
                of the box obstacles.
            buffer (float): the minimum required distance to the obstacle, i.e., if
                the distance is smaller than `buffer`, the cost will be positive as
                well. Defaults to 0.
        """
        super().__init__()
        # Box obstacle
        self.box_center = jnp.array([[box_spec[0]], [box_spec[1]]])
        self.box_yaw = box_spec[2]
        # rotate clockwise (to move the world frame to obstacle frame)
        self.obs_rot_mat = jnp.array([[
            jnp.cos(self.box_yaw), jnp.sin(self.box_yaw)
        ], [-jnp.sin(self.box_yaw), jnp.cos(self.box_yaw)]])
        self.box_halflength = box_spec[3]
        self.box_halfwidth = box_spec[4]
        self.buffer = buffer

    @partial(jax.jit, static_argnames='self')
    def get_stage_margin(
        self, state: DeviceArray, ctrl: DeviceArray,
    ) -> DeviceArray:
        pos = state[0:2].reshape(2, -1)
        pos_final = self.obs_rot_mat @ (pos - self.box_center)

        diff_x = jnp.maximum(
            pos_final[0] - self.box_halflength,
            - self.box_halflength - pos_final[0]
        )
        diff_y = jnp.maximum(
            pos_final[1] - self.box_halfwidth,
            - self.box_halfwidth - pos_final[1]
        )
        diff = jnp.maximum(diff_x, diff_y)
        diff = diff.squeeze()

        return diff - self.buffer

    @partial(jax.jit, static_argnames='self')
    def get_target_stage_margin(
        self, state: DeviceArray, ctrl: DeviceArray
    ) -> DeviceArray:
        return self.get_stage_margin(state, ctrl)


class SoftBoxObsMargin(BaseMargin):

    def __init__(
        self, box_spec: np.ndarray, buffer: float = 0.
    ):
        """
        Args:
            box_spec (np.ndarray): [x, y, heading, half_length, half_width], spec
                of the box obstacles.
            buffer (float): the minimum required distance to the obstacle, i.e., if
                the distance is smaller than `buffer`, the cost will be positive as
                well. Defaults to 0.
        """
        super().__init__()
        # Box obstacle
        self.box_center = jnp.array([[box_spec[0]], [box_spec[1]]])
        self.box_yaw = box_spec[2]
        # rotate clockwise (to move the world frame to obstacle frame)
        self.obs_rot_mat = jnp.array([[
            jnp.cos(self.box_yaw), jnp.sin(self.box_yaw)
        ], [-jnp.sin(self.box_yaw), jnp.cos(self.box_yaw)]])
        self.box_halflength = box_spec[3]
        self.box_halfwidth = box_spec[4]
        self.box_kappa = 4.0
        self.buffer = buffer

    @partial(jax.jit, static_argnames='self')
    def get_stage_margin(
        self, state: DeviceArray, ctrl: DeviceArray,
    ) -> DeviceArray:
        pos = state[0:2].reshape(2, -1)
        pos_final = self.obs_rot_mat @ (pos - self.box_center)

        diff = 0.
        diff += jnp.exp(self.box_kappa*jnp.minimum(pos_final[0] - self.box_halflength, 5.0))
        diff += jnp.exp(self.box_kappa*jnp.minimum(-self.box_halflength - pos_final[0], 5.0))
        diff += jnp.exp(self.box_kappa*jnp.minimum(pos_final[1] - self.box_halfwidth, 5.0))
        diff += jnp.exp(self.box_kappa*jnp.minimum(-self.box_halfwidth - pos_final[1], 5.0))
        diff = jnp.log(diff)/self.box_kappa

        diff = diff.squeeze()

        return diff - self.buffer

    @partial(jax.jit, static_argnames='self')
    def get_target_stage_margin(
        self, state: DeviceArray, ctrl: DeviceArray
    ) -> DeviceArray:
        return self.get_stage_margin(state, ctrl)

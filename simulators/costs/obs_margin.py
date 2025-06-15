import numpy as np
import jax
import jax.numpy as jnp
from jax import Array as DeviceArray
from functools import partial

from .base_margin import BaseMargin


class CircleObsMargin(BaseMargin):
    """
    Distance-based cost to circular obstacle.
    """

    def __init__(
        self, circle_spec=np.ndarray,
        buffer: float = 0.
    ):
        """
        Args:
            circle_spec (np.ndarray): [x, y, radius], spec of the circular obstacles.
            buffer (float): the minimum required distance to the obstacle, i.e., if
                the distance is smaller than `buffer`, the cost will be negative as
                well. Defaults to 0.
        """
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
                the distance is smaller than `buffer`, the cost will be negative as
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
                the distance is smaller than `buffer`, the cost will be negative as
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

class EllipseObsMargin(BaseMargin):

    def __init__(
        self, ellipse_spec: np.ndarray, buffer: float = 0.
    ):
        """
        Args:
            ellipse_spec (np.ndarray): [x, y, heading, half_length, half_width], spec
                of the box obstacles.
            buffer (float): the minimum required distance to the obstacle, i.e., if
                the distance is smaller than `buffer`, the cost will be negative as
                well. Defaults to 0.
        """
        super().__init__()
        self.ellipse_center = jnp.array([[ellipse_spec[0]], [ellipse_spec[1]]])
        self.ellipse_yaw = ellipse_spec[2]
        self.ellipse_half_length = ellipse_spec[3]
        self.ellipse_half_width = ellipse_spec[4]
        self.obs_rot_mat = jnp.array([[jnp.cos(self.ellipse_yaw), jnp.sin(self.ellipse_yaw)], 
                            [-jnp.sin(self.ellipse_yaw), jnp.cos(self.ellipse_yaw)]])
        self.buffer = buffer
        self.max_radius = jnp.maximum(self.ellipse_half_length, self.ellipse_half_width)

    @partial(jax.jit, static_argnames='self')
    def get_stage_margin(
        self, state: DeviceArray, ctrl: DeviceArray,
    ) -> DeviceArray:
        """
        This is not exactly the distance to the ellipse but only a cost function which is negative inside the ellipse
        and positive outside.
        """
        pos = state[0:2].reshape(2, -1)
        relative_vector = (pos - self.ellipse_center)
        pos_final = self.obs_rot_mat @ relative_vector
        buffer_margin_vector = self.buffer*pos_final /jnp.linalg.norm(pos_final , axis=0, keepdims=True)
        pos_final = pos_final - buffer_margin_vector
        obs_margin = jnp.sqrt((pos_final[0]/self.ellipse_half_length)**2 + (pos_final[1]/self.ellipse_half_width)**2) - 1.0

        return obs_margin.squeeze()*self.max_radius

    @partial(jax.jit, static_argnames='self')
    def get_target_stage_margin(
        self, state: DeviceArray, ctrl: DeviceArray
    ) -> DeviceArray:
        return self.get_stage_margin(state, ctrl)


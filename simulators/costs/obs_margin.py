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
    """
    This is not exactly the L2 distance to the box but only a cost function which is negative inside the box
    and positive outside.
    First, rotate the coordinate axes along the perpendicular axes of the box. Then, we find the distance along each
    axis from the box to the center of the circular footprint. Then, subtract the radius of circle.
    """

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
        # rotate clockwise (to move the point from world frame to obstacle frame)
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
    """
    This is a soft approximation of BoxObsMargin which is strictly negative inside the box.
    """
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
        # rotate clockwise (to move the point from world frame to obstacle frame)
        self.obs_rot_mat = jnp.array([[
            jnp.cos(self.box_yaw), jnp.sin(self.box_yaw)
        ], [-jnp.sin(self.box_yaw), jnp.cos(self.box_yaw)]])
        self.box_halflength = box_spec[3]
        self.box_halfwidth = box_spec[4]
        self.box_kappa = 5.0
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
        diff = (jnp.log(diff) - jnp.log(4))/self.box_kappa

        diff = diff.squeeze()

        return diff - self.buffer

    @partial(jax.jit, static_argnames='self')
    def get_target_stage_margin(
        self, state: DeviceArray, ctrl: DeviceArray
    ) -> DeviceArray:
        return self.get_stage_margin(state, ctrl)

class EllipseObsMargin(BaseMargin):
    """
    This is not exactly the distance to the ellipse but only a cost function which is negative inside the ellipse
    and positive outside. We check whether (x^2/a^2 + y^2/b^2 - 1.0)<=0.0 to determine whether a point is inside 
    the ellipse. The circular footprint is provided a buffer corresponding to ego-radius. This buffer is converted to
    cost function units by dividing by minimum(a, b). This approximation can be costly if the ellipse is skewed.
    """

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
        # rotate clockwise (to move the point from world frame to obstacle frame)
        self.obs_rot_mat = jnp.array([[jnp.cos(self.box_yaw), jnp.sin(self.box_yaw)], 
                    [-jnp.sin(self.box_yaw), jnp.cos(self.box_yaw)]])
        # This is a conservative approximation of how much units of cost margin we provide to the buffer.
        # A better approximation would be the to sample footprint and choose minimum or smooth minimum.
        # We use this approximaiton for speed and smoothness.
        self.buffer_margin = buffer/jnp.minimum(self.ellipse_half_length, self.ellipse_half_width)
        self.avg_radius = (self.ellipse_half_length + self.ellipse_half_width)/2.0

    @partial(jax.jit, static_argnames='self')
    def get_stage_margin(
        self, state: DeviceArray, ctrl: DeviceArray,
    ) -> DeviceArray:
        pos = state[0:2].reshape(2, -1)
        relative_vector = (pos - self.ellipse_center)
        pos_final = self.obs_rot_mat @ relative_vector
        obs_margin = jnp.sqrt((pos_final[0]/self.ellipse_half_length)**2 + (pos_final[1]/self.ellipse_half_width)**2) - 1.0 - self.buffer_margin
        return obs_margin.squeeze()*self.avg_radius

    @partial(jax.jit, static_argnames='self')
    def get_target_stage_margin(
        self, state: DeviceArray, ctrl: DeviceArray
    ) -> DeviceArray:
        return self.get_stage_margin(state, ctrl)


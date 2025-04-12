from typing import Optional
import numpy as np

from matplotlib import pyplot as plt


class RectangleFootprint:

    def __init__(
        self, center: Optional[np.ndarray] = None,
        angle: Optional[np.ndarray] = None,
        ego_radius: float = 0
    ) -> None:
        if center is None:
            self.center = np.zeros(3)
        else:
            self.center = center.copy()
        self.ego_radius = ego_radius
        self.angle = angle

    def set_center(self, center: Optional[np.ndarray] = None):
        self.center = center
    
    def set_angle(self, angle: Optional[np.ndarray] = None):
        self.angle = angle

    def plot(self, ax, color='r', lw=1.5, alpha=1.):
        ego_circle = plt.Rectangle(
            [self.center[0], self.center[1]], 0.1, 2*self.ego_radius, angle=self.angle, alpha=0.4, color=color)
        ax.add_patch(ego_circle)

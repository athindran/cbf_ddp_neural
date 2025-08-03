import numpy as np
import math

def pvtol_linear_task_policy(obs, dyn):
    lqr_tuned_gain = np.array([[-1.00000000e+00, 1.34334159e-16, 7.85405998e+00, -1.60495816e+00, 1.31261662e-15,  2.06849834e+00],
        [-1.74437640e-17, 1.00000000e+00, 1.79332809e-15, -1.62742386e-17, 2.95041664e+00,  6.31223160e-17]])

    control_task = -lqr_tuned_gain @ obs
    control_task[1] = control_task[1] + dyn.mass * dyn.g

    return control_task

def bicycle_linear_task_policy(run_env_obs):
    lookahead_distance = 4.0
    direction_waypoint = math.atan2(-run_env_obs[1], lookahead_distance)

    if run_env_obs.size == 5:
        control_task = np.zeros((2, ))
        vref = 2.8
        control_task[0] = -1.0 * (run_env_obs[2] - vref)
        # Use only unwrapped yaw phase for this subtraction
        delta_theta = (run_env_obs[3] - direction_waypoint)

        if np.abs(delta_theta) < 0.1:
            control_task[1] = -1.0 * run_env_obs[4]
        else:
            control_task[1] = -1.2 * delta_theta - 0.9 * run_env_obs[4]
    elif run_env_obs.size == 4:
        control_task = np.zeros((2, ))
        vref = 2.0
        control_task[0] = -1.0 * (run_env_obs[2] - vref)
        # Use only unwrapped yaw phase for this subtraction
        delta_theta = (run_env_obs[3] - direction_waypoint)

        control_task[1] = -1.0 * delta_theta

    return control_task

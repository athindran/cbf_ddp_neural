from typing import Optional, Dict
import time

from jax import numpy as jp
from jax import Array as DeviceArray
from typing import List

import copy

from .brax_ilqr_reachability_policy import iLQRBraxReachability
from .brax_ilqr_reachavoid_policy import iLQRBraxReachAvoid
from simulators import BasePolicy
from brax_utils import WrappedBraxEnv
from simulators.costs.base_margin import BaseMargin

class LRBraxSafetyFilter(BasePolicy):

    def __init__(self, id: str, config, brax_envs: List[WrappedBraxEnv],
                 cost: BaseMargin) -> None:
        super().__init__(id, config)
        self.config = config

        self.filter_type = config.FILTER_TYPE
        self.lr_threshold = 0.0

        self.filter_steps = 0

        self.brax_env = brax_envs[0]
        self.cost = copy.deepcopy(cost)

        self.dim_x = self.brax_env.dim_x
        self.dim_u = self.brax_env.dim_u
        self.N = config.N

        # Two ILQR solvers
        if self.config.COST_TYPE == "Reachavoid":
            self.solver_0 = iLQRBraxReachAvoid(
                self.id, self.config, brax_envs[1], self.cost)
            self.solver_1 = iLQRBraxReachAvoid(
                self.id, self.config, brax_envs[2], self.cost)
            self.solver_2 = iLQRBraxReachAvoid(
                self.id, self.config, brax_envs[3], self.cost)
        elif self.config.COST_TYPE == "Reachability":
            self.solver_0 = iLQRBraxReachability(
                self.id, self.config, brax_envs[1], self.cost)
            self.solver_1 = iLQRBraxReachability(
                self.id, self.config, brax_envs[2], self.cost)
            self.solver_2 = iLQRBraxReachability(
                self.id, self.config, brax_envs[3], self.cost)

    def get_action(
        self, 
        obs: DeviceArray, 
        state: DeviceArray, 
        task_ctrl: DeviceArray,
        prev_sol: Optional[Dict] = None, 
        prev_ctrl = None, 
        warmup=False,
    ):

        task_ctrl_jp = jp.array(task_ctrl)
        obs = jp.array(obs)
        state = jp.array(state)
        prev_ctrl = jp.array(prev_ctrl)

        # Find safe policy from step 0
        if prev_sol is not None:
            controls_initialize = jp.array(prev_sol['reinit_controls'])
        else:
            controls_initialize = None

        if prev_sol is None or prev_sol['resolve']:
            control_0, solver_info_0 = self.solver_0.get_action(
                obs=obs, controls=controls_initialize, state=state)
        else:
            # Potential source of acceleration. We don't need to resolve both ILQs as we can reuse
            # solution from previous time. - Unused currently.
            solver_info_0 = prev_sol['bootstrap_next_solution']
            control_0 = solver_info_0['controls'][:, 0]
            # Closed loop solution
            #solver_info_0['controls'] = jp.array( solver_info_0['reinit_controls'] )
            #solver_info_0['states'] = jp.array( solver_info_0['reinit_states'] )
            #solver_info_0['Vopt'] = solver_info_0['Vopt_next']
            #solver_info_0['marginopt'] = solver_info_0['marginopt_next']
            #solver_info_0['is_inside_target'] = solver_info_0['is_inside_target_next']

        solver_info_0['safe_opt_ctrl'] =  jp.array(control_0)
        solver_info_0['task_ctrl'] = jp.array(task_ctrl)

        solver_info_0['mark_barrier_filter'] = False
        solver_info_0['mark_complete_filter'] = False
        # Find safe policy from step 1
        state_imaginary = self.brax_env.step(
            state, task_ctrl_jp
        )
        boot_controls = jp.array(solver_info_0['controls'])

        _, solver_info_1 = self.solver_1.get_action(
            obs=state_imaginary, controls=boot_controls, state=state_imaginary)

        solver_info_0['Vopt_next'] = solver_info_1['Vopt']
        solver_info_0['marginopt_next'] = solver_info_1['marginopt']

        if solver_info_1['Vopt'] > 0 or warmup:
            solver_info_0['mark_complete_filter'] = False
            solver_info_0['resolve'] = True
            solver_info_0['bootstrap_next_solution'] = solver_info_1
            solver_info_0['reinit_controls'] = jp.array(
                solver_info_1['controls'])
            solver_info_0['num_iters'] = -1

            return task_ctrl_jp.ravel(), solver_info_0
        else:
            # Safe policy
            solver_info_0['resolve'] = True
            solver_info_0['reinit_controls'] = jp.zeros((self.dim_u, self.N))
            solver_info_0['reinit_controls'] = solver_info_0['reinit_controls'].at[:, 0:self.N - 1].set(
                solver_info_0['controls'][:, 1:self.N])
            solver_info_0['mark_complete_filter'] = True
            safety_control = solver_info_0['controls'][:, 0]
            solver_info_0['num_iters'] = -1

            return safety_control.ravel(), solver_info_0

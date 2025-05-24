from typing import Optional, Dict
import time

from jax import numpy as jnp

import copy
import numpy as np

from .ilqr_reachavoid_policy import iLQRReachAvoid
from .ilqr_reachability_policy import iLQRReachability
from .base_policy import BasePolicy
from .solver_utils import barrier_filter_linear, barrier_filter_quadratic_two, barrier_filter_quadratic_eight
from simulators.dynamics.base_dynamics import BaseDynamics
from simulators.costs.base_margin import BaseMargin

class iLQRSafetyFilter(BasePolicy):

    def __init__(self, id: str, config, dyn: BaseDynamics,
                 cost: BaseMargin) -> None:
        super().__init__(id, config)
        self.config = config

        self.filter_type = config.FILTER_TYPE
        self.constraint_type = config.CONSTRAINT_TYPE
        if self.filter_type == 'CBF':
            self.gamma = config.CBF_GAMMA
        elif self.filter_type == 'SoftCBF':
            self.gamma = config.SOFT_CBF_GAMMA
        else:
            self.gamma = None

        self.lr_threshold = config.LR_THRESHOLD

        self.filter_steps = 0
        self.barrier_filter_steps = 0

        self.dyn = copy.deepcopy(dyn)
        self.rollout_dyn_0 = copy.deepcopy(dyn)
        self.rollout_dyn_1 = copy.deepcopy(dyn)
        self.rollout_dyn_2 = copy.deepcopy(dyn)

        self.cost = copy.deepcopy(cost)

        self.dim_x = dyn.dim_x
        self.dim_u = dyn.dim_u
        self.N = config.N

        # Two ILQR solvers
        if self.config.COST_TYPE == "Reachavoid":
            self.solver_0 = iLQRReachAvoid(
                self.id, self.config, self.rollout_dyn_0, self.cost)
            self.solver_1 = iLQRReachAvoid(
                self.id, self.config, self.rollout_dyn_1, self.cost)
            self.solver_2 = iLQRReachAvoid(
                self.id, self.config, self.rollout_dyn_1, self.cost)
        elif self.config.COST_TYPE == "Reachability":
            self.solver_0 = iLQRReachability(
                self.id, self.config, self.rollout_dyn_0, self.cost)
            self.solver_1 = iLQRReachability(
                self.id, self.config, self.rollout_dyn_1, self.cost)
            self.solver_2 = iLQRReachability(
                self.id, self.config, self.rollout_dyn_1, self.cost)

    def get_action(
        self, obs: np.ndarray, state:np.ndarray, 
        task_ctrl: np.ndarray = np.array([0.0, 0.0]),
        prev_sol: Optional[Dict] = None, 
        prev_ctrl: np.ndarray = np.array([0.0, 0.0]), 
        warmup=False,
    ) -> np.ndarray:

        # Linear feedback policy
        initial_state = np.array(state)
        stopping_ctrl = np.array([self.dyn.ctrl_space[0, 0], 0])
        task_ctrl = np.array(task_ctrl)

        # Find safe policy from step 0
        if prev_sol is not None:
            controls_initialize = prev_sol['reinit_controls']
        else:
            controls_initialize = None

        if prev_sol is None or prev_sol['resolve']:
            control_0, solver_info_0 = self.solver_0.get_action(
                obs=obs, controls=controls_initialize, state=state)
        else:
            # Potential source of acceleration. We don't need to resolve both ILQs as we can reuse
            # solution from previous time. - Unused currently.
            solver_info_0 = prev_sol['bootstrap_next_solution']
            control_0 = (solver_info_0['controls'][:, 0] 
                            + solver_info_0['K_closed_loop'][:, :, 0] @ (initial_state - solver_info_0['states'][:, 0]))
            # Closed loop solution
            #solver_info_0['controls'] = jnp.array( solver_info_0['reinit_controls'] )
            #solver_info_0['states'] = jnp.array( solver_info_0['reinit_states'] )
            #solver_info_0['Vopt'] = solver_info_0['Vopt_next']
            #solver_info_0['marginopt'] = solver_info_0['marginopt_next']
            #solver_info_0['is_inside_target'] = solver_info_0['is_inside_target_next']

        solver_info_0['safe_opt_ctrl'] =  jnp.array(control_0)
        solver_info_0['task_ctrl'] = jnp.array(task_ctrl)

        solver_info_0['mark_barrier_filter'] = False
        solver_info_0['mark_complete_filter'] = False
        # Find safe policy from step 1
        state_imaginary, task_ctrl = self.dyn.integrate_forward(
            state=initial_state, control=task_ctrl
        )
        boot_controls = jnp.array(solver_info_0['controls'])

        _, solver_info_1 = self.solver_1.get_action(
            obs=state_imaginary, controls=boot_controls, state=state_imaginary)

        solver_info_0['Vopt_next'] = solver_info_1['Vopt']
        solver_info_0['marginopt_next'] = solver_info_1['marginopt']
        solver_info_0['is_inside_target_next'] = solver_info_1['is_inside_target']

        if(self.filter_type == "LR" or self.filter_type=="SoftLR"):
            solver_info_0['barrier_filter_steps'] = self.barrier_filter_steps
            if(solver_info_1['Vopt'] <= self.lr_threshold):
                self.filter_steps += 1
                solver_info_0['filter_steps'] = self.filter_steps
                solver_info_0['resolve'] = True
                solver_info_0['reinit_controls'] = jnp.zeros(
                    (self.dim_u, self.N))
                # Warm start for next cycle
                solver_info_0['reinit_controls'] = solver_info_0['reinit_controls'].at[:,
                                                                                       0:self.N - 1].set(solver_info_0['controls'][:, 1:self.N])
                if self.dyn.id ==  "PVTOL6D":
                    solver_info_0['reinit_controls'] = solver_info_0['reinit_controls'].at[1, -1].set(self.dyn.mass * self.dyn.g)
                else:
                    solver_info_0['reinit_controls'] = solver_info_0['reinit_controls'].at[:, -1].set(self.dyn.ctrl_space[0, 0])

                solver_info_0['mark_complete_filter'] = True
                solver_info_0['num_iters'] = 0
                solver_info_0['deviation'] = np.linalg.norm(
                    control_0 - task_ctrl, ord=1)
                #print("Filtered control safe", control_0)
                if solver_info_0['is_inside_target']:
                    # Render the target set controlled invariant
                    return stopping_ctrl + solver_info_0['K_closed_loop'][:, :, 0] @ (initial_state - solver_info_0['states'][:, 0]), solver_info_0
                else:
                    return control_0 + solver_info_0['K_closed_loop'][:, :, 0] @ (initial_state - solver_info_0['states'][:, 0]), solver_info_0
            else:
                solver_info_0['filter_steps'] = self.filter_steps
                solver_info_0['resolve'] = True
                solver_info_0['bootstrap_next_solution'] = solver_info_1
                solver_info_0['reinit_controls'] = jnp.array(
                    solver_info_1['controls'])
                solver_info_0['reinit_states'] = jnp.array(
                    solver_info_1['states'])
                solver_info_0['num_iters'] = 0
                solver_info_0['deviation'] = 0
                return task_ctrl, solver_info_0
        elif(self.filter_type == "CBF" or self.filter_type == "SoftCBF"):
            gamma = self.gamma
            cutoff = gamma * solver_info_0['Vopt']

            control_cbf_cand = task_ctrl

            solver_initial = np.zeros((self.dim_u,))
            if prev_sol is not None:
                solver_initial = (prev_ctrl - control_cbf_cand)

            # Define initial state and initial performance policy
            initial_state_jnp = jnp.array(initial_state[:, np.newaxis])
            control_cbf_cand_jnp = jnp.array(control_cbf_cand[:, np.newaxis])
            num_iters = 0

            # Setting tolerance to zero does not cause big improvements at the
            # cost of more unnecessary looping
            cbf_tol = -1e-5
            # Conditioning parameter out of abundance of caution
            eps_reg = 1e-8

            # Checking CBF constraint violation
            constraint_violation = solver_info_1['Vopt'] - cutoff
            scaled_c = constraint_violation

            # Scaling parameter
            scaling_factor = 0.8

            # Exit loop once CBF constraint satisfied or maximum iterations
            # violated
            control_bias_term = np.zeros((self.dim_u,))
            while((constraint_violation < cbf_tol or warmup) and num_iters < 5):
                num_iters = num_iters + 1

                # Extract information from solver for enforcing constraint
                grad_x = jnp.array(solver_info_1['grad_x'])
                _, B0 = self.dyn.get_jacobian(
                    initial_state_jnp, control_cbf_cand_jnp)

                if self.constraint_type == 'quadratic':
                    grad_xx = np.array(solver_info_1['grad_xx'])

                    # Get jacobian at initial point
                    B0u = B0[:, :, 0]

                    # Compute P, p
                    P = B0u.T @ grad_xx @ B0u - eps_reg * jnp.eye(self.dim_u)

                    # For some reason, hessian from jax is only approxiamtely
                    # symmetric
                    P = 0.5 * (P + P.T)
                    p = grad_x.T @ B0u
                    # Controls improvement direction
                    # limits = np.array( [[self.dyn.ctrl_space[0, 0] - control_cbf_cand[0], self.dyn.ctrl_space[0, 1] - control_cbf_cand[0]],
                    #          [self.dyn.ctrl_space[1, 0] - control_cbf_cand[1], self.dyn.ctrl_space[1, 1] - control_cbf_cand[1]]] )
                    if self.dim_u==2:
                        control_correction = barrier_filter_quadratic_two(
                            P, p, scaled_c, initialize=solver_initial, control_bias_term=control_bias_term)
                    elif self.dim_u==8:
                        control_correction = barrier_filter_quadratic_eight(
                            P, p, scaled_c, initialize=solver_initial, control_bias_term=control_bias_term)    
                elif self.constraint_type == 'linear':
                    control_correction = barrier_filter_linear(
                        grad_x, B0, scaled_c)

                control_bias_term = control_bias_term + control_correction
                control_cbf_cand = control_cbf_cand + \
                    np.array(control_correction)

                # Restart from current point and run again
                solver_initial = (prev_ctrl - control_cbf_cand)

                state_imaginary, control_cbf_cand = self.dyn.integrate_forward(
                    state=initial_state, control=control_cbf_cand
                )
                _, solver_info_1 = self.solver_2.get_action(obs=state_imaginary,
                                                            controls=jnp.array(
                                                                solver_info_1['controls']),
                                                            state=state_imaginary)
                solver_info_0['Vopt_next'] = solver_info_1['Vopt']
                solver_info_0['marginopt_next'] = solver_info_1['marginopt']
                solver_info_0['is_inside_target_next'] = solver_info_1['is_inside_target']

                control_cbf_cand_jnp = jnp.array(control_cbf_cand[:, np.newaxis])

                # CBF constraint violation
                constraint_violation = solver_info_1['Vopt'] - cutoff
                scaled_c = scaling_factor * constraint_violation

            if solver_info_1['Vopt'] > 0:
                if num_iters > 0:
                    self.barrier_filter_steps += 1
                    solver_info_0['mark_barrier_filter'] = True
                solver_info_0['barrier_filter_steps'] = self.barrier_filter_steps
                solver_info_0['filter_steps'] = self.filter_steps
                solver_info_0['resolve'] = False
                solver_info_0['bootstrap_next_solution'] = solver_info_1
                solver_info_0['reinit_controls'] = jnp.array(
                    solver_info_1['controls'])
                solver_info_0['reinit_states'] = jnp.array(
                    solver_info_1['states'])
                #solver_info_0['reinit_J'] = solver_info_1['Vopt']
                solver_info_0['num_iters'] = num_iters
                solver_info_0['deviation'] = np.linalg.norm(
                    control_cbf_cand - task_ctrl, ord=1)
                solver_info_0['qcqp_initialize'] = control_cbf_cand - task_ctrl
                return control_cbf_cand.ravel() + solver_info_0['K_closed_loop'][:, :, 0] @ (initial_state - solver_info_0['states'][:, 0]), solver_info_0

        self.filter_steps += 1
        # Safe policy
        solver_info_0['barrier_filter_steps'] = self.barrier_filter_steps
        solver_info_0['filter_steps'] = self.filter_steps
        solver_info_0['resolve'] = True
        solver_info_0['num_iters'] = num_iters
        solver_info_0['reinit_controls'] = jnp.zeros((self.dim_u, self.N))
        solver_info_0['reinit_controls'] = solver_info_0['reinit_controls'].at[:, 0:self.N - 1].set(
            solver_info_0['controls'][:, 1:self.N])
        if self.dyn.id ==  "PVTOL6D":
            solver_info_0['reinit_controls'] = solver_info_0['reinit_controls'].at[:, 1].set(self.dyn.mass * self.dyn.g)
        else:
            solver_info_0['reinit_controls'] = solver_info_0['reinit_controls'].at[:, -1].set(self.dyn.ctrl_space[0, 0])
        solver_info_0['mark_complete_filter'] = True
        solver_info_0['deviation'] = np.linalg.norm(control_0 - task_ctrl)
        if solver_info_0['is_inside_target']:
            # Render the target set controlled invariant
            safety_control = stopping_ctrl
        else:
            safety_control = solver_info_0['controls'][:, 0] + solver_info_0['K_closed_loop'][:, :, 0] @ (
                initial_state - solver_info_0['states'][:, 0])

        solver_info_0['qcqp_initialize'] = safety_control - task_ctrl

        return safety_control, solver_info_0

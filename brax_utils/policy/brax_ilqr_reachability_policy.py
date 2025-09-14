from typing import Tuple, Optional, Dict
import time
import jax
from jax import numpy as jp
from jax import Array as DeviceArray
from functools import partial

from .brax_ilqr_policy import iLQRBrax


class iLQRBraxReachability(iLQRBrax):

  def get_action(
      self, obs, state, controls = None, **kwargs
  ):
    status = 0
    self.min_alpha = 1e-12
    # `controls` include control input at timestep N-1, which is a dummy
    # control of zeros.
    if controls is None:
        controls = jp.zeros((self.dim_u, self.N))

    # Rolls out the nominal trajectory and gets the initial cost.
    gc_states, controls, pipeline_states = self.rollout_nominal(
        state, controls
    )

    failure_margins = self.cost.constraint.get_mapped_margin(
        gc_states, controls
    )
    ctrl_costs = self.cost.ctrl_cost.get_mapped_margin(gc_states, controls)
    critical, reachable_margin = self.get_critical_points(failure_margins)
    J = (reachable_margin + jp.sum(ctrl_costs)).astype(float)

    converged = False
    time0 = time.time()

    for i in range(self.max_iter):
      # We need cost derivatives from 0 to N-1, but we only need dynamics
      # jacobian from 0 to N-2.
      c_x, c_u, c_xx, c_uu, c_ux = self.cost.get_derivatives(
          gc_states, controls
      )
      fx, fu = self.brax_env.get_batched_generalized_coordinates_grad(pipeline_states, controls)
      V_x, V_xx, k_open_loop, K_closed_loop, _, _, Q_u = self.backward_pass(
          c_x=c_x, c_u=c_u, c_xx=c_xx, c_uu=c_uu, c_ux=c_ux, fx=fx, fu=fu,
          critical=critical
      )
      
      # Choose the best alpha scaling using appropriate line search methods
      alpha_chosen = 1.0
      # Choose the best alpha scaling using appropriate line search methods
      if self.line_search == 'baseline':
          alpha_chosen = self.baseline_line_search(state, gc_states, controls, K_closed_loop, k_open_loop, J)
      elif self.line_search == 'armijo':
          alpha_chosen = self.armijo_line_search( state=state, gc_states=gc_states, controls=controls, Ks1=K_closed_loop, ks1=k_open_loop, 
                                                  critical=critical, J=J, Q_u=Q_u)
      elif self.line_search == 'trust_region_constant_margin':
          alpha_chosen = self.trust_region_search_constant_margin( state=state, gc_states=gc_states, controls=controls, Ks1=K_closed_loop,
                                                                   ks1=k_open_loop, critical=critical, J=J, Q_u=Q_u)
      elif self.line_search == 'trust_region_tune_margin':
          alpha_chosen = self.trust_region_search_tune_margin(state=state, gc_states=gc_states, controls=controls, Ks1=K_closed_loop, 
                                                              ks1=k_open_loop, critical=critical, J=J,  
                                                              c_x=c_x, c_xx=c_xx, Q_u=Q_u) 
      else:
          raise Exception(f'{self.line_search} does not match any implemented line search')
      
      gc_states, controls, pipeline_states, J_new, critical, failure_margins, reachable_margin = self.forward_pass(state, gc_states, controls, K_closed_loop, k_open_loop, alpha_chosen) 
      if (jp.abs((J-J_new) / J) < self.tol):  # Small improvement.
        status = 1
        if J_new>0:
          converged = True

      J = J_new
      
      if alpha_chosen<self.min_alpha:
          status = 2
          break

      # Terminates early if the objective improvement is negligible.
      if converged:
        #print(f"Converged in {i + 1} iterations with {alpha_chosen, J}")
        status = 1
        break
    
    # Retrieve gradients of final converged controls.
    c_x, c_u, c_xx, c_uu, c_ux = self.cost.get_derivatives(
        gc_states, controls
    )
    fx, fu = self.brax_env.get_batched_generalized_coordinates_grad(pipeline_states, controls)
    V_x, V_xx, k_open_loop, K_closed_loop, _, _, Q_u = self.backward_pass(
        c_x=c_x, c_u=c_u, c_xx=c_xx, c_uu=c_uu, c_ux=c_ux, fx=fx, fu=fu,
        critical=critical
    )

    t_process = time.time() - time0
    #print(f"Reachability solver took {t_process} seconds with status {status}")
    solver_info = dict(
        gc_states=gc_states, controls=controls, reinit_controls=controls, t_process=t_process, status=status, Vopt=J, marginopt=reachable_margin,
        grad_x=V_x, grad_xx=V_xx, B0=fu[:, :, 0], is_inside_target=False,  K_closed_loop=K_closed_loop, k_open_loop=k_open_loop, num_ddp_iters=i + 1,
    )

    return controls[:, 0], solver_info

  @partial(jax.jit, static_argnames='self')
  def baseline_line_search(self, state, gc_states, controls, K_closed_loop, k_open_loop, J, beta=0.7, alpha_initial=1.0):
    alpha = alpha_initial
    J_new = -jp.inf

    @jax.jit
    def run_forward_pass(args):
      alpha, J, J_new = args
      alpha = beta*alpha
      _, _, _, J_new, _, _, _ = self.forward_pass(state, gc_states, controls, K_closed_loop, k_open_loop, alpha)
      return alpha, J, J_new

    @jax.jit
    def check_terminated(args):
      alpha, J, J_new = args
      return jp.logical_and( alpha>self.min_alpha, J_new<J )
    
    alpha, J, J_new = jax.lax.while_loop(check_terminated, run_forward_pass, (alpha, J, J_new))

    return alpha

  @partial(jax.jit, static_argnames='self')
  def armijo_line_search(self, state, gc_states, controls, Ks1,
                          ks1, critical, J, Q_u, alpha_init=1.0, beta=0.8):
    @jax.jit
    def run_forward_pass(args):
      state, gc_states, controls, Ks1, ks1, J, _, _, _, Q_u, alpha = args
      alpha = beta * alpha
      X, U, _, J_new, critical_new, _, _ = self.forward_pass(state, gc_states, controls, Ks1, ks1, alpha)
      # critical point
      t_star = jp.argwhere(critical_new != 0, size=self.N - 1)[0][0]

      # Calculate gradient for armijo decrease condition
      grad_u = Ks1[:, :, t_star] @ (X[:, t_star] - gc_states[:, t_star]) + ks1[:, t_star]

      # update gradient along alpha
      grad_alpha = Q_u @ grad_u

      return state, gc_states, controls, Ks1, ks1, J, J_new, t_star, grad_alpha, Q_u, alpha

    @jax.jit
    def check_continue(args):
      _, _, _, _, _, J, J_new, _, grad_alpha, _, alpha = args
      armijo_check = ( J_new < J + 0.5 * grad_alpha * alpha )
      return jp.logical_and(alpha > self.min_alpha, armijo_check)

    alpha = alpha_init
    J_new = -jp.inf
    grad_alpha = 0
    t_star = jp.argwhere(critical != 0, size=self.N - 1)[0][0]

    state, gc_states, controls, Ks1, ks1, J, J_new, _, _, _, alpha = jax.lax.while_loop(check_continue, run_forward_pass, (state, gc_states, controls,
                                                                                                                        Ks1, ks1, J, J_new, t_star, grad_alpha, 
                                                                                                                          Q_u, alpha))

    return alpha

  @partial(jax.jit, static_argnames='self')
  def trust_region_search_constant_margin(self, state, gc_states, controls, Ks1, ks1, critical, J, Q_u, alpha_init=1.0, beta=0.8):
    @jax.jit
    def run_forward_pass(args):
      state, gc_states, controls, Ks1, ks1, J, _, _, _, _, Q_u, alpha = args
      alpha = beta * alpha
      X, U, _, J_new, critical_new, _, _ = self.forward_pass(state, gc_states, controls, Ks1, ks1, alpha)

      # critical point
      t_star = jp.argwhere(critical_new != 0, size=self.N - 1)[0][0]
      traj_diff = jp.max(jp.array([jp.linalg.norm(x_new - x_old)
                          for x_new, x_old in zip(X[:2, :], gc_states[:2, :])]))

      # Calculate gradient for armijo decrease condition
      grad_u = Ks1[:, :, t_star] @ (X[:, t_star] - gc_states[:, t_star]) + ks1[:, t_star]

      # update returns
      grad_alpha = Q_u @ grad_u

      return state, gc_states, controls, Ks1, ks1, J, J_new, t_star, grad_alpha, traj_diff, Q_u, alpha

    @jax.jit
    def check_continue(args):
      _, _, _, _, _, J, J_new, _, grad_alpha, traj_diff, _, alpha = args
      armijo_violation = ( J_new < J + 0.5 * grad_alpha * alpha )
      trust_region_violation = (traj_diff > self.margin)
      return jp.logical_and(alpha > self.min_alpha, jp.logical_or(armijo_violation, trust_region_violation))

    alpha = alpha_init
    J_new = -jp.inf
    grad_alpha = 0
    t_star = jp.argwhere(critical != 0, size=self.N - 1)[0][0]
    self.margin = 2.0
    traj_diff = 0.0

    _, gc_states, controls, Ks1, ks1, J, J_new, _, _, _, _, alpha = jax.lax.while_loop(check_continue, run_forward_pass, (state, gc_states, controls,
                                                                                                                        Ks1, ks1, J, J_new, t_star, grad_alpha, 
                                                                                                                        traj_diff, Q_u, alpha))

    return alpha

  @partial(jax.jit, static_argnames='self')
  def trust_region_search_tune_margin(self, state, gc_states, controls, Ks1,
                          ks1, critical, J, c_x, c_xx, Q_u, alpha_init=1.0, beta=0.8):
    @jax.jit
    def decrease_margin(args):
      traj_diff, rho, margin = args
      margin = 0.85 * margin
      return traj_diff, rho, margin

    @jax.jit
    def increase_margin(args):
      traj_diff, rho, margin = args
      margin = 1.3 * margin
      return traj_diff, rho, margin

    @jax.jit
    def fix_margin(args):
      return args

    @jax.jit
    def increase_or_fix_margin(args):
      traj_diff, rho, margin = args
      traj_diff, rho, margin = jax.lax.cond(jp.logical_and(jp.abs(traj_diff - margin) < 0.1, rho > 0.75), increase_margin,
                          fix_margin, (traj_diff, rho, margin))
      return traj_diff, rho, margin

    @jax.jit
    def run_forward_pass(args):
      state, gc_states, controls, Ks1, ks1, alpha, J, _, _, _, _, _, margin = args
      alpha = beta * alpha
      X, U, _, J_new, critical_new, _, _ = self.forward_pass(state, gc_states, controls, Ks1, ks1, alpha)
      t_star = jp.argwhere(critical_new != 0, size=self.N - 1)[0][0]

      # Calculate gradient for armijo decrease condition
      grad_u = Ks1[:, :, t_star] @ (X[:, t_star] - gc_states[:, t_star]) + ks1[:, t_star]
      # update gradient along alpha
      grad_alpha = Q_u @ grad_u

      # find margin
      traj_diff = jp.max(jp.array([jp.linalg.norm(x_new - x_old)
                          for x_new, x_old in zip(X[:2, :], gc_states[:2, :])]))

      # use the quality of approximation to increase or decrease margin
      x_diff = X[:, t_star] - gc_states[:, t_star]
      delta_cost_quadratic_approx = 0.5 * \
          (x_diff @ c_xx[:, :, t_star] + 2 * c_x[:, t_star]) @ x_diff
      delta_cost_actual = J_new - J
      # old_cost_error = jp.abs(cost_error)
      # cost_error = jp.abs(
      #     delta_cost_quadratic_approx -
      #     delta_cost_actual)
      rho = jp.abs(delta_cost_actual / delta_cost_quadratic_approx)

      return state, gc_states, controls, Ks1, ks1, alpha, J, t_star, J_new, traj_diff, rho, grad_alpha, margin

    @jax.jit
    def check_continue(args):
      _, _, _, _, _, alpha, J, _, J_new, traj_diff, rho, grad_alpha, margin = args
      # Check trust region constraint.
      trust_region_violation = (traj_diff > margin)
      # Check armijo constraint.
      armijo_violation = ( J_new < J + 0.5 * grad_alpha * alpha )
      # update margin.
      traj_diff, rho, margin = jax.lax.cond(rho <= 0.15, decrease_margin,
                                                                increase_or_fix_margin, (traj_diff, rho, margin))
      return jp.logical_and(alpha > self.min_alpha, jp.logical_or(
          trust_region_violation, armijo_violation))

    alpha = alpha_init
    J_new = -jp.inf
    t_star = jp.where(critical != 0, size=self.N - 1)[0][0]
    grad_alpha = 0.0
    traj_diff = 0.0

    # set these to not cause change in margin after first iteration.
    margin = 2.5
    traj_diff = 0.2
    rho = 0.5

    _, _, _, _, _, alpha, _, _, _, _, _, _, _ = (
        jax.lax.while_loop(check_continue, run_forward_pass, (state, gc_states, controls,
                                                              Ks1, ks1, alpha, J, t_star, J_new,
                                                              traj_diff, rho, grad_alpha, margin)))

    return alpha

  
  @partial(jax.jit, static_argnames='self')
  def get_critical_points(
      self, failure_margins: DeviceArray
  ) -> Tuple[DeviceArray, DeviceArray]:

    @jax.jit
    def true_func(args):
      idx, critical, cur_margin, reachable_margin = args
      critical = critical.at[idx].set(True)
      return critical, cur_margin

    @jax.jit
    def false_func(args):
      idx, critical, cur_margin, reachable_margin = args
      return critical, reachable_margin

    @jax.jit
    def critical_pt(i, _carry):
      idx = self.N - 1 - i
      critical, reachable_margin = _carry
      critical, reachable_margin = jax.lax.cond(
          failure_margins[idx] < reachable_margin, true_func, false_func,
          (idx, critical, failure_margins[idx], reachable_margin)
      )
      return critical, reachable_margin

    critical = jp.zeros(shape=(self.N,), dtype=bool)
    critical = critical.at[self.N - 1].set(True)
    critical, reachable_margin = jax.lax.fori_loop(
        1, self.N, critical_pt, (critical, failure_margins[-1])
    )  # backward until timestep 1
    return critical, reachable_margin

  @partial(jax.jit, static_argnames='self')
  def forward_pass(
      self, state, nominal_gc_states: DeviceArray, nominal_controls: DeviceArray,
      K_closed_loop: DeviceArray, k_open_loop: DeviceArray, alpha: float
  ) -> Tuple[DeviceArray, DeviceArray, float, DeviceArray, DeviceArray,
             DeviceArray]:
    X, U, pipeline_states = self.rollout(
        state, nominal_gc_states, nominal_controls, K_closed_loop, k_open_loop, alpha
    )

    # J = self.cost.get_traj_cost(X, U, closest_pt, slope, theta)
    failure_margins = self.cost.constraint.get_mapped_margin(X, U)
    ctrl_costs = self.cost.ctrl_cost.get_mapped_margin(X, U)

    critical, reachable_margin = self.get_critical_points(failure_margins)
    J = (reachable_margin + jp.sum(ctrl_costs)).astype(float)
    return X, U, pipeline_states, J, critical, failure_margins, reachable_margin

  @partial(jax.jit, static_argnames='self')
  def backward_pass(
      self, c_x: DeviceArray, c_u: DeviceArray, c_xx: DeviceArray,
      c_uu: DeviceArray, c_ux: DeviceArray, fx: DeviceArray, fu: DeviceArray,
      critical: DeviceArray
  ) -> Tuple[DeviceArray, DeviceArray, DeviceArray, DeviceArray, DeviceArray,DeviceArray, DeviceArray]:
    """
    Jitted backward pass looped computation.

    Args:
        c_x (DeviceArray): (dim_x, N)
        c_u (DeviceArray): (dim_u, N)
        c_xx (DeviceArray): (dim_x, dim_x, N)
        c_uu (DeviceArray): (dim_u, dim_u, N)
        c_ux (DeviceArray): (dim_u, dim_x, N)
        fx (DeviceArray): (dim_x, dim_x, N-1)
        fu (DeviceArray): (dim_x, dim_u, N-1)

    Returns:
        Ks (DeviceArray): gain matrices (dim_u, dim_x, N - 1)
        ks (DeviceArray): gain vectors (dim_u, N - 1)
    """

    @jax.jit
    def true_func(args):
      idx, V_x, V_xx, ks, Ks, _, _, _ = args

      #! Q_x, Q_xx are not used if this time step is critical.
      # Q_x = c_x[:, idx] + fx[:, :, idx].T @ V_x
      # Q_xx = c_xx[:, :, idx] + fx[:, :, idx].T @ V_xx @ fx[:, :, idx]
      Q_ux = c_ux[:, :, idx] + fu[:, :, idx].T @ V_xx @ fx[:, :, idx]
      Q_u = c_u[:, idx] + fu[:, :, idx].T @ V_x
      Q_uu = c_uu[:, :, idx] + fu[:, :, idx].T @ V_xx @ fu[:, :, idx]

      Q_uu_inv = jp.linalg.inv(Q_uu + reg_mat)
      Ks = Ks.at[:, :, idx].set(-Q_uu_inv @ Q_ux)
      ks = ks.at[:, idx].set(-Q_uu_inv @ Q_u)

      return c_x[:, idx], c_xx[:, :, idx], ks, Ks, c_x[:, idx], c_xx[:, :, idx], Q_u

    @jax.jit
    def false_func(args):
      idx, V_x, V_xx, ks, Ks, V_x_critical, V_xx_critical, _ = args

      Q_x = fx[:, :, idx].T @ V_x
      Q_xx = fx[:, :, idx].T @ V_xx @ fx[:, :, idx]
      Q_ux = c_ux[:, :, idx] + fu[:, :, idx].T @ V_xx @ fx[:, :, idx]
      Q_u = c_u[:, idx] + fu[:, :, idx].T @ V_x
      Q_uu = c_uu[:, :, idx] + fu[:, :, idx].T @ V_xx @ fu[:, :, idx]

      Q_uu_inv = jp.linalg.inv(Q_uu + reg_mat)
      Ks = Ks.at[:, :, idx].set(-Q_uu_inv @ Q_ux)
      ks = ks.at[:, idx].set(-Q_uu_inv @ Q_u)

      # The terms will cancel out but for the regularization added. 
      # See https://studywolf.wordpress.com/2016/02/03/the-iterative-linear-quadratic-regulator-method/ and references therein.
      # V_x = Q_x + Q_ux.T @ ks[:, idx]
      # V_xx = Q_xx + Q_ux.T @ Ks[:, :, idx]

      V_x = Q_x + Ks[:, :, idx].T @ Q_u + Q_ux.T @ ks[:, idx] + Ks[:, :, idx].T @ Q_uu @ ks[:, idx]
      V_xx = (Q_xx + Ks[:, :, idx].T @ Q_ux + Q_ux.T @ Ks[:, :, idx]
            + Ks[:, :, idx].T @ Q_uu @ Ks[:, :, idx])

      return V_x, V_xx, ks, Ks, V_x_critical, V_xx_critical, Q_u

    @jax.jit
    def backward_pass_looper(i, _carry):
      V_x, V_xx, ks, Ks, critical, V_x_critical, V_xx_critical, Q_u = _carry
      idx = self.N - 2 - i

      V_x, V_xx, ks, Ks, V_x_critical, V_xx_critical, Q_u = jax.lax.cond(
          critical[idx], true_func, false_func, (idx, V_x, V_xx, ks, Ks, V_x_critical, V_xx_critical, Q_u)
      )

      return V_x, V_xx, ks, Ks, critical, V_x_critical, V_xx_critical, Q_u

    # Initializes.
    Ks = jp.zeros((self.dim_u, self.dim_x, self.N - 1))
    ks = jp.zeros((self.dim_u, self.N - 1))
    
    V_x_critical = jp.zeros((self.dim_x, ))
    V_xx_critical = jp.zeros((self.dim_x, self.dim_x, ))

    V_x = c_x[:, -1]
    V_xx = c_xx[:, :, -1]
    
    reg_mat = self.eps * jp.eye(self.dim_u)

    V_x, V_xx, ks, Ks, critical, V_x_critical, V_xx_critical, Q_u = jax.lax.fori_loop(
        0, self.N - 1, backward_pass_looper, (V_x, V_xx, ks, Ks, critical, V_x_critical, V_xx_critical, c_u[:, self.N - 1])
    )
    return V_x, V_xx, ks, Ks, V_x_critical, V_xx_critical, Q_u


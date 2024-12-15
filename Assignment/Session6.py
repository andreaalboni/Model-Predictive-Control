
from typing import Callable
from given.homework import problem
import matplotlib.pyplot as plt
import numpy as np 

import os 
WORKING_DIR = os.path.split(__file__)[0]

def lqr_factor_step(N: int, nl: problem.NewtonLagrangeQP) -> problem.NewtonLagrangeFactors:
    #Begin TODO----------------------------------------------------------

    P = [None] * (N + 1)  
    s = [None] * (N + 1)  
    K = [None] * N        
    e = [None] * N       
    
    P[-1] = nl.QN        
    s[-1] = nl.qN        

    for k in range(N-1, -1, -1):
        Ak = nl.Ak[k]
        Bk = nl.Bk[k]
        Qk = nl.Qk[k]
        Rk = nl.Rk[k]
        Sk = nl.Sk[k]
        qk = nl.qk[k]
        rk = nl.rk[k]
        ck = nl.ck[k]

        R_hat = Rk + Bk.T @ P[k + 1] @ Bk
        S_hat = Sk + Bk.T @ P[k + 1] @ Ak
        y = P[k + 1] @ ck + s[k + 1]
        K[k] = -np.linalg.solve(R_hat, S_hat)
        e[k] = -np.linalg.solve(R_hat, Bk.T@y + rk)
        s[k] = qk + Ak.T @ y + S_hat.T @ e[k]
        P[k] = Qk + Ak.T @ P[k + 1] @ Ak + S_hat.T @ K[k]

    #End TODO -----------------------------------------------------------
    return problem.NewtonLagrangeFactors(K, s, P, e)

def symmetric(P):
    return 0.5 * (P.T + P)

def lqr_solve_step(
    prob: problem.Problem,
    nl: problem.NewtonLagrangeQP,
    fac: problem.NewtonLagrangeFactors
) -> problem.NewtonLagrangeUpdate: 
    #Begin TODO----------------------------------------------------------
    
    dx = np.zeros((prob.N + 1, prob.ns))  
    du = np.zeros((prob.N, prob.nu))      
    p = np.zeros((prob.N, prob.ns))       
    dx[0] = 0 
    
    for k in range(prob.N-1):
        Kk = fac.K[k]
        sk1 = fac.s[k+1]
        Pk1 = fac.P[k+1]
        ek = fac.e[k]
    
        Ak = np.array(nl.Ak[k])
        Bk = np.array(nl.Bk[k])
        ck = np.array(nl.ck[k].flatten())

        du[k] = ((Kk @ dx[k]).reshape(-1,1) + ek).flatten()
        dx[k + 1] = Ak @ dx[k] + Bk @ du[k] + ck
        p[k+1] = np.array(Pk1 @ dx[k+1] + sk1).flatten()

    #End TODO -----------------------------------------------------------
    return problem.NewtonLagrangeUpdate(dx, du, p)

def armijo_condition(merit: problem.FullCostFunction, x_plus, u_plus, x, u, dx, du, c, σ, α):
    φ, g, dJdx, dJdu = merit.phi, merit.h, merit.dJdx, merit.dJdu
    return φ(c, x_plus, u_plus) <= φ(c, x, u) + σ * α * (dJdx(x, u) @ dx + dJdu(x,u)@du - c * g(x,u))

def armijo_linesearch(zk: problem.NLIterate, update: problem.NewtonLagrangeUpdate, merit: problem.FullCostFunction, *, σ=1e-4) -> problem.NLIterate:
    #Begin TODO----------------------------------------------------------
    
    alpha = 1.0     # Initial step size
    beta = 0.5      # Step size reduction factor
    c = 10        # Armijo condition constant

    dx, du, p = update.dx, update.du, update.p

    for _ in range(100):
        xplus = zk.x + alpha * dx
        uplus = zk.u + alpha * du
        pplus = alpha * p

        # Check Armijo condition
        #x_plus = xplus.reshape(-1,1)
        #u_plus = uplus.reshape(-1,1)
        #x = zk.x.reshape(-1,1)
        #u = zk.u.reshape(-1,1)
        #_dx = dx.reshape(-1,1)
        #_du = du.reshape(-1,1)
        #if _ == 0:
        #    print(x_plus.shape)
        #    print(u_plus.shape)
        #    print(x.shape)
        #    print(u.shape)
        #    print(_dx.shape)
        #    print(_du.shape)
        if armijo_condition(merit, xplus.reshape(-1,1), uplus.reshape(-1,1), 
                            zk.x.reshape(-1,1), zk.u.reshape(-1,1), 
                            dx.reshape(-1,1), du.reshape(-1,1), c, σ, alpha):
            print(f"alpha: {alpha}")
            print(f"dp: {_}")
            break
        alpha *= beta

    xplus[0] = zk.x[0]

    #End TODO -----------------------------------------------------------
    return problem.NLIterate(xplus, uplus, pplus)

def update_iterate(zk: problem.NLIterate, update: problem.NewtonLagrangeUpdate, *, linesearch: bool, merit_function: problem.FullCostFunction=None) -> problem.NLIterate:
    """Take the current iterate zk and the Newton-Lagrange update and return a new iterate. 

    If linesearch is True, then also perform a linesearch procedure. 

    Args:
        zk (problem.NLIterate): Current iterate 
        update (problem.NewtonLagrangeUpdate): Newton-Lagrange step 
        linesearch (bool): Perform line search or not? 
        merit_function (problem.FullCostFunction, optional): The merit function used for linesearch. Defaults to None.

    Raises:
        ValueError: If no merit function was passed, but linesearch was requested. 

    Returns:
        problem.NLIterate: Next Newton-Lagrange iterate.
    """
    
    if linesearch:
        if merit_function is None:
            raise ValueError("No merit function was passed but line search was requested")
        return armijo_linesearch(zk, update, merit_function)
    #Begin TODO----------------------------------------------------------
    
    dx, du, p = update.dx, update.du, update.p
    alpha = 1.0
    
    xnext = zk.x + alpha * dx
    unext = zk.u + alpha * du
    pnext = alpha * p

    xnext[0] = zk.x[0]
    return problem.NLIterate(
        x = xnext,
        u = unext, 
        p = pnext 
    )

    #End TODO -----------------------------------------------------------

def is_posdef(M):
    return np.min(np.linalg.eigvalsh(M)) > 0

def regularize(qp: problem.NewtonLagrangeQP):
    """Regularize the problem.

    If the given QP (obtained as a linearization of the problem) is nonconvex, 
    add an increasing multiple of the identity to the Hessian 
    until it is positive definite. 

    Side effects: the passed qp is modified by the regularization!

    Args:
        qp (problem.NewtonLagrangeQP): Linearization of the optimal control problem
    """
    #Begin TODO----------------------------------------------------------

    for _ in range(qp.Qk.shape[0]):
        Q_bar = qp.Qk[_]
        _lambda = 1e-6
        while not is_posdef(Q_bar):
            Q_bar += _lambda * np.eye(Q_bar.shape[0])
            _lambda *= 2

    #End TODO -----------------------------------------------------------

def newton_lagrange(p: problem.Problem,
         initial_guess = problem.NLIterate, cfg: problem.NewtonLagrangeCfg = None, *,
         log_callback: Callable = lambda *args, **kwargs: ...
) -> problem.NewtonLagrangeStats:
    """Newton Lagrange method for nonlinear OCPs
    Args:
        p (problem.Problem): The problem description 
        initial_guess (NLIterate, optional): Initial guess. Defaults to problem.NewtonLagrangeIterate.
        cfg (problem.NewtonLagrangeCfg, optional): Settings. Defaults to None.
        log_callback (Callable): A function that takes the iteration count and the current iterate. Useful for logging purposes. 

    Returns:
        Solver stats  
    """
    stats = problem.NewtonLagrangeStats(0, initial_guess)
    # Set the default config if None was passed 
    if cfg is None:
        cfg = problem.NewtonLagrangeCfg()

    # Get the merit function ingredients in case line search was requested 
    if cfg.linesearch:
        full_cost = problem.build_cost_and_constraint(p)
    else: 
        full_cost = None # We don't need it in this case 
    
    QP_sym = problem.construct_newton_lagrange_qp(p)
    zk = initial_guess

    for it in range(cfg.max_iter):
        qp_it = QP_sym(zk)

        if cfg.regularize:
            regularize(qp_it)

        factor = lqr_factor_step(p.N, qp_it)

        update = lqr_solve_step(p, qp_it, factor)

        zk = update_iterate(zk, update, linesearch=cfg.linesearch, merit_function=full_cost)

        stats.n_its = it 
        stats.solution = zk 
        # Call the logger. 
        log_callback(stats)

        # Sloppy heuristics as termination criteria.
        # In a real application, it's better to check the violation of the KKT conditions.
        # e.g., terminate based on the norm of the gradients of the Lagrangian.
        if np.linalg.norm(update.du.squeeze(), ord=np.inf)/np.linalg.norm(zk.u) < 1e-4:
            stats.exit_message = "Converged\n"
            stats.success = True 
            return stats

        elif np.any(np.linalg.norm(update.du) > 1e4): 
            stats.exit_message = "Diverged\n"
            return stats
        
    stats.exit_message = "Maximum number of iterations exceeded\n"
    return stats

def exercise1():
    print("Assignment 6.1.")
    p = problem.Problem()
    qp = problem.construct_newton_lagrange_qp(p)

def fw_euler(f, Ts):
    return lambda x,u,t: x + Ts*f(x,u)

def test_linear_system():

    p = problem.ToyProblem(cont_dynamics = problem.LinearSystem(), N=100)
    u0 = np.zeros((p.N, p.nu))
    x0 = np.zeros((p.N+1, p.ns))
    x0[0] = p.x0
    initial_guess = problem.NLIterate(x0, u0, np.zeros_like(x0))

    logger = problem.Logger(p, initial_guess)
    result = newton_lagrange(p, initial_guess, log_callback=logger)
    assert result.success, "Newton Lagrange did not converge on a linear system! Something is wrong!"
    assert result.n_its < 2, "Newton Lagrange took more than 2 iterations!"



def exercise2():
    print("Assignment 6.2.")
    from rcracers.simulator.core import simulate
    
    # Build the problem 
    p = problem.ToyProblem()

    # Select initial guess by running an open-loop simulation
    x0 = np.zeros((p.N+1, p.ns))
    u0 = np.zeros((p.N, p.nu))
    x0[0] = p.x0

    initial_guess = problem.NLIterate(x0, u0, np.zeros_like(x0))
    logger = problem.Logger(p, initial_guess)
    stats = newton_lagrange(p, initial_guess, log_callback=logger)
    print(stats.exit_message)

    from given.homework import animate
    plt.rcParams["animation.writer"] = "ffmpeg"
    animate.animate_iterates(logger.iterates, os.path.join(WORKING_DIR, "Assignment6-2"))

def exercise34(linesearch:bool):
    print("Assignment 6.3 and 6.4.")
    from rcracers.simulator.core import simulate
    f = problem.ToyDynamics(False)

    # Build the problem 
    p = problem.ToyProblem()

    # Select initial guess by running an open-loop simulation
    #Begin TODO----------------------------------------------------------
    
    def open_loop_policy():
        return np.zeros(p.nu)
    
    x_trajectory = simulate(x0=p.x0, dynamics=fw_euler(f, p.Ts), n_steps=p.N, policy=open_loop_policy)
    u_trajectory = np.zeros((p.N, p.nu)) 
    initial_guess = problem.NLIterate(x=x_trajectory, u=u_trajectory, p=np.zeros_like(x_trajectory))
    
    #End TODO -----------------------------------------------------------
    
    logger = problem.Logger(p, initial_guess)
    cfg = problem.NewtonLagrangeCfg(linesearch=linesearch, max_iter=100)
    final_iterate = newton_lagrange(p, initial_guess, log_callback=logger, cfg=cfg)
    print(final_iterate.exit_message)
    from given.homework import animate
    if not linesearch:
        animate.animate_iterates(logger.iterates, os.path.join(WORKING_DIR, "Assignment6-3"))
    else:
        animate.animate_iterates(logger.iterates, os.path.join(WORKING_DIR, "Assignment6-4"))

def exercise56(regularize=False):
    from rcracers.simulator.core import simulate
    f = problem.KinematicBicycle()
    
    # Build the problem 
    p = problem.ParkingProblem()

    # Select initial guess by running an open-loop simulation
    #Begin TODO----------------------------------------------------------
    
    #x0 = np.zeros((p.N+1, p.ns))
    #u0 = np.zeros((p.N, p.nu))
    #x0[0] = p.x0
    #initial_guess = problem.NLIterate(x0, u0, np.zeros_like(x0))
    
    def open_loop_policy():
        return np.zeros(p.nu)
    
    x_trajectory = simulate(x0=p.x0, dynamics=fw_euler(f, p.Ts), n_steps=p.N, policy=open_loop_policy)
    u_trajectory = np.zeros((p.N, p.nu)) 
    initial_guess = problem.NLIterate(x=x_trajectory, u=u_trajectory, p=np.zeros_like(x_trajectory))

    #End TODO -----------------------------------------------------------

    logger = problem.Logger(p, initial_guess)
    cfg = problem.NewtonLagrangeCfg(linesearch=True, max_iter=50, regularize=regularize)
    final_iterate = newton_lagrange(p, initial_guess, log_callback=logger, cfg=cfg)
    print(final_iterate.exit_message)

    from given.homework import animate
    animate.animate_iterates(logger.iterates, os.path.join(WORKING_DIR, f"Assignment6-5-reg{regularize}"))
    animate.animate_positions(logger.iterates, os.path.join(WORKING_DIR, f"parking_regularize-{regularize}"))


if __name__ == "__main__":
    #test_linear_system()
    #exercise2()
    #exercise34(False)
    #exercise34(True)
    #exercise56(regularize=False)
    exercise56(regularize=True)
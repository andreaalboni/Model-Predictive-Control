import sys, os
sys.path.append(os.path.split(__file__)[0])  # Allow relative imports
import numpy as np
import numpy.random as npr
import casadi as cs
import matplotlib.pyplot as plt
from given.problem import (
    system_info,
    get_linear_dynamics,
    get_system_equations,
    build_mhe,
    default_config,
    simulate,
    Config,
    ObserverLog,
    LOGGER,
)

def parse_covariances(Q: np.ndarray, R: np.ndarray, P: np.ndarray):
    """Parse covariance arguments."""
    f, h = get_system_equations(symbolic=True, noise=True)
    nx, nw, nv = system_info(f, h)
    return (
        np.eye(nw) * Q if isinstance(Q, float) else Q,
        np.eye(nv) * R if isinstance(R, float) else R,
        np.eye(nx) * P if isinstance(P, float) else P,
    )

class EKF:
    def __init__(
        self,
        f: cs.Function,
        h: cs.Function,
        x0: np.ndarray,
        *,
        Q: np.ndarray = 0.002**2,
        R: np.ndarray = 0.25**2,
        P: np.ndarray = 0.5**2,
        clipping: bool = False,
    ):
        Q, R, P = parse_covariances(Q, R, P)
        self.dfdx, self.dfdw, self.dhdx = get_linear_dynamics(f, h)
        self.f, self.h = f, h
        self.Q, self.R, self.P = Q, R, P
        self.x = x0
        self.clipping = clipping

    def __call__(self, y: np.ndarray, log: LOGGER = None):
        P = self.P
        C: np.ndarray = cs.DM.full(self.dhdx(self.x))

        # measurement update
        L = np.linalg.solve(C @ P @ C.T + self.R, C @ P).T
        P = P - L @ C @ P
        _x = cs.DM.full(self.x + L @ (y - self.h(self.x)))
        if self.clipping:
            _x = np.maximum(_x, 0.0)

        # time update
        w = np.zeros(self.Q.shape[0])
        A: np.ndarray = cs.DM.full(self.dfdx(_x, w))
        # general expression: P = A @ P @ A.T + G @ Q @ G.T
        # but in this case G = np.eye(3), so we can omit it.
        self.P = A @ P @ A.T + self.Q
        self.x = np.squeeze(cs.DM.full(self.f(_x, w)))

        # log results
        #if log is not None:
        #    log("y", y)
        #    log("x", np.squeeze(_x))

class MHE:
    def __init__(
        self,
        f: cs.Function,
        h: cs.Function,
        horizon: int,
        *,
        Q: float = 0.002**2,
        R: float = 0.25**2,
        ekf_initial_state: np.ndarray,
        ekf_clipping: bool = False
    ):
        Q, R, _ = parse_covariances(Q, R, 0.0)
        self.f, self.h = f, h
        self.horizon = horizon

        self.loss = lambda w, v: (
            w.T @ np.linalg.inv(Q) @ w + v.T @ np.linalg.inv(R) @ v
        )
        self.lbx, self.ubx = 0.0, 10.0

        self.y = []
        self.solver = self.build(horizon)

        # Initialize EKF
        self.ekf = EKF(f, h, ekf_initial_state, Q=Q, R=R, clipping=ekf_clipping)

    @property
    def nx(self):
        nx, _, _ = system_info(self.f, self.h)
        return nx

    @property
    def nw(self):
        _, nw, _ = system_info(self.f, self.h)
        return nw

    def build(self, horizon: int):
        return build_mhe(
            self.loss,
            self.f,
            self.h,
            horizon,
            lbx=self.lbx,
            ubx=self.ubx,
            use_prior=True
        )

    def __call__(self, y: np.ndarray, log: LOGGER):
        # Update EKF with new measurement
        self.ekf(y, log)

        # store the new measurement
        self.y.append(y)
        if len(self.y) > self.horizon:
            self.y.pop(0)

        # get solver and bounds
        solver = self.solver
        if len(self.y) < self.horizon:
            solver = self.build(len(self.y))

        # Use EKF state as initial guess for MHE
        initial_state = self.ekf.x
        P = self.ekf.P

        # update mhe
        x, _ = solver(P, initial_state, self.y)

        # update log
        log("x", x[-1, :])
        log("y", y)

def show_result(t: np.ndarray, x: np.ndarray, x_: np.ndarray):
    _, ax = plt.subplots(1, 1)
    c = ["C0", "C1", "C2"]
    h = []
    for i, c in enumerate(c):
        h += ax.plot(t, x_[..., i], "--", color=c)
        h += ax.plot(t, x[..., i], "-", color=c)
    ax.set_xlim(t[0], t[-1])
    if np.max(x_) >= 10.0:
        ax.set_yscale('log')
    else:
        ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Time")
    ax.set_ylabel("Concentration")
    ax.legend(
        h,
        [
            "$A_{\mathrm{est}}$",
            "$A$",
            "$B_{\mathrm{est}}$",
            "B",
            "$C_{\mathrm{est}}$",
            "C",
        ],
        loc="lower left",
        mode="expand",
        ncol=6,
        bbox_to_anchor=(0, 1.02, 1, 0.2),
        borderaxespad=0,
    )
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    plt.show()



def Assignment51(clipping=False):
    if not clipping:
        print("Assignment 5.1:")
    else:
        print("\nAssignment 5.4:")
    
    for horizon in [10, 25]:
        # problem setup
        cfg = default_config()
        n_steps = 400

        # gather dynamics
        fs, hs = get_system_equations(symbolic=True, noise=True, Ts=cfg.Ts)

        # setup the moving horizon estimator
        mhe = MHE(fs, hs, horizon=horizon, ekf_initial_state=cfg.x0_est, ekf_clipping=clipping)

        # prepare log
        log = ObserverLog()
        log.append("x", cfg.x0_est)  # add initial estimate

        # simulate
        f, h = get_system_equations(noise=(0.0, cfg.sig_v), Ts=cfg.Ts, rg=cfg.rg)
        x = simulate(cfg.x0, f, n_steps=n_steps, policy=mhe, measure=h, log=log)
        t = np.arange(0, n_steps+1) * cfg.Ts

        # plot output in `x` and `log.x`
        show_result(t, x, log.x)


if __name__ == "__main__":
    Assignment51()
    Assignment51(clipping=True)
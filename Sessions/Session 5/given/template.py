import numpy as np
import numpy.random as npr
import casadi as cs
from rcracers.simulator.core import rk4
import matplotlib.pyplot as plt

from problem import (
    Config,
    get_system_equations,
    get_linear_dynamics,
    build_mhe,
    default_config,
    simulate,
    ObserverLog,
    LOGGER,
)


class EKF:
    def __init__(self) -> None:
        # Create an instance of this `EKF`
        self.fs, self.hs = get_system_equations(symbolic=True, noise=True)
        self.A, self.G, self.C = get_linear_dynamics(self.fs, self.hs)
        params = Config()
        self.Q = params.sig_p ** 2 * np.eye(3)
        self.R = params.sig_w ** 2 * np.eye(1)
        self.P_hat = params.sig_v ** 2 * np.eye(3)
        self.x_hat = params.x0_est.reshape(-1, 1)  # Ensure x_hat is a column vector

    def __call__(self, y: np.ndarray, log: LOGGER):
        A = self.A
        G = self.G
        C = self.C
        Q = self.Q
        R = self.R
        P = self.P_hat

        P = P - P @ C(self.x_hat).T @ np.linalg.inv(C(self.x_hat) @ P @ C(self.x_hat).T + R) @ C(self.x_hat) @ P
        self.L = P @ C(self.x_hat).T @ np.linalg.inv(C(self.x_hat) @ P @ C(self.x_hat).T + R)
        self.x_hat = self.x_hat + self.L @ (y - self.hs(self.x_hat))

        self.x_hat = self.fs(self.x_hat, 0)
        self.P_hat = A(self.x_hat, 0) @ P @ A(self.x_hat, 0).T + G(self.x_hat, 0) @ Q @ G(self.x_hat, 0).T

        # Log the state estimate and the measurement for plotting
        log("y", y)
        log("x", self.x_hat)


class MHE:
    def __init__(self) -> None:
        """Create an instance of this `MHE`.
        TODO: Pass required arguments and build the MHE problem using
            `build_mhe`. You can use the output of `get_system_equations`.
        """
        print("UNIMPLEMENTED: MHE.")
        print("Try using:  `build_mhe`:")
        print(build_mhe.__doc__)

    def __call__(self, y: np.ndarray, log: LOGGER):
        """Process a measurement
            TODO: Implement MHE using the solver produced by `build_mhe`.

        :param y: the measurement
        :param log: the logger for output
        """
        # log the state estimate and the measurement for plotting
        log("y", y)
        log("x", np.zeros(3))


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

def part_1():
    """Implementation for Exercise 1."""
    print("\nExecuting Exercise 1\n" + "-" * 80)
    # problem setup
    cfg = default_config()
    n_steps = 400

    # setup the extended kalman filter
    ekf = EKF()

    # prepare log
    log = ObserverLog()
    log.append("x", cfg.x0_est)  # add initial estimate

    # simulate
    f, h = get_system_equations(noise=(0.0, cfg.sig_v), Ts=cfg.Ts, rg=cfg.rg)
    x = simulate(cfg.x0, f, n_steps=n_steps, policy=ekf, measure=h, log=log)
    t = np.arange(0, n_steps + 1) * cfg.Ts

    # plot output in `x` and `log.x`
    print(f'{x.shape=}')
    print(f'{log.x.shape=}')

    show_result(t, x, log.x)


def part_2():
    """Implementation for Exercise 2."""
    print("\nExecuting Exercise 2\n" + "-" * 80)
    # problem setup
    cfg = default_config()

    # setup the extended kalman filter
    mhe = MHE()

    # prepare log
    log = ObserverLog()
    log.append("x", cfg.x0_est)  # add initial estimate

    # simulate
    f, h = get_system_equations(noise=(0.0, cfg.sig_v), Ts=cfg.Ts, rg=cfg.rg)
    x = simulate(cfg.x0, f, n_steps=400, policy=mhe, measure=h, log=log)

    # plot output in `x` and `log.x`
    print(f'{x.shape=}')
    print(f'{log.x.shape=}')


def part_3():
    """Implementation for Homework."""
    print("\nExecuting Homework\n" + "-" * 80)
    # problem setup
    cfg = default_config()

    # setup the extended kalman filter
    mhe = MHE()

    # prepare log
    log = ObserverLog()
    log.append("x", cfg.x0_est)  # add initial estimate

    # simulate
    f, h = get_system_equations(noise=(0.0, cfg.sig_v), Ts=cfg.Ts, rg=cfg.rg)
    x = simulate(cfg.x0, f, n_steps=400, policy=mhe, measure=h, log=log)

    # plot output in `x` and `log.x`
    print(f'{x.shape=}')
    print(f'{log.x.shape=}')


if __name__ == "__main__":
    part_1()
    part_2()
    part_3()

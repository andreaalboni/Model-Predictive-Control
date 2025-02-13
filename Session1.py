import numpy as np 
import numpy.linalg as la
import matplotlib.pyplot as plt
from typing import Tuple , Callable
from scipy.linalg import solve_discrete_are


def get_dynamics_continuous() -> Tuple[np.ndarray]: 
    """Get the continuous-time dynamics represented as 
    
    ..math::
        \dot{x} = A x + B u 
    
    """
    A = np.array(
        [[0., 1.],
         [0., 0.]]
        )
    B = np.array(
        [[0],
         [-1]]
    )
    return A, B

def get_dynamics_discrete(ts: float) -> Tuple[np.ndarray]: 
    """Get the dynamics of the cars in discrete-time:

    ..math::
        x_{k+1} = A x_k + B u_k  
    
    Args: 
        ts [float]: sample time [s]
    """
    A, B = get_dynamics_continuous()
    Ad = np.eye(2) + A * ts 
    Bd = B * ts 
    return Ad, Bd

def setup():
    ts = 0.5 
    C = np.array([[1, -2./3]])
    Q = C.T@C + 1e-3 * np.eye(2)
    R = np.array([[0.1]])

    A, B = get_dynamics_discrete(ts)
    
    return A, B, Q, R

def riccati_recursion(A: np.ndarray, B: np.ndarray, R: np.ndarray, Q: np.ndarray, Pf: np.ndarray, N: int):
    """Solve the finite-horizon LQR problem through recursion

    Args:
        A: System A-matrix 
        B: System B-matrix 
        R: weights on the input (positive definite)
        Q: weights on the states (positive semidefinite)
        Pf: Initial value for the Hessian of the cost-to-go (positive definite)
        N: Control horizon  
    """

    P = [Pf] 
    K = [] 
    for _ in range(N):
        Kk = -la.solve(R + B.T@P[-1]@B, B.T@P[-1]@A)
        K.append(Kk)
        Pk = Q + A.T@P[-1]@(A + B@K[-1])
        P.append(Pk)

    return P[::-1], K[::-1]  # Reverse the order for easier indexing later.

def simulate(x0: np.ndarray, f: Callable, policy: Callable, steps: int) -> Tuple[np.ndarray, bool]:
    """Generic simulation loop.
    
    Simulate the discrete-time dynamics f: (x, u) -> x
    using policy `policy`: (x, t) -> u 
    for `steps` steps ahead and return the sequence of states.

    Returns 
        x: sequence of states 
        instability_occurred: whether or not the state grew to a large norm, indicating instability 
    """
    instability_occured = False  # Keep a flag that indicates whenever we detected instability. 
    x = [x0]
    for t in range(steps):
        xt = x[-1]
        ut = policy(xt, t)
        xnext = f(xt, ut)
        x.append(xnext)
        if np.linalg.norm(xnext) > 100 and not instability_occured:  
            # If the state become very large, we flag instability. 
            instability_occured = True 
    
    return np.array(x), instability_occured

def plot_sim(x0: np.ndarray, A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray, gains: list, Kinf: np.ndarray, sim_time=10): 
    """Plot the simulated states, when applying the given state feedback gains.
    
    Args: 
        x0: Initial state 
        A: A-matrix of the system
        B: B-matrix of the system
        R: weights on the input (positive definite)
        Q: weights on the states (positive semidefinite)
        gains: List of state-feedback gains
        sim_time: Number of simulated samples of the closed-loop system
    """

    def f(x,u):
        """Dynamics"""
        return A@x + B@u

    def κ(x,t):
        """Control policy (receding horizon)"""
        return gains[0]@x
    
    ## -------------------- Finite Horizon Controller -------------------- ##
    horizon = len(gains)
    x_closed_loop, cl_unstable = simulate(x0, f, κ, sim_time)

    if cl_unstable: 
        print(" The state grew quite large under the closed-loop policy, which indicates instability!")
    plt.figure()
    plt.plot(x_closed_loop[:, 0], x_closed_loop[:, 1], marker=".", color="k", linewidth=2)

    def κ_pred(x,t): 
        """Control policy (receding horizon)"""
        return gains[t]@x

    for xt in x_closed_loop:
        x_pred, _ = simulate(xt, f, κ_pred, horizon)
        plt.plot(x_pred[:, 0], x_pred[:, 1], color="tab:red", linestyle="--", marker=".", alpha=0.5)
    
    ## ----------------- Infinite Horizon LQR Controller ----------------- ##
    def κ_inf(x,t):
        """Control policy (receding horizon)"""
        return Kinf@x
    
    x_cl_inf, _ = simulate(x0, f, κ_inf, sim_time)
    plt.plot(x_cl_inf[:, 0], x_cl_inf[:, 1], marker=".", color="blue", linewidth=0.5)
    
    plt.annotate("$x_0$", x0)
    plt.title(f"State trajectory (real: black | predicted: red | infinite: blue) for N = {horizon}")
    plt.xlabel("Position")
    plt.ylabel("Velocity")
    #filename = f"images/Assignment_11_N{horizon}.png"
    #plt.savefig(filename, dpi=700, format='png', bbox_inches='tight')
    plt.show()

def finite_horizon_cost(x0: np.ndarray, A: np.ndarray, B: np.ndarray, K: np.ndarray, Q: np.ndarray, R: np.ndarray, sim_time=10):
    def f(x,u):
        """Dynamics"""
        return A@x + B@u

    def κ(x,t):
        """Control policy (receding horizon)"""
        return K@x
    
    x_closed_loop, _ = simulate(x0, f, κ, sim_time)
    M = Q + K.T @ R @ K

    cost = 0
    for k in range(sim_time):
        xk = x_closed_loop[k,:].T
        cost += xk.T @ M @ xk
    
    return cost

def Assignment11():
    A, B, Q, R = setup()
    x0 = 10 * np.ones(2)
    Pf = Q

    Pinf = solve_discrete_are(A, B, Q, R)
    Kinf = -np.linalg.solve(R + B.T @ Pinf @ B, B.T @ Pinf @ A)

    for N in [4, 6, 10, 20]:
        print(f"Running recursion for N = {N}.")
        _, K = riccati_recursion(A, B, R, Q, Pf, N)
        plot_sim(x0, A, B, Q, R, K, Kinf, sim_time=30)

    print("Kinf: ", Kinf)

def Assignment12():
    ## -------------------- Value Function vs. N -------------------- ##
    A, B, Q, R = setup()
    x0 = 10 * np.ones(2)
    Vn = []
    Vinf = []
    Vhat = []
    Pf = Q

    Pinf = solve_discrete_are(A, B, Q, R)
    horizons = [_ for _ in range(1,11)]

    for N in horizons:
        P, K = riccati_recursion(A, B, R, Q, Pf, N)
        Vn.append(x0.T@P[0]@x0)
        Vinf.append(x0.T@Pinf@x0)
        Vhat.append(finite_horizon_cost(x0, A, B, K[0], Q, R, sim_time=100))
        #print(f"For N = {N}, Vn is {Vn[-1]}, Vinf is {Vinf[-1]} and Vhat is {Vhat[-1]}.")

    plt.figure()
    line1, = plt.plot(horizons, Vn, color="cyan", linewidth=1, label="$V_N$")
    plt.scatter(horizons, Vn, color="blue", marker="^")
    line2, = plt.plot(horizons, Vinf, color="violet", linewidth=1, label="$V_{\infty}$")
    plt.scatter(horizons, Vinf, color="purple", marker=".")
    line3, = plt.plot(horizons, Vhat, color="lightgreen", linewidth=1, label="$\hat{V}_N$")
    plt.scatter(horizons, Vhat, color="darkgreen", marker="x")
    plt.legend(handles=[line1, line2, line3])
    plt.title("Assignment 1.2")
    plt.xlabel("Horizon Length (N)")
    plt.ylabel("Value Function")
    plt.ylim(0, 2000)

    #filename = "Assignment_12.png"
    #plt.savefig(filename, dpi=700, format='png', bbox_inches='tight')

    plt.show()


def main():
    Assignment11()
    Assignment12()

if __name__ == "__main__": 
    main()
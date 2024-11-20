import numpy as np 
import cvxpy as cp
import time
import matplotlib.pyplot as plt
from typing import Tuple , Callable


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

def Assignment23(initial_states, pmax: float, umin: float, Ts: float):
    bd = []
    # initial states: [-10, 0] <= x0=[p0, v0] <= [1, 25]
    for x0 in initial_states:
        _, initial_vel = x0
        bd.append(pmax + initial_vel**2/umin - 0.5*initial_vel*(initial_vel + Ts*umin)/umin)

    return bd

# TO CHECK WHETHER IT MAKES SENSE
def check_initial_feasibility(initial_state, horizon, pmin, pmax, umin, umax, vmin, vmax):
    A, B, _, _ = setup()

    # Define the optimization variables
    x = cp.Variable((A.shape[0], horizon + 1))
    u = cp.Variable((B.shape[1], horizon))

    # Define the constraints
    xmax = np.array([pmax, vmax])
    xmin = np.array([pmin, vmin])
    umax = np.array([umax])
    umin = np.array([umin])

    constraints = [x[:, 0] == initial_state]
    for t in range(horizon):
        constraints += [x[:, t + 1] == A @ x[:, t] + B @ u[:, t]]
        constraints += [x[:, t] <= xmax]
        constraints += [x[:, t] >= xmin]
        constraints += [u[:, t] <= umax]
        constraints += [u[:, t] >= umin]

    # Define the optimization problem
    problem = cp.Problem(cp.Minimize(0), constraints)

    # Solve the optimization problem
    try:
        problem.solve()
        feasible = problem.status == cp.OPTIMAL
    except cp.error.SolverError:
        feasible = False

    return feasible

def check_feasibility(initial_state, horizon, pmin, pmax, umin, umax, vmin, vmax):
    A, B, _, _ = setup()

    # Define the optimization variables
    x = cp.Variable((A.shape[0], horizon + 1))
    u = cp.Variable((B.shape[1], horizon))

    # Define the constraints
    xmax = np.array([pmax, vmax])
    xmin = np.array([pmin, vmin])
    umax = np.array([umax])
    umin = np.array([umin])

    constraints = [x[:, 0] == initial_state]
    for t in range(horizon):
        constraints += [x[:, t + 1] == A @ x[:, t] + B @ u[:, t]]
        constraints += [x[:, t] <= xmax]
        constraints += [x[:, t] >= xmin]
        constraints += [u[:, t] <= umax]
        constraints += [u[:, t] >= umin]

    # Define the optimization problem
    problem = cp.Problem(cp.Minimize(0), constraints)

    # Solve the optimization problem with a specific solver and warm start
    try:
        problem.solve(solver=cp.ECOS, warm_start=False)
        feasible = problem.status == cp.OPTIMAL
    except cp.error.SolverError:
        feasible = False

    return feasible

def Assigment24():
    pmin = -150
    pmax = 1.0
    umin = -5.0
    umax = 10.0
    vmin = -20.0
    vmax = 25.0
    ts = 0.5
    initial_states_feasible = []
    initial_states_feasible1 = []

    # Generate grid of initial states
    p0_range = np.linspace(-10, 1, 30)
    v0_range = np.linspace(0, 25, 30)
    p0, v0 = np.meshgrid(p0_range, v0_range)
    initial_states = np.vstack([p0.ravel(), v0.ravel()]).T

    # Assignement 2.3
    break_distances = Assignment23(initial_states=initial_states, pmax=pmax, umin=umin, Ts=ts)
   
    # Check initial feasibility of each initial state given MPC of different horizons {2, 5, 10}
    for N in [2, 5, 10]:
        start_time = time.time()

        for x0 in initial_states:
            initial_feasibility = check_initial_feasibility(initial_state=x0, horizon=N, pmin=pmin, pmax=pmax, umin=umin, umax=umax, vmin=vmin, vmax=vmax)
            initial_states_feasible.append([x0, initial_feasibility])
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Method 1, horizon {N}: {elapsed_time:.2f} seconds")



        start_time = time.time()

        #for x0 in initial_states:
        #    initial_feasibility1 = check_feasibility(initial_state=x0, horizon=N, pmin=pmin, pmax=pmax, umin=umin, umax=umax, vmin=vmin, vmax=vmax)
        #    initial_states_feasible1.append([x0, initial_feasibility1])
        #
        #end_time = time.time()
        #elapsed_time = end_time - start_time
        #print(f"Method 2, horizon {N}: {elapsed_time:.2f} seconds")



        # Separate feasible and non-feasible points
        feasible_points = np.array([point[0] for point in initial_states_feasible if point[1]])
        non_feasible_points = np.array([point[0] for point in initial_states_feasible if not point[1]])

        # Plot initial states
        plt.figure()
        if feasible_points.size > 0:
            plt.scatter(feasible_points[:, 0], feasible_points[:, 1], color='green', marker='o', s=10, label='Feasible')
        if non_feasible_points.size > 0:
            plt.scatter(non_feasible_points[:, 0], non_feasible_points[:, 1], color='red', marker='o', s=10, label='Non-Feasible')
        plt.plot(break_distances, np.linspace(0, 25, 900), color='cyan', linewidth=2, label='Constraint\'s boundary')
        plt.title("Initial feasibility analysis for MPC of horizon {}".format(N))
        plt.xlabel("Position ($p_0$)")
        plt.ylabel("Velocity ($v_0$)")
        plt.grid(True)
        plt.ylim(-1, 26)
        plt.xlim(-10.5, 1.5)
        plt.legend()
        plt.show()



def main():
    Assigment24()

if __name__ == "__main__": 
    main()
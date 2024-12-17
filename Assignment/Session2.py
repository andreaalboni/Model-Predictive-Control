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
         [1]]
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
    ts = 0.3 
    Q = np.array([[10, 0], [0, 1]])
    R = np.array([[0.01]])

    A, B = get_dynamics_discrete(ts)
    
    return A, B, Q, R

def Assignment23(initial_states, pmax: float, umin: float, Ts: float):
    bd = []
    velocities = initial_states[:, 1]
    v_min = np.min(velocities)
    v_max = np.max(velocities)
    velocities_mesh = np.linspace(v_min, v_max, 1000)

    for initial_vel in velocities_mesh:
        bd.append(pmax + initial_vel**2/umin - 0.5*initial_vel*(initial_vel + Ts*umin)/umin)

    return bd, velocities_mesh

def check_initial_feasibility(initial_state, horizon: int, pmin: float, pmax: float, umin: float, umax: float, vmin: float, vmax: float):
    A, B, Q, R = setup()

    # Define the optimization variables
    x = cp.Variable((A.shape[0], horizon + 1))
    u = cp.Variable((B.shape[1], horizon))

    # Make symbolic variables for the states and inputs 
    x = [cp.Variable((A.shape[0],), name=f"x_{i}") for i in range(horizon+1)]
    u = [cp.Variable((B.shape[1],), name=f"u_{i}") for i in range(horizon)]
    
    x_init = cp.Parameter((A.shape[0],), name="x_init")

    # Define the constraints
    xmax = np.array([pmax, vmax])
    xmin = np.array([pmin, vmin])
    umax = np.array([umax])
    umin = np.array([umin])

    # Sum of stage costs 
    cost = cp.sum([cp.quad_form(xt, Q) + cp.quad_form(ut, R) for (xt, ut) in zip(x,u)])
    
    constraints = [ uk <= umax for uk in u ] + \
                  [ uk >= umin for uk in u ] + \
                  [ xk <= xmax for xk in x ] + \
                  [ xk >= xmin for xk in x ] + \
                  [ x[0] == x_init] + \
                  [ xk1 == A@xk + B@uk for xk1, xk, uk in zip(x[1:], x, u)] 
    
    # Define the optimization problem
    problem = cp.Problem(cp.Minimize(cost), constraints)
    problem.param_dict["x_init"].value = initial_state

    # Solve the optimization problem
    try:
        problem.solve()
        feasible = problem.status == cp.OPTIMAL
    except cp.error.SolverError:
        feasible = False

    return feasible

def Assignment24():
    pmin = -150
    pmax = 1.0
    umin = -5.0
    umax = 10.0
    vmin = -20.0
    vmax = 25.0
    ts = 0.3
    initial_states_feasible = []

    # Generate grid of initial states
    p0_range = np.linspace(-10, 1, 30)
    v0_range = np.linspace(0, 25, 30)
    p0, v0 = np.meshgrid(p0_range, v0_range)
    initial_states = np.vstack([p0.ravel(), v0.ravel()]).T

    # Assignement 2.3
    break_distances, mesh = Assignment23(initial_states=initial_states, pmax=pmax, umin=umin, Ts=ts)
   
    # Check initial feasibility of each initial state given MPC of different horizons {2, 5, 10}
    for N in [2, 5, 10]:
        start_time = time.time()
        for x0 in initial_states:
            initial_feasibility = check_initial_feasibility(initial_state=x0, horizon=N, pmin=pmin, pmax=pmax, umin=umin, umax=umax, vmin=vmin, vmax=vmax)
            initial_states_feasible.append([x0, initial_feasibility])
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Initial feasibility analysis for MPC of horizon {N} took {elapsed_time:.2f} seconds")

        # Separate feasible and non-feasible points
        feasible_points = np.array([point[0] for point in initial_states_feasible if point[1]])
        non_feasible_points = np.array([point[0] for point in initial_states_feasible if not point[1]])

        # Plot initial states
        plt.figure()
        if feasible_points.size > 0:
            plt.scatter(feasible_points[:, 0], feasible_points[:, 1], color='green', marker='o', s=9, label='Feasible')
        if non_feasible_points.size > 0:
            plt.scatter(non_feasible_points[:, 0], non_feasible_points[:, 1], color='red', marker='o', s=9, label='Infeasible')
        
        # Plot all the break distances
        #plt.plot(break_distances, mesh, color='cyan', linewidth=2, label='Constraint\'s boundary')
        # Plot just the break distances that are feasible
        mask = np.array(break_distances) >= min(p0_range)
        filtered_break_distances = np.array(break_distances)[mask]
        filtered_velocity_mesh = mesh[mask]
        plt.plot(filtered_break_distances, filtered_velocity_mesh, color='cyan', linewidth=2, label="Constraint's boundary")

        plt.title("Initial feasibility analysis for MPC of horizon {}".format(N))
        plt.xlabel("Position ($p_0$)")
        plt.ylabel("Velocity ($v_0$)")
        #plt.grid(True)
        #plt.ylim(-1, 26)
        #plt.xlim(-10.5, 1.5)
        plt.legend()

        filename = f"Assignment_24_N{N}.png"
        plt.savefig(filename, dpi=700, format='png', bbox_inches='tight')

        plt.show()


def main():
    Assignment24()

if __name__ == "__main__": 
    main()
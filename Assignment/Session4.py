import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from typing import Callable, Tuple
import sys 
import casadi as cs 
import os
WORKING_DIR = os.path.split(__file__)[0]
sys.path.append(os.path.join(WORKING_DIR, os.pardir))
from given.parameters import VehicleParameters
from given.animation import AnimateParking
from given.plotting import *

from rcracers.simulator.dynamics import KinematicBicycle
from rcracers.simulator import simulate


def forward_euler(f, ts) -> Callable:
    def fw_eul(x,u):
        return x + f(x,u) * ts
    return fw_eul

def runge_kutta4(f, ts) -> Callable:
    def rk4_dyn(x,u):
        s1 = f(x, u)
        s2 = f(x + ts/2 * s1, u)
        s3 = f(x + ts/2 * s2, u)
        s4 = f(x + ts * s3, u)
        return x + ts/6 * (s1 + 2*s2 + 2*s3 + s4)
    return rk4_dyn

def exact_integration(f, ts) -> Callable:
    """Ground truth for the integration
    
    Integrate the given dynamics using scipy.integrate.odeint, which is very accurate in 
    comparison to the methods we implement in this settings, allowing it to serve as a 
    reference to compare against.

    Args:
        f (dynamics): The dynamics to integrate (x,u) -> xdot
        ts (_type_): Sampling time 

    Returns:
        Callable: Discrete-time dynamics (x, u) -> x+ 
    """
    from scipy.integrate import odeint  # Load scipy integrator as a ground truth
    def dt_dyn(x, u):
        f_wrap = lambda x, t: np.array(f(x, u)).reshape([x.size])
        y = odeint(f_wrap, x.reshape([x.size]), [0, ts])
        return y[-1].reshape((x.size,))
    return dt_dyn

def build_test_policy():
    # Define a policy to test the system
    acceleration = 1 # Fix a constant longitudinal acceleration 
    policy = lambda y, t: np.array([acceleration, 0.1 * np.sin(t)])
    return policy


class MPCController:

    def __init__(self, N: int, ts: float, *, params: VehicleParameters):
        """Constructor.

        Args:
            N (int): Prediction horizon
            ts (float): sampling time [s]
        """
        self.N = N
        self.ts = ts 
        nlp_dict, self.bounds = self.build_ocp(params)
        
        opts = {"ipopt": {"print_level": 1}, "print_time": False}
        self.ipopt_solver = cs.nlpsol("solver", "ipopt", nlp_dict, opts) 
        
    def solve(self, x) -> dict:
        return self.ipopt_solver(p=x, **self.bounds)
        
    def build_ocp(self, params: VehicleParameters) -> Tuple[dict, dict]:
        """
        Args: 
            VehicleParameters [params]: vehicle parameters
        Returns: 
            solver [dict]: the nonlinear program as a dictionary: 
                {"f": [cs.SX] cost (as a function of the decision variables, built as an expression, e.g., x + y, where x and y are CasADi SX.sym objects),
                "g": [cs.Expression] nonlinear constraint function as 
                an expression of the variables and the parameters. 
                These constraints represent the bounds on the state. 
                "x": [cs.SX] decision_vars (all control actions over the prediction horizon (concatenated into a long vector)), 
                "p": [cs.SX] parameters (initial state vector)} 
            bounds [dict]: the bounds on the constraints 
                {"lbx": [np.ndarray] Lower bounds on the decision variables, 
                "ubx": [np.ndarray] Upper bounds on the decision variables, 
                "lbg": [np.ndarray] Lower bounds on the nonlinear constraint g, 
                "ubg": [np.ndarray] Upper bounds on the nonlinear constraint g 
                }
        """
        # Create a parameter for the initial state. 
        x0 = cs.SX.sym("x0", (4,1))

        x = x0
        u = [cs.SX.sym(f"u_{t}", (2,1)) for t in range(self.N)]
        self.u = u
        
        # Add upper− and lower bounds to x and u
        lb_states = np.array([params.min_pos_x, params.min_pos_y, params.min_heading, params.min_vel])
        ub_states= np.array([params.max_pos_x, params.max_pos_y, params.max_heading, params.max_vel])
        
        lbu = np.array([params.min_drive,-params.max_steer])
        ubu = np.array([params.max_drive,params.max_steer])
        
        Q = cs.diagcat(1, 3, 0.1, 0.01)
        R = cs.diagcat(1, 0.01)
        Q_N = 5 * Q
        
        cost = 0
        lbx = []
        ubx = []
        g = []
        lbg = []
        ubg = []

        ode = KinematicBicycle(params, symbolic=True)
        f = runge_kutta4(ode, self.ts)

        # Compute circles belonging to the stationary car
        parkedcar_position = (0.25, 0.)
        parkedcar_yaw = 0.
        centers_parkedcar, radius_parkedcar = compute_circles(3, params.length, params.width, parkedcar_position, parkedcar_yaw)

        for k in range(self.N):
            cost += x.T @ Q @ x + u[k].T @ R @ u[k]
            x = f(x,u[k])
            lbx.append(lbu)
            ubx.append(ubu)
            
            # Added state to g and the state's bounds to lbg and ubg
            g.append(x)
            lbg.append(lb_states)
            ubg.append(ub_states)

            centers_movingcar, radius_movingcar = compute_circles(3, params.length, params.width, (x[0], x[1]), x[2])
            # Compute the distance between the center of the circles and the ones belonging to the stationary car
            for cmc in centers_movingcar:
                for cpc in centers_parkedcar:
                    # Squared distance between the centers
                    sq_dist = (cmc[0] - cpc[0])**2 + (cmc[1] - cpc[1])**2
                    
                    # Sqaured distance between the radii
                    sq_rad = (radius_movingcar + radius_parkedcar)**2

                    # Add the constraint to the list of constraints and the relative bounds
                    g.append(sq_rad - sq_dist)
                    lbg.append(-cs.inf)
                    ubg.append(0)
          
        cost += x.T @ Q_N @ x
        variables = cs.vertcat(*u)
        
        # Create the solver
        nlp = {"f": cost, "x": variables, "g" : cs.vertcat(*g), "p" :x0}
        
        bounds = {
            "lbx": cs.vertcat(*lbx),
            "ubx": cs.vertcat(*ubx),
            "lbg": cs.vertcat(*lbg),
            "ubg": cs.vertcat(*ubg)
            }
        
        return nlp, bounds

    def reshape_input(self, sol):
        return np.reshape(sol["x"], ((-1, 2)))

    def __call__(self, y):
        """Solve the OCP for initial state y.

        Args:
            y (np.ndarray): Measured state 
        """
        solution = self.solve(y)
        u = self.reshape_input(solution)
        return u[0]

def build_test_policy():
    # Define a policy to test the system
    acceleration = 1 # Fix a constant longitudinal acceleration 
    policy = lambda y, t: np.array([acceleration, 0.1 * np.sin(t)])
    return policy

def rel_error(val, ref):
    """Compute the relative errors between `val` and `ref`, taking the ∞-norm along axis 1. 
    """
    return np.linalg.norm(
        val - ref, axis=1, ord=np.inf,
    )/0.5*(1e-12 + np.linalg.norm(val, axis=1, ord=np.inf) + np.linalg.norm(ref, axis=1, ord=np.inf))



def compute_circles(circles_number, length, width, car_position, car_yaw):
    centers = []

    # Definition of the car's rotation matrix
    R = np.array([[np.cos(car_yaw), -np.sin(car_yaw)], [np.sin(car_yaw), np.cos(car_yaw)]])
    
    d = length / (2 * circles_number)
    radius = math.sqrt(d**2 + (width/2)**2)

    for i in range(circles_number):
        xc = - length/2 + (d + i*2*d)
        center = R @ np.array([xc, 0])
        centers.append((center[0]+car_position[0], center[1]+car_position[1]))

    return centers, radius

def plot(nc, length, width, assignment, car_position=(0.0, 0.0), car_yaw=0.0):
    R = Rectangle(
            (-length/2+car_position[0], -width/2+car_position[1]), 
            length, 
            width, 
            angle=car_yaw*180/math.pi,
            rotation_point=car_position,
            facecolor='none', 
            edgecolor='black', 
            linewidth=1,
            label="Car"
        )
    centers, radius = compute_circles(nc, length, width, car_position, car_yaw)

    # Create a figure and axes
    _, ax = plt.subplots()

    for center in centers:
        circle = Circle(center, radius, facecolor='none', edgecolor='cyan', linewidth=1, label="Circles")
        ax.add_patch(circle)

    # Add the rectangle to the axes
    ax.add_patch(R)

    # Set the title and labels
    ax.set_title(assignment)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    ax.set_xlabel("$p_x$ [m]")
    ax.set_ylabel("$p_y$ [m]")

    # Set the plot limits
    lim = np.max([length, width])
    ax.set_xlim(-lim + car_position[0], lim + car_position[0])
    ax.set_ylim(-lim + car_position[1], lim + car_position[1])

    # Set aspect ratio to be equal
    ax.set_aspect('equal')

    # Save the plot as an image
    #filename = f"{assignment}.png"
    #plt.savefig(filename, dpi=700, format='png', bbox_inches='tight')

    # Display the plot
    plt.show()



def Assignment41():
    num_circles = 3
    assignment = "Assignment 4.1"
    length = 4
    width = 2
    plot(num_circles, length, width, assignment)

def Assignment42():
    num_circles = 3
    assignment = "Assignment 4.2"
    length = 4
    width = 2
    car_position = (2.0, 2.0)
    car_yaw = math.pi / 4
    plot(num_circles, length, width, assignment, car_position, car_yaw)

def Assignment44():
    N = 30
    ts = 0.08
    x0 = np.array([0.3, -0.1, 0, 0])
    params = VehicleParameters()

    print("--Set up the MPC controller")
    controller = MPCController(N=N, ts=ts, params=params)

    print(f"--Solve the OCP for x0 = {x0}")
    solution = controller.solve(x0)
    controls = controller.reshape_input(solution)

    def open_loop_policy(t):
        return controls[t]

    # Build the assumed model 
    bicycle = KinematicBicycle(params)
    dynamics_assumed = runge_kutta4(bicycle, ts)
    #dynamics_assumed = forward_euler(bicycle, ts)

    print(f"--Simulate under the assumed model")
    x_open_loop_model = simulate(x0, dynamics_assumed, n_steps=N, policy=open_loop_policy)

    # With more accurate predictions: 
    print(f"--Simulate using more precise integration")
    dynamics_accurate = exact_integration(bicycle, ts)
    x_open_loop_exact = simulate(x0, dynamics_accurate, n_steps=N, policy=open_loop_policy)

    print(f"--Plotting the results")

    print(f"---Plot Controls")
    plot_input_sequence(controls, VehicleParameters())
    plt.show()
    print(f"---Plot trajectory under the predictions")
    plot_state_trajectory(x_open_loop_model, color="tab:blue", label="Predicted")
    print("---Plot the trajectory under the more accurate model")
    plot_state_trajectory(x_open_loop_exact, color="tab:red", label="Real")
    plt.title("Trajectory (integration error)")
    plt.show()

    print(f"---Plot trajectory under the predictions")
    plt.figure()
    plt.plot(rel_error(x_open_loop_model, x_open_loop_exact) * 100)
    plt.xlabel("Time step")
    plt.ylabel("$\| x - x_{pred} \| / \| x \| \\times 100$")
    plt.title("Relative prediction error (integration error) [%]")
    plt.show()

def main():
    Assignment41()
    Assignment42()    

    Assignment44()

if __name__ == "__main__":
    main()
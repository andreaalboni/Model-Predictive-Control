# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 15:35:32 2024

@author: carmi
"""

from typing import Callable, Tuple
import sys 
import casadi as cs 
import os
import string
import math
WORKING_DIR = os.path.split(__file__)[0]
sys.path.append(os.path.join(WORKING_DIR, os.pardir))
from given.parameters import VehicleParameters
from given.animation import AnimateParking
from given.plotting import *

from rcracers.simulator.dynamics import KinematicBicycle
from rcracers.simulator import simulate
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.patches import Rectangle, Circle

def forward_euler(f, ts) -> Callable:
    def fw_eul(x,u):
        return x + f(x,u) * ts
    return fw_eul

def runge_kutta4(f, ts) -> Callable:
    def rk4_dyn(x,u):
        #Begin TODO----------------------------------------------------------
        #raise NotImplementedError("Implement the RK4 integrator!")
        s1=f(x,u)
        s2=f(x+ts/2*s1,u)
        s3=f(x+ts/2*s2,u)
        s4=f(x+ts*s3,u)
        x_next = x+ts/6*(s1+2*s2+2*s3+s4)
        #End TODO -----------------------------------------------------------
        return x_next 
    
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

#-----------------------------------------------------------
# MPC CONTROLLER
#-----------------------------------------------------------


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
        TODO IMPLEMENT

        Build a nonlinear program that represents the parametric optimization problem described above, with the initial state x as a parameter. Use a single shooting formulation, i.e., do not define a new decision variable for the states, but rather write them as functions of the initial state and the control variables. Also return the lower bound and upper bound on the decision variables and constraint functions:

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
        
        #Begin TODO----------------------------------------------------------
        #raise NotImplementedError("Implement the build step of your optimal control problem!")
        N=self.N
        x = x0
        u=[cs.SX.sym(f"u_{t}", (2,1)) for t in range(N)]
        self.u=u
        # Add upper− and lower bounds to x and u.
        
        lb_states = np.array([params.min_pos_x,params.min_pos_y,params.min_heading,params.min_vel])
        ub_states= np.array([params.max_pos_x,params.max_pos_y,params.max_heading,params.max_vel])
        
        lbu = np.array([params.min_drive,-params.max_steer])
        ubu = np.array([params.max_drive,params.max_steer])
        
        # Define the cost
        
        Q=cs.diagcat(1,3,0.1,0.01)
        R=cs.diagcat(1,0.01)
        Q_N=5*Q
        cost=0
        lbx=[]
        ubx=[]
        g=[]
        lbg=[]
        ubg=[]
        
        ts=self.ts
        ode=KinematicBicycle(params,symbolic=True)
        f=runge_kutta4(ode, ts)
        
        centers_obstacle,r_obstacle=compute_circles(params.width,params.length,3,(0.25,0),0)
        for k in range(N):
          cost = cost + x.T @ Q @ x + u[k].T @ R @ u[k]
          x=f(x,u[k])
          lbx.append(lbu)
          ubx.append(ubu)
          g.append(x)
          lbg.append(lb_states)
          ubg.append(ub_states)
          centers_controlled,r_controlled=compute_circles(params.width,params.length,3,(x[0],x[1]),x[2])
          for center_controlled in centers_controlled:
              for center_obstacle in centers_obstacle:
                 distance=(center_controlled[0]-center_obstacle[0])**2+(center_controlled[1]-center_obstacle[1])**2
                 squared_rad=(r_controlled+r_obstacle)**2
                
                 g.append(squared_rad-distance)
                 lbg.append(-cs.inf)
                 ubg.append(0)
        
            
        cost+=x.T @ Q_N @ x
        variables=cs.vertcat(*u)
        
        # Create the solver
        nlp = {"f": cost,"x": variables,"g" : cs.vertcat(*g),"p" :x0}
        
        bounds ={
            "lbx": cs.vertcat(*lbx),
            "ubx": cs.vertcat(*ubx),
            "lbg": cs.vertcat(*lbg),
            "ubg": cs.vertcat(*ubg)
            }
        #End TODO -----------------------------------------------------------
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


#-----------------------------------------------------------
# UTILITIES
#-----------------------------------------------------------

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

def compute_circles(w,l,n_c,car_position=(0,0),heading_angle=0):
    rotation_matrix=np.array([[np.cos(heading_angle),-np.sin(heading_angle)],[np.sin(heading_angle),np.cos(heading_angle)]])
    r=math.sqrt((l**2/(4*n_c**2))+(w/2)**2)
    c=[]
    for i in range(n_c):
        x_c=-l/2+(l/(2*n_c)+i*2*l/(2*n_c))
        c.append(rotation_matrix@np.array([x_c,0])+np.array(car_position))
    return c,r
    
def assignment4_1(w,l,n_c,car_position=(0,0),heading_angle=0):
    c,r=compute_circles(w,l,n_c,car_position,heading_angle)
    fig, ax=plt.subplots()
    rectangle=Rectangle([-l/2+car_position[0],-w/2+car_position[1]],l,w,angle=heading_angle*180/np.pi,rotation_point=car_position,facecolor='none',edgecolor='red')
    for center in c:
        circle=Circle(center,r,facecolor='none',edgecolor='blue')
        ax.add_patch(circle)
    ax.add_patch(rectangle)
    limit=np.max([l,w])
    ax.set_xlim(-limit+car_position[0],limit+car_position[0])
    ax.set_ylim(-limit+car_position[1],limit+car_position[1])
    ax.set_aspect('equal')
    plt.show()
    return c,r
def assignment4_2():
    p=(2,2)
    psi=np.pi/4
    assignment4_1(w=2,l=4,n_c=3,car_position=p,heading_angle=psi)
    
def assignment4_4():
    
    N=30
    ts = 0.08
    x0 = np.array([0.3, -0.1, 0, 0])
    params=VehicleParameters()
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
    assignment4_1(w=2,l=4,n_c=3)
    assignment4_2()
    assignment4_4()
    
if __name__ == "__main__":
    main()
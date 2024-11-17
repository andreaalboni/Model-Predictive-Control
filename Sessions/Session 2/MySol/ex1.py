import cvxpy as cp
import numpy as np
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

# Set up the numberical values
K = np.array([[1., 2.]])

A, B, Q, R = setup() 
x0 = 10 * np.ones(2)

Abar = A + B@K
Qbar = Q + K.T@R@K

ns = 2
x = 10*np.ones(2)

P = cp.Variable((ns, ns), PSD=True)
cost = x.T@P@x # Set the cost. Remember, P is the variable here!

# Hint: semidefinite constraints are implemented using `<<' and `>>'.
# `<=' and `>=' are element−wise!
constraints = [ Abar.T@P@Abar - P + Qbar << 0 ] # Set the constraints
optimizer = cp.Problem(cp.Minimize(cost), constraints) # build optimizer
solution = optimizer.solve(verbose=True)

print(f"Optimal cost: {solution}")
print(f"Number of iterations: {optimizer.solver_stats.num_iters}")
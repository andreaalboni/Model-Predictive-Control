import numpy as np 
import cvxpy as cp
import numpy.linalg as la
import matplotlib.pyplot as plt
from typing import Tuple , Callable
from scipy.linalg import solve_discrete_are
from rcracers.utils.geometry import Polyhedron, plot_polytope
from rcracers.utils.geometry import Ellipsoid, plot_ellipsoid

 
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
    C = np.array([[1, -2./3]])
    Q = np.array([[10, 0], [0, 1]])
    R = np.array([[0.01]])

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



def Assignment32():
    A, B, Q, R = setup()

    Hx = np.vstack((np.eye(2), -np.eye(2)))
    hx = np.array([1, 25, 120, 50])
    X = Polyhedron.from_inequalities(Hx, hx)
    #plot_polytope(X, color='cyan', label="X")

    P, K = riccati_recursion(A, B, R, Q, Q, 5)
    Hu = np.vstack((K[0], -K[0]))
    hu = np.array([10, 20])

    H = np.vstack((Hx, Hu))
    h = np.hstack((hx, hu))
    Xk = Polyhedron.from_inequalities(H, h)
    plot_polytope(Xk, color='b', label="$X_k$")

    print(Xk.vertices())

    shape_matrix = Q
    center = np.array([0, 0])
    ellipsoid = Ellipsoid(shape_matrix, center)
    plot_ellipsoid(ellipsoid, color='violet', label="$\epsilon$")

    alpha = 2 # to be computed
    ellipsoid_alpha = ellipsoid.__rmul__(alpha)
    plot_ellipsoid(ellipsoid_alpha, color='violet', label="$\epsilon_{\alpha}$")

    plt.legend()
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.show()


def Assignment33():
    pass



def main():
    Assignment32()

if __name__ == "__main__": 
    main()
import numpy as np 
import numpy.linalg as la
import matplotlib.pyplot as plt
from typing import Tuple , Callable
from rcracers.utils.geometry import Polyhedron, plot_polytope
 

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
    ts = 0.5 
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

def invariant_set(A: np.ndarray, B: np.ndarray, K: np.ndarray, H: np.ndarray, h: np.ndarray):
    Omega=Polyhedron.from_inequalities(H, h)
    Hu = np.vstack((K[0], -K[0]))
    hu = np.array([10, 20])
    U = Polyhedron.from_inequalities(Hu, hu)

    print((H@A).shape)
    print((U.H@B).shape)

    H_pre = np.vstack((H@A, U.H@B))
    h_pre = np.vstack((Omega.h, U.h))
    Pre_omega = Polyhedron.from_inequalities(H_pre,h_pre)
    for i in range(100):
        Omega_new = Omega.intersect(Pre_omega)
        if Omega_new == Omega:
            break
        Omega = Omega_new
    return Omega


def ex2():
    A, B, Q, R = setup()

    Hx = np.vstack((np.eye(2), -np.eye(2)))
    h = np.array([1, 25, 120, 50])
    X = Polyhedron.from_inequalities(Hx, h)
    plot_polytope(X)

def ex3():
    A, B, Q, R = setup()
    Hx = np.vstack((np.eye(2), -np.eye(2)))
    h = np.array([1, 25, 120, 50, 10, 20])
    _, K = riccati_recursion(A, B, R, Q, Q, 5)

    Hu = np.vstack((K[0], -K[0]))
    H = np.vstack((Hx, Hu))
    X = Polyhedron.from_inequalities(H, h)
    plot_polytope(X, color='b')
    ex2()
    Xinv = invariant_set(A, B, K[0], H, h)
    plot_polytope(Xinv, color='g')
    plt.show()


def main():
    #ex2()
    ex3()

if __name__ == "__main__": 
    main()
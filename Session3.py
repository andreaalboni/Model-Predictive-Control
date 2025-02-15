import numpy as np 
import cvxpy as cp
from typing import Tuple
import numpy.linalg as la
import matplotlib.pyplot as plt
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

def alpha(H: np.ndarray, h: np.ndarray, shape_matrix: np.ndarray) -> float:
    alphas = []
    for i in range(H.shape[0]):
        h_i = H[i,:]
        alpha = h[i]**2/(h_i @ la.inv(shape_matrix) @ h_i.T)
        alphas.append(alpha)
    return min(alphas)

def max_vol_ellipsoid(A: np.ndarray, B: np.ndarray, K: np.ndarray, Hx: np.ndarray, hx: np.ndarray, Hu: np.ndarray, hu: np.ndarray):
    # Decision variable, positive definite matrix
    S = cp.Variable((A.shape[0], A.shape[1]), PSD=True)

    # Objective: Maximize the volume of the ellipsoid (proportional to log(det(P^-1)))
    objective = cp.Minimize(-cp.log_det(S))

    # Constraints
    constraints = []
    F = K @ S

    # Constraints developed in previuos assignments
    schur_matrix = cp.bmat([[S, (A @ S + B @ F).T], 
                            [(A @ S + B @ F), S]])
    constraints.append(schur_matrix >> 0)

    for i in range(Hx.shape[0]):
        h_i = Hx[i,:]
        constraints.append(h_i @ S @ h_i.T - hx[i]**2 <= 0)

    for i in range(Hu.shape[0]):
        h_i = Hu[i,:].reshape(-1, 1)
        first_raw = cp.hstack([cp.reshape(hu[i]**2, (1, 1)), h_i.T @ S])
        last_raws = cp.hstack([S.T @ h_i, S])
        constraints.append(cp.vstack([first_raw, last_raws]) >> 0)

    # Solve the problem
    problem = cp.Problem(objective, constraints)
    problem.solve(solver='SCS', verbose=False, max_iters=1000000)

    return S.value


def Assignment32():
    A, B, Q, R = setup()

    Hx = np.vstack((np.eye(2), -np.eye(2)))
    hx = np.array([1, 25, 120, 50])
    X = Polyhedron.from_inequalities(Hx, hx)
    #plot_polytope(X, color='cyan', label="X")

    P, K = riccati_recursion(A, B, R, Q, Q, 10)
    Hu = np.vstack((K[0], -K[0]))
    hu = np.array([10, 20])

    H = np.vstack((Hx, Hu))
    h = np.hstack((hx, hu))
    Xk = Polyhedron.from_inequalities(H, h)
    plot_polytope(Xk, color=(0.,0.,0.), label="$X_k$")
    
    shape_matrix = P[0]
    center = np.array([0, 0])
    alpha_ellipsoid = alpha(H, h, shape_matrix)
    ellipsoid = Ellipsoid(shape_matrix/alpha_ellipsoid, center)
    plot_ellipsoid(ellipsoid, color='yellow', label="$\epsilon$")

    plt.title("Ellipsoid invariant set")
    plt.legend()
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    #filename = f"images/Assignment_32.png"
    #plt.savefig(filename, dpi=700, format='png', bbox_inches='tight')
    plt.show()

def Assignment37():
    A, B, Q, R = setup()
    
    Hx = np.vstack((np.eye(2), -np.eye(2)))
    hx = np.array([1, 25, 120, 50])

    _, K = riccati_recursion(A, B, R, Q, Q, 10)
    Hu = np.vstack((K[0], -K[0]))
    hu = np.array([10, 20])

    H = np.vstack((Hx, Hu))
    h = np.hstack((hx, hu))
    Xk = Polyhedron.from_inequalities(H, h)
    plot_polytope(Xk, color='black', label="$X_k$")

    S = max_vol_ellipsoid(A, B, K[0], Hx, hx, Hu, hu)
    print("Optimal S:", S)
    ellipsoid = Ellipsoid(la.inv(S), np.array([0, 0]))
    plot_ellipsoid(ellipsoid, color='violet', label="$\epsilon$ max volume")

    P, K = riccati_recursion(A, B, R, Q, Q, 10)
    shape_matrix = P[0]
    center = np.array([0, 0])
    alpha_ellipsoid = alpha(H, h, shape_matrix)
    ellipsoid = Ellipsoid(shape_matrix/alpha_ellipsoid, center)
    plot_ellipsoid(ellipsoid, color='yellow', label="$\epsilon$")

    plt.legend()
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    #filename = f"images/Assignment_37.png"
    #plt.savefig(filename, dpi=700, format='png', bbox_inches='tight')
    plt.show()

def main():
    Assignment32()
    Assignment37()

if __name__ == "__main__": 
    main()
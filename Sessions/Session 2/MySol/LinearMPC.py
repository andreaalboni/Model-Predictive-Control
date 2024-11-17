import cvxpy as cp
import numpy as np
from given.problem import Problem

pb = Problem()
A = pb.A
B = pb.B
Q = pb.Q
R = pb.R
x0 = np.array([-100, 0])  # Initial state
N = pb.N    # Horizon length

# Define the constraints (assuming they are defined in the problem.py file)
p_max = pb.p_max
v_max = pb.v_max
u_min = pb.u_min
u_max = pb.u_max

# Define the optimization variables
x = cp.Variable((A.shape[0], N + 1))
u = cp.Variable((B.shape[1], N))

# Define the cost function
cost = 0
constraints = [x[:, 0] == x0]

for t in range(N):
    cost += cp.quad_form(x[:, t], Q) + cp.quad_form(u[:, t], R)
    constraints += [x[:, t + 1] == A @ x[:, t] + B @ u[:, t]]
    constraints += [x[0, t] <= p_max]
    constraints += [x[1, t] <= v_max]
    constraints += [u_min <= u[:, t], u[:, t] <= u_max]

# Terminal cost
cost += cp.quad_form(x[:, N], Q)

# Define the optimization problem
problem = cp.Problem(cp.Minimize(cost), constraints)

# Solve the optimization problem
solution = problem.solve()
print("Optimal cost:\n", solution)

# Extract the optimal control inputs and states
optimal_x = x.value
optimal_u = u.value

print("Optimal states:\n", optimal_x)
print("Optimal control inputs:\n", optimal_u)
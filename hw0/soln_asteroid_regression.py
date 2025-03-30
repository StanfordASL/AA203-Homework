"""
Solution code for the problem "Asteroid regression".

Autonomous Systems Lab (ASL), Stanford University
"""

import numpy as np
import matplotlib.pyplot as plt

# Load the data and form `(A, y)`
data = np.genfromtxt('data_asteroid_regression.csv', delimiter=',')
d, y = data.T
A = np.column_stack([d, d**2, d**3])

# Solve the l2-problem and print the solution
x_l2 = np.linalg.solve(A.T@A, A.T@y)
print('l2 solution:', x_l2)

# Solve the l1-problem via subgradient descent and print the solution
alpha = 1e-4
max_iters = 10000
x = np.zeros(A.shape[1])
x_best = np.copy(x)
f_best = np.sum(np.abs(A@x - y))
for _ in range(max_iters):
    g = A.T @ np.sign(A@x - y)
    x -= alpha*g
    f = np.sum(np.abs(A@x - y))
    if f < f_best:
        x_best = np.copy(x)
        f_best = np.copy(f)
x_l1 = x_best
print('l1 solution:', x_l1)

# Plot each fit
fig, ax = plt.subplots(1, 1, dpi=150)
ds = np.linspace(d.min(), d.max())
A = np.column_stack([ds, ds**2, ds**3])
ax.plot(ds, A@x_l2, '-', label=r'$\ell_2$ fit')
ax.plot(ds, A@x_l1, '-', label=r'$\ell_1$ fit')
ax.plot(d, y, 'x', label='data')
ax.set_xlabel(r'$d$')
ax.set_ylabel(r'$m$')
ax.legend()
fig.savefig('../figures/soln_asteroid_regression.png', bbox_inches='tight')
plt.show()
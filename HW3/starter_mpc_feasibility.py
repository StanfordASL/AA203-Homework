"""
Starter code for the problem "MPC feasibility".

Autonomous Systems Lab (ASL), Stanford University
"""

import numpy as np
import cvxpy as cvx
import matplotlib.pyplot as plt
from scipy.linalg import solve_discrete_are
from tqdm.auto import tqdm
from itertools import product


def do_mpc(x0: np.ndarray, A: np.ndarray, B: np.ndarray,
           P: np.ndarray, Q: np.ndarray, R: np.ndarray,
           N: int, rx: float, ru: float,
           rf: float):
    """Solve the MPC problem starting at state `x0`."""
    n, m = Q.shape[0], R.shape[0]
    x_cvx = cvx.Variable((N + 1, n))
    u_cvx = cvx.Variable((N, m))

    # ####################### PART (a): YOUR CODE BELOW #######################

    # INSTRUCTIONS: Construct and solve the MPC problem using CVXPY.

    # TODO: Replace the two lines below with your code.
    cost = 0.
    constraints = []

    # ############################# END PART (a) ##############################

    prob = cvx.Problem(cvx.Minimize(cost), constraints)
    prob.solve()
    x = x_cvx.value
    u = u_cvx.value
    status = prob.status

    return x, u, status


def compute_roa(A: np.ndarray, B: np.ndarray,
                P: np.ndarray, Q: np.ndarray, R: np.ndarray,
                N: int, rx: float, ru: float, rf: float,
                grid_dim: int = 21, max_steps: int = 20,
                tol: float = 1e-2):
    """Compute a region of attraction."""
    roa = np.zeros((grid_dim, grid_dim))
    xs = np.linspace(-rx, rx, grid_dim)
    for i, x1 in enumerate(xs):
        for j, x2 in enumerate(xs):
            x = np.array([x1, x2])
            # ################### PART (b): YOUR CODE BELOW ###################

            # INSTRUCTIONS: Simulate the closed-loop system for `max_steps`,
            #               stopping early only if the problem becomes
            #               infeasible or the state has converged close enough
            #               to the origin. If the state converges, flag the
            #               corresponding entry of `roa` with a value of `1`.

            # ######################### END PART (b) ##########################
    return roa


# Part (a): Simulate and plot trajectories of the closed-loop system
n, m = 2, 1
A = np.array([[1., 1.], [0., 1.]])
B = np.array([[0.], [1.]])
Q = np.eye(n)
R = 10.*np.eye(m)
P_dare = solve_discrete_are(A, B, Q, R)
N = 3
T = 20
rx = 5.
ru = 0.5
rf = np.inf

Ps = (np.eye(n), P_dare)
titles = (r'$P = I$', r'$P = P_\mathrm{DARE}$')
x0s = (np.array([-4.5, 2.]), np.array([-4.5, 3.]))

fig, ax = plt.subplots(2, len(Ps), dpi=150, figsize=(10, 8),
                       sharex='row', sharey='row')
for i, (P, title) in enumerate(zip(Ps, titles)):
    for x0 in x0s:
        x = np.copy(x0)
        x_mpc = np.zeros((T, N + 1, n))
        u_mpc = np.zeros((T, N, m))
        for t in range(T):
            x_mpc[t], u_mpc[t], status = do_mpc(x, A, B, P, Q, R, N,
                                                rx, ru, rf)
            if status == 'infeasible':
                x_mpc = x_mpc[:t]
                u_mpc = u_mpc[:t]
                break
            x = A@x + B@u_mpc[t, 0, :]
            ax[0, i].plot(x_mpc[t, :, 0], x_mpc[t, :, 1], '--*', color='k')
        ax[0, i].plot(x_mpc[:, 0, 0], x_mpc[:, 0, 1], '-o')
        ax[1, i].plot(u_mpc[:, 0], '-o')
    ax[0, i].set_title(title)
    ax[0, i].set_xlabel(r'$x_{k,1}$')
    ax[1, i].set_xlabel(r'$k$')
ax[0, 0].set_ylabel(r'$x_{k,2}$')
ax[1, 0].set_ylabel(r'$u_k$')
fig.savefig('mpc_feasibility_sim.png', bbox_inches='tight')
plt.show()


# Part (b): Compute and plot regions of attraction for different MPC parameters
print('Computing regions of attraction (this may take a while) ... ',
      flush=True)
Ns = (2, 6)
rfs = (0., np.inf)
fig, axes = plt.subplots(len(Ns), len(rfs), dpi=150, figsize=(10, 10),
                         sharex=True, sharey=True)
prog_bar = tqdm(product(Ns, rfs), total=len(Ns)*len(rfs))
for flat_idx, (N, rf) in enumerate(prog_bar):
    i, j = np.unravel_index(flat_idx, (len(Ns), len(rfs)))
    roa = compute_roa(A, B, P_dare, Q, R, N, rx, ru, rf, grid_dim=30)
    axes[i, j].imshow(roa.T, origin='lower', extent=[-rx, rx, -rx, rx],
                      interpolation='none')
    axes[i, j].set_title(r'$N = {},\ r_f = $'.format(N)
                         + (r'$\infty$' if rf == np.inf else str(rf)))
for ax in axes[-1, :]:
    ax.set_xlabel(r'$x_{0,1}$')
for ax in axes[:, 0]:
    ax.set_ylabel(r'$x_{0,2}$')
fig.savefig('mpc_feasibility_roa.png', bbox_inches='tight')
plt.show()

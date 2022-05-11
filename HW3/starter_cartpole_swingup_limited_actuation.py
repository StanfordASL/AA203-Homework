"""
Starter code for the problem "Cart-pole swing-up with limited actuation".

Autonomous Systems Lab (ASL), Stanford University
"""

import numpy as np
import cvxpy as cvx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from tqdm import tqdm
from functools import partial
from animations import animate_cartpole


@partial(jax.jit, static_argnums=(0,))
@partial(jax.vmap, in_axes=(None, 0, 0))
def linearize(fd: callable,
              s: jnp.ndarray,
              u: jnp.ndarray):
    """Linearize the function `fd(s,u)` around `(s,u)`."""
    # ####################### PART (b): YOUR CODE BELOW #######################

    # INSTRUCTIONS: Use JAX to linearize `fd` around `(s,u)`.

    # TODO: Replace the four lines below with your code.
    n, m = s.size, u.size
    A = jnp.zeros((n, n))
    B = jnp.zeros((n, m))
    c = jnp.zeros((n,))

    # ############################# END PART (b) ##############################

    return A, B, c


def solve_swingup_scp(fd: callable,
                      P: np.ndarray,
                      Q: np.ndarray,
                      R: np.ndarray,
                      N: int,
                      s_goal: np.ndarray,
                      s0: np.ndarray,
                      ru: float,
                      ρ: float,
                      tol: float,
                      max_iters: int):
    """Solve the cart-pole swing-up problem via SCP."""
    n = Q.shape[0]  # state dimension
    m = R.shape[0]  # control dimension

    # Initialize nominal trajectories
    u_bar = np.zeros((N, m))
    s_bar = np.zeros((N + 1, n))
    s_bar[0] = s0
    for k in range(N):
        s_bar[k+1] = fd(s_bar[k], u_bar[k])

    # Do SCP until convergence or maximum number of iterations is reached
    converged = False
    obj_prev = np.inf
    for i in (prog_bar := tqdm(range(max_iters))):
        s, u, obj = scp_iteration(fd, P, Q, R, N, s_bar, u_bar, s_goal, s0,
                                  ru, ρ)
        diff_obj = np.abs(obj - obj_prev)
        prog_bar.set_postfix({'objective change': '{:.5f}'.format(diff_obj)})

        if diff_obj < tol:
            converged = True
            print('SCP converged after {} iterations.'.format(i))
            break
        else:
            obj_prev = obj
            np.copyto(s_bar, s)
            np.copyto(u_bar, u)

    if not converged:
        raise RuntimeError('SCP did not converge!')

    return s, u


def scp_iteration(fd: callable, P: np.ndarray, Q: np.ndarray, R: np.ndarray,
                  N: int, s_bar: np.ndarray, u_bar: np.ndarray,
                  s_goal: np.ndarray, s0: np.ndarray,
                  ru: float, ρ: float):
    """Solve a single SCP sub-problem for the cart-pole swing-up problem."""
    A, B, c = linearize(fd, s_bar[:-1], u_bar)
    A, B, c = np.array(A), np.array(B), np.array(c)
    n = Q.shape[0]
    m = R.shape[0]
    s_cvx = cvx.Variable((N + 1, n))
    u_cvx = cvx.Variable((N, m))

    # ####################### PART (c): YOUR CODE BELOW #######################

    # INSTRUCTIONS: Construct and solve the convex sub-problem for SCP.

    # TODO: Replace the two lines below with your code.
    objective = 0.
    constraints = []

    # ############################# END PART (c) ##############################

    prob = cvx.Problem(cvx.Minimize(objective), constraints)
    prob.solve()

    if prob.status != 'optimal':
        raise RuntimeError('SCP solve failed. Problem status: ' + prob.status)

    s = s_cvx.value
    u = u_cvx.value
    obj = prob.objective.value

    return s, u, obj


def cartpole(s, u):
    """Compute the cart-pole state derivative."""
    mp = 1.     # pendulum mass
    mc = 4.     # cart mass
    ℓ = 1.      # pendulum length
    g = 9.81    # gravitational acceleration

    x, θ, dx, dθ = s
    sinθ, cosθ = jnp.sin(θ), jnp.cos(θ)
    h = mc + mp*(sinθ**2)
    ds = jnp.array([
        dx,
        dθ,
        (mp*sinθ*(ℓ*(dθ**2) + g*cosθ) + u[0]) / h,
        -((mc + mp)*g*sinθ + mp*ℓ*(dθ**2)*sinθ*cosθ + u[0]*cosθ) / (h*ℓ)
    ])
    return ds


def discretize(f, dt):
    """Discretize continuous-time dynamics `f` via Runge-Kutta integration."""

    def integrator(s, u, dt=dt):
        k1 = dt * f(s, u)
        k2 = dt * f(s + k1 / 2, u)
        k3 = dt * f(s + k2 / 2, u)
        k4 = dt * f(s + k3, u)
        return s + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return integrator


# Cartpole swing-up simulation parameters
n = 4                                # state dimension
m = 1                                # control dimension
s_goal = np.array([0, np.pi, 0, 0])  # desired upright pendulum state
s0 = np.array([0, 0, 0, 0])          # initial downright pendulum state
dt = 0.1                             # discrete time resolution
T = 10.                              # total simulation time

# Dynamics
fd = jax.jit(discretize(cartpole, dt))

# SCP parameters
P = 1e3*np.eye(n)                    # terminal state cost matrix
Q = np.diag([1e-2, 1., 1e-3, 1e-3])  # state cost matrix
R = 1e-3*np.eye(m)                   # control cost matrix
ρ = 1.                               # trust region parameter
ru = 8.                              # control effort bound
tol = 5e-1                           # convergence tolerance
max_iters = 100                      # maximum number of SCP iterations

# Solve the swing-up problem with SCP
t = np.arange(0., T + dt, dt)
N = t.size - 1
s, u = solve_swingup_scp(fd, P, Q, R, N, s_goal, s0, ru, ρ, tol, max_iters)

# Simulate open-loop control
for k in range(N):
    s[k+1] = fd(s[k], u[k])

# Plot state and control trajectories
fig, ax = plt.subplots(1, n + 1, figsize=(15, 3), dpi=150)
plt.subplots_adjust(wspace=0.55)
ylabels = (r'$x(t)$', r'$\theta(t)$', r'$\dot{x}(t)$', r'$\dot{\theta}(t)$',
           r'$u(t)$')
for i in range(n):
    ax[i].plot(t, s[:, i], color='tab:blue')
    ax[i].axhline(s_goal[i], linestyle='--', color='tab:orange')
    ax[i].set_xlabel(r'$t$')
    ax[i].set_ylabel(ylabels[i])
ax[n].plot(t[0:N], u)
ax[n].set_xlabel(r'$t$')
ax[n].set_ylabel(ylabels[n])
plt.savefig('cartpole_swingup_limited_actuation.png',
            bbox_inches='tight')
plt.show()

# Animate the solution
fig, ani = animate_cartpole(t, s[:, 0], s[:, 1])
ani.save('cartpole_scp_swingup.mp4', writer='ffmpeg')
plt.show()

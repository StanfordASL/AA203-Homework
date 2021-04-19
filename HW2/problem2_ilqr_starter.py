"""
Starter code for the problem "Cart-pole swing-up".
​
Author: Spencer M. Richards
        Autonomous Systems Lab (ASL), Stanford
        (GitHub: spenrich)
"""
​
import numpy as np
from scipy.integrate import odeint
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from animations import animate_cartpole
import time
​
​
@jax.partial(jax.jit, static_argnums=(0,))
def linearize(f, s, u):
    """Linearize the function `f(s,u)` around `(s,u)`."""
    # WRITE YOUR CODE BELOW ###################################################
    # A, B = ...
    ###########################################################################
    return A, B
​
​
def ilqr(f, s0, s_goal, N, Q, R, Qf):
    """Compute the iLQR set-point tracking solution.
​
    Arguments
    ---------
    f : Callable
        A function describing the discrete-time dynamics, such that
        `s[k+1] = f(s[k], u[k])`.
    s0 : numpy.ndarray
        The initial state (1-D).
    s_goal : numpy.ndarray
        The goal state (1-D).
    N : int
        The time horizon of the LQR cost function.
    Q : numpy.ndarray
        The state cost matrix (2-D).
    R : numpy.ndarray
        The control cost matrix (2-D).
    Qf : numpy.ndarray
        The terminal state cost matrix (2-D).
​
    Returns
    -------
    s_bar : numpy.ndarray
        A 3-D array where `s_bar[k]` is the nominal state at time step `k`,
        for `k = 0, 1, ..., N-1`
    u_bar : numpy.ndarray
        A 3-D array where `u_bar[k]` is the nominal control at time step `k`,
        for `k = 0, 1, ..., N-1`
    L : numpy.ndarray
        A 3-D array where `L[k]` is the matrix gain term of the iLQR control
        law at time step `k`, for `k = 0, 1, ..., N-1`
    l : numpy.ndarray
        A 3-D array where `l[k]` is the offset term of the iLQR control law
        at time step `k`, for `k = 0, 1, ..., N-1`
    """
    n = Q.shape[0]        # state dimension
    m = R.shape[0]        # control dimension
    eps = 0.001           # termination threshold for iLQR
    max_iters = int(1e3)  # maximum number of iLQR iterations
​
    # Initialize control law terms `L` and `l`
    L = np.zeros((N, m, n))
    l = np.zeros((N, m))
​
    # Initialize `u`, `u_bar`, `s`, and `s_bar` with a forward pass
    u_bar = np.zeros((N, m))
    s_bar = np.zeros((N + 1, n))
    s_bar[0] = s0
    for k in range(N):
        s_bar[k+1] = f(s_bar[k], u_bar[k])
    u = np.copy(u_bar)
    s = np.copy(s_bar)
​
    # iLQR loop
    converged = False
    for _ in range(max_iters):
        # Linearize the dynamics at each step `k` of `(s_bar, u_bar)`
        A, B = jax.vmap(linearize, in_axes=(None, 0, 0))(f, s_bar[:-1], u_bar)
        A, B = np.array(A), np.array(B)
​
        # WRITE YOUR CODE BELOW ###############################################
        # Update the arrays `L`, `l`, `s`, and `u`.
        #######################################################################
        if np.max(np.abs(u - u_bar)) < eps:
            converged = True
            break
        else:
            u_bar[:] = u
            s_bar[:] = s
    if not converged:
        raise RuntimeError('iLQR did not converge!')
    return s_bar, u_bar, L, l
​
​
def cartpole(s, u):
    """Compute the cart-pole state derivative."""
    mp = 2.     # pendulum mass
    mc = 10.    # cart mass
    ℓ = 1.      # pendulum length
    g = 9.81    # gravitational acceleration
​
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
​
​
if __name__ == "__main__":
    n = 4                                      # state dimension
    m = 1                                      # control dimension
    Q = np.diag(np.array([10., 10., 2., 2.]))  # state cost matrix
    R = 1e-2*np.eye(m)                         # control cost matrix
    Qf = 1e2*np.eye(n)                         # terminal state cost matrix
    s0 = np.array([0., 0., 0., 0.])            # initial state
    s_goal = np.array([0., np.pi, 0., 0.])     # goal state
    T = 10.                                    # simulation time
    dt = 0.1                                   # sampling time
​
    # Initialize continuous-time and discretized dynamics
    f = jax.jit(cartpole)
    f_discrete = jax.jit(lambda s, u, dt=dt: s + dt*f(s, u))
​
    # Compute the iLQR solution with the discretized dynamics
    print('Computing iLQR solution ... ', end='', flush=True)
    start = time.time()
    t = np.arange(0., T, dt)
    N = t.size - 1
    s_bar, u_bar, L, l = ilqr(f_discrete, s0, s_goal, N, Q, R, Qf)
    print('done! ({:.2f} s)'.format(time.time() - start), flush=True)
​
    # Simulate on the true continuous-time system
    print('Simulating ...', end='', flush=True)
    simulate_continuous_time_dynamics = False  # change to `True` in part (d)
    start = time.time()
    s = np.zeros((N + 1, n))
    u = np.zeros((N, m))
    s[0] = s0
    for k in range(N):
        if simulate_continuous_time_dynamics:
            # WRITE YOUR CODE BELOW ###########################################
            # Update `u[k]` using the final LQR policy `L`, `l` output by
            # `ilqr` above to track the planned trajectory when we simulate the
            # continuous-time dynamics.
            u[k] = ...
            ###################################################################
            s[k+1] = odeint(lambda s, t: f(s, u[k]), s[k], t[k:k+2])[1]
        else:
            u[k] = u_bar[k]
            s[k+1] = f_discrete(s[k], u[k])
    print('done! ({:.2f} s)'.format(time.time() - start), flush=True)
​
    # Plot
    fig, axes = plt.subplots(1, n, dpi=100, figsize=(12, 2))
    plt.subplots_adjust(wspace=0.35)
    ylabels = (r'$x(t)$', r'$\theta(t)$',
               r'$\dot{x}(t)$', r'$\dot{\theta}(t)$')
    for i in range(n):
        axes[i].plot(t, s[:, i])
        axes[i].set_xlabel(r'$t$')
        axes[i].set_ylabel(ylabels[i])
    plt.savefig('cartpole_ilqr_swingup.pdf', bbox_inches='tight')
​
    fig, ani = animate_cartpole(t, s[:, 0], s[:, 1])
    ani.save('cartpole_ilqr_swingup.mp4', writer='ffmpeg')
    plt.show()

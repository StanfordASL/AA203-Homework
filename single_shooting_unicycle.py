"""
Starter code for the problem "Single shooting for a unicycle".

Autonomous Systems Lab (ASL), Stanford University
"""

import jax
import jax.numpy as jnp
from jax.experimental.ode import odeint

import matplotlib.pyplot as plt

# NOTE: It can be good to use both `jax.numpy` and regular `numpy`, as `numpy`
# operations are typically much faster than their `jax.numpy` counterparts (at
# least before `jax.jit`) whenever JAX-specific magic (e.g., automatic
# differentiation, vectorization) is not required. However, since this may be
# your first experience with JAX, for this problem we encourage you to use
# `jax.numpy` for everything unless you know what you are doing.


def dynamics(x: jnp.array, u: jnp.array) -> jnp.array:
    """Evaluate the continuous-time dynamics of a unicycle.

    Args:
        x:  An array of shape (3,) containing the unicycle pose `(x, y, θ)`.
        u:  An array of shape (2,) containing the velocity controls `(v, ω)`.

    Returns:
        An array of shape (3,) containing the time derivative of the state
        `dx/dt = (dx/dt, dy/dt, dθ/dt)`.
    """
    raise NotImplementedError()


def hamiltonian(x: jnp.array, p: jnp.array, u: jnp.array) -> float:
    """Evaluate the Hamiltonian.

    Args:
        x:  An array of shape (3,) containing the unicycle pose `(x, y, θ)`.
        p:  An array of shape (3,) containing the co-state `(px, py, pθ)`.
        u:  An array of shape (2,) containing the velocity controls `(v, ω)`.

    Returns:
        The value of the Hamiltonian `H`.
    """
    raise NotImplementedError()


def optimal_control(x: jnp.array, p: jnp.array) -> jnp.array:
    """Compute an optimal control as a function of the state and co-state.

    Args:
        x:  An array of shape (3,) containing the unicycle pose `(x, y, θ)`.
        p:  An array of shape (3,) containing the co-state `(px, py, pθ)`.

    Returns:
        An array of shape (2,) containing an optimal control `u = (v, ω)`.
    """
    raise NotImplementedError()


def pmp_ode(x_and_p: tuple[jnp.array, jnp.array],
            t: float) -> tuple[jnp.array, jnp.array]:
    """Evaluate the ODE that an optimal state and co-state must obey.

    Args:
        x_and_p:    A tuple of arrays `(x, p)`, where `x` is an array of shape
                    (3,) containing the unicycle pose `(x, y, θ)`, and `p` is
                    an array of shape (3,) containing the co-state
                    `(px, py, pθ)`.
        t:          The current time (required for use with `odeint`, but can
                    be ignored here).

    Returns:
        A tuple of arrays `(dx, dp)` containing the time derivatives of the
        state and co-state, respectively.
    """
    raise NotImplementedError()


def pmp_trajectories(x0: jnp.array,
                     p0: jnp.array,
                     T: float,
                     N: int = 20) -> tuple[jnp.array, jnp.array, jnp.array,
                                           jnp.array]:
    """Integrate the optimal state and co-state ODE forward in time.

    Args:
        x0: An array of shape (3,) containing the initial state
            `(x(0), y(0), θ(0))`.
        p0: An array of shape (3,) containing the initial co-state
            `(px(0), py(0), pθ(0))`.
        T:  The final time `T`.
        N:  The number of nodes along the ODE solution at which to report the
            solution values.

    Returns:
        A tuple of arrays `(ts, xs, us, ps)` where:
            ts: An array of shape (N,) containing a sequence of times
                spanning `[0, T]`.
            xs: An array of shape (N, 3) containing the states at `ts`.
            us: An array of shape (N, 2) containing the optimal control
                inputs at `ts`.
            ps: An array of shape (N, 3) containing the co-states at `ts`.
    """
    raise NotImplementedError()


def boundary_residual(p0: jnp.array,
                      T: float,
                      x0: jnp.array,
                      xT: jnp.array) -> jnp.array:
    """Compute the residual error of the boundary conditions for the PMP.

    Args:
        p0: An array of shape (3,) containing the initial co-state
            `(px(0), py(0), pθ(0))` estimate.
        T:  The final time `T` estimate.
        x0: An array of shape (3,) containing the initial state
            `(x(0), y(0), θ(0))`.
        xT: An array of shape (3,) containing the final state
            `(x(T), y(T), θ(T))`.

    Returns:
        The array of shape (4,) we want to drive to zero through appropriate
        selection of `p0` and `T`.
    """
    # Hint: Use `pmp_trajectories` here.
    raise NotImplementedError()


# Uncomment `@jax.jit` for a speedier runtime per iteration (post-compilation),
# but a harder time debugging.
# @jax.jit
def newton_step(p0: jnp.array,
                T: float,
                x0: jnp.array,
                xT: jnp.array) -> tuple[jnp.array, float]:
    """Implement a step of the Newton-Raphson method for `boundary_residual`.

    Args:
        p0: An array of shape (3,) containing the current initial co-state
            `(px(0), py(0), pθ(0))` estimate.
        T:  The current final time `T` estimate.
        x0: An array of shape (3,) containing the initial state
            `(x(0), y(0), θ(0))`.
        xT: An array of shape (3,) containing the final state
            `(x(T), y(T), θ(T))`.

    Returns:
        A tuple containing the next estimate of `p0` and `T` computed by the
        Newton-Raphson method.
    """
    # Hint: Use `jax.jacobian` and `jnp.linalg.solve`.
    raise NotImplementedError()


def single_shooting(p0: jnp.array,
                    T: float,
                    x0: jnp.array,
                    xT: jnp.array,
                    max_iters: int = 10,
                    tol: float = 1e-4) -> tuple[jnp.array, float]:
    """Implement single shooting for the unicycle.

    Args:
        p0:         An array of shape (3,) containing an initial guess for the
                    initial co-state `(px(0), py(0), pθ(0))`.
        T:          An initial guess for the final time `T`.
        x0:         An array of shape (3,) containing the initial state
                    `(x(0), y(0), θ(0))`.
        xT:         An array of shape (3,) containing the desired final state
                    `(x(T), y(T), θ(T))`.
        max_iters:  The maximum number of Newton-Raphson steps to take.
        tol:        The convergence tolerance.

    Returns:
        A tuple containing the optimized initial co-state `p0` and final
        time `T`.
    """
    raise NotImplementedError()


p0_init = NotImplementedError()
T_init = NotImplementedError()

x0 = jnp.array([0, 0, jnp.pi / 2])
xT = jnp.array([5, 5, jnp.pi / 2])
p0, T = single_shooting(p0_init, T_init, x0, xT)
ts, xs, us, ps = pmp_trajectories(x0, p0, T)

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(xs[:, 0], xs[:, 1], 'k-', linewidth=2)
ax[0].quiver(xs[:, 0], xs[:, 1], jnp.cos(xs[:, 2]), jnp.sin(xs[:, 2]))
ax[0].plot(x0[0], x0[1], 'go', markerfacecolor='green', markersize=15)
ax[0].plot(xT[0], xT[1], 'ro', markerfacecolor='red', markersize=15)
ax[0].grid(True)
ax[0].axis([-1, 6, -1, 6])
ax[0].set_xlabel(r'$x$ [m]')
ax[0].set_ylabel(r'$y$ [m]')
ax[0].set_title('Optimal trajectory')

ax[1].plot(ts, us[:, 0], linewidth=2)
ax[1].plot(ts, us[:, 1], linewidth=2)
ax[1].grid(True)
ax[1].set_xlabel(r'$t$ [s]')
ax[1].legend([r'$v^*$ [m/s]', r'$\omega^*$ [rad/s]'])
ax[1].set_title('Optimal control sequence')

plt.tight_layout()
plt.savefig('single_shooting_unicycle.png')
plt.show()

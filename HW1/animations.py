"""
Animations for various dynamical systems using `matplotlib`.

Author: Spencer M. Richards
        Autonomous Systems Lab (ASL), Stanford
        (GitHub: spenrich)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation


def animate_cartpole(t, x, θ):
    """Animate the cart-pole system from given position data.

    The arguments `t`, `x`, and `θ` are assumed to be 1-D Numpy arrays
    describing the degrees of freedom (i.e., `x` and `θ`) of the cart-pole over
    time (i.e., `t`).

    Example usage:
        import matplotlib.pyplot as plt
        from animations import animate_cartpole
        fig, ani = animate_cartpole(t, x, θ)
        ani.save('cartpole_balance.mp4', writer='ffmpeg')
        plt.show()
    """
    # Geometry
    cart_width = 2.
    cart_height = 1.
    wheel_radius = 0.3
    wheel_sep = 1.
    pole_length = 5.
    mass_radius = 0.25

    # Figure and axis
    fig, ax = plt.subplots(dpi=100)
    x_lim = 1.1*np.max(np.abs(x))
    y_lim = 1.1*(wheel_radius + cart_height + pole_length)
    ax.set_xlim([-1.1*x_lim, 1.1*x_lim])
    ax.set_ylim([0., y_lim])
    ax.set_yticks([])
    ax.set_aspect(1.)

    # Artists
    cart = mpatches.FancyBboxPatch((0., 0.), cart_width, cart_height,
                                   facecolor='tab:blue', edgecolor='k',
                                   boxstyle='Round,pad=0.,rounding_size=0.05')
    wheel_left = mpatches.Circle((0., 0.), wheel_radius, color='k')
    wheel_right = mpatches.Circle((0., 0.), wheel_radius, color='k')
    mass = mpatches.Circle((0., 0.), mass_radius, color='k')
    pole = ax.plot([], [], '-', linewidth=3, color='k')[0]
    trace = ax.plot([], [], '--', linewidth=2, color='tab:orange')[0]
    timestamp = ax.text(-x_lim, 0.8*y_lim, '')

    ax.add_patch(cart)
    ax.add_patch(wheel_left)
    ax.add_patch(wheel_right)
    ax.add_patch(mass)

    def animate(k, t, x, θ):
        # Geometry
        cart_corner = np.array([x[k] - cart_width/2, wheel_radius])
        wheel_left_center = np.array([x[k] - wheel_sep/2, wheel_radius])
        wheel_right_center = np.array([x[k] + wheel_sep/2, wheel_radius])
        pole_start = np.array([x[k], wheel_radius + cart_height])
        pole_end = pole_start + pole_length*np.array([np.sin(θ[k]),
                                                      -np.cos(θ[k])])

        # Cart
        cart.set_x(cart_corner[0])
        cart.set_y(cart_corner[1])

        # Wheels
        wheel_left.set_center(wheel_left_center)
        wheel_right.set_center(wheel_right_center)

        # Pendulum
        pole.set_data([pole_start[0], pole_end[0]],
                      [pole_start[1], pole_end[1]])
        mass.set_center(pole_end)
        mass_x = x[:k+1] + pole_length*np.sin(θ[:k+1])
        mass_y = wheel_radius + cart_height - pole_length*np.cos(θ[:k+1])
        trace.set_data(mass_x, mass_y)

        # Time-stamp
        timestamp.set_text('t = {:.1f} s'.format(t[k]))

        artists = (cart, wheel_left, wheel_right, pole, mass, trace, timestamp)
        return artists

    dt = t[1] - t[0]
    ani = animation.FuncAnimation(fig, animate, t.size, fargs=(t, x, θ),
                                  interval=dt*1000, blit=True)
    return fig, ani

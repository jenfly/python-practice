"""
Matplotlib Animation Example

author: Jake Vanderplas
email: vanderplas@astro.washington.edu
website: http://jakevdp.github.com
license: BSD
Please feel free to use and modify this, but keep the above information. Thanks!

-----------------------------------------------------------------------
JMW Note:
Make sure an incoder is installed, e.g.
sudo apt-get install mencoder

Use writer='mencoder' when saving, e.g.
anim.save('movie.mp4', writer='mencoder')
"""

from __future__ import division
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from mpl_toolkits.basemap import Basemap
import xray
from datetime import datetime, timedelta

# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure()
ax = plt.axes(xlim=(0, 2), ylim=(-2, 2))
line, = ax.plot([], [], lw=2)

# initialization function: plot the background of each frame
def init():
    line.set_data([], [])
    return line,

# animation function.  This is called sequentially
def animate(i):
    x = np.linspace(0, 2, 1000)
    y = np.sin(2 * np.pi * (x - 0.01 * i))
    line.set_data(x, y)
    return line,

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=200, interval=20, blit=True)

# save the animation as an mp4.  This requires ffmpeg or mencoder to be
# installed.  The extra_args ensure that the x264 codec is used, so that
# the video can be embedded in html5.  You may need to adjust this for
# your system: for more information, see
# http://matplotlib.sourceforge.net/api/animation_api.html
#anim.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
anim.save('basic_animation.mp4', writer='mencoder', fps=30)
plt.show()

# -----------------------------------------------------------------------------

# Create a grid of points
points = np.arange(-5, 5, 0.01)
xs, ys = np.meshgrid(points, points)

# Data time points
times = np.arange(-5, 5, 0.1)
nt = len(times)
data = np.zeros([nt] + list(xs.shape), dtype=float)
for t, val in enumerate(times):
    data[t] = np.sqrt((xs-val)**2 + ys**2)


fig = plt.figure()

# animation function
def animate(t):
    pc = plt.pcolormesh(xs, ys, data[t])
    plt.title('t = %.1f' % times[t])
    return pc

anim = animation.FuncAnimation(fig, animate, frames=200)

import math
import random

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
from mpl_toolkits.mplot3d.art3d import Line3DCollection


A = np.array([
    [4.0, 4.0],
    [1.0, 1.0],
    [8.0, 8.0],
    [6.0, 6.0],
    [3.0, 7.0]
], dtype=float)

c = np.array([0.1, 0.2, 0.2, 0.4, 0.4], dtype=float)


def shekel5_value(x, y):
    s = np.zeros_like(x, dtype=float) if isinstance(x, np.ndarray) else 0.0
    for i in range(5):
        s += 1.0 / ((x - A[i, 0])**2 + (y - A[i, 1])**2 + c[i])
    return s


def objective(x, y):
    return -shekel5_value(x, y)


def neighbour(x, y, step=0.35, lower=0.0, upper=10.0):
    xn = x + random.uniform(-step, step)
    yn = y + random.uniform(-step, step)
    xn = max(lower, min(upper, xn))
    yn = max(lower, min(upper, yn))
    return xn, yn


def simulated_annealing():
    t = 3.0
    t_min = 0.001
    alpha = 0.985
    step = 0.35

    x = random.uniform(0.0, 10.0)
    y = random.uniform(0.0, 10.0)

    best_x, best_y = x, y
    best_f = objective(x, y)

    path_x = [x]
    path_y = [y]
    path_z = [shekel5_value(x, y)]
    path_t = [t]

    best_path_x = [best_x]
    best_path_y = [best_y]
    best_path_z = [shekel5_value(best_x, best_y)]

    while t > t_min:
        xn, yn = neighbour(x, y, step=step)
        delta = objective(xn, yn) - objective(x, y)

        if delta < 0 or random.random() < math.exp(-delta / t):
            x, y = xn, yn

        current_f = objective(x, y)
        if current_f < best_f:
            best_x, best_y = x, y
            best_f = current_f

        path_x.append(x)
        path_y.append(y)
        path_z.append(shekel5_value(x, y))
        path_t.append(t)

        best_path_x.append(best_x)
        best_path_y.append(best_y)
        best_path_z.append(shekel5_value(best_x, best_y))

        t *= alpha

    return {
        "path_x": np.array(path_x),
        "path_y": np.array(path_y),
        "path_z": np.array(path_z),
        "path_t": np.array(path_t),
        "best_x": np.array(best_path_x),
        "best_y": np.array(best_path_y),
        "best_z": np.array(best_path_z),
    }


def set_scatter_point(scatter_obj, xval, yval, zval):
    scatter_obj._offsets3d = ([xval], [yval], [zval])


def clear_scatter(scatter_obj):
    scatter_obj._offsets3d = ([], [], [])


def make_segments_3d(x, y, z):
    points = np.array([x, y, z]).T.reshape(-1, 1, 3)
    return np.concatenate([points[:-1], points[1:]], axis=1)


random.seed(42)
data = simulated_annealing()

path_x = data["path_x"]
path_y = data["path_y"]
path_z = data["path_z"]
path_t = data["path_t"]
best_x = data["best_x"]
best_y = data["best_y"]
best_z = data["best_z"]

x = np.linspace(0.0, 10.0, 220)
y = np.linspace(0.0, 10.0, 220)
X, Y = np.meshgrid(x, y)
Z = shekel5_value(X, Y)

z_offset = 0.9

fig = plt.figure(figsize=(11, 8))
plt.subplots_adjust(bottom=0.18, top=0.82)

ax = fig.add_subplot(111, projection="3d")

surf = ax.plot_surface(
    X, Y, Z,
    cmap="jet",
    linewidth=0,
    antialiased=True,
    alpha=0.55
)

surface_cbar = fig.colorbar(surf, ax=ax, shrink=0.68, pad=0.08)
surface_cbar.set_label("f(x)")

traj_collection = Line3DCollection(
    np.empty((0, 2, 3)),
    linewidths=4,
    cmap="plasma"
)

traj_added = [False]

current_outer = ax.scatter([], [], [], s=260, c="black", depthshade=False)
current_inner = ax.scatter([], [], [], s=120, c="white", depthshade=False)

best_outer = ax.scatter([], [], [], s=300, c="black", depthshade=False)
best_inner = ax.scatter([], [], [], s=150, c="white", depthshade=False)

norm = plt.Normalize(vmin=np.min(path_t), vmax=np.max(path_t))
mappable = plt.cm.ScalarMappable(norm=norm, cmap="plasma")
mappable.set_array(path_t)
temp_cbar = fig.colorbar(mappable, ax=ax, shrink=0.68, pad=0.02)
temp_cbar.set_label("Temperature")

ax.set_title("Simulated Annealing for Shekel-5", pad=18)
ax.set_xlabel("x1", fontsize=16, labelpad=12)
ax.set_ylabel("x2", fontsize=16, labelpad=12)
ax.set_zlabel("f(x)", fontsize=16, labelpad=10)

ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_zlim(0, max(np.max(Z) + z_offset + 0.5, 10))
ax.view_init(elev=36, azim=-60)

info_text = fig.text(
    0.5,
    0.92,
    "",
    ha="center",
    fontsize=11,
    bbox=dict(facecolor="white", edgecolor="black"),
)

frame = [0]
running = [False]


def update(_):
    if running[0] and frame[0] < len(path_x):
        i = frame[0]

        x_curr = path_x[i]
        y_curr = path_y[i]
        z_curr = path_z[i] + z_offset

        x_best_curr = best_x[i]
        y_best_curr = best_y[i]
        z_best_curr = best_z[i] + z_offset

        if i >= 1:
            z_path = path_z[:i + 1] + z_offset
            segments = make_segments_3d(path_x[:i + 1], path_y[:i + 1], z_path)
            seg_temps = np.asarray(path_t[:i], dtype=float)

            traj_collection.set_segments(segments)
            traj_collection.set_array(seg_temps)
            traj_collection.set_norm(norm)

            if not traj_added[0]:
                ax.add_collection3d(traj_collection)
                traj_added[0] = True

        set_scatter_point(current_outer, x_curr, y_curr, z_curr)
        set_scatter_point(current_inner, x_curr, y_curr, z_curr)

        set_scatter_point(best_outer, x_best_curr, y_best_curr, z_best_curr)
        set_scatter_point(best_inner, x_best_curr, y_best_curr, z_best_curr)

        info_text.set_text(
            f"Iteracja: {i}    T = {path_t[i]:.4f}\n"
            f"Aktualne: x1={path_x[i]:.3f}  x2={path_y[i]:.3f}  f={path_z[i]:.3f}\n"
            f"Najlepsze: x1*={best_x[i]:.3f}  x2*={best_y[i]:.3f}  f*={best_z[i]:.3f}"
        )

        frame[0] += 1

    return traj_collection, current_outer, current_inner, best_outer, best_inner, info_text


def start(event):
    running[0] = True
    fig.canvas.draw_idle()


def stop(event):
    running[0] = False
    fig.canvas.draw_idle()


def reset(event):

    global data
    global path_x, path_y, path_z, path_t
    global best_x, best_y, best_z

    running[0] = False
    frame[0] = 0

    # nowy losowy przebieg algorytmu
    data = simulated_annealing()

    path_x = data["path_x"]
    path_y = data["path_y"]
    path_z = data["path_z"]
    path_t = data["path_t"]

    best_x = data["best_x"]
    best_y = data["best_y"]
    best_z = data["best_z"]

    # wyczyszczenie trajektorii
    traj_collection.set_segments(np.empty((0, 2, 3)))
    traj_collection.set_array(np.array([]))

    clear_scatter(current_outer)
    clear_scatter(current_inner)
    clear_scatter(best_outer)
    clear_scatter(best_inner)

    info_text.set_text("New random starting point")

    fig.canvas.draw_idle()


ax_start = plt.axes([0.14, 0.05, 0.18, 0.07])
ax_stop = plt.axes([0.41, 0.05, 0.18, 0.07])
ax_reset = plt.axes([0.68, 0.05, 0.18, 0.07])

btn_start = Button(ax_start, "Start")
btn_stop = Button(ax_stop, "Stop")
btn_reset = Button(ax_reset, "Reset")

btn_start.on_clicked(start)
btn_stop.on_clicked(stop)
btn_reset.on_clicked(reset)

anim = FuncAnimation(
    fig,
    update,
    interval=80,
    blit=False,
    repeat=False,
    cache_frame_data=False,
)

plt.show()
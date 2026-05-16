import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation as R

class OrientationViewer:
    def __init__(self,Q, yaw, pitch, roll, tof_fvz, gyro_fvz, tof_total_duration):
        self.Q = Q
        self.yaw = yaw
        self.pitch = pitch
        self.roll = roll
        self.tof_fvz = tof_fvz
        self.gyro_fvz = gyro_fvz
        self.tof_total_duration = tof_total_duration
        self.N = len(yaw)
        self.gyro_duration= self.N / self.gyro_fvz

        self.path_x, self.path_y = self._dead_reckon()

        self.fig = plt.figure("Orientation Viewer", figsize=(6, 11))
        self.fig.patch.set_facecolor("#1e1e2e")

        gs = gridspec.GridSpec(
            3, 1, height_ratios=[1, 1.3, 1],
            hspace=0.5, left=0.13, right=0.95,
            top=0.95, bottom=0.06
        )

        self.ax_angles = self.fig.add_subplot(gs[0])
        self.ax_3d  = self.fig.add_subplot(gs[1], projection='3d')
        self.ax_path = self.fig.add_subplot(gs[2])

        self._style_ax(self.ax_angles)
        self._style_ax(self.ax_path)
        self._style_ax_3d(self.ax_3d)

        self._init_angle_traces()
        self._init_3d()
        self._init_path()

        self.fig.canvas.draw()

    def _style_ax(self, ax):
        ax.set_facecolor("#13131f")
        ax.tick_params(colors="#aaaacc", labelsize=7)
        ax.xaxis.label.set_color("#aaaacc")
        ax.yaxis.label.set_color("#aaaacc")
        ax.title.set_color("#e0e0ff")
        for spine in ax.spines.values():
            spine.set_edgecolor("#33334a")

    def _style_ax_3d(self, ax):
        ax.set_facecolor("#13131f")
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor("#33334a")
        ax.yaxis.pane.set_edgecolor("#33334a")
        ax.zaxis.pane.set_edgecolor("#33334a")
        ax.tick_params(colors="#aaaacc", labelsize=6)
        ax.xaxis.label.set_color("#aaaacc")
        ax.yaxis.label.set_color("#aaaacc")
        ax.zaxis.label.set_color("#aaaacc")
        ax.title.set_color("#e0e0ff")

    def _dead_reckon(self):
        dt = 1.0 / self.gyro_fvz
        heading = np.deg2rad(self.yaw)
        x = np.cumsum(np.cos(heading) * dt)
        y = np.cumsum(np.sin(heading) * dt)
        return x, y


    def _init_angle_traces(self):
        ax = self.ax_angles
        t  = np.arange(self.N) / self.gyro_fvz

        ax.plot(t, self.yaw,   color="#e06c75", lw=0.8, label="Yaw")
        ax.plot(t, self.pitch, color="#61afef", lw=0.8, label="Pitch")
        ax.plot(t, self.roll,  color="#98c379", lw=0.8, label="Roll")

        gap = self.tof_total_duration - self.gyro_duration
        if gap > 0.5:
            ax.axvspan(self.gyro_duration, self.tof_total_duration,
                       color="#ffffff", alpha=0.04, zorder=0)
            mid = (self.gyro_duration + self.tof_total_duration) / 2
            ax.text(mid, 0.5, "no IMU data",
                    color="#555577", fontsize=6, ha="center", va="center",
                    transform=ax.get_xaxis_transform())

        x_max = max(self.tof_total_duration, self.gyro_duration)
        ax.set_xlim(0, x_max)

        ax.set_ylabel("degrees", fontsize=7)
        ax.set_xlabel("time (s)", fontsize=7)
        ax.set_title("Yaw / Pitch / Roll", fontsize=9, fontweight="bold")
        ax.legend(loc="upper right", fontsize=6,
                  facecolor="#1e1e2e", edgecolor="#33334a",
                  labelcolor="#ccccdd")
        ax.grid(True, lw=0.3, alpha=0.4, color="#33334a")

        self.cursor_line = ax.axvline(
            0, color="#ffffff", lw=1.0, ls="--", alpha=0.7
        )
    @staticmethod
    def _cube_data():
        s = 0.7
        vertices = np.array([
            [-s, -s, -s],
            [ s, -s, -s],
            [ s,  s, -s],
            [-s,  s, -s],
            [-s, -s,  s],
            [ s, -s,  s],
            [ s,  s,  s],
            [-s,  s,  s]
        ])

        faces = [
            [vertices[j] for j in [0,1,2,3]],
            [vertices[j] for j in [4,5,6,7]],
            [vertices[j] for j in [0,1,5,4]],
            [vertices[j] for j in [2,3,7,6]],
            [vertices[j] for j in [1,2,6,5]],
            [vertices[j] for j in [4,7,3,0]]
        ]

        return vertices, faces

    def _draw_cube(self, rot):
        artists = []
        ax = self.ax_3d

        vertices, faces = self._cube_data()

        rotated_faces = []
        for face in faces:
            rotated_face = [rot.apply(v) for v in face]
            rotated_faces.append(rotated_face)

        cube = Poly3DCollection(
            rotated_faces,
            facecolors="#7a7aaa",
            edgecolors="#c0c0ff",
            linewidths=0.8,
            alpha=0.7
        )

        ax.add_collection3d(cube)
        artists.append(cube)

        fwd = rot.apply(np.array([1.2, 0.0, 0.0]))

        arr = ax.quiver(
            0, 0, 0,
            fwd[0], fwd[1], fwd[2],
            color="#e06c75",
            linewidth=2.5,
            arrow_length_ratio=0.25
        )

        artists.append(arr)

        return artists

    def _init_slider(self):
        ax_slider = self.fig.add_axes([0.15, 0.02, 0.7, 0.03])

        self.slider = Slider(
            ax=ax_slider,
            label='Time',
            valmin=0,
            valmax=self.tof_total_duration,
            valinit=0,
            valstep=1 / self.tof_fvz,
            color="#61afef"
        )

        self.slider.on_changed(self._on_slider_change)

    def _on_slider_change(self, val):
        sample_idx = int(val * self.tof_fvz)
        self.update(sample_idx)

    def _init_3d(self):
        ax = self.ax_3d
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_zlim(-1.5, 1.5)

        ax.set_box_aspect([1, 1, 1])

        ax.set_title("Orientation (body frame)", fontsize=9, fontweight="bold")
        ax.set_xlabel("X", fontsize=7)
        ax.set_ylabel("Y", fontsize=7)
        ax.set_zlabel("Z", fontsize=7)

        ax.grid(True, lw=0.3, alpha=0.3)

        self.cube_artists = self._draw_cube(
            R.from_euler('xyz', [0, 0, 0], degrees=True)
        )

        ax.plot([], [], color="#7a7aaa", label="body", lw=1)
        ax.plot([], [], color="#e06c75", label="forward", lw=2)

        ax.legend(
            loc="upper left",
            fontsize=6,
            facecolor="#1e1e2e",
            edgecolor="#33334a",
            labelcolor="#ccccdd"
        )

    def _init_path(self):
        self._init_slider()
        ax = self.ax_path
        ax.plot(self.path_x, self.path_y,
                color="#4a4a6a", lw=0.7, alpha=0.7)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title("2D Heading Path", fontsize=9, fontweight="bold")
        ax.set_xlabel("X", fontsize=7)
        ax.set_ylabel("Y", fontsize=7)
        ax.grid(True, lw=0.3, alpha=0.4, color="#33334a")
        self.path_dot, = ax.plot(
            [self.path_x[0]], [self.path_y[0]],
            "o", color="#e5c07b", markersize=7, zorder=5
        )
        self.path_tail, = ax.plot(
            [], [], color="#e5c07b", lw=1.8, alpha=0.8
        )


    def update(self, tof_sample_index):
        t_click = tof_sample_index / self.tof_fvz
        i = int(t_click * self.gyro_fvz)
        i = max(0, min(i, self.N - 1))

        self.cursor_line.set_xdata([t_click, t_click])

        q = self.Q[i]  
        rot = R.from_quat([q[1], q[2], q[3], q[0]])  

        for artist in self.cube_artists:
            artist.remove()
        self.cube_artists = self._draw_cube(rot)

        self.ax_3d.set_xlim(-1.5, 1.5)
        self.ax_3d.set_ylim(-1.5, 1.5)
        self.ax_3d.set_zlim(-1.5, 1.5)

        self.path_dot.set_data([self.path_x[i]], [self.path_y[i]])

        tail_start = max(0, i - 200)
        self.path_tail.set_data(
            self.path_x[tail_start:i + 1],
            self.path_y[tail_start:i + 1]
        )

        if hasattr(self, "slider"):
            if abs(self.slider.val - t_click) > 1e-3:
                self.slider.set_val(t_click)

        self.fig.canvas.draw_idle()
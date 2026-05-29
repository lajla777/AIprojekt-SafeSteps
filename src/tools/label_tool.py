import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from scipy.spatial.transform import Rotation as R
from scipy.signal import find_peaks
from ahrs.filters import Madgwick

from tools.visualization import prikazi_signal
from tools.orientation_viewer import OrientationViewer

LABELS = {
    "0": "no_obstacle",
    "1": "obstacle_left",
    "2": "obstacle_right",
    "3": "obstacle_center",
    "4": "obstacle_right_left"
}

LABEL_COLORS = {
    "no_obstacle":"#2ecc71", 
    "obstacle_left":"#3498db",  
    "obstacle_right":"#9b59b6", 
    "obstacle_center":"#f9841d",  
    "obstacle_right_left":"#e74c3c",  
}

def _resample_to_rate(arr, src_fvz, dst_fvz, dst_n):

    # časovna os originalnega signala
    t_src = np.arange(len(arr)) / src_fvz
    # časovna os novega
    t_dst = np.arange(dst_n)/ dst_fvz

    # 1D signal
    if arr.ndim == 1:
        out = np.zeros(dst_n)
        # omejimo območje
        mask = t_dst <= t_src[-1]
        #interpolacija
        out[mask] = np.interp(t_dst[mask], t_src, arr)
        return out

    # več kanalni
    out = np.zeros((dst_n, arr.shape[1]))
    mask = t_dst <= t_src[-1]
    # interpolacija vsakega posebej
    for ch in range(arr.shape[1]):
        out[mask, ch] = np.interp(t_dst[mask], t_src, arr[:, ch])
    return out



def calibrate_mag(mag):
    """
    Kalibracija magnetometra
    Odstrani soft and hard iron distorcijo:
        stalni zamik
        različno skalirane osi

    Hard iron disortion:
        odstranitev outajerjev 
      
    Soft iron distortion :
        avg_delta_x = (max(x) - min(x)) / 2
        ... isto za y in z

        avg_delta = (avg_delta_x + avg_delta_y + avg_delta_z) / 3

        scale_x = avg_delta / avg_delta_x
        ... isto za y in z

        corrected_x = (sensor_x - offset_x) * scale_x
        ... isto za y in z
        
    """
 
    # odstrani ekstremne outliere (šum) zgornji 2% in spodnji 2%
    p_low, p_high = np.percentile(mag, [2, 98], axis=0)
    mag_clipped = np.clip(mag, p_low, p_high)
    
    # omeji vredost na interval low-high
    offset = (mag_clipped.max(axis=0) + mag_clipped.min(axis=0)) / 2.0

    # centriranje 
    mag_centered = mag - offset

    ranges = (mag.max(axis=0) - mag.min(axis=0)) / 2.0
    # povprečen razpon okrog vseh osi 
    avg_range = ranges.mean()
    # osi z manjšim razponom povečamo in obratno
    scale = avg_range / ranges
    # omejitev da se ne skalira preveč
    scale = np.clip(scale, 0.1, 5.0)  

    #skaliranje teh podatkov
    mag_cal = mag_centered * scale
    print(f"Mag hard-iron offset — X:{offset[0]:.3f}  Y:{offset[1]:.3f}  Z:{offset[2]:.3f}")
    print(f"Mag soft-iron scale  — X:{scale[0]:.3f}  Y:{scale[1]:.3f}  Z:{scale[2]:.3f}")
    print(f"Mag cal range — X:{mag_cal[:,0].max()-mag_cal[:,0].min():.3f}  "
          f"Y:{mag_cal[:,1].max()-mag_cal[:,1].min():.3f}  "
          f"Z:{mag_cal[:,2].max()-mag_cal[:,2].min():.3f}")
    return mag_cal


def orientation(gyro, accel, mag, gyro_fvz, accel_fvz, mag_fvz):
    """
    Izračun kvanterniona + Eulerjevi koti

        - odstrani bias žiroskopa -- poiskusno ne spremeni veliko
        - izračuna orientacijo z Madgwick
        - odstrani začetno referenco (zeroing)
        - pretvori v Eulerjeve kote

        Vrne: 
            Q - kvanternioni [w,x,y,z]
            yaw - zasuk okoli Z osi
            pitch - naklon
            roll - nagib

    """
    gyro_raw = gyro.copy()
    N = len(gyro_raw)

    # Če je frekvenca ali dolžina drugačna interpolacija

    if abs(accel_fvz - gyro_fvz) > 0.5 or len(accel) != N:
        accel = _resample_to_rate(accel, accel_fvz, gyro_fvz, N)
    if abs(mag_fvz - gyro_fvz) > 0.5 or len(mag) != N:
        mag = _resample_to_rate(mag, mag_fvz, gyro_fvz, N)

    # apliciranje biasa če je na začetku senzor pri miru 
    # preiskus--- bolj malo spremeni
    bias_samples = int(gyro_fvz * 2.0) # vzame prve 2 sekundi preveri če je primiru

    gyro_window = gyro_raw[:bias_samples]
    if gyro_window.std(axis=0).max() < 0.5:
        gyro_bias = gyro_window.mean(axis=0)
    else:
        gyro_bias = np.zeros(3)
        print("WARNING: sensor wasn't still at start — bias not applied")


    gyro_debiased = gyro_raw - gyro_bias # odstranitev biasa bolj poiskus ni spremenilo veliko
    gyro_rad = np.deg2rad(gyro_debiased) # žiroskop v radiane 
    print(f"Gyro bias — X:{gyro_bias[0]:.3f}  Y:{gyro_bias[1]:.3f}  Z:{gyro_bias[2]:.3f} °/s")

    filter_ = Madgwick(frequency=gyro_fvz, beta=0.07) #filter 
    Q = np.zeros((N, 4))
    Q[0] = [1.0, 0.0, 0.0, 0.0] #začetna orientacija

    for t in range(1, N):
        Q[t] = filter_.updateMARG(
            Q[t - 1],
            gyr=gyro_rad[t],
            acc=accel[t],
            mag=mag[t]
        )

    q0 = Q[0] # začetni kvanternion
    r0_inv = R.from_quat([q0[1], q0[2], q0[3], q0[0]]).inv() #inverz 

    # vse rotacije relativno na začetek
    for i in range(N):
        qi = Q[i]
        ri = R.from_quat([qi[1], qi[2], qi[3], qi[0]])
        r_zeroed = r0_inv * ri
        q_new = r_zeroed.as_quat()  
        Q[i] = [q_new[3], q_new[0], q_new[1], q_new[2]]  

     # SciPy uporablja format [x, y, z, w]
    rots = R.from_quat(Q[:, [1, 2, 3, 0]])
     # ZYX vrstni red (yaw-pitch-roll)
    euler = rots.as_euler('ZYX', degrees=True) # eulerjevi koti
    yaw   = euler[:, 0]
    pitch = euler[:, 1]
    roll  = euler[:, 2]

    print(f"Yaw range — min:{yaw.min():.1f}°  max:{yaw.max():.1f}°  span:{yaw.max()-yaw.min():.1f}°")

    return Q, yaw, pitch, roll

def pair_angle_distance(yaw, tof_matrix, gyro_fvz, tof_fvz):
    """Naredi pare za json datoteko [angle, distance]"""
    N_tof = len(tof_matrix)
    N_yaw = len(yaw)

    t_yaw = np.arange(N_yaw) / gyro_fvz
    t_tof = np.arange(N_tof) / tof_fvz

    t_max = min(t_yaw[-1], t_tof[-1])
    mask  = t_tof <= t_max

    yaw_resampled = np.interp(t_tof[mask], t_yaw, yaw)

    if tof_matrix.ndim == 1:
        distances = tof_matrix[mask]
    else:
        distances = np.nanmin(tof_matrix[mask], axis=1)

    return list(zip(yaw_resampled.tolist(), distances.tolist()))

class LabelTool:
    def __init__(self, signals: dict, save_path, imu_signals: dict = None):
        self.signals = {}
        self.fvz_map = {}
        for name, (arr, fvz) in signals.items():
            self.signals[name] = arr
            self.fvz_map[name] = fvz

        self.names = list(signals.keys())
        self.Fvz = self.fvz_map[self.names[0]]
        self.save_path = save_path

        tof_samples = len(self.signals[self.names[0]])
        self.tof_duration = tof_samples / self.Fvz
        self.total_samples = tof_samples

        self.Q = None
        self.yaw = None
        self.pitch = None
        self.roll = None
        self.gyro_fvz = None
        self.viewer = None

        if imu_signals and all(k in imu_signals for k in ("gyro", "accel", "mag")):
            gyro, self.gyro_fvz = imu_signals["gyro"]
            accel, accel_fvz = imu_signals["accel"]
            mag, mag_fvz = imu_signals["mag"]

            mag_cal = calibrate_mag(mag)

            print(f"IMU durations  — "
                  f"gyro: {len(gyro)/self.gyro_fvz:.2f}s  "
                  f"accel: {len(accel)/accel_fvz:.2f}s  "
                  f"mag: {len(mag)/mag_fvz:.2f}s")
            print(f"ToF duration   — {self.tof_duration:.2f}s")
            print("Computing orientation…")
            mag_cal = calibrate_mag(mag)

            self.Q, self.yaw, self.pitch, self.roll = orientation(
                gyro, accel, mag_cal,
                gyro_fvz=self.gyro_fvz,
                accel_fvz=accel_fvz,
                mag_fvz=mag_fvz,
            )

            print(f"Orientation: {len(self.yaw)} samples @ {self.gyro_fvz:.1f} Hz  "
                  f"({len(self.yaw)/self.gyro_fvz:.2f}s)")

            self.viewer = OrientationViewer(
                Q=self.Q,
                yaw=self.yaw,
                pitch=self.pitch,
                roll=self.roll,
                tof_fvz=self.Fvz,
                gyro_fvz=self.gyro_fvz,
                tof_total_duration=self.tof_duration,
            )

        n = len(self.names)
        self.fig, self.axes = plt.subplots(
            n, 1, figsize=(14, 4 * n), sharex=True,
            num="Label Tool"
        )
        if n == 1:
            self.axes = [self.axes]
        for ax, name in zip(self.axes, self.names):
            prikazi_signal(self.signals[name], Fvz=self.fvz_map[name],
                           y_label=name, ax=ax)
            ax.grid(True)

        self.segments = []
        self.selected = None
        self.select_mode = False
        self.sweep_pending = None       

        self.sweeps = self._find_sweeps()
        self.draw_sweeps()

        self.fig.canvas.mpl_connect("button_press_event", self.onclick)
        self.fig.canvas.mpl_connect("key_press_event", self.onkey)
        self.update_title()
        self.fig.canvas.draw()
    
    def _find_sweeps(self):
        """
        Najde min in max yaw signala (peaks and valleys)
        z funkcijo find_peaks in tvori sweeps (gibe levo-desno)
        """
        if self.yaw is None:
            return []
        yaw_tof = np.interp(np.arange(self.total_samples)/self.Fvz, np.arange(len(self.yaw)) / self.gyro_fvz, self.yaw)
        peaks, _ = find_peaks(yaw_tof, prominence=15)
        valleys, _ = find_peaks(-yaw_tof, prominence=15)
        turns = np.sort(np.concatenate([peaks, valleys]))
        if len(turns) == 0:
            return []
        if turns[0] > 0:
            turns = np.concatenate([[0], turns])
        if turns[-1] < self.total_samples - 1:
            turns = np.concatenate([turns, [self.total_samples - 1]])

        sweeps = []
        for i in range(len(turns) - 1):
            sweeps.append((turns[i], turns[i + 1]))

        # merga prvi in zadnji sweep z ostalimi, ker navadno niso celotni  
        if len(sweeps) >= 4:
            first_merged = (sweeps[0][0], sweeps[1][1])
            last_merged = (sweeps[-2][0], sweeps[-1][1])
            sweeps = [first_merged] + sweeps[2:-2] + [last_merged]
        elif len(sweeps) == 3:
            first_merged = (sweeps[0][0], sweeps[1][1])
            last_merged = (sweeps[1][0], sweeps[2][1])
            sweeps = [first_merged, last_merged]
        elif len(sweeps) == 2:
            sweeps = [(sweeps[0][0], sweeps[1][1])]

        return sweeps
    
    def draw_sweeps(self):
        for i, (s,e) in enumerate(self.sweeps):
            s_time = s / self.Fvz
            e_time = e / self.Fvz
            color = "#2000f3" if i % 2 == 0 else "#ff6200"

            for ax in self.axes:
                ax.axvline(s_time, color=color, lw=1.4, linestyle=":", alpha=0.9)
            mid_time = (s_time + e_time) / 2
            self.axes[0].text(
                mid_time, 0.97, f"S{i+1}",
                transform=self.axes[0].get_xaxis_transform(),
                ha="center", va="top",
                fontsize=6, color=color, alpha=0.8
            )


    def _find_gap(self, x):
        """ni trenutno uporabljeno--- uporabljeno v prejšnji verziji"""
        for seg in self.segments:
            if seg["start"] <= x <= seg["end"]:
                return None

        covered = sorted([(s["start"], s["end"]) for s in self.segments])

        gap_start = 0
        for s, e in covered:
            if e < x:
                gap_start = e + 1

        gap_end = self.total_samples - 1
        for s, e in covered:
            if s > x:
                gap_end = s - 1
                break

        return (gap_start, gap_end)

    def update_title(self):
        if self.select_mode:
            title = "[SELECT MODE] Click segment | 0/1/2/3/4=change | d=delete | m=exit | h=save"
            if self.selected:
                title += f"  |  Selected: {self.selected['label']}"
        else:
            title = "[SWEEP MODE] Click sweep → 0/1/2/3/4 to label | m=select | h=save"
            if self.sweep_pending:
                s, e = self.sweep_pending
                title += f"  |  Sweep: {s/self.Fvz:.2f}s – {e/self.Fvz:.2f}s → press label key"
        self.axes[0].set_title(title)
        self.fig.canvas.draw()


    def onclick(self, event):
        if event.inaxes not in self.axes or event.xdata is None:  
            return
        x_time = event.xdata
        x = int(x_time * self.Fvz) 

        if self.viewer is not None:                             
            self.viewer.update(x)
        if event.inaxes not in self.axes or event.xdata is None:
            return
        
        x_time = event.xdata
        x = int(x_time * self.Fvz)

        if self.select_mode:
            self.selected = None
            for seg in self.segments:
                if seg["start"] <= x <= seg["end"]:
                    self.selected = seg
                    print(f"Selected: {seg['label']} ({seg['start']}–{seg['end']})")
                    break
            if self.selected is None:
                print("No segment at click position")
            self.update_title()
        else:
            clicked_sweep = None
            for s, e in self.sweeps:
                if s <= x <= e:
                    clicked_sweep = (s,e)
                    break
            if clicked_sweep is None:
                print("Click inside a sweep region")
                self.sweep_pending = None
                self.redraw()
                return 
            
            self.sweep_pending = clicked_sweep
            s_time = clicked_sweep[0] / self.Fvz
            e_time = clicked_sweep[1] / self.Fvz

            self.redraw()

            for ax in self.axes:
                ax.axvspan(s_time, e_time, alpha=0.15, color="white", zorder=0)
                ax.axvline(s_time, color="white", lw=1.5, linestyle="--")
                ax.axvline(e_time, color="white", lw=1.5, linestyle="--")

            self.fig.canvas.draw()
            print(f"Sweep selected: {s_time:.2f}s – {e_time:.2f}s → press 0/1/2/3/4 to label")
            self.update_title()    

    def onkey(self, event):
        if event.key == "h":
            self.save()
            return

        if event.key == "m":
            self.select_mode = not self.select_mode
            self.selected = None
            self.sweep_pending = None
            print(f"{'Entered' if self.select_mode else 'Exited'} select mode")
            self.redraw()
            return

        if self.select_mode:
            if event.key in LABELS:
                if self.selected is None:
                    print("No segment selected")
                    return
                self.selected["label"] = LABELS[event.key]
                print(f"Updated label → {self.selected['label']}")
                self.selected = None
                self.redraw()
            elif event.key == "d":
                self.delete_selected()
        
        if event.key in LABELS:
            if self.sweep_pending is None:
                print("Click a sweep first")
                return
            s, e  = self.sweep_pending
            label = LABELS[event.key]
            seg = {
                "start":s,
                "end":e,
                "start_time": s / self.Fvz,
                "end_time": e / self.Fvz,
                "label": label,
                "span": [],
                "start_line": [],
                "end_line": []
            }
            self.segments.append(seg)
            self.sweep_pending = None
            self.redraw()
            print(f"Saved: {label} ({s}–{e})")

    def draw_segment(self, seg):
        color = LABEL_COLORS[seg["label"]]
        is_selected = (seg is self.selected)
        seg["span"] = []; seg["start_line"] = []; seg["end_line"] = []
        s_time = seg["start"] / self.Fvz
        e_time = seg["end"]   / self.Fvz
        for ax in self.axes:
            seg["span"].append(ax.axvspan(
                s_time, e_time,
                alpha=0.4 if is_selected else 0.25, color=color
            ))
            seg["start_line"].append(ax.axvline(
                s_time, color=color,
                linewidth=2 if is_selected else 1,
                linestyle="--" if is_selected else "-"
            ))
            seg["end_line"].append(ax.axvline(
                e_time, color=color,
                linewidth=2 if is_selected else 1,
                linestyle="--" if is_selected else "-"
            ))

    def redraw(self):
        for ax, name in zip(self.axes, self.names):
            ax.clear()
            prikazi_signal(self.signals[name], Fvz=self.fvz_map[name],
                           y_label=name, ax=ax)
            ax.grid(True)
        self.draw_sweeps()
        for seg in self.segments:
            self.draw_segment(seg)
        self.fig.canvas.draw()
        self.update_title()

    def delete_selected(self):
        if self.selected is None:
            print("No segment selected")
            return
        self.segments.remove(self.selected)
        self.selected = None
        self.redraw()
        print("Deleted segment")

    def save(self):
        tof_matrix = self.signals[self.names[0]]
        tof_fvz = self.fvz_map[self.names[0]]
        primary = tof_matrix[:, 0] if tof_matrix.ndim > 1 else tof_matrix

        clean = []
        for s in self.segments:
            seg_tof = tof_matrix[s["start"]:s["end"] + 1]
            angle_distance = None

            if self.yaw is not None:
                pairs = pair_angle_distance(
                    self.yaw, seg_tof,
                    gyro_fvz=self.gyro_fvz,
                    tof_fvz=tof_fvz
                )
                angle_distance = [
                    [None if np.isnan(a) else round(a, 4),
                     None if np.isnan(d) else round(d, 2)]
                    for a, d in pairs
                ]

            clean.append({
                # "start": s["start"],
                # "end": s["end"],
                # "start_time": s["start_time"],
                # "end_time": s["end_time"],
                "label": s["label"],
                # "samples": [
                #     None if np.isnan(v) else v
                #     for v in primary[s["start"]:s["end"] + 1].tolist()
                # ],
                "angle_distance": angle_distance
            })

        with open(self.save_path, "w") as f:
            json.dump(clean, f, indent=2)

        print(f"Saved {len(clean)} segments {self.save_path}")

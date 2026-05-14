import json
import numpy as np
import matplotlib.pyplot as plt

from visualisation import prikazi_signal

LABELS = {
    "0": "no_obstacle",
    "1": "obstacle",
    "2": "very_close"
}

LABEL_COLORS = {
    "no_obstacle": "blue",
    "obstacle": "orange",
    "very_close":"red"
}

class LabelTool:
    def __init__(self, signals: dict, save_path):
        self.signals = {}
        self.fvz_map = {}
        for name, (arr, fvz) in signals.items():
            self.signals[name] = arr
            self.fvz_map[name] = fvz

        self.names     = list(signals.keys())
        self.Fvz       = self.fvz_map[self.names[0]]
        self.save_path = save_path

        tof_samples = len(self.signals[self.names[0]])
        self.tof_duration = tof_samples / self.Fvz

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

        self.start = None
        self.end = None
        self.segments = []
        self.selected = None
        self.select_mode = False

        self.fig.canvas.mpl_connect("button_press_event", self.onclick)
        self.fig.canvas.mpl_connect("key_press_event",   self.onkey)
        self.update_title()

    def update_title(self):
        if self.select_mode:
            title = "[SELECT MODE] Click segment | 0/1/2=change | d=delete | m=exit | h=save"
            if self.selected:
                title += f"  |  Selected: {self.selected['label']}"
        else:
            title = "[DRAW MODE] Click start/end → 0/1/2 to label | m=select mode | h=save"
        self.axes[0].set_title(title)
        self.fig.canvas.draw()

    def onclick(self, event):
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
            if self.start is None:
                self.start = x
                for ax in self.axes:
                    ax.axvline(x / self.Fvz, color="green")
                self.fig.canvas.draw()
                print(f"Start: {x}")
            else:
                self.end = x
                for ax in self.axes:
                    ax.axvline(x / self.Fvz, color="black")
                self.fig.canvas.draw()
                print(f"End: {x}")

    def onkey(self, event):
        if event.key == "h":
            self.save()
            return

        if event.key == "m":
            self.select_mode = not self.select_mode
            self.selected    = None
            self.start       = None
            self.end         = None
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
        else:
            if event.key in LABELS:
                if self.start is None or self.end is None:
                    print("Set start and end point first")
                    return
                label = LABELS[event.key]
                s = min(self.start, self.end)
                e = max(self.start, self.end)
                seg = {
                    "start": s,
                    "end": e,
                    "start_time": s / self.Fvz,
                    "end_time": e / self.Fvz,
                    "label": label,
                    "span": [],
                    "start_line": [],
                    "end_line": []
                }
                self.segments.append(seg)
                self.redraw()
                print(f"Saved: {label} ({s}–{e})")
                self.start = None
                self.end   = None

    def draw_segment(self, seg):
        color       = LABEL_COLORS[seg["label"]]
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
        primary    = tof_matrix[:, 0] if tof_matrix.ndim > 1 else tof_matrix
        clean = []
        for s in self.segments:
            clean.append({
                "start": s["start"],
                "end": s["end"],
                "start_time": s["start_time"],
                "end_time": s["end_time"],
                "label": s["label"],
                "samples": [
                    None if np.isnan(v) else v
                    for v in primary[s["start"]:s["end"] + 1].tolist()
                ],
            })

        with open(self.save_path, "w") as f:
            json.dump(clean, f, indent=2)

        print(f"Saved {len(clean)} segments → {self.save_path}")

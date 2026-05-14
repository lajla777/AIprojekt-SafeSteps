import json
import numpy as np
import matplotlib.pyplot as plt

LABELS = {
    "0": "no_obstacle",
    "1": "obstacle",
    "2": "very_close",
}

LABEL_COLORS = {
    "no_obstacle": "blue",
    "obstacle": "orange",
    "very_close": "red"
}

class LabelTool:
    def __init__(self, signal, Fvz, save_path):
        self.signal = signal.flatten()
        self.Fvz = Fvz
        self.save_path = save_path

        self.fig, self.ax = plt.subplots(figsize=(14, 5))
        self.ax.plot(self.signal, linewidth=0.8)
        self.ax.grid(True)

        self.start = None
        self.end = None

        self.segments = []
        self.selected = None
        self.select_mode = False

        self.fig.canvas.mpl_connect("button_press_event", self.onclick)
        self.fig.canvas.mpl_connect("key_press_event", self.onkey)

        self.update_title()

    def update_title(self):
        if self.select_mode:
            title = "[SELECT MODE] Click segment | 0/1/2=change | d=delete | m=exit | h=save"
            if self.selected:
                title += f"  |  Selected: {self.selected['label']}"
        else:
            title = "[DRAW MODE] Click start/end → 0/1/2 to label | m=select mode | h=save"

        self.ax.set_title(title)
        self.fig.canvas.draw()

    def onclick(self, event):
        if event.inaxes != self.ax:
            return

        x = int(event.xdata)

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
                self.ax.axvline(x, color="green")
                self.fig.canvas.draw()
                print(f"Start: {x}")
            else:
                self.end = x
                self.ax.axvline(x, color="black")
                self.fig.canvas.draw()
                print(f"End: {x}")

    def onkey(self, event):

        if event.key == "h":
            self.save()
            return

        if event.key == "m":
            self.select_mode = not self.select_mode
            self.selected = None
            self.start = None
            self.end = None
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
                    "span": None,
                    "start_line": None,
                    "end_line": None
                }

                self.segments.append(seg)
                self.redraw()
                print(f"Saved: {label} ({s}–{e})")

                self.start = None
                self.end = None

    def draw_segment(self, seg):
        color = LABEL_COLORS[seg["label"]]
        is_selected = (seg is self.selected)

        seg["span"] = self.ax.axvspan(
            seg["start"], seg["end"],
            alpha=0.4 if is_selected else 0.25,
            color=color
        )
        seg["start_line"] = self.ax.axvline(
            seg["start"], color=color,
            linewidth=2 if is_selected else 1,
            linestyle="--" if is_selected else "-"
        )
        seg["end_line"] = self.ax.axvline(
            seg["end"], color=color,
            linewidth=2 if is_selected else 1,
            linestyle="--" if is_selected else "-"
        )

    def redraw(self):
        self.ax.clear()
        self.ax.plot(self.signal, linewidth=0.8)
        self.ax.grid(True)

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
        clean = [
            {
                "start": s["start"],
                "end": s["end"],
                "start_time": s["start_time"],
                "end_time": s["end_time"],
                "label": s["label"],
                "samples": [
                    None if np.isnan(v) else v
                    for v in self.signal[s["start"]:s["end"] + 1].tolist()
                ]
            }
            for s in self.segments
        ]

        with open(self.save_path, "w") as f:
            json.dump(clean, f, indent=2)

        print(f"Saved to {self.save_path}")

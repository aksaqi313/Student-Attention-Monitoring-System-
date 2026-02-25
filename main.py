"""
main.py – Student Attention Monitoring System
Launch point: py main.py
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import cv2
from PIL import Image, ImageTk
import numpy as np
import os
import platform
import winsound

from attention_engine import AttentionEngine
import report_generator


# ─── Alert sounds (winsound – built into Windows) ─────────────────────────────
_SOUND_DROWSY = (880, 400)   # (freq Hz, duration ms)
_SOUND_AWAY   = (660, 250)
_SOUND_YAWN   = (440, 300)

def _play_beep(freq, dur):
    """Play a beep in a daemon thread so GUI stays responsive."""
    def _beep():
        try:
            winsound.Beep(freq, dur)
        except Exception:
            pass
    threading.Thread(target=_beep, daemon=True).start()


# ─── Color palette ─────────────────────────────────────────────────────────────
CLR_BG       = "#0d1117"
CLR_CARD     = "#161b22"
CLR_CARD2    = "#1c2230"
CLR_ACCENT   = "#00bcd4"
CLR_GREEN    = "#00e676"
CLR_YELLOW   = "#ffeb3b"
CLR_ORANGE   = "#ff9800"
CLR_RED      = "#f44336"
CLR_TEXT     = "#e6edf3"
CLR_MUTED    = "#8b949e"
CLR_BORDER   = "#30363d"

FONT_HEAD    = ("Segoe UI", 20, "bold")
FONT_METRIC  = ("Segoe UI", 28, "bold")
FONT_LABEL   = ("Segoe UI", 9)
FONT_SMALL   = ("Segoe UI", 8)
FONT_ALERT   = ("Segoe UI", 10, "bold")
FONT_BTN     = ("Segoe UI", 10, "bold")


def score_color(score):
    if score >= 80:  return CLR_GREEN
    if score >= 55:  return CLR_YELLOW
    if score >= 30:  return CLR_ORANGE
    return CLR_RED


# ─── Mini Canvas graph (Composition over Inheritance) ─────────────────────────
class MiniGraph:
    def __init__(self, master, width=280, height=70):
        self.frame = tk.Frame(master, bg=CLR_CARD2)
        self.canvas = tk.Canvas(self.frame, width=width, height=height,
                                bg=CLR_CARD2, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self._w = width
        self._h = height
        self.history = []

    def pack(self, **kwargs):
        self.frame.pack(**kwargs)

    def update_data(self, history):
        self.history = list(history)[-self._w:]
        self._redraw()

    def _redraw(self):
        c = self.canvas
        c.delete("all")
        h, w = self._h, self._w
        if len(self.history) < 2:
            return

        # Grid lines
        for pct in [25, 50, 75, 100]:
            y = h - int(h * pct / 100)
            c.create_line(0, y, w, y, fill="#222", width=1)

        # Zone tints (darker colors)
        for lo, hi, clr in [(80, 100, "#013220"), (55, 80, "#2a2a00"),
                            (30, 55, "#3d2b00"),  (0, 30, "#3d0000")]:
            y1 = h - int(h * hi / 100)
            y2 = h - int(h * lo / 100)
            c.create_rectangle(0, y1, w, y2, fill=clr, outline="")

        # Line
        xs = np.linspace(0, w, len(self.history))
        ys = [h - int(h * v / 100) for v in self.history]
        pts = []
        for x, y in zip(xs, ys):
            pts += [float(x), float(y)]
        if len(pts) >= 4:
            c.create_line(*pts, fill=CLR_ACCENT, width=2, smooth=True)

        # Last point dot
        c.create_oval(xs[-1]-4, ys[-1]-4, xs[-1]+4, ys[-1]+4,
                         fill=CLR_ACCENT, outline="white", width=1)


# ─── Main App ──────────────────────────────────────────────────────────────────
class AttentionApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Student Attention Monitoring System")
        self.configure(bg=CLR_BG)
        self.resizable(True, True)
        self.geometry("1200x750")
        self.minsize(1000, 650)

        self.engine         = AttentionEngine()
        self.cap            = None
        self.running        = False
        self._cam_thread    = None
        self._last_result   = {}
        self._alert_cool    = {}
        self.student_name   = "Student"

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(500, self._ask_student_name)

    def _build_ui(self):
        # Header
        hdr = tk.Frame(self, bg=CLR_CARD, height=56)
        hdr.pack(fill="x", side="top")
        hdr.pack_propagate(False)

        tk.Label(hdr, text="🎓 Attention Monitoring System",
                 font=FONT_HEAD, fg=CLR_ACCENT, bg=CLR_CARD).pack(side="left", padx=20, pady=10)

        btn_frame = tk.Frame(hdr, bg=CLR_CARD)
        btn_frame.pack(side="right", padx=16)

        self.btn_start = tk.Button(btn_frame, text="▶  Start", font=FONT_BTN, bg="#1a6b3a", fg="white",
                                   relief="flat", padx=14, pady=6, command=self._start_monitoring)
        self.btn_start.pack(side="left", padx=4)

        self.btn_stop = tk.Button(btn_frame, text="■  Stop", font=FONT_BTN, bg="#7a1c1c", fg="white",
                                  relief="flat", padx=14, pady=6, state="disabled", command=self._stop_monitoring)
        self.btn_stop.pack(side="left", padx=4)

        self.btn_report = tk.Button(btn_frame, text="📄  Report", font=FONT_BTN, bg="#1a3a6b", fg="white",
                                    relief="flat", padx=14, pady=6, command=self._export_report)
        self.btn_report.pack(side="left", padx=4)

        self.btn_reset = tk.Button(btn_frame, text="↺  Reset", font=FONT_BTN, bg="#3a2f00", fg=CLR_YELLOW,
                                   relief="flat", padx=14, pady=6, command=self._reset_session)
        self.btn_reset.pack(side="left", padx=4)

        # Body
        body = tk.Frame(self, bg=CLR_BG)
        body.pack(fill="both", expand=True, padx=12, pady=8)
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(0, weight=1)

        # Video Card
        vid_card = self._card(body)
        vid_card.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        self.vid_label = tk.Label(vid_card, bg="#000", text="Camera preview ready", fg=CLR_MUTED)
        self.vid_label.pack(fill="both", expand=True, padx=4, pady=4)

        # Right Panel
        right = tk.Frame(body, bg=CLR_BG)
        right.grid(row=0, column=1, sticky="nsew")

        # Score Card
        score_card = self._card(right)
        score_card.pack(fill="x", pady=(0, 6))
        tk.Label(score_card, text="ATTENTION SCORE (REAL-TIME)", font=("Segoe UI", 9, "bold"), fg=CLR_MUTED, bg=CLR_CARD).pack(pady=(10,0))
        self.lbl_score = tk.Label(score_card, text="—", font=("Segoe UI", 52, "bold"), fg=CLR_ACCENT, bg=CLR_CARD)
        self.lbl_score.pack()
        self.lbl_status = tk.Label(score_card, text="Not Started", font=("Segoe UI", 12), fg=CLR_MUTED, bg=CLR_CARD)
        self.lbl_status.pack(pady=(0, 4))

        self.score_bar_var = tk.DoubleVar(value=0)
        style = ttk.Style()
        style.theme_use("default")
        style.configure("Score.Horizontal.TProgressbar", background=CLR_ACCENT, troughcolor=CLR_CARD2, thickness=10)
        pb = ttk.Progressbar(score_card, variable=self.score_bar_var, maximum=100, length=240, style="Score.Horizontal.TProgressbar")
        pb.pack(pady=(10, 10))

        # Graph
        graph_card = self._card(right)
        graph_card.pack(fill="x", pady=(0, 6))
        self.mini_graph = MiniGraph(graph_card, width=320, height=80)
        self.mini_graph.pack(fill="x", padx=10, pady=10)

        # Metrics
        metrics_card = self._card(right)
        metrics_card.pack(fill="x", pady=(0, 6))
        mg = tk.Frame(metrics_card, bg=CLR_CARD)
        mg.pack(fill="x", padx=8, pady=8)
        self.metric_widgets = {}
        defs = [("blinks", "Blinks"), ("session", "Session"), ("distracted", "Distracted"),
                ("avg_score", "Avg Score"), ("ear", "EAR"), ("yaw", "Yaw"), 
                ("pitch", "Pitch")]
        for i, (key, label) in enumerate(defs):
            cell = tk.Frame(mg, bg=CLR_CARD2, padx=6, pady=5)
            cell.grid(row=i//3, column=i%3, padx=2, pady=2, sticky="nsew")
            tk.Label(cell, text=label, font=FONT_SMALL, fg=CLR_MUTED, bg=CLR_CARD2).pack()
            val_lbl = tk.Label(cell, text="—", font=("Segoe UI", 12, "bold"), fg=CLR_TEXT, bg=CLR_CARD2)
            val_lbl.pack()
            self.metric_widgets[key] = val_lbl
        for c in range(3): mg.columnconfigure(c, weight=1)

        # Alerts
        alerts_card = self._card(right)
        alerts_card.pack(fill="both", expand=True)
        self.alert_labels = {}
        for key, text in [("alert_drowsy", "Eyes Closed"), ("alert_looking_away", "Looking Away"),
                          ("alert_yawning", "Yawning"), ("alert_no_face", "No Face")]:
            lbl = tk.Label(alerts_card, text=text, font=FONT_ALERT, fg="#333", bg=CLR_CARD, anchor="w", padx=10, pady=4)
            lbl.pack(fill="x", padx=8, pady=1)
            self.alert_labels[key] = lbl

        self.lbl_name = tk.Label(self, text=f"Monitoring: {self.student_name}", font=FONT_SMALL, fg=CLR_MUTED, bg=CLR_BG)
        self.lbl_name.pack(side="bottom", pady=4)

    def _card(self, parent):
        f = tk.Frame(parent, bg=CLR_CARD, bd=0, relief="flat", highlightbackground=CLR_BORDER, highlightthickness=1)
        return f

    def _ask_student_name(self):
        dlg = tk.Toplevel(self)
        dlg.title("Student Name")
        dlg.configure(bg=CLR_CARD)
        dlg.geometry("400x150")
        dlg.grab_set()
        tk.Label(dlg, text="Enter student name:", fg=CLR_TEXT, bg=CLR_CARD, pady=10).pack()
        entry = tk.Entry(dlg, font=("Segoe UI", 11), bg=CLR_CARD2, fg=CLR_TEXT, insertbackground=CLR_ACCENT)
        entry.pack(pady=5)
        entry.insert(0, "Student")
        def _ok():
            self.student_name = entry.get().strip() or "Student"
            self.lbl_name.config(text=f"Monitoring: {self.student_name}")
            dlg.destroy()
        tk.Button(dlg, text="Start", bg=CLR_ACCENT, command=_ok, padx=10, pady=5).pack(pady=10)

    def _start_monitoring(self):
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Webcam not found")
            return
        self.running = True
        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")
        threading.Thread(target=self._cam_loop, daemon=True).start()

    def _stop_monitoring(self):
        self.running = False
        self.btn_start.config(state="normal")
        self.btn_stop.config(state="disabled")

    def _reset_session(self):
        self.engine.reset_session()
        self._ask_student_name()

    def _export_report(self):
        if not self._last_result: return
        path = report_generator.generate(self._last_result, self.student_name)
        messagebox.showinfo("Saved", f"Report saved:\n{path}")
        if platform.system() == "Windows": os.startfile(path)

    def _cam_loop(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok: break
            frame = cv2.flip(frame, 1)
            ann, res = self.engine.process(frame)
            self.after(0, self._update_ui, ann, res)
            time.sleep(0.01)
        if self.cap: self.cap.release()

    def _update_ui(self, frame, res):
        self._last_result = res
        img = Image.fromarray(cv2.cvtColor(cv2.resize(frame, (640, 480)), cv2.COLOR_BGR2RGB))
        photo = ImageTk.PhotoImage(img)
        self.vid_label.config(image=photo); self.vid_label._img = photo
        
        score = res.get("attention_score", 0)
        avg_score = res.get("average_score", 0)
        
        self.lbl_score.config(text=str(int(score)), fg=score_color(score))
        self.lbl_status.config(text=res.get("status_label", ""), fg=score_color(score))
        self.score_bar_var.set(score)
        self.mini_graph.update_data(res.get("history", []))
        
        self.metric_widgets["blinks"].config(text=str(res.get("blinks", 0)))
        s = res.get("session_sec", 0); self.metric_widgets["session"].config(text=f"{int(s//60)}:{int(s%60):02d}")
        d = res.get("distracted_sec", 0); self.metric_widgets["distracted"].config(text=f"{int(d//60)}:{int(d%60):02d}")
        
        self.metric_widgets["avg_score"].config(text=f"{avg_score:.1f}", fg=score_color(avg_score))
        self.metric_widgets["ear"].config(text=f"{res.get('ear',0):.2f}")
        self.metric_widgets["yaw"].config(text=f"{res.get('yaw',0):.0f}°")
        self.metric_widgets["pitch"].config(text=f"{res.get('pitch',0):.0f}°")
        
        for key, lbl in self.alert_labels.items():
            active = res.get(key, False)
            lbl.config(fg=CLR_RED if active else "#333")
            if active and key != "alert_no_face":
                last = self._alert_cool.get(key, 0)
                if (time.time() - last) > 5:
                    _play_beep(600, 200); self._alert_cool[key] = time.time()

    def _on_close(self):
        self.running = False; self.destroy()

if __name__ == "__main__":
    AttentionApp().mainloop()

import tkinter as tk
from tkinter import colorchooser
import mss
import mss.tools
import numpy as np
import platform
import socket
import pyautogui #type: ignore
import time
import threading

# --- ZERO LATENCY ---
pyautogui.PAUSE = 0.0 

# --- CONFIGURATION ---
SERVER_IP = "localhost"
SERVER_PORT = 5000

DEFAULT_CONFIG = {
    "grid_rows": 12,       
    "grid_cols": 20,       
    "offset_x": 40, "offset_y": 120,
    "spacing_x": 38, "spacing_y": 38,
    
    # --- COLORS (snake-game.io) ---
    "snake_head_color": [78, 124, 246], 
    "snake_tail_color": [25, 69, 161],  
    "food_color":       [229, 25, 26],  
    
    "color_threshold": 50,          
    "sample_radius": 0,
    "zoom_factor": 1.0, 
    "auto_play": False,
    "debug_hitboxes": False
}

KEY_MAP = {0: 'up', 1: 'right', 2: 'down', 3: 'left'}

class ScreenTranslationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Snake AI | Signal Boosted Client")
        self.root.geometry("1200x850")
        
        self.is_native_windows = platform.system() == "Windows"
        if self.is_native_windows:
            self.transparent_color = "#ff00ff"
            self.root.attributes("-transparentcolor", self.transparent_color)
        else:
            self.transparent_color = "#000000"
            self.root.attributes("-alpha", 0.7)

        self.config = DEFAULT_CONFIG.copy()
        self.sct = mss.mss()
        self.sock = None
        self.connected = False
        
        self.update_color_vectors()
        self.setup_ui()
        
        threading.Thread(target=self.network_loop, daemon=True).start()
        self.update_loop()

    def update_color_vectors(self):
        self.head_vec = np.array(self.config["snake_head_color"], dtype=float)
        self.tail_vec = np.array(self.config["snake_tail_color"], dtype=float)
        self.food_vec = np.array(self.config["food_color"], dtype=float)
        self.body_vec = self.head_vec - self.tail_vec
        self.body_vec_len_sq = np.dot(self.body_vec, self.body_vec)

    def setup_ui(self):
        self.frame_left = tk.Frame(self.root, bg="#101010", width=320)
        self.frame_left.pack(side=tk.LEFT, fill=tk.Y)
        self.frame_left.pack_propagate(False)
        
        tk.Label(self.frame_left, text="AI Vision (Boosted)", bg="#101010", fg="white", font=("Arial", 12, "bold")).pack(pady=10)
        self.canvas_vis = tk.Canvas(self.frame_left, width=280, height=180, bg="black", highlightthickness=1, highlightbackground="#333")
        self.canvas_vis.pack(padx=20)
        
        self.lbl_prediction = tk.Label(self.frame_left, text="Action: -", bg="#101010", fg="#888", font=("Arial", 14, "bold"))
        self.lbl_prediction.pack(pady=15)
        
        self.lbl_latency = tk.Label(self.frame_left, text="Ping: --ms", bg="#101010", fg="#aaa", font=("Courier", 10))
        self.lbl_latency.pack(side=tk.BOTTOM, pady=5)
        self.lbl_status = tk.Label(self.frame_left, text="Connecting...", bg="#101010", fg="#aaa", font=("Arial", 10))
        self.lbl_status.pack(side=tk.BOTTOM, pady=5)

        self.frame_center = tk.Frame(self.root, bg=self.transparent_color)
        self.frame_center.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas_overlay = tk.Canvas(self.frame_center, bg=self.transparent_color, highlightthickness=0)
        self.canvas_overlay.pack(fill=tk.BOTH, expand=True)
        self.canvas_overlay.bind("<Configure>", lambda e: self.draw_overlay())

        self.frame_right = tk.Frame(self.root, bg="white", width=300)
        self.frame_right.pack(side=tk.RIGHT, fill=tk.Y)
        self.frame_right.pack_propagate(False)
        self.setup_config_panel()

    def setup_config_panel(self):
        tk.Label(self.frame_right, text="Configuration", bg="white", font=("Arial", 12, "bold")).pack(pady=10)
        
        f_debug = tk.Frame(self.frame_right, bg="#eef", bd=1, relief="solid")
        f_debug.pack(fill=tk.X, padx=10, pady=5)
        
        self.var_hitboxes = tk.BooleanVar(value=self.config["debug_hitboxes"])
        tk.Checkbutton(f_debug, text="Show Sample Areas", variable=self.var_hitboxes, bg="#eef",
                       command=lambda: self.update_bool_config("debug_hitboxes", self.var_hitboxes)).pack(anchor="w", padx=5)
        
        self.lbl_debug_rgb = tk.Label(f_debug, text="RGB: 0,0,0", bg="black", fg="white", font=("Courier", 10))
        self.lbl_debug_rgb.pack(pady=2)

        self.var_autoplay = tk.BooleanVar(value=self.config["auto_play"])
        tk.Checkbutton(self.frame_right, text="ENABLE AUTO-PLAY", variable=self.var_autoplay, 
                       bg="white", fg="red", font=("Arial", 10, "bold"),
                       command=lambda: self.update_bool_config("auto_play", self.var_autoplay)).pack(pady=10)

        def add_slider(label, key, min_v, max_v, res=1):
            f = tk.Frame(self.frame_right, bg="white")
            f.pack(fill=tk.X, padx=10, pady=1)
            tk.Label(f, text=label, bg="white", anchor="w", font=("Arial", 8)).pack(fill=tk.X)
            s = tk.Scale(f, from_=min_v, to=max_v, orient=tk.HORIZONTAL, bg="white", resolution=res,
                         command=lambda v: self.update_config(key, float(v)))
            s.set(self.config[key])
            s.pack(fill=tk.X)

        tk.Label(self.frame_right, text="Geometry", bg="#eee").pack(fill=tk.X, pady=2)
        add_slider("Start X", "offset_x", 0, 500)
        add_slider("Start Y", "offset_y", 0, 500)
        add_slider("Spacing X", "spacing_x", 10, 100)
        add_slider("Spacing Y", "spacing_y", 10, 100)
        add_slider("Sample Radius", "sample_radius", 0, 10)
        
        tk.Label(self.frame_right, text="Gradient Detection", bg="#eee").pack(fill=tk.X, pady=2)
        add_slider("Color Tolerance", "color_threshold", 5, 150)
        
        f_col = tk.Frame(self.frame_right, bg="white")
        f_col.pack(pady=5)
        def make_col_btn(txt, key, hex_c):
            tk.Button(f_col, text=txt, bg=hex_c, fg="white", width=8,
                      command=lambda: self.pick_color(key)).pack(side=tk.LEFT, padx=2)
        make_col_btn("Head", "snake_head_color", "#4e7cf6")
        make_col_btn("Tail", "snake_tail_color", "#1945a1")
        make_col_btn("Food", "food_color", "#e5191a")

    def update_config(self, key, value):
        if key in ["grid_rows", "grid_cols", "sample_radius"]: value = int(value)
        self.config[key] = value
        self.draw_overlay()

    def update_bool_config(self, key, var):
        self.config[key] = var.get()
        self.draw_overlay()

    def pick_color(self, key):
        c = colorchooser.askcolor(color=tuple(self.config[key]))[0] #type: ignore
        if c: 
            self.config[key] = [int(x) for x in c]
            self.update_color_vectors() 

    def draw_overlay(self):
        self.canvas_overlay.delete("all")
        rows, cols = self.config["grid_rows"], self.config["grid_cols"]
        ox, oy = self.config["offset_x"], self.config["offset_y"]
        sx, sy = self.config["spacing_x"], self.config["spacing_y"]
        rad = self.config["sample_radius"]
        gap = rad + 2
        for r in range(rows):
            for c in range(cols):
                px, py = ox + c * sx, oy + r * sy
                self.canvas_overlay.create_line(px-(gap+4), py, px-gap, py, fill="red", width=1)
                self.canvas_overlay.create_line(px+gap, py, px+(gap+4), py, fill="red", width=1)
                self.canvas_overlay.create_line(px, py-(gap+4), px, py-gap, fill="red", width=1)
                self.canvas_overlay.create_line(px, py+gap, px, py+(gap+4), fill="red", width=1)
        if self.config["debug_hitboxes"]:
            for r in range(rows):
                for c in range(cols):
                    px, py = ox + c * sx, oy + r * sy
                    self.canvas_overlay.create_rectangle(px-rad, py-rad, px+rad, py+rad, outline="yellow", width=1)

    def calculate_gradient_value(self, detected_rgb):
        vec_to_pixel = detected_rgb - self.tail_vec
        t = np.dot(vec_to_pixel, self.body_vec) / (self.body_vec_len_sq + 1e-6)
        
        # --- SIGNAL BOOSTING ---
        # Remap [0.1 ... 1.0] -> [0.5 ... 1.0]
        # This ensures the model sees the tail as a "Real Object" not noise
        raw_val = float(np.clip(t, 0.1, 1.0))
        boosted_val = 0.5 + (0.5 * raw_val)
        
        return boosted_val

    def network_loop(self):
        """Persistent TCP connection"""
        while True:
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1) 
                self.sock.connect((SERVER_IP, SERVER_PORT))
                self.connected = True
                self.root.after(0, lambda: self.lbl_status.config(text="Status: Connected (TCP)", fg="#00ff00"))
                while self.connected: time.sleep(1)
            except Exception as e:
                self.connected = False
                self.root.after(0, lambda: self.lbl_status.config(text=f"Reconnecting... {e}", fg="red"))
                time.sleep(2)

    def update_loop(self):
        start_time = time.time()
        try:
            win_x, win_y = self.frame_center.winfo_rootx(), self.frame_center.winfo_rooty()
            win_w, win_h = self.frame_center.winfo_width(), self.frame_center.winfo_height()
            
            if win_w > 10:
                monitor = {"top": int(win_y), "left": int(win_x), "width": int(win_w), "height": int(win_h)}
                img = np.array(self.sct.grab(monitor))

                rows, cols = self.config["grid_rows"], self.config["grid_cols"]
                ox, oy = self.config["offset_x"], self.config["offset_y"]
                sx, sy = self.config["spacing_x"], self.config["spacing_y"]
                rad = self.config["sample_radius"]
                zoom = self.config["zoom_factor"]
                thresh = self.config["color_threshold"]
                
                final_grid = np.zeros((rows, cols), dtype=np.float32)

                for r in range(rows):
                    for c in range(cols):
                        px = int((ox + c * sx) * zoom)
                        py = int((oy + r * sy) * zoom)
                        
                        if px-rad < 0 or py-rad < 0 or px+rad >= img.shape[1] or py+rad >= img.shape[0]:
                            continue

                        region = img[py-rad : py+rad+1, px-rad : px+rad+1, :3]
                        avg_bgra = np.mean(region, axis=(0,1))
                        avg_rgb = avg_bgra[::-1] 

                        # Food check
                        if np.linalg.norm(avg_rgb - self.food_vec) < thresh:
                            final_grid[r, c] = -1.0
                            continue
                            
                        # Body check
                        dist_head = np.linalg.norm(avg_rgb - self.head_vec)
                        dist_tail = np.linalg.norm(avg_rgb - self.tail_vec)
                        
                        if dist_head < thresh * 1.5 or dist_tail < thresh * 1.5:
                            val = self.calculate_gradient_value(avg_rgb)
                            final_grid[r, c] = val

                # Debug Probe
                d_px, d_py = int(ox * zoom), int(oy * zoom)
                if d_py-rad >= 0 and d_px-rad >= 0 and d_py+rad < img.shape[0] and d_px+rad < img.shape[1]:
                    region = img[d_py-rad:d_py+rad+1, d_px-rad:d_px+rad+1, :3]
                    if region.size > 0:
                        p = np.mean(region, axis=(0,1))[::-1].astype(int)
                        self.lbl_debug_rgb.config(text=f"{p[0]},{p[1]},{p[2]}", bg=f"#{p[0]:02x}{p[1]:02x}{p[2]:02x}")
                
                self.draw_model_view(final_grid)
                
                if self.config["auto_play"] and self.connected and self.sock:
                    try:
                        self.sock.sendall(final_grid.tobytes())
                        response = self.sock.recv(1)
                        if response:
                            action = int.from_bytes(response, 'big')
                            key = KEY_MAP[action]
                            pyautogui.press(key)
                            lat = int((time.time() - start_time) * 1000)
                            self.lbl_prediction.config(text=f"Action: {key.upper()}", fg="#00ff00")
                            self.lbl_latency.config(text=f"Ping: {lat}ms", fg="#00ff00" if lat < 40 else "#ff0000")
                    except Exception:
                        self.connected = False

        except Exception:
            pass
        
        self.root.after(15, self.update_loop)

    def draw_model_view(self, grid):
        self.canvas_vis.delete("all")
        rows, cols = grid.shape
        w = self.canvas_vis.winfo_width()
        h = self.canvas_vis.winfo_height()
        cw, ch = w/max(cols, 1), h/max(rows, 1)
        self.canvas_vis.create_rectangle(0,0,w,h, fill="black")
        for r in range(rows):
            for c in range(cols):
                val = grid[r, c]
                x1, y1 = c*cw, r*ch
                x2, y2 = (c+1)*cw, (r+1)*ch
                if val < -0.5: 
                    self.canvas_vis.create_oval(x1+2, y1+2, x2-2, y2-2, fill="red", outline="")
                elif val > 0.05:
                    b = int(val * 255)
                    color = f"#{0:02x}{b:02x}{b:02x}" 
                    self.canvas_vis.create_rectangle(x1+1, y1+1, x2-1, y2-1, fill=color, outline="")

if __name__ == "__main__":
    root = tk.Tk()
    app = ScreenTranslationApp(root)
    root.mainloop()
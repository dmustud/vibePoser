import os
import sys
import threading
import time
import itertools
import traceback
import warnings
import gc
import tkinter as tk
from tkinter import messagebox, PanedWindow, ttk, scrolledtext
import keyboard
import socket
import logging
import winsound
from logging.handlers import RotatingFileHandler

# --- Global Variables for Optional Modules ---
ARUCO_AVAILABLE = False
aruco = None

# --- Singleton Check ---
# Try to bind to a specific port to ensure only one instance is running
try:
    _lock_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    _lock_socket.bind(('127.0.0.1', 65432)) # Random unused port
except socket.error:
    # Another instance is already running
    root = tk.Tk()
    root.withdraw()
    messagebox.showwarning("Vibe Poser", "Vibe Poser가 이미 실행 중입니다.\n이미 열려 있는 창을 확인해 주세요.")
    sys.exit(0)

# --- Splash Screen ---
splash = tk.Tk()
splash.title("Vibe Poser Loading")
# Center the splash window
sw = splash.winfo_screenwidth()
sh = splash.winfo_screenheight()
ww, wh = 350, 150
x = (sw - ww) // 2
y = (sh - wh) // 2
splash.geometry(f"{ww}x{wh}+{x}+{y}")
splash.overrideredirect(True) # Remove title bar
splash.attributes("-topmost", True)
splash.configure(bg='#2c3e50')

frame = tk.Frame(splash, bg='#2c3e50', highlightthickness=2, highlightbackground='#34495e')
frame.pack(fill='both', expand=True)

label_title = tk.Label(frame, text="Vibe Poser", font=("Arial", 18, "bold"), fg="white", bg='#2c3e50', pady=10)
label_title.pack()

label_msg = tk.Label(frame, text="라이브러리를 불러오는 중입니다...\n잠시만 기다려 주세요 (약 10-20초)", font=("Arial", 10), fg="#ecf0f1", bg='#2c3e50')
label_msg.pack(pady=5)

style = ttk.Style(splash)
style.theme_use('clam')
style.configure("custom.Horizontal.TProgressbar", troughcolor='#34495e', bordercolor='#34495e', background='#3498db', lightcolor='#3498db', darkcolor='#3498db')

progress = ttk.Progressbar(frame, mode='indeterminate', length=280, style="custom.Horizontal.TProgressbar")
progress.pack(pady=10)
progress.start(15)

splash.update()

# Close splash screen after imports are finished
def perform_imports():
    global torch, cv2, np, trimesh, udp_client, Image, ImageTk, R_scipy, matplotlib, FigureCanvasTkAgg, plt, Axes3D, aruco, ARUCO_AVAILABLE
    
    import torch
    import cv2
    try:
        import cv2.aruco as aruco
        ARUCO_AVAILABLE = True
    except ImportError:
        ARUCO_AVAILABLE = False
        print("OpenCV ArUco module not found. Install opencv-contrib-python.")
    import numpy as np
    import trimesh 
    from pythonosc import udp_client
    from PIL import Image, ImageTk
    from scipy.spatial.transform import Rotation as R_scipy

    # Matplotlib Tkinter 백엔드
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    splash.quit()

# Start imports in a thread
threading.Thread(target=perform_imports, daemon=True).start()
splash.mainloop()

try:
    splash.destroy()
except:
    pass

# 경고 무시
warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --- Logging Setup ---
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

LOG_FILE = os.path.join(LOG_DIR, "vibe_poser.log")
# Rotating file handler: Max 10MB per file, keep 5 backups
file_handler = RotatingFileHandler(LOG_FILE, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8')
file_handler.setFormatter(logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s'))

logger = logging.getLogger("VibePoser")
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)

def get_vram_info():
    """Helper to get current VRAM usage if torch/cuda available."""
    try:
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024 ** 2)
            reserved = torch.cuda.memory_reserved() / (1024 ** 2)
            return f" | VRAM: {allocated:.1f}MB / {reserved:.1f}MB"
    except:
        pass
    return ""

def log_cmd(msg, level="INFO", app_instance=None):
    vram = get_vram_info()
    text = f"[{time.strftime('%H:%M:%S')}] [{level}] {msg}{vram}"
    
    # Log to file
    if level == "ERROR":
        logger.error(f"{msg}{vram}")
    elif level == "WARNING":
        logger.warning(f"{msg}{vram}")
    else:
        logger.info(f"{msg}{vram}")

    if app_instance and hasattr(app_instance, 'log_to_ui'):
        # If UI exists, send directly to UI
        app_instance.root.after(0, lambda: app_instance.log_to_ui(text))
    else:
        try:
            print(text)
        except:
            pass

# Redirect stdout/stderr for pythonw.exe
class LogRedirector:
    def __init__(self, callback, level="INFO"):
        self.callback = callback
        self.level = level
    def write(self, s):
        if s.strip():
            msg = s.strip()
            self.callback(msg)
            # Also log redirects to file
            if self.level == "ERROR":
                logger.error(f"[REDIRECT] {msg}")
            else:
                logger.info(f"[REDIRECT] {msg}")
    def flush(self):
        pass

# --- 경로 설정 ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)

SOURCE_CODE_DIR = os.path.join(CURRENT_DIR, "sam-3d-body")
if os.path.exists(SOURCE_CODE_DIR): sys.path.append(SOURCE_CODE_DIR)

MODEL_WEIGHTS_DIR = os.path.join(CURRENT_DIR, "sam-3d-body-dinov3")

MHR_DIR = os.path.join(CURRENT_DIR, "MHR-main")
if os.path.exists(MHR_DIR):
    sys.path.append(MHR_DIR)
    sys.path.append(os.path.join(MHR_DIR, "tools", "mhr_smpl_conversion"))

SMPL_DIR = os.path.join(CURRENT_DIR, "smplx")

MHR70_SKELETON_PAIRS = [
    (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6),
    (5, 6), (5, 9), (6, 10), (9, 10), (5, 7), (7, 62),
    (6, 8), (8, 41), (9, 11), (11, 13), (13, 17),
    (10, 12), (12, 14), (14, 20), (5, 69), (6, 69), (69, 0)
]

SMPL_SKELETON_PAIRS = [] # Deprecated, using parents from model

class DirectMHRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Vibe Poser")
        self.root.geometry("1250x900") 
        icon_path = os.path.join(CURRENT_DIR, "assets", "vibeposer-app.ico")
        if os.path.exists(icon_path):
            try:
                self.root.iconbitmap(icon_path)
            except Exception:
                pass
        
        # --- Handle pythonw.exe redirection ---
        if sys.stdout is None or sys.stderr is None:
            sys.stdout = LogRedirector(lambda msg: self.log_to_ui(f"[STDOUT] {msg}"), level="INFO")
            sys.stderr = LogRedirector(lambda msg: self.log_to_ui(f"[STDERR] {msg}"), level="ERROR")
        
        self.osc_client = udp_client.SimpleUDPClient("127.0.0.1", 9000)
        self.ui_thread_id = threading.get_ident()
        
        self.webcam_running = False
        self.cap = None
        self.webcam_thread = None
        self.webcam_generation = 0
        self.webcam_lock = threading.Lock()
        self.webcam_frame_pending = False
        self.current_image = None
        
        self.estimator = None
        self.processed_data = None
        
        self.mhr_model = None
        self.smplx_model = None
        self.converter = None
        self.smpl_result = None
        
        self.is_loading = True
        self.load_failed = False
        self.status_base_text = "시스템 초기화"
        
        self.orig_kpts_3d = None # Original MHR output
        self.orig_smpl_joints = None # Original SMPL joints
        self.view_init_done = False # Track if initial perspective is set
        self.cam_sync_lock = False # Prevent feedback loop
        self.cam_elev = -20.0 
        self.cam_azim = 130.0 
        self.cam_roll = 180.0
        
        self.hotkey_oneshot = "-" # Default hotkey for SMPL conversion button
        self.is_recording_hotkey = False
        self.hotkey_hook = None
        self.rotation_changed_flag = False
        self.osc_after_id = None
        self.rotation_dragging = False
        self.mesh_preview_face_stride = 8
        
        # Handle Close Window
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        self.setup_ui()
        self.zoom_factor = 1.0 # Scene zoom
        
        # Set Initial Proportions (1/2 width, 4/5 height for 3D)
        self.root.update() # Get actual dimensions
        try:
            w = self.root.winfo_width()
            h = self.root.winfo_height()
            # 1. Main horizontal split (mid)
            self.paned_main.sash_place(0, w // 2, 0)
            # 2. Right vertical split (4/5 for 3D, 1/5 for Log)
            usable_h = h - 60 
            self.paned_right.sash_place(0, 0, int(usable_h * 0.8))
            # 3. Left vertical split (1/2 for Webcam, 1/2 for 2D)
            self.paned_left.sash_place(0, 0, usable_h // 2)
        except:
            pass # Fail gracefully if sash not ready
            
        # Start loading in background thread to keep UI reactive
        threading.Thread(target=self.start_background_tasks, daemon=True).start()
        self.setup_hotkeys()

    def on_close(self):
        """Explicitly stop background threads and release resources on exit."""
        with self.webcam_lock:
            self.webcam_running = False
            self.webcam_generation += 1
            cap = self.cap
            self.cap = None
        if cap:
            try:
                cap.release()
            except:
                pass
        self.root.destroy()
        # Force terminate all processes (especially background loading threads)
        os._exit(0)

    def set_button_state(self, btn, state, normal_bg):
        """Helper to set button state and background color for better visibility."""
        if state == tk.DISABLED:
            btn.config(state=tk.DISABLED, bg="#d3d3d3") # Gray
        else:
            btn.config(state=tk.NORMAL, bg=normal_bg)

    def is_ui_thread(self):
        return threading.get_ident() == self.ui_thread_id

    def run_on_ui(self, callback, *args, delay=0):
        if self.is_ui_thread() and delay == 0:
            callback(*args)
        else:
            self.root.after(delay, lambda: callback(*args))

    def set_status(self, text=None, color=None):
        if text is not None:
            self.status_var.set(text)
        if color is not None:
            self.status_label.config(fg=color)

    def log_to_ui(self, msg):
        if not self.is_ui_thread():
            self.run_on_ui(self.log_to_ui, msg)
            return
        if hasattr(self, 'txt_log'):
            self.txt_log.config(state=tk.NORMAL)
            self.txt_log.insert(tk.END, msg + "\n")
            self.txt_log.see(tk.END)
            self.txt_log.config(state=tk.DISABLED)

    def setup_ui(self):
        control_frame = tk.Frame(self.root, pady=5, bg="#f0f0f0")
        control_frame.pack(side=tk.TOP, fill=tk.X)
        
        # --- Row 1: Execution Controls ---
        row1_frame = tk.Frame(control_frame, bg="#f0f0f0", pady=5)
        row1_frame.pack(side=tk.TOP, fill=tk.X)
        
        btn_font = ("Arial", 10, "bold")

        self.btn_extract = tk.Button(row1_frame, text="1. 포즈 추출", font=btn_font, command=self.btn_extract_click, bg="#e6fffa", state=tk.DISABLED)
        self.btn_extract.pack(side=tk.LEFT, padx=10)
        
        self.btn_convert = tk.Button(row1_frame, text="2. SMPL 변환 (+전송)", font=btn_font, command=self.btn_convert_click, bg="#fff9e6", state=tk.DISABLED)
        self.btn_convert.pack(side=tk.LEFT, padx=5)

        self.btn_send = tk.Button(row1_frame, text="3. OSC 전송", font=btn_font, command=self.btn_send_click, bg="#e6f2ff", state=tk.DISABLED)
        self.btn_send.pack(side=tk.LEFT, padx=5)
        
        tk.Label(row1_frame, text="|", bg="#f0f0f0").pack(side=tk.LEFT, padx=10)

        self.status_var = tk.StringVar(value="시스템 로딩 중...")
        self.status_label = tk.Label(row1_frame, textvariable=self.status_var, font=("Arial", 11, "bold"), fg="red", bg="#f0f0f0")
        self.status_label.pack(side=tk.LEFT, padx=20)

        # --- Row 2: Settings and Adjustments ---
        row2_frame = tk.Frame(control_frame, bg="#f0f0f0", pady=5)
        row2_frame.pack(side=tk.TOP, fill=tk.X)

        # Rotation Control Frame
        self.rotation_frame = tk.LabelFrame(row2_frame, text="회전 보정 (Rotation Adjustment)", bg="#f0f0f0", font=("Arial", 9, "bold"))
        self.rotation_frame.pack(side=tk.LEFT, padx=5, fill=tk.Y) # Reduced padding and removed expand=True

        def create_slider_group(parent, label_text, variable, str_var, col):
            f = tk.Frame(parent, bg="#f0f0f0")
            f.grid(row=0, column=col, padx=10, sticky="nsew")
            
            lbl_f = tk.Frame(f, bg="#f0f0f0")
            lbl_f.pack(side=tk.TOP, fill=tk.X)
            tk.Label(lbl_f, text=label_text, bg="#f0f0f0", font=("Arial", 8)).pack(side=tk.LEFT)
            tk.Label(lbl_f, textvariable=str_var, bg="#f0f0f0", font=("Arial", 8, "bold")).pack(side=tk.LEFT, padx=5)
            
            s = tk.Scale(f, from_=-180, to=180, orient=tk.HORIZONTAL, variable=variable, 
                         resolution=1, length=125, bg="#f0f0f0", font=("Arial", 8), showvalue=False, command=self.on_rotation_change)
            s.bind("<ButtonPress-1>", self.on_rotation_drag_start)
            s.bind("<ButtonRelease-1>", self.on_rotation_drag_end)
            s.pack(side=tk.TOP, fill=tk.X, expand=True)
            return f

        self.pitch_var = tk.DoubleVar(value=0.0)
        self.yaw_var = tk.DoubleVar(value=0.0)
        self.roll_var = tk.DoubleVar(value=0.0)
        
        self.pitch_str = tk.StringVar(value="0")
        self.yaw_str = tk.StringVar(value="0")
        self.roll_str = tk.StringVar(value="0")

        create_slider_group(self.rotation_frame, "Pitch(X):", self.pitch_var, self.pitch_str, 0)
        create_slider_group(self.rotation_frame, "Yaw(Y):", self.yaw_var, self.yaw_str, 1)
        create_slider_group(self.rotation_frame, "Roll(Z):", self.roll_var, self.roll_str, 2)

        # ArUco Calibration Button (Simplified)
        self.btn_calib = tk.Button(self.rotation_frame, text="바닥 보정 (마커)", command=self.calibrate_floor, font=("Arial", 8))
        self.btn_calib.grid(row=0, column=3, padx=15, pady=5)
        
        self.hand_mode_var = tk.BooleanVar(value=False)
        self.chk_hand_mode = tk.Checkbutton(row2_frame, text="손 모드 (Hands Only)", variable=self.hand_mode_var, 
                                            font=("Arial", 10), bg="#f0f0f0", activebackground="#f0f0f0")
        self.chk_hand_mode.pack(side=tk.LEFT, padx=5)
        
        # Mesh Preview Toggle
        self.show_mesh_var = tk.BooleanVar(value=False)
        self.chk_show_mesh = tk.Checkbutton(row2_frame, text="메시 보기 (Light)", variable=self.show_mesh_var,
                                            font=("Arial", 10), bg="#f0f0f0", activebackground="#f0f0f0", command=self.on_rotation_change)
        self.chk_show_mesh.pack(side=tk.LEFT, padx=5)

        tk.Label(row2_frame, text="| 핫키:", bg="#f0f0f0").pack(side=tk.LEFT, padx=5)
        self.btn_hotkey = tk.Button(row2_frame, text=self.hotkey_oneshot, font=("Arial", 9), command=self.btn_hotkey_click, bg="#f9f9f9", width=12)
        self.btn_hotkey.pack(side=tk.LEFT, padx=5)

        tk.Label(row2_frame, text="| 캠:", bg="#f0f0f0").pack(side=tk.LEFT, padx=5)
        self.cam_var = tk.StringVar()
        self.cam_combo = ttk.Combobox(row2_frame, textvariable=self.cam_var, state="readonly", width=10)
        self.cam_combo.pack(side=tk.LEFT, padx=5)
        self.cam_combo.bind("<<ComboboxSelected>>", self.on_camera_select)

        # Camera View UI removed as requested

        # --- Hidden Configuration Variables (Hardcoded Defaults) ---
        self.off_x = 0.0
        self.off_y = 0.0
        self.off_z = 0.0
        self.map_x, self.map_y, self.map_z = "X", "Y", "Z"
        self.inv_x, self.inv_y, self.inv_z = False, True, False
        self.fix_xy = True

        # --- UI Layout: Main Split (Left/Right) ---
        self.paned_main = PanedWindow(self.root, orient=tk.HORIZONTAL, bg="#d9d9d9", sashwidth=6, sashrelief=tk.RAISED)
        self.paned_main.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left Side (Vertical Split: Webcam / 2D)
        self.paned_left = PanedWindow(self.paned_main, orient=tk.VERTICAL, bg="#d9d9d9", sashwidth=6, sashrelief=tk.RAISED)
        # Use a weight or specific width/height to encourage 50/50
        self.paned_main.add(self.paned_left, minsize=100) 

        self.frame_webcam = tk.Frame(self.paned_left, bg="black")
        self.lbl_webcam = tk.Label(self.frame_webcam, text="Webcam Input", bg="black", fg="white")
        self.lbl_webcam.pack(fill=tk.BOTH, expand=True)
        self.paned_left.add(self.frame_webcam, minsize=200)

        self.frame_2d = tk.Frame(self.paned_left, bg="#222")
        self.lbl_result = tk.Label(self.frame_2d, text="Estimation Result", bg="#222", fg="white")
        self.lbl_result.pack(fill=tk.BOTH, expand=True)
        self.paned_left.add(self.frame_2d, minsize=200)

        # Right Side (3D Preview + Terminal)
        self.paned_right = tk.PanedWindow(self.paned_main, orient=tk.VERTICAL, bg="#ccc", sashwidth=6)
        self.paned_main.add(self.paned_right, minsize=100)

        # Top-Right: 3D Preview
        self.frame_3d = tk.Frame(self.paned_right, bg="white")
        self.paned_right.add(self.frame_3d, minsize=200)
        
        # Bottom-Right: System Logs (Terminal)
        self.frame_log = tk.Frame(self.paned_right, bg="#1e1e1e")
        # Header "System Logs" removed as requested
        self.txt_log = scrolledtext.ScrolledText(self.frame_log, bg="#1e1e1e", fg="white", font=("Consolas", 8), height=4, state=tk.DISABLED)
        self.txt_log.pack(fill=tk.BOTH, expand=True)
        self.paned_right.add(self.frame_log, minsize=60)
        
        self.fig = plt.figure(figsize=(4, 4))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_box_aspect([1,1,1])
        self.ax.set_proj_type('persp')
        if not self.ax.yaxis_inverted():
            self.ax.invert_yaxis()

        self.ground_transform = None # Store 4x4 calibration matrix
        
        # UI Polish: Figure Adjustments for "Full Frame"
        self.fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        self.ax.grid(True)
        try:
            self.ax.view_init(elev=self.cam_elev, azim=self.cam_azim, roll=self.cam_roll)
        except TypeError:
            self.ax.view_init(elev=self.cam_elev, azim=self.cam_azim)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame_3d)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Bind mouse wheel for zooming
        self.canvas.get_tk_widget().bind("<MouseWheel>", self.on_mouse_wheel)
        # Connect motion event to sync sliders if user rotates manually
        self.ax.figure.canvas.mpl_connect('button_release_event', self.on_mpl_view_change)

    def on_camera_change(self, *args):
        pass # Removed

    def on_mpl_view_change(self, event):
        """Track camera state even if UI is gone (Internal use/Zoom stabilization)."""
        if event.inaxes != self.ax: return
        self.cam_elev = self.ax.elev
        self.cam_azim = self.ax.azim
        if hasattr(self.ax, 'roll'):
            self.cam_roll = self.ax.roll

    def start_background_tasks(self):
        # Detect available cameras
        cams = self.list_available_cameras()
        if cams:
            self.run_on_ui(self.apply_camera_list, cams)
        else:
            log_cmd("No cameras detected!", "ERROR", app_instance=self)
            self.run_on_ui(self.apply_no_camera)

        threading.Thread(target=self.init_models, daemon=True).start()
        threading.Thread(target=self.loading_animation, daemon=True).start()

    def apply_camera_list(self, cams):
        self.cam_combo['values'] = [f"Camera {i}" for i in cams]
        self.cam_combo.current(0)
        self.start_webcam(cams[0])

    def apply_no_camera(self):
        self.cam_combo['values'] = ["No Camera"]
        self.cam_combo.current(0)

    def list_available_cameras(self, max_to_test=5):
        available_indices = []
        for i in range(max_to_test):
            temp_cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if temp_cap.isOpened():
                available_indices.append(i)
                temp_cap.release()
        return available_indices

    def on_camera_select(self, event=None):
        selection = self.cam_var.get()
        if "Camera" in selection:
            idx = int(selection.split(" ")[1])
            log_cmd(f"Switching to Camera {idx}...", "INFO", app_instance=self)
            self.start_webcam(idx)

    def auto_level_pose(self):
        pass # Deprecated

    def calibrate_floor(self):
        """Detect ArUco marker (ID 0) and set ground plane transformation."""
        if not ARUCO_AVAILABLE:
            messagebox.showerror("Error", "OpenCV ArUco 모듈이 필요합니다.\npip install opencv-contrib-python")
            return
            
        if self.current_image is None:
            messagebox.showwarning("Warning", "카메라 화면이 없습니다.")
            return

        gray = cv2.cvtColor(self.current_image, cv2.COLOR_RGB2GRAY)
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        parameters = aruco.DetectorParameters()
        
        # Detect
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        
        if ids is not None and 0 in ids:
            # Index of ID 0
            idx = list(ids).index(0)
            corner_0 = corners[idx] # [1, 4, 2]
            
            # Approximate Intrinsic Matrix (w/o calibration)
            h, w = gray.shape
            focal_length = w # Typical approximation
            center = (w/2, h/2)
            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype=np.float32)
            dist_coeffs = np.zeros(5) # Assume no distortion
            
            # Estimate Pose
            marker_length = 0.20 # 20cm (Arbitrary, affects scale but not rotation)
            # Actually, scale matters for translation Z (height). 
            # If we don't know the real marker size, height estimation will be off-scale.
            # But the rotation will be correct.
            # Let's assume standard A4 printed marker around 15-20cm? 
            # Doesn't strictly matter for alignment if we just want "Zero" to be ON the marker.
            
            # EstimatePoseSingleMarkers
            rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corner_0, marker_length, camera_matrix, dist_coeffs)
            rvec = rvec[0][0]
            tvec = tvec[0][0]
            
            # Draw axis for visual feedback
            vis = self.current_image.copy()
            cv2.drawFrameAxes(vis, camera_matrix, dist_coeffs, rvec, tvec, 0.1)
            self.display_image(vis, self.lbl_result)
            
            # Compute Transformation Matrix (Camera -> Marker)
            R, _ = cv2.Rodrigues(rvec)
            
            # Extract Euler Angles (xyz)
            euler = R_scipy.from_matrix(R).as_euler('xyz', degrees=True)
            
            # --- Hardcoded "Golden" Logic ---
            # Axis Mapping: P=X(0), Y=Z(2), R=Y(1)
            # Inversions: All Inverted (*-1)
            
            p_val = (euler[0] * -1) + 90.0
            y_val = (euler[2] * -1)
            r_val = (euler[1] * -1)
            
            # Normalize to [-180, 180]
            def norm(a):
                while a > 180: a -= 360
                while a < -180: a += 360
                return a
            
            target_pitch = norm(p_val)
            target_yaw = norm(y_val)
            target_roll = norm(r_val)
            
            self.pitch_var.set(target_pitch)
            self.yaw_var.set(target_yaw)
            self.roll_var.set(target_roll)
            
            log_cmd(f"바닥 보정 완료 (P={target_pitch:.1f}, Y={target_yaw:.1f}, R={target_roll:.1f})", "SUCCESS", app_instance=self)
            
            # Apply immediately to preview
            self.on_rotation_change()
            
            # Store full transform for height correction
            T_cam_to_marker_space = np.eye(4)
            T_cam_to_marker_space[:3, :3] = R
            T_cam_to_marker_space[:3, 3] = tvec
            self.ground_transform = np.linalg.inv(T_cam_to_marker_space)
            
        else:
            log_cmd("마커(ID 0)를 찾을 수 없습니다.", "WARNING", app_instance=self)

    def loading_animation(self):
        chars = itertools.cycle(['|', '/', '-', '\\'])
        while self.is_loading:
            try:
                self.run_on_ui(self.status_var.set, f"{self.status_base_text} {next(chars)}")
                time.sleep(0.1)
            except: break

    def init_models(self):
        try:
            log_cmd("모델 초기화 시작 (온디맨드 모드)...", "INFO", app_instance=self)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            log_cmd(f"디바이스: {device}", "INFO", app_instance=self)

            import smplx
            smplx_path = os.path.join(SMPL_DIR, "SMPLX_NEUTRAL.npz")
            smpl_pkl_path = os.path.join(SMPL_DIR, "SMPL_NEUTRAL.pkl")
            
            self.smplx_model = None
            
            # Attempt 1: SMPLX
            if os.path.exists(smplx_path):
                 try:
                     self.smplx_model = smplx.SMPLX(
                        model_path=SMPL_DIR,
                        model_type='smplx',
                        gender='neutral',
                        use_pca=False,
                        flat_hand_mean=True
                    ).to(device)
                     log_cmd("SMPL-X 모델 로드 성공", "INFO", app_instance=self)
                 except Exception as e:
                     log_cmd(f"SMPL-X 로드 실패 ({e}), SMPL 폴백 시도...", "WARNING", app_instance=self)
            
            # Attempt 2: SMPL (Fallback)
            if self.smplx_model is None:
                if os.path.exists(smpl_pkl_path):
                     try:
                         self.smplx_model = smplx.create(
                             model_path=smpl_pkl_path,
                             model_type='smpl',
                             gender='neutral'
                         ).to(device)
                         log_cmd("SMPL 모델 (Fallback) 로드 성공", "INFO", app_instance=self)
                     except Exception as e2:
                        log_cmd(f"SMPL 로드 실패: {e2}", "ERROR", app_instance=self)
            
            if self.smplx_model is None:
                log_cmd("SMPL 모델 로드 실패", "FATAL", app_instance=self)

        except Exception as e:
            log_cmd(f"초기화 에러: {e}", "FATAL", app_instance=self)
            traceback.print_exc()
            self.load_failed = True
        finally:
            self.is_loading = False
            self.update_ui_state()

    def update_ui_state(self):
        if self.load_failed:
            self.run_on_ui(self.set_status, "로딩 실패 (로그 확인)", "red")
        elif self.smplx_model:
            self.run_on_ui(self.set_status, "준비완료 (온디맨드)", "green")
            self.run_on_ui(self.enable_all_buttons)
        else:
            self.run_on_ui(self.set_status, "준비완료 (변환 불가)", "orange")
            self.run_on_ui(lambda: self.btn_extract.config(state=tk.NORMAL))

    def enable_all_buttons(self):
        self.set_button_state(self.btn_extract, tk.NORMAL, "#e6fffa")
        self.set_button_state(self.btn_convert, tk.NORMAL, "#fff9e6")
        self.set_button_state(self.btn_send, tk.NORMAL, "#e6f2ff")

    def disable_all_buttons(self):
        for btn in [self.btn_extract, self.btn_convert, self.btn_send]:
            self.set_button_state(btn, tk.DISABLED, btn.cget('bg'))

    def get_model_device(self):
        try:
            return next(self.smplx_model.parameters()).device
        except Exception:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def sync_cuda_if_needed(self):
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass

    def perf_now(self):
        self.sync_cuda_if_needed()
        return time.perf_counter()

    def log_perf(self, label, start_time):
        elapsed = time.perf_counter() - start_time
        log_cmd(f"[PERF] {label}: {elapsed:.3f}s", "INFO", app_instance=self)
        return self.perf_now()

    def cache_smpl_result_on_cpu(self, result_parameters):
        cached = {}
        for k, v in result_parameters.items():
            if isinstance(v, torch.Tensor):
                cached[k] = v.detach().cpu().clone()
            elif isinstance(v, np.ndarray):
                cached[k] = v.copy()
            else:
                cached[k] = v
        return cached

    def smpl_result_to_device(self, device):
        params = {}
        for k, v in self.smpl_result.items():
            if isinstance(v, torch.Tensor):
                params[k] = v.to(device)
            elif isinstance(v, np.ndarray):
                params[k] = torch.from_numpy(v).float().to(device)
            else:
                params[k] = v
        return params

    def smpl_result_to_numpy(self):
        data = {}
        for k, v in self.smpl_result.items():
            if isinstance(v, torch.Tensor):
                data[k] = v.detach().cpu().numpy()[0]
            elif isinstance(v, np.ndarray):
                data[k] = v[0]
            else:
                data[k] = v
        return data

    def btn_extract_click(self):
        self.disable_all_buttons()
        self.status_base_text = "1. 포즈 추출 중"
        self.is_loading = True
        self.run_on_ui(self.set_status, None, "orange")
        threading.Thread(target=self.loading_animation, daemon=True).start()
        threading.Thread(target=self.process_pose, daemon=True).start()

    def btn_convert_click(self):
        self.disable_all_buttons()
        self.status_base_text = "2. SMPL 변환 중 0%"
        self.is_loading = True
        self.run_on_ui(self.set_status, None, "orange")
        threading.Thread(target=self.loading_animation, daemon=True).start()
        threading.Thread(target=self.process_conversion, daemon=True).start()

    def btn_send_click(self):
        ui_state = self.get_osc_ui_state()
        threading.Thread(target=self.send_osc_data, args=(ui_state,), daemon=True).start()

    def btn_oneshot_click(self):
        self.btn_convert_click()

    def run_oneshot(self):
        # Redirect the old oneshot logic to the new combined button 2 logic
        self.run_on_ui(self.btn_convert_click)

    def notify_conversion_complete(self):
        """Play a short completion sound after SMPL conversion finishes."""
        try:
            winsound.MessageBeep(winsound.MB_ICONASTERISK)
        except Exception:
            try:
                self.root.bell()
            except Exception:
                pass

    def process_pose(self, silent=False):
        image = self.current_image.copy() if self.current_image is not None else None
        if image is None:
            self.is_loading = False
            self.run_on_ui(self.enable_all_buttons, delay=1000)
            return False
            
        try:
            # Load estimator only once and keep it in memory
            if self.estimator is None:
                if not silent: log_cmd("SAM3DBody 모델 최초 로딩 중... (메모리 유지)", "INFO", app_instance=self)
                
                os.environ["MOMENTUM_ENABLED"] = "0"
                from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
                ckpt = os.path.join(MODEL_WEIGHTS_DIR, "model.ckpt")
                mhr_pt = os.path.join(MODEL_WEIGHTS_DIR, "mhr_model.pt")
                
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model, cfg = load_sam_3d_body(ckpt, device=device, mhr_path=mhr_pt)
                
                # Extract faces for MHR initialization later
                if hasattr(model, 'head_pose') and hasattr(model.head_pose, 'faces'):
                    # Must detach, move to CPU, and convert to numpy to break the graph completely.
                    self.mhr_faces_tensor = model.head_pose.faces.detach().cpu().clone()
                    self.mhr_faces = self.mhr_faces_tensor.numpy()
                else:
                    self.mhr_faces_tensor = None
                    self.mhr_faces = None
                    
                self.estimator = SAM3DBodyEstimator(model, cfg, None, None, None)
            
            if not silent: log_cmd("포즈 추론 중...", "INFO", app_instance=self)
            outputs = self.estimator.process_one_image(image, bbox_thr=0.5, use_mask=False)
            
            if not outputs:
                if not silent: log_cmd("사람을 감지하지 못했습니다.", "WARNING", app_instance=self)
                self.is_loading = False
                return False
                
            self.processed_data = outputs[0]
            self.smpl_result = None 
            
            vis = image.copy()
            for i1, i2 in MHR70_SKELETON_PAIRS:
                k = self.processed_data['pred_keypoints_2d']
                if i1 < len(k) and i2 < len(k):
                    cv2.line(vis, (int(k[i1][0]), int(k[i1][1])), (int(k[i2][0]), int(k[i2][1])), (0,255,0), 2)
            self.display_image(vis, self.lbl_result)
            
            if 'pred_keypoints_3d' in self.processed_data:
                self.orig_kpts_3d = self.processed_data['pred_keypoints_3d'].copy()
                self.orig_smpl_joints = None # Clear old SMPL joints
                
                # Store MHR Vertices for Preview (Pre-SMPL)
                if 'pred_vertices' in self.processed_data:
                    self.orig_mhr_verts = self.processed_data['pred_vertices'] # Full array (V, 3)
                else:
                    self.orig_mhr_verts = None
                
                # Apply current rotation sliders immediately (Reset View for new Pose)
                self.run_on_ui(lambda: self.on_rotation_change(reset_view=True, auto_send=False))
            
            # --- Break reference to raw outputs to free VRAM immediately ---
            del outputs
            
            # --- Stop Loading Animation ---
            self.is_loading = False
            
            if not silent:
                self.run_on_ui(self.set_status, "포즈 추출 완료", "green")
                self.run_on_ui(self.enable_all_buttons)
            
            return True
        except Exception as e:
            self.is_loading = False
            log_cmd(f"추론 에러: {e}", "ERROR")
            self.run_on_ui(self.enable_all_buttons)
            return False
        finally:
            self.is_loading = False # Ensure loading animation stops

    def process_conversion(self, silent=False):
        # Auto-run pose extraction if no data exists
        if not self.processed_data:
            if not silent: log_cmd("포즈 데이터가 없어 자동 추출을 먼저 진행합니다.", "INFO", app_instance=self)
            self.status_base_text = "1. 포즈 자동 추출 중"
            self.run_on_ui(self.set_status, None, "orange")
            success = self.process_pose(silent=True)
            if not success:
                self.is_loading = False
                self.run_on_ui(self.enable_all_buttons, delay=1000)
                return False
            
            self.status_base_text = "포즈 자동 추출 완료 -> SMPL 변환 중 0%"
            self.run_on_ui(self.set_status, "포즈 자동 추출 완료 -> SMPL 변환 중 0%", "orange")

        if not self.processed_data or not self.smplx_model:
            if not silent: log_cmd("데이터 없음 / SMPL 모델 없음", "ERROR", app_instance=self)
            self.is_loading = False
            self.run_on_ui(self.enable_all_buttons, delay=1000)
            return False
            
        if not hasattr(self, 'mhr_model') or self.mhr_model is None:
            try:
                self.run_on_ui(self.disable_all_buttons)
                if not silent: log_cmd("MHR 변환기 로딩 중... (최초 1회만, 메모리 유지)", "INFO", app_instance=self)
                
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                mhr_pt = os.path.join(MODEL_WEIGHTS_DIR, "mhr_model.pt")
                from mhr.mhr import MHR 
                from conversion import Conversion
                
                if hasattr(self, 'mhr_faces_tensor') and self.mhr_faces_tensor is not None:
                     self.mhr_model = MHR.from_jit(jit_path=mhr_pt, faces=self.mhr_faces_tensor.to(device), device=device)
                else:
                     self.mhr_model = MHR.from_files(lod=1, device=device)
                     
                self.converter = Conversion(mhr_model=self.mhr_model, smpl_model=self.smplx_model, method="pytorch")
            except Exception as e:
                log_cmd(f"MHR 로드 에러: {e}", "ERROR", app_instance=self)
                self.is_loading = False
                self.run_on_ui(self.enable_all_buttons, delay=1000)
                return False
        else:
            self.run_on_ui(self.disable_all_buttons)
            if not silent: log_cmd("캐시된 MHR 변환기 사용 중...", "INFO", app_instance=self)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            
            if not silent: log_cmd("SMPL 변환 시작...", "INFO", app_instance=self)
            total_t = self.perf_now()
            self.status_base_text = "2. SMPL 변환 중 0%"
            self.run_on_ui(self.set_status, "2. SMPL 변환 중 0%", None)
            
            step_t = self.perf_now()
            verts = torch.from_numpy(self.processed_data['pred_vertices']).float().to(device).unsqueeze(0)
            verts_cm = verts * 100.0 
            step_t = self.log_perf("SMPL convert/input vertices to device", step_t)
            
            def progress_cb(p):
                try:
                    progress_value = float(p)
                    if progress_value <= 1.0:
                        progress_value *= 100.0
                    progress_percent = max(0, min(100, int(round(progress_value))))
                except Exception:
                    progress_percent = 0
                self.status_base_text = f"2. SMPL 변환 중 {progress_percent}%"
                self.run_on_ui(self.set_status, f"2. SMPL 변환 중 {progress_percent}%", "orange")

            result = self.converter.convert_mhr2smpl(
                mhr_vertices=verts_cm,
                return_smpl_parameters=True,
                return_smpl_meshes=False,
                single_identity=True,
                batch_size=1,
                progress_callback=progress_cb
            )
            step_t = self.log_perf("SMPL convert/convert_mhr2smpl", step_t)
            
            # Keep converted parameters on CPU so the persistent cache does not pin extra VRAM.
            self.smpl_result = self.cache_smpl_result_on_cpu(result.result_parameters)
            step_t = self.log_perf("SMPL convert/cache result on CPU", step_t)
                    
            log_cmd("SMPL 파라미터 생성 완료!", "SUCCESS", app_instance=self)
            
            # Reset UI to ready
            self.run_on_ui(self.set_status, "2. SMPL 변환 완료", "green")
            self.run_on_ui(self.notify_conversion_complete)
            self.run_on_ui(self.enable_all_buttons)
            
            # --- Update 3D Preview with SMPL Joints ---
            with torch.no_grad():
                smpl_params = self.smpl_result_to_device(device)
                step_t = self.log_perf("SMPL convert/params back to device for preview", step_t)
                smpl_out = self.smplx_model(**smpl_params)
                step_t = self.log_perf("SMPL convert/smplx preview forward", step_t)
                joints = smpl_out.joints.detach().cpu().numpy()[0] # [J, 3]
                self.orig_smpl_joints = joints.copy()
                
                # Store Vertices for Mesh Preview
                if hasattr(smpl_out, 'vertices'):
                    self.orig_smpl_verts = smpl_out.vertices.detach().cpu().numpy()[0] # [V, 3]
                else:
                    self.orig_smpl_verts = None
                
                self.run_on_ui(lambda: self.on_rotation_change(reset_view=True, auto_send=False))
                step_t = self.log_perf("SMPL convert/preview arrays to CPU", step_t)
                
            # Auto OSC send
            self.send_osc_data()
            step_t = self.log_perf("SMPL convert/auto OSC send", step_t)
            
            # 🛑 MEMORY LEAK FIX: Clear huge processed_data dictionaries 
            # after conversion since all SMPL/3D visuals are now independently Cached.
            if hasattr(self, 'processed_data') and self.processed_data is not None:
                del self.processed_data
                self.processed_data = None

            del result, verts, verts_cm, smpl_params, smpl_out
            self.log_perf("SMPL convert/total", total_t)
            
            self.is_loading = False
            return True
        
        except Exception as e:
            self.is_loading = False
            self.run_on_ui(self.enable_all_buttons)
            log_cmd(f"변환 에러: {e}", "ERROR", app_instance=self)
            traceback.print_exc()
            return False
        finally:
            self.run_on_ui(self.enable_all_buttons)
            
            # --- MEMORY LEAK FIX BY PINNING MODELS ---
            # Do NOT delete self.estimator (SAM3D) or self.mhr_model (MHR)!
            # DINOv3 / PyTorch Hub leaks ~665MB RAM/VRAM permanently every time you load and delete it 
            # within the same python process. By just retaining them globally, we prevent the leak.
            # Base VRAM will hit ~5GB-6GB but it will stay flat instead of growing indefinitely.
                
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            import gc
            gc.collect()
            
            if not silent: log_cmd("변환 완료 (모델은 메모리에 유지되어 누수를 방지합니다)", "INFO", app_instance=self)
            
    # --- Hotkey Management ---
    def setup_hotkeys(self):
        """Register the global hotkey."""
        try:
            if self.hotkey_hook:
                keyboard.remove_hotkey(self.hotkey_hook)
            self.hotkey_hook = keyboard.add_hotkey(self.hotkey_oneshot, self.trigger_oneshot_from_hotkey)
        except Exception as e:
            log_cmd(f"핫키 등록 실패: {e}", "ERROR", app_instance=self)

    def trigger_oneshot_from_hotkey(self):
        """Callback for the global hotkey."""
        if not self.is_ui_thread():
            self.run_on_ui(self.trigger_oneshot_from_hotkey)
            return

        # Check if buttons are enabled (not already processing)
        if self.btn_convert['state'] == tk.NORMAL:
            log_cmd(f"핫키 ({self.hotkey_oneshot}) 입력됨", "INFO", app_instance=self)
            self.run_on_ui(self.btn_convert_click)

    def btn_hotkey_click(self):
        """Start recording a new hotkey."""
        if self.is_recording_hotkey: return
        
        self.is_recording_hotkey = True
        self.btn_hotkey.config(text="키 입력 대기...", bg="#fff3cd")
        
        # Listen for the next key press
        threading.Thread(target=self.record_hotkey, daemon=True).start()

    def record_hotkey(self):
        """Wait for a key press and update the hotkey."""
        try:
            # block until a key is pressed
            key_event = keyboard.read_event(suppress=True)
            if key_event.event_type == keyboard.KEY_DOWN:
                new_key = key_event.name
                log_cmd(f"새 핫키 감지: {new_key}", "INFO", app_instance=self)
                self.hotkey_oneshot = new_key
                self.run_on_ui(self.update_hotkey_ui)
                self.setup_hotkeys()
        finally:
            self.is_recording_hotkey = False

    def update_hotkey_ui(self):
        self.btn_hotkey.config(text=self.hotkey_oneshot, bg="#f9f9f9")

    def on_rotation_drag_start(self, event=None):
        self.rotation_dragging = True

    def on_rotation_drag_end(self, event=None):
        self.rotation_dragging = False
        self.on_rotation_change(reset_view=False)

    def schedule_osc_send(self, delay_ms=150):
        if self.osc_after_id is not None:
            try:
                self.root.after_cancel(self.osc_after_id)
            except Exception:
                pass
        self.osc_after_id = self.root.after(delay_ms, self.flush_scheduled_osc)

    def flush_scheduled_osc(self):
        self.osc_after_id = None
        ui_state = self.get_osc_ui_state()
        threading.Thread(target=self.send_osc_data, args=(ui_state,), daemon=True).start()

    def get_osc_ui_state(self):
        return {
            "pitch": self.pitch_var.get(),
            "yaw": self.yaw_var.get(),
            "roll": self.roll_var.get(),
            "hand_mode": self.hand_mode_var.get(),
        }

    def on_rotation_change(self, *args, reset_view=False, auto_send=True):
        """Callback to update 3D plot in real-time when sliders move."""
        if not self.is_ui_thread():
            self.run_on_ui(lambda: self.on_rotation_change(*args, reset_view=reset_view, auto_send=auto_send))
            return
        try:
            # Update Integer Labels
            self.pitch_str.set(str(int(self.pitch_var.get())))
            self.yaw_str.set(str(int(self.yaw_var.get())))
            self.roll_str.set(str(int(self.roll_var.get())))

            rx = self.pitch_var.get() - 90.0 
            ry = self.yaw_var.get()
            rz = self.roll_var.get()
            # Revert to xyz order (cancel swap) and apply stoj offset for preview
            r_offset = R_scipy.from_euler('xyz', [rx, ry, rz], degrees=True)

            # --- Automatic OSC Transmission on Slider Change ---
            if auto_send and self.smpl_result is not None:
                self.schedule_osc_send()

            # Case A: We have SMPL joints (Prioritize for precision)
            if self.orig_smpl_joints is not None:
                # Rotate around pelvis (root)
                pelvis = self.orig_smpl_joints[0]
                rotated_joints = (r_offset.as_matrix() @ (self.orig_smpl_joints - pelvis).T).T + pelvis
                
                rotated_verts = None
                if self.show_mesh_var.get() and not self.rotation_dragging and hasattr(self, 'orig_smpl_verts') and self.orig_smpl_verts is not None:
                    rotated_verts = (r_offset.as_matrix() @ (self.orig_smpl_verts - pelvis).T).T + pelvis
                
                self.update_3d_plot(rotated_joints, is_smpl=True, verts=rotated_verts)
                return

            # Case B: We only have MHR 3D keypoints
            if self.orig_kpts_3d is not None:
                # Rotate MHR joints
                pelvis = self.orig_kpts_3d[0] # Approximated root
                rotated_kpts = (r_offset.as_matrix() @ (self.orig_kpts_3d - pelvis).T).T + pelvis
                
                rotated_verts = None
                if self.show_mesh_var.get() and not self.rotation_dragging and hasattr(self, 'orig_mhr_verts') and self.orig_mhr_verts is not None:
                    rotated_verts = (r_offset.as_matrix() @ (self.orig_mhr_verts - pelvis).T).T + pelvis
                    
                self.update_3d_plot(rotated_kpts, is_smpl=False, verts=rotated_verts, update_limits=reset_view)

        except Exception as e:
            print(f"Rotation/Plot Error: {e}")
            traceback.print_exc()
            pass # Avoid spamming errors during slider move

    def send_osc_data(self, ui_state=None):
        if not self.smpl_result: return
        if ui_state is None:
            if not self.is_ui_thread():
                self.run_on_ui(lambda: threading.Thread(target=self.send_osc_data, args=(self.get_osc_ui_state(),), daemon=True).start())
                return
            ui_state = self.get_osc_ui_state()

        log_cmd("OSC 전송 준비 (배열 통합 방식)...", "INFO", app_instance=self)
        try:
            total_t = self.perf_now()
            step_t = total_t
            device = self.get_model_device()
            data = self.smpl_result_to_numpy()
            step_t = self.log_perf("OSC/prepare numpy data", step_t)
            
            # --- 변환 행렬 P (Unreal 좌표계) ---
            axis_indices = {'X': 0, 'Y': 1, 'Z': 2}
            P = np.zeros((3, 3))
            P[0, axis_indices[self.map_x]] = -1.0 if self.inv_x else 1.0
            P[1, axis_indices[self.map_y]] = -1.0 if self.inv_y else 1.0
            P[2, axis_indices[self.map_z]] = -1.0 if self.inv_z else 1.0

            # --- 회전 보정 (UI 슬라이더) ---
            orig_global_orient = data['global_orient'].copy()
            rx = ui_state["pitch"] - 90.0
            ry = ui_state["yaw"]
            rz = ui_state["roll"]
            
            r_offset = R_scipy.from_euler('xyz', [rx, ry, rz], degrees=True)
            r_global = r_offset * R_scipy.from_rotvec(orig_global_orient)
            data['global_orient'] = r_global.as_rotvec()

            # --- Unreal 바닥 접지(Grounding) 계산 ---
            with torch.no_grad():
                 smpl_params = self.smpl_result_to_device(device)
                 smpl_out = self.smplx_model(
                     betas=smpl_params.get('betas'),
                     body_pose=smpl_params.get('body_pose'),
                     global_orient=torch.from_numpy(data['global_orient']).float().to(device).unsqueeze(0),
                     transl=smpl_params.get('transl')
                 )
                 joints_unreal = (P @ (smpl_out.joints[0].cpu().numpy().T * 100.0)).T
                 pel_unreal = joints_unreal[0].copy()
                 min_z_unreal = np.min(joints_unreal[:, 2])
                 pel_unreal[2] -= min_z_unreal
                 del smpl_params, smpl_out
                 step_t = self.log_perf("OSC/smplx grounding forward", step_t)

            # ==========================================
            # [수정된 핵심 로직] 한 번에 보낼 문자열 리스트
            # ==========================================
            all_pose_strings =[]

            # (기존 send_combined 대신, 텍스트를 만들어 리스트에 넣는 함수)
            def add_to_payload(name, axis_angle, transl_unreal=None):
                r_orig = R_scipy.from_rotvec(axis_angle)
                mat_unreal = P @ r_orig.as_matrix() @ P.T
                quat_unreal = R_scipy.from_matrix(mat_unreal).as_quat()
                
                if transl_unreal is not None:
                    tx = transl_unreal[0] + self.off_x
                    ty = transl_unreal[1] + self.off_y
                    tz = transl_unreal[2] + self.off_z
                    if self.fix_xy:
                        tx = 0.0 + self.off_x
                        ty = 0.0 + self.off_y
                else:
                    tx, ty, tz = 0.0, 0.0, 0.0

                # "이름,X,Y,Z,qX,qY,qZ,qW" 형태의 한 줄 텍스트로 생성
                bone_str = f"{name},{tx:.4f},{ty:.4f},{tz:.4f},{quat_unreal[0]:.4f},{quat_unreal[1]:.4f},{quat_unreal[2]:.4f},{quat_unreal[3]:.4f}"
                all_pose_strings.append(bone_str)

            # --- 1. 전신 뼈대 (Body Bones) ---
            BONE_HIERARCHY =[
                'pelvis', 'spine1', 'spine2', 'spine3', 'neck', 'head',
                'left_collar', 'left_shoulder', 'left_elbow', 'left_wrist',
                'right_collar', 'right_shoulder', 'right_elbow', 'right_wrist',
                'left_hip', 'left_knee', 'left_ankle', 'left_foot',
                'right_hip', 'right_knee', 'right_ankle', 'right_foot'
            ]
            
            SMPL_IDX_MAP = {
                'left_hip': 1, 'right_hip': 2, 'spine1': 3, 'left_knee': 4, 'right_knee': 5,
                'spine2': 6, 'left_ankle': 7, 'right_ankle': 8, 'spine3': 9, 'left_foot': 10,
                'right_foot': 11, 'neck': 12, 'left_collar': 13, 'right_collar': 14, 'head': 15,
                'left_shoulder': 16, 'right_shoulder': 17, 'left_elbow': 18, 'right_elbow': 19,
                'left_wrist': 20, 'right_wrist': 21
            }

            body_pose = data['body_pose'].reshape(-1, 3)
            
            # --- [손 모드 체크 유지] ---
            if not ui_state["hand_mode"]:
                for name in BONE_HIERARCHY:
                    if name == 'pelvis':
                        add_to_payload('pelvis', data['global_orient'], pel_unreal)
                        continue
                    
                    idx = SMPL_IDX_MAP.get(name)
                    if idx is not None and (idx-1) < len(body_pose):
                        pose_val = body_pose[idx-1]
                        if name in['spine1', 'spine2', 'spine3']:
                            pose_val = pose_val * 1.2 # 스파인 증폭 로직 유지
                        add_to_payload(name, pose_val)
            else:
                log_cmd("손 모드 활성화됨: 전신 포즈 생략 (배열 전송)", "INFO", app_instance=self)

            # --- 2. 손가락 (Hands) ---
            if 'left_hand_pose' in data:
                l_hand = data['left_hand_pose'].reshape(-1, 3)
                L_HAND_NAMES =[
                    'left_index1', 'left_index2', 'left_index3', 'left_middle1', 'left_middle2', 'left_middle3', 
                    'left_pinky1', 'left_pinky2', 'left_pinky3', 'left_ring1', 'left_ring2', 'left_ring3', 
                    'left_thumb1', 'left_thumb2', 'left_thumb3'
                ]
                for i, name in enumerate(L_HAND_NAMES):
                    if i < len(l_hand): add_to_payload(name, l_hand[i])

            if 'right_hand_pose' in data:
                r_hand = data['right_hand_pose'].reshape(-1, 3)
                R_HAND_NAMES =[
                    'right_index1', 'right_index2', 'right_index3', 'right_middle1', 'right_middle2', 'right_middle3', 
                    'right_pinky1', 'right_pinky2', 'right_pinky3', 'right_ring1', 'right_ring2', 'right_ring3', 
                    'right_thumb1', 'right_thumb2', 'right_thumb3'
                ]
                for i, name in enumerate(R_HAND_NAMES):
                    if i < len(r_hand): add_to_payload(name, r_hand[i])
            
            # ==========================================
            #[최종 발사] 모아둔 리스트를 한 번에 언리얼로 쏨
            # ==========================================
            self.osc_client.send_message("/pose/all", all_pose_strings)
            step_t = self.log_perf("OSC/build payload and send", step_t)
            
            log_cmd(f"OSC 전송 완료! (총 {len(all_pose_strings)}개 뼈대 배열 전송)", "SUCCESS", app_instance=self)
            self.log_perf("OSC/total", total_t)
            self.run_on_ui(self.set_status, "3. OSC 전송 완료", "green")
        
        except Exception as e:
            log_cmd(f"OSC 실패: {e}", "ERROR", app_instance=self)
            import traceback
            traceback.print_exc()

    def start_webcam(self, idx):
        with self.webcam_lock:
            self.webcam_generation += 1
            generation = self.webcam_generation
            self.webcam_running = False
            old_cap = self.cap
            old_thread = self.webcam_thread
            self.cap = None

        if old_thread and old_thread.is_alive():
            old_thread.join(timeout=0.5)

        if old_cap:
            try:
                old_cap.release()
            except Exception:
                pass

        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(idx + 1, cv2.CAP_DSHOW)
        if not cap.isOpened():
            log_cmd(f"Camera {idx} open failed", "ERROR", app_instance=self)
            cap.release()
            return

        with self.webcam_lock:
            if generation != self.webcam_generation:
                cap.release()
                return
            self.cap = cap
            self.webcam_running = True
            self.webcam_thread = threading.Thread(target=self.loop_webcam, args=(generation, cap), daemon=True)
            self.webcam_thread.start()

    def loop_webcam(self, generation, cap):
        while True:
            with self.webcam_lock:
                should_run = self.webcam_running and generation == self.webcam_generation and cap is self.cap
            if not should_run:
                break
            ret, frame = cap.read()
            if ret:
                self.current_image = frame.copy()
                self.display_image(frame, self.lbl_webcam)
            time.sleep(0.03)

    def display_image(self, cv_img, lbl):
        if not self.is_ui_thread():
            if hasattr(self, 'lbl_webcam') and lbl is self.lbl_webcam:
                if self.webcam_frame_pending:
                    return
                self.webcam_frame_pending = True

                def render_webcam_frame(img=cv_img.copy(), label=lbl):
                    try:
                        self.display_image(img, label)
                    finally:
                        self.webcam_frame_pending = False

                self.run_on_ui(render_webcam_frame)
                return

            self.run_on_ui(self.display_image, cv_img.copy(), lbl)
            return
        try:
            h, w = cv_img.shape[:2]
            scale = min(lbl.winfo_width()/w, lbl.winfo_height()/h)
            if scale > 0:
                img = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(cv2.resize(cv_img, (int(w*scale), int(h*scale))), cv2.COLOR_BGR2RGB)))
                lbl.config(image=img)
                lbl.image = img
        except: pass

    def update_3d_plot(self, kpts, is_smpl=False, verts=None, update_limits=True):
        if not self.is_ui_thread():
            kpts_copy = kpts.copy()
            verts_copy = verts.copy() if verts is not None else None
            self.run_on_ui(self.update_3d_plot, kpts_copy, is_smpl, verts_copy, update_limits)
            return

        # DO NOT USE self.ax.clear() - It resets the camera and causes jumps.
        # Instead, remove only the added plot artists.
        for artist in list(self.ax.collections):
            artist.remove()
        for artist in list(self.ax.lines):
            artist.remove()

        self.current_kpts = kpts # Store for zoom
        
        # Extract coordinates first to determine bounds
        xs, ys, zs = kpts[:, 0], kpts[:, 1], -kpts[:, 2] # Apply Z-flip
        
        # Determine Floor Height (Visual Bottom = Numerical Max in this inverted system)
        all_z = [zs]
        if verts is not None:
            all_z.append(-verts[:, 2])
        
        # Merge and find absolute max across all plotted points
        combined_zs = np.concatenate(all_z)
        floor_z = np.max(combined_zs) 
        
        # --- Floor & Shadow ---
        # Define floor bounds based on character position
        center_x, center_y = np.mean(xs), np.mean(ys)
        f_size = 1.0 # 1 meter radius
        xx, yy = np.meshgrid([center_x - f_size, center_x + f_size], [center_y - f_size, center_y + f_size])
        zz = np.full_like(xx, floor_z)
        
        self.ax.plot_surface(xx, yy, zz, color='blue', alpha=0.1, shade=False) # Blueish floor
        
        # --- Mesh Render ---
        if verts is not None:
             # Draw Mesh (Actual)
             faces = None
             if is_smpl and self.smplx_model is not None:
                faces = self.smplx_model.faces
             elif not is_smpl and hasattr(self, 'mhr_faces') and self.mhr_faces is not None:
                faces = self.mhr_faces
            
             if faces is not None:
                if self.mesh_preview_face_stride > 1:
                    faces = faces[::self.mesh_preview_face_stride]
                vx, vy, vz = verts[:, 0], verts[:, 1], -verts[:, 2] # Apply Z-flip
                self.ax.plot_trisurf(vx, vy, vz, triangles=faces, color='lightgray', alpha=0.5, edgecolor='none', shade=True)
                
                pass

        # Draw Joint Shadows (Projected to Floor Z)
        self.ax.scatter(xs, ys, np.full_like(zs, floor_z), c='black', alpha=0.2, s=20, marker='o') # Shadow
        color = 'g' if is_smpl else 'r'
        line_color = 'cyan' if is_smpl else 'blue'
        
        # Draw on the EXISTING axis (preserving view and settings)
        self.ax.scatter(xs, ys, zs, c=color, s=15 if is_smpl else 10)
        
        if is_smpl and self.smplx_model is not None:
             parents = self.smplx_model.parents.cpu().numpy()
             for i in range(1, len(parents)):
                 p = parents[i]
                 if p != -1 and i < len(kpts) and p < len(kpts):
                      if i > 24: continue 
                      self.ax.plot([xs[i], xs[p]], [ys[i], ys[p]], [zs[i], zs[p]], c=line_color, linewidth=2)
        else:
             for i1, i2 in MHR70_SKELETON_PAIRS:
                 if i1 < len(kpts) and i2 < len(kpts):
                     self.ax.plot([xs[i1], xs[i2]], [ys[i1], ys[i2]], [zs[i1], zs[i2]], c=line_color, linewidth=1)
        
        # --- Centering & Scaling ---
        if update_limits:
            mid_x = (xs.max()+xs.min()) * 0.5
            mid_y = (ys.max()+ys.min()) * 0.5
            mid_z = (zs.max()+zs.min()) * 0.5
            
            # Track previous range to avoid jitter
            if not hasattr(self, 'prev_bbox_range'): self.prev_bbox_range = 0.5
            
            bbox_range = np.array([xs.max()-xs.min(), ys.max()-ys.min(), zs.max()-zs.min()]).max() / 2.0
            if bbox_range < 0.05: bbox_range = 0.5
            
            # Smoothly adapt range or keep it stable
            if abs(bbox_range - self.prev_bbox_range) > 0.1:
                self.prev_bbox_range = bbox_range
                
            max_range = self.prev_bbox_range * self.zoom_factor
            
            self.ax.set_xlim(mid_x - max_range, mid_x + max_range)
            self.ax.set_ylim(mid_y - max_range, mid_y + max_range)
            self.ax.set_zlim(mid_z - max_range, mid_z + max_range)
            
            # Ensure Y-axis stays inverted
            if not self.ax.yaxis_inverted():
                self.ax.invert_yaxis()
        
        # Show grid and axes for reference
        self.ax.grid(True)
        self.ax.set_axis_on()
        
        # IMPORTANT: Never call view_init here to respect user manual rotation
        
        self.canvas.draw_idle()
        
    def on_mouse_wheel(self, event):
        # Zoom factor logic: 
        # delta > 0 (Wheel UP) -> Closer -> zoom_factor gets SMALLER
        # delta < 0 (Wheel DOWN) -> Further -> zoom_factor gets LARGER
        if event.num == 4 or event.delta > 0: # Scroll Up
            self.zoom_factor *= 0.8
        elif event.num == 5 or event.delta < 0: # Scroll Down
            self.zoom_factor *= 1.2
            
        # Limit zoom factor
        self.zoom_factor = max(0.01, min(self.zoom_factor, 10.0))
        
        # Redraw
        if hasattr(self, 'current_kpts') and self.current_kpts is not None:
             self.update_3d_plot(self.current_kpts, is_smpl=True if self.smpl_result else False)

if __name__ == "__main__":
    root = tk.Tk()
    app = DirectMHRApp(root)
    root.mainloop()

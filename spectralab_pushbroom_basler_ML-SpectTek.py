#!/usr/bin/env python3
"""
SpectraLab Push-Broom Imaging — SpectTek Co.

Main cube model:
    hsi_cube_yxw[Y, X, wavelength]
"""

import os
import json
import time
import pickle as pk
import threading
import webbrowser
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable

try:
    from PIL import Image, ImageTk
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

try:
    from pypylon import pylon
    PYLON_AVAILABLE = True
except Exception:
    PYLON_AVAILABLE = False

try:
    from astropy.io import fits
    ASTROPY_AVAILABLE = True
except Exception:
    ASTROPY_AVAILABLE = False

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.pipeline import make_pipeline
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.cluster import KMeans
    from sklearn.metrics import mean_squared_error, accuracy_score
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


CAMERA_NAME = "Basler daA1600-60um"
LINKEDIN_URL = "https://www.linkedin.com/in/specttek/"
SENSOR_WIDTH = 1600
SENSOR_HEIGHT = 1200
PIXEL_SIZE_UM = 4.5

REGION_COLORS = [
    "red", "lime", "cyan", "yellow", "magenta",
    "orange", "deepskyblue", "white", "violet", "springgreen"
]


class baslerData:
    def __init__(self,numOfFrame=None, roiX=None,roiY=None, lookUp_x=None, lookUp_y=None\
                 , refUsed=None, darkRefUsed=False, data=None, scannerType=None, matFileLen=None, refData=None\
                 ,deviceModel=None, dateAndTime=None, exposureTime= None, frameRate=None
                 , Gain=None, Gamma=None, desScannerType=None, scanLength=None, averagingTime=None
                 , refFrameUsed=None, sizeofDataCube=None, describtion=None, cropped=None):
        self.numOfFrame=numOfFrame
        self.roiX=roiX
        self.roiY=roiY
        self.lookUp_x=lookUp_x
        self.lookUp_y=lookUp_y
        self.refUsed=refUsed
        self.darkRefUsed=darkRefUsed
        self.data=data
        self.scannerType=scannerType
        self.matFileLen=matFileLen
        self.refData=refData
        self.desDeviceModel=deviceModel
        self.desDateAndTime=dateAndTime
        self.desExposureTime=exposureTime
        self.desFrameRate=frameRate
        self.desGain=Gain
        self.desGamma=Gamma
        self.desScannerType=desScannerType
        self.desScanLength=scanLength
        self.desAveragingTime=averagingTime
        self.desRefFrameUsed=refFrameUsed
        self.desSizeofDataCube=sizeofDataCube
        self.desDescribtion=describtion
        self.cropped=cropped


class BaslerCamera:
    def __init__(self):
        self.camera = None
        self.connected = False
        self.dummy_mode = False

    def enable_dummy(self):
        self.close()
        self.connected = True
        self.dummy_mode = True
        return "Dummy mode enabled. Working without camera."

    def connect(self):
        if not PYLON_AVAILABLE:
            return self.enable_dummy()

        try:
            factory = pylon.TlFactory.GetInstance()
            devices = factory.EnumerateDevices()

            if len(devices) == 0:
                return self.enable_dummy()

            self.camera = pylon.InstantCamera(factory.CreateFirstDevice())
            self.camera.Open()

            try:
                self.camera.PixelFormat.SetValue("Mono8")
            except Exception:
                pass

            self.connected = True
            self.dummy_mode = False
            return f"Connected: {self.camera.GetDeviceInfo().GetModelName()}"

        except Exception as e:
            self.enable_dummy()
            return f"Camera failed. Dummy mode enabled. Error: {e}"

    def set_exposure_us(self, exposure_us):
        if self.dummy_mode or self.camera is None:
            return
        try:
            self.camera.ExposureAuto.SetValue("Off")
            self.camera.ExposureTime.SetValue(float(exposure_us))
        except Exception as e:
            print("Exposure error:", e)

    def set_gain(self, gain):
        if self.dummy_mode or self.camera is None:
            return
        try:
            self.camera.GainAuto.SetValue("Off")
            self.camera.Gain.SetValue(float(gain))
        except Exception:
            pass

    def get_frame(self):
        if self.dummy_mode:
            return self.synthetic_frame()

        if self.camera is None:
            return None

        try:
            grab = self.camera.GrabOne(5000)
            if grab.GrabSucceeded():
                arr = grab.Array.astype(np.float32)
                grab.Release()
                return arr
            grab.Release()
        except Exception as e:
            print("Grab error:", e)

        return None

    def synthetic_frame(self):
        y = np.arange(SENSOR_HEIGHT)[:, None]
        x = np.arange(SENSOR_WIDTH)[None, :]
        spectrum = (
            2500*np.exp(-((x-250)/45)**2) +
            6500*np.exp(-((x-620)/80)**2) +
            4200*np.exp(-((x-1050)/120)**2) +
            2500*np.exp(-((x-1380)/70)**2)
        )
        spatial = (
            0.45 +
            0.40*np.exp(-((y-450)/180)**2) +
            0.25*np.exp(-((y-850)/140)**2)
        )
        noise = np.random.normal(0, 80, (SENSOR_HEIGHT, SENSOR_WIDTH))
        return np.clip(spectrum*spatial + noise + 300, 0, 65535).astype(np.float32)

    def close(self):
        try:
            if self.camera is not None:
                self.camera.Close()
        except Exception:
            pass
        self.camera = None
        self.connected = False


class SpectraLabPushBroomGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("SpectraLab Push-Broom Imaging — SpectTek Co.")
        self.root.geometry("1880x1040")

        self.camera_password = "SpectTek"
        self.logo_image = None

        self.cam = BaslerCamera()
        self.scan_running = False

        self.hsi_cube_yxw = None
        self.display_image = None
        self.raw_loaded_data = None
        self.cube = None

        self.x_map = None
        self.y_map = None
        self.wavelength_axis = None

        self.lookUpX = None
        self.lookUpY = None
        self.numOfFrame = None
        self.roiX = None
        self.roiY = None

        self.regions = []
        self.references = []
        self.class_map = None
        self.last_image_artist = None
        self.colorbar = None

        self.section_frames = {}
        self.build_ui()

    def build_ui(self):
        top = ttk.Frame(self.root)
        top.pack(fill="x", padx=8, pady=5)

        ttk.Label(
            top,
            text="SpectraLab Push-Broom Imaging",
            font=("Arial", 16, "bold")
        ).pack(side="left")

        company = tk.Label(
            top,
            text="SpectTek Co.",
            fg="blue",
            cursor="hand2",
            font=("Arial", 12, "bold underline")
        )
        company.pack(side="right")
        company.bind("<Button-1>", lambda e: webbrowser.open(LINKEDIN_URL))

        main = ttk.Frame(self.root)
        main.pack(fill="both", expand=True)

        left = ttk.Frame(main, width=500)
        left.pack(side="left", fill="y", padx=5, pady=6)
        left.pack_propagate(False)

        right = ttk.Frame(main)
        right.pack(side="right", fill="both", expand=True, padx=5, pady=6)

        selector_box = ttk.LabelFrame(left, text="Control section")
        selector_box.pack(fill="x", padx=6, pady=4)

        self.section_var = tk.StringVar(value="Camera")
        self.section_selector = ttk.Combobox(
            selector_box,
            textvariable=self.section_var,
            state="readonly",
            values=[
                "Camera",
                "Scan",
                "Dummy",
                "Regions",
                "Save / Load",
                "Display",
                "Analysis",
                "References",
            ]
        )
        self.section_selector.pack(fill="x", padx=6, pady=6)
        self.section_selector.bind("<<ComboboxSelected>>", lambda e: self.show_section())

        self.left_body = ttk.Frame(left)
        self.left_body.pack(fill="both", expand=True, padx=4, pady=4)

        for name in self.section_selector["values"]:
            frame = ttk.Frame(self.left_body)
            self.section_frames[name] = frame

        self.build_camera_section(self.section_frames["Camera"])
        self.build_scan_section(self.section_frames["Scan"])
        self.build_dummy_section(self.section_frames["Dummy"])
        self.build_region_section(self.section_frames["Regions"])
        self.build_save_load_section(self.section_frames["Save / Load"])
        self.build_display_section(self.section_frames["Display"])
        self.build_analysis_section(self.section_frames["Analysis"])
        self.build_reference_section(self.section_frames["References"])

        self.build_plot_panel(right)
        self.show_section()

    def show_section(self):
        for frame in self.section_frames.values():
            frame.pack_forget()
        self.section_frames[self.section_var.get()].pack(fill="both", expand=True)

    def make_entry(self, parent, label, variable):
        frame = ttk.Frame(parent)
        frame.pack(fill="x", padx=8, pady=3)
        ttk.Label(frame, text=label).pack(anchor="w")
        ttk.Entry(frame, textvariable=variable).pack(fill="x")

    def run_loading_task(self, loading_function):
        self.status_var.set("Status: waiting... loading data")
        self.root.update_idletasks()
        try:
            loading_function()
            if "waiting" in self.status_var.get().lower():
                self.status_var.set("Status: loading complete.")
        except Exception as e:
            self.status_var.set("Status: loading failed.")
            messagebox.showerror("Loading error", str(e))

    def build_camera_section(self, parent):
        ttk.Label(
            parent,
            text="SpecTek Hyperspectral Imager",
            font=("Arial", 13, "bold")
        ).pack(pady=(10, 4))

        ttk.Label(
            parent,
            text=f"{CAMERA_NAME}\n1600 × 1200 | Mono | USB3\nGlobal shutter | 4.5 µm pixels",
            justify="center"
        ).pack(pady=(0, 10))

        ttk.Button(parent, text="Connect Camera", command=self.connect_camera).pack(fill="x", padx=8, pady=5)
        ttk.Button(parent, text="Enable Dummy/Test Mode", command=self.enable_dummy_mode).pack(fill="x", padx=8, pady=5)

        self.status_var = tk.StringVar(value="Status: not connected")
        ttk.Label(parent, textvariable=self.status_var, wraplength=460).pack(padx=8, pady=10)

        ttk.Separator(parent).pack(fill="x", padx=8, pady=8)

        self.exposure_var = tk.DoubleVar(value=10000.0)
        self.gain_var = tk.DoubleVar(value=0.0)
        self.average_var = tk.IntVar(value=3)

        self.make_entry(parent, "Integration / exposure time [µs]", self.exposure_var)
        self.make_entry(parent, "Gain", self.gain_var)
        self.make_entry(parent, "Frame average", self.average_var)

        ttk.Button(parent, text="Apply Camera Settings", command=self.apply_settings).pack(fill="x", padx=8, pady=8)

        if PIL_AVAILABLE:
            try:
                img = Image.open("logo.png")
                img.thumbnail((220, 220))
                self.logo_image = ImageTk.PhotoImage(img)
                ttk.Label(parent, image=self.logo_image).pack(pady=(15, 10))
            except Exception:
                ttk.Label(parent, text="[logo.png not found]").pack(pady=10)
        else:
            ttk.Label(parent, text="[Install pillow to show logo.png]").pack(pady=10)

    def build_scan_section(self, parent):
        ttk.Label(parent, text="Push-Broom Scan Settings", font=("Arial", 12, "bold")).pack(pady=10)

        self.steps_var = tk.IntVar(value=100)
        self.delay_var = tk.DoubleVar(value=0.05)
        self.scan_line_y_var = tk.IntVar(value=SENSOR_HEIGHT // 2)
        self.scan_line_height_var = tk.IntVar(value=5)

        self.make_entry(parent, "Scan steps", self.steps_var)
        self.make_entry(parent, "Delay per step [s]", self.delay_var)
        self.make_entry(parent, "Detector line center Y [px]", self.scan_line_y_var)
        self.make_entry(parent, "Line averaging height [px]", self.scan_line_height_var)

        ttk.Button(parent, text="Start Push-Broom Scan", command=self.start_scan).pack(fill="x", padx=8, pady=10)
        ttk.Button(parent, text="Stop Scan", command=self.stop_scan).pack(fill="x", padx=8, pady=5)

        self.progress = ttk.Progressbar(parent, orient="horizontal", mode="determinate")
        self.progress.pack(fill="x", padx=8, pady=10)

    def build_dummy_section(self, parent):
        ttk.Label(parent, text="Dummy Test Data", font=("Arial", 12, "bold")).pack(pady=10)

        self.dummy_y_var = tk.IntVar(value=120)
        self.dummy_x_var = tk.IntVar(value=80)
        self.dummy_w_var = tk.IntVar(value=220)
        self.dummy_noise_var = tk.DoubleVar(value=0.03)

        self.make_entry(parent, "Dummy Y rows", self.dummy_y_var)
        self.make_entry(parent, "Dummy X columns", self.dummy_x_var)
        self.make_entry(parent, "Dummy wavelength bands", self.dummy_w_var)
        self.make_entry(parent, "Noise level", self.dummy_noise_var)

        ttk.Button(parent, text="Generate Dummy Y-X-W Cube", command=self.generate_dummy_cube).pack(fill="x", padx=8, pady=10)
        ttk.Button(parent, text="Enable Dummy Camera Mode", command=self.enable_dummy_mode).pack(fill="x", padx=8, pady=5)

    def build_region_section(self, parent):
        ttk.Label(parent, text="Region Comparison", font=("Arial", 12, "bold")).pack(pady=10)

        ttk.Label(
            parent,
            text="Drag on image to add training/analysis regions.\nSpectra use full Y-X-wavelength cube.",
            wraplength=460,
            justify="left"
        ).pack(anchor="w", padx=8, pady=8)

        ttk.Button(parent, text="Replot Region Spectra", command=self.replot_all_region_spectra).pack(fill="x", padx=8, pady=5)
        ttk.Button(parent, text="Clear All Regions", command=self.clear_regions).pack(fill="x", padx=8, pady=5)

        self.region_list = tk.Listbox(parent, height=14)
        self.region_list.pack(fill="both", expand=True, padx=8, pady=10)

    def build_save_load_section(self, parent):
        ttk.Label(parent, text="Save / Load", font=("Arial", 12, "bold")).pack(pady=8)

        file_box = ttk.LabelFrame(parent, text="File I/O")
        file_box.pack(fill="x", padx=8, pady=5)

        for s, sc, l, lc in [
            ("Save NPZ", self.save_npz, "Load NPZ", self.load_npz),
            ("Save CUBE", self.save_cube_file, "Load CUBE", self.load_cube_file),
            ("Save FITS", self.save_fits_file, "Load FITS", self.load_fits_file),
        ]:
            row = ttk.Frame(file_box)
            row.pack(fill="x", padx=4, pady=3)
            ttk.Button(row, text=s, command=sc).pack(side="left", fill="x", expand=True, padx=2)
            ttk.Button(row, text=l, command=lambda f=lc: self.run_loading_task(f)).pack(side="left", fill="x", expand=True, padx=2)

        ttk.Button(file_box, text="Import wavelength vector CSV/TXT", command=self.load_wavelength_vector).pack(fill="x", padx=6, pady=5)
        ttk.Button(file_box, text="Deload / Clear All", command=self.deload_all).pack(fill="x", padx=6, pady=5)

        dim_box = ttk.LabelFrame(parent, text="Current Dimensions")
        dim_box.pack(fill="both", expand=True, padx=8, pady=5)

        self.dimension_report_var = tk.StringVar(value="No data loaded.")
        ttk.Label(dim_box, textvariable=self.dimension_report_var, wraplength=460, justify="left").pack(anchor="w", padx=6, pady=5)

    def build_display_section(self, parent):
        ttk.Label(parent, text="Display Controls", font=("Arial", 12, "bold")).pack(pady=8)

        mode_box = ttk.LabelFrame(parent, text="Display mode")
        mode_box.pack(fill="x", padx=8, pady=5)

        self.display_mode_var = tk.StringVar(value="RGB")
        ttk.Radiobutton(mode_box, text="RGB preview", variable=self.display_mode_var, value="RGB", command=self.update_display_image).pack(anchor="w", padx=8)
        ttk.Radiobutton(mode_box, text="Single wavelength/channel", variable=self.display_mode_var, value="Single", command=self.update_display_image).pack(anchor="w", padx=8)

        self.single_wave_var = tk.DoubleVar(value=550.0)
        self.single_band_var = tk.IntVar(value=0)

        self.make_entry(mode_box, "Single wavelength target / nm", self.single_wave_var)
        self.make_entry(mode_box, "Single channel index fallback", self.single_band_var)

        rgb_box = ttk.LabelFrame(parent, text="RGB bands and contrast")
        rgb_box.pack(fill="x", padx=8, pady=5)

        self.red_wave_var = tk.DoubleVar(value=650.0)
        self.green_wave_var = tk.DoubleVar(value=550.0)
        self.blue_wave_var = tk.DoubleVar(value=450.0)
        self.brightness_var = tk.DoubleVar(value=0.0)
        self.contrast_var = tk.DoubleVar(value=1.0)

        for label, var in [
            ("Red wavelength / nm", self.red_wave_var),
            ("Green wavelength / nm", self.green_wave_var),
            ("Blue wavelength / nm", self.blue_wave_var),
            ("Brightness offset", self.brightness_var),
            ("Contrast scale", self.contrast_var),
        ]:
            self.make_entry(rgb_box, label, var)

        ttk.Button(rgb_box, text="Apply Display", command=self.update_display_image).pack(fill="x", padx=8, pady=5)

        orient_box = ttk.LabelFrame(parent, text="Orientation")
        orient_box.pack(fill="x", padx=8, pady=5)

        self.transform_var = tk.StringVar(value="Original")
        ttk.Combobox(
            orient_box,
            textvariable=self.transform_var,
            state="readonly",
            values=["Original", "Transpose X/Y", "Flip X", "Flip Y", "Rotate 90", "Rotate 180", "Rotate 270"]
        ).pack(fill="x", padx=6, pady=4)

        ttk.Button(orient_box, text="Apply Orientation", command=self.apply_display_transform).pack(fill="x", padx=6, pady=4)

    def build_analysis_section(self, parent):
        ttk.Label(parent, text="ML Classification", font=("Arial", 12, "bold")).pack(pady=8)

        box = ttk.LabelFrame(parent, text="Method")
        box.pack(fill="x", padx=8, pady=5)

        self.ml_method_var = tk.StringVar(value="RF")
        ttk.Combobox(
            box,
            textvariable=self.ml_method_var,
            state="readonly",
            values=["RF", "SVM", "GB", "NN", "KMeans"]
        ).pack(fill="x", padx=6, pady=4)

        self.ml_classes_var = tk.IntVar(value=3)
        self.ml_pca_var = tk.IntVar(value=10)
        self.ml_level_var = tk.StringVar(value="Medium")

        self.make_entry(box, "Number of classes", self.ml_classes_var)
        self.make_entry(box, "PCA components / spectral level", self.ml_pca_var)

        ttk.Label(box, text="Model level").pack(anchor="w", padx=8)
        ttk.Combobox(box, textvariable=self.ml_level_var, state="readonly", values=["Fast", "Medium", "High"]).pack(fill="x", padx=8, pady=4)

        ttk.Label(
            box,
            text="Classification uses the active X/Y plot range from the top-right range controls.",
            wraplength=430,
            justify="left"
        ).pack(anchor="w", padx=8, pady=4)

        ttk.Button(box, text="Run Classification in X/Y Range", command=self.run_classification).pack(fill="x", padx=8, pady=5)
        ttk.Button(box, text="Show Current Display Again", command=self.update_display_image).pack(fill="x", padx=8, pady=5)

        log_box = ttk.LabelFrame(parent, text="Log / RMSE")
        log_box.pack(fill="both", expand=True, padx=8, pady=5)

        self.ml_log = tk.Text(log_box, height=18, wrap="word")
        self.ml_log.pack(fill="both", expand=True, padx=4, pady=4)

    def build_reference_section(self, parent):
        ttk.Label(parent, text="Spectral References", font=("Arial", 12, "bold")).pack(pady=8)

        self.ref_normalize_var = tk.BooleanVar(value=True)
        self.ref_overlay_var = tk.BooleanVar(value=True)

        ttk.Button(parent, text="Load Reference CSV", command=self.load_reference_csv).pack(fill="x", padx=8, pady=5)
        ttk.Checkbutton(parent, text="Overlay references", variable=self.ref_overlay_var, command=self.replot_all_region_spectra).pack(anchor="w", padx=8)
        ttk.Checkbutton(parent, text="Normalize selected/reference to max", variable=self.ref_normalize_var, command=self.replot_all_region_spectra).pack(anchor="w", padx=8)

        ttk.Button(parent, text="Clear References", command=self.clear_references).pack(fill="x", padx=8, pady=5)

        self.ref_list = tk.Listbox(parent, height=18)
        self.ref_list.pack(fill="both", expand=True, padx=8, pady=10)

    def build_plot_panel(self, parent):
        range_bar = ttk.LabelFrame(parent, text="X/Y plot range — also used by ML classification")
        range_bar.pack(fill="x", padx=4, pady=(0, 6))

        self.xmin_var = tk.StringVar()
        self.xmax_var = tk.StringVar()
        self.ymin_var = tk.StringVar()
        self.ymax_var = tk.StringVar()

        ttk.Label(range_bar, text="X min").pack(side="left", padx=(8, 2))
        ttk.Entry(range_bar, textvariable=self.xmin_var, width=9).pack(side="left", padx=2)
        ttk.Label(range_bar, text="X max").pack(side="left", padx=(8, 2))
        ttk.Entry(range_bar, textvariable=self.xmax_var, width=9).pack(side="left", padx=2)

        ttk.Label(range_bar, text="Y min").pack(side="left", padx=(18, 2))
        ttk.Entry(range_bar, textvariable=self.ymin_var, width=9).pack(side="left", padx=2)
        ttk.Label(range_bar, text="Y max").pack(side="left", padx=(8, 2))
        ttk.Entry(range_bar, textvariable=self.ymax_var, width=9).pack(side="left", padx=2)

        ttk.Button(range_bar, text="Apply range", command=self.apply_plot_limits).pack(side="left", padx=10)
        ttk.Button(range_bar, text="Reset", command=self.reset_plot_limits).pack(side="left", padx=4)

        plot_frame = ttk.Frame(parent)
        plot_frame.pack(fill="both", expand=True)

        self.fig = Figure(figsize=(15.8, 9.6), dpi=100)
        gs = self.fig.add_gridspec(
            2,
            1,
            height_ratios=[4.2, 1.8],
            hspace=0.40,
            left=0.055,
            right=0.93,
            top=0.95,
            bottom=0.08
        )

        self.ax_img = self.fig.add_subplot(gs[0])
        self.ax_spec = self.fig.add_subplot(gs[1])

        self.ax_img.set_title("X-Y Image from HSI Cube")
        self.ax_img.set_xlabel("X spatial pixel")
        self.ax_img.set_ylabel("Y spatial pixel")

        self.ax_spec.set_title("Spectrum from selected X,Y pixel or region")
        self.ax_spec.set_xlabel("Wavelength / spectral channel")
        self.ax_spec.set_ylabel("Intensity")

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.canvas.mpl_connect("button_press_event", self.on_image_click)

        self.rect_selector = RectangleSelector(
            self.ax_img,
            self.on_select_area,
            useblit=False,
            button=[1],
            minspanx=2,
            minspany=2,
            spancoords="pixels",
            interactive=True
        )

    def connect_camera(self):
        entered = simpledialog.askstring(
            "SpectTek Security",
            "Enter camera connection password:",
            show="*"
        )

        if entered is None:
            return

        if entered != self.camera_password:
            messagebox.showerror("Access Denied", "Invalid password.")
            return

        self.status_var.set("Status: " + self.cam.connect())

    def enable_dummy_mode(self):
        self.status_var.set("Status: " + self.cam.enable_dummy())

    def apply_settings(self):
        if not self.cam.connected:
            messagebox.showwarning("Camera", "Connect camera or enable dummy mode first.")
            return

        self.cam.set_exposure_us(self.exposure_var.get())
        self.cam.set_gain(self.gain_var.get())
        self.status_var.set(
            f"Settings applied: exposure={self.exposure_var.get()} µs, gain={self.gain_var.get()}, average={self.average_var.get()}"
        )

    def averaged_frame(self):
        frames = []
        for _ in range(max(1, int(self.average_var.get()))):
            fr = self.cam.get_frame()
            if fr is not None:
                frames.append(fr)
            time.sleep(0.002)
        return np.mean(frames, axis=0) if frames else None

    def extract_line_spectrum(self, frame, y0, height):
        y0 = int(np.clip(y0, 0, frame.shape[0] - 1))
        half = max(0, int(height) // 2)
        return np.mean(
            frame[max(0, y0-half):min(frame.shape[0], y0+half+1), :],
            axis=0
        )

    def start_scan(self):
        if not self.cam.connected:
            messagebox.showwarning("Camera", "Connect camera or enable dummy mode first.")
            return

        self.clear_regions()
        self.scan_running = True
        self.progress["value"] = 0
        threading.Thread(target=self.scan_loop, daemon=True).start()

    def stop_scan(self):
        self.scan_running = False

    def scan_loop(self):
        steps = max(1, int(self.steps_var.get()))
        delay = max(0.0, float(self.delay_var.get()))
        y0 = int(self.scan_line_y_var.get())
        h = max(1, int(self.scan_line_height_var.get()))
        scan_lines = []

        self.status_var.set("Status: push-broom scan running...")

        for i in range(steps):
            if not self.scan_running:
                break

            frame = self.averaged_frame()
            if frame is not None:
                scan_lines.append(self.extract_line_spectrum(frame, y0, h))

            self.root.after(0, self.progress.configure, {"value": 100*(i+1)/steps})
            time.sleep(delay)

        self.scan_running = False

        if scan_lines:
            image_yw = np.asarray(scan_lines, dtype=np.float32)
            self.hsi_cube_yxw = image_yw[:, None, :]
            self.raw_loaded_data = self.hsi_cube_yxw.copy()
            self.cube = self.hsi_cube_yxw
            self.x_map = np.arange(1, dtype=np.float32)
            self.y_map = np.arange(self.hsi_cube_yxw.shape[0], dtype=np.float32)
            self.wavelength_axis = np.arange(self.hsi_cube_yxw.shape[2], dtype=np.float32)
            self.update_display_image()
            self.status_var.set(f"Status: scan finished. HSI Y,X,W={self.hsi_cube_yxw.shape}")

    def generate_dummy_cube(self):
        self.clear_regions()

        ny = max(2, int(self.dummy_y_var.get()))
        nx = max(2, int(self.dummy_x_var.get()))
        nw = max(10, int(self.dummy_w_var.get()))
        noise = max(0.0, float(self.dummy_noise_var.get()))

        wavelength = np.linspace(400, 900, nw).astype(np.float32)
        yy, xx = np.mgrid[0:ny, 0:nx]
        cube = np.zeros((ny, nx, nw), dtype=np.float32)

        centers = [
            (nx*.30, ny*.35, 550, .8),
            (nx*.70, ny*.50, 680, 1.1),
            (nx*.50, ny*.75, 820, .75)
        ]

        for i, wl in enumerate(wavelength):
            img = np.zeros((ny, nx), dtype=np.float32)
            for cx, cy, cwl, amp in centers:
                spatial = np.exp(-((xx-cx)**2 + (yy-cy)**2)/(0.06*nx*ny))
                spectral = np.exp(-((wl-cwl)/45)**2)
                img += amp * spatial * spectral

            cube[:, :, i] = img + 0.15*np.exp(-((wl-600)/180)**2) + np.random.normal(0, noise, (ny, nx))

        cube -= np.nanmin(cube)
        if np.nanmax(cube) > 0:
            cube /= np.nanmax(cube)

        self.hsi_cube_yxw = cube
        self.raw_loaded_data = cube.copy()
        self.cube = cube.copy()
        self.x_map = np.arange(nx, dtype=np.float32)
        self.y_map = np.arange(ny, dtype=np.float32)
        self.wavelength_axis = wavelength
        self.update_display_image()
        self.status_var.set(f"Status: dummy Y-X-wavelength cube generated: {cube.shape}")

    def nearest_band(self, target):
        if self.wavelength_axis is None:
            return 0
        wl = np.asarray(self.wavelength_axis, dtype=np.float32)
        if np.nanmax(wl) > 100:
            return int(np.argmin(np.abs(wl-target)))
        return int(np.clip(round(target), 0, len(wl)-1))

    def adjust_brightness_contrast(self, image):
        out = (image - 0.5) * float(self.contrast_var.get()) + 0.5 + float(self.brightness_var.get())
        return np.clip(out, 0, 1)

    def make_rgb_from_cube(self):
        cube = np.asarray(self.hsi_cube_yxw, dtype=np.float32)
        bands = [
            self.nearest_band(float(self.red_wave_var.get())),
            self.nearest_band(float(self.green_wave_var.get())),
            self.nearest_band(float(self.blue_wave_var.get()))
        ]
        rgb = np.stack([cube[:, :, bands[0]], cube[:, :, bands[1]], cube[:, :, bands[2]]], axis=2)
        lo, hi = np.nanpercentile(rgb, 1), np.nanpercentile(rgb, 99)
        return self.adjust_brightness_contrast(np.clip((rgb-lo)/(hi-lo+1e-12), 0, 1))

    def make_single_band_image(self):
        idx = self.nearest_band(float(self.single_wave_var.get()))
        idx = int(np.clip(idx, 0, self.hsi_cube_yxw.shape[2]-1))
        img = self.hsi_cube_yxw[:, :, idx]
        lo, hi = np.nanpercentile(img, 1), np.nanpercentile(img, 99)
        return self.adjust_brightness_contrast(np.clip((img-lo)/(hi-lo+1e-12), 0, 1)), idx

    def update_display_image(self):
        if self.hsi_cube_yxw is None:
            return

        if self.display_mode_var.get() == "Single":
            self.display_image, idx = self.make_single_band_image()
            title = f"Single wavelength/channel image | band={idx}, λ={self.wavelength_axis[idx]:.3g}"
            self.show_display_image(title=title, cmap="gray")
        else:
            self.display_image = self.make_rgb_from_cube()
            self.show_display_image(title="RGB mapped image from HSI cube", cmap=None)

    def clear_colorbar(self):
        if self.colorbar is not None:
            try:
                self.colorbar.remove()
            except Exception:
                pass
            self.colorbar = None

    def show_display_image(self, title="X-Y Image from HSI Cube", cmap=None):
        if self.display_image is None or self.hsi_cube_yxw is None:
            return

        ny, nx, nw = self.hsi_cube_yxw.shape
        self.ax_img.clear()
        self.clear_colorbar()

        self.last_image_artist = self.ax_img.imshow(
            self.display_image,
            cmap=cmap,
            origin="upper",
            aspect="auto",
            extent=[0, nx-1, ny-1, 0]
        )

        if self.display_image.ndim == 2:
            divider = make_axes_locatable(self.ax_img)
            cax = divider.append_axes("right", size="2.5%", pad=0.12)
            self.colorbar = self.fig.colorbar(self.last_image_artist, cax=cax)

        self.ax_img.set_title(f"{title} | Y={ny}, X={nx}, λ={nw}")
        self.ax_img.set_xlabel("X spatial pixel")
        self.ax_img.set_ylabel("Y spatial pixel")
        self.apply_plot_limits(draw=False)

        self.redraw_region_rectangles()

        self.ax_spec.clear()
        self.ax_spec.set_title("Select X,Y pixel or region to show wavelength spectrum")
        self.ax_spec.set_xlabel("Wavelength / spectral channel")
        self.ax_spec.set_ylabel("Intensity")

        if self.regions:
            self.replot_all_region_spectra(draw=False)
        else:
            self.plot_references_if_enabled()

        self.update_dimension_report()
        self.canvas.draw_idle()

    def redraw_region_rectangles(self):
        for reg in self.regions:
            self.ax_img.add_patch(Rectangle(
                (reg["x1"], reg["y1"]),
                reg["x2"]-reg["x1"],
                reg["y2"]-reg["y1"],
                edgecolor=reg["color"],
                facecolor="none",
                linewidth=2.2
            ))

    def get_xy_bounds(self):
        if self.hsi_cube_yxw is None:
            return None

        ny, nx, _ = self.hsi_cube_yxw.shape

        try:
            x1 = int(float(self.xmin_var.get())) if self.xmin_var.get() else 0
            x2 = int(float(self.xmax_var.get())) + 1 if self.xmax_var.get() else nx
            y1 = int(float(self.ymin_var.get())) if self.ymin_var.get() else 0
            y2 = int(float(self.ymax_var.get())) + 1 if self.ymax_var.get() else ny
        except Exception:
            x1, x2, y1, y2 = 0, nx, 0, ny

        x1 = int(np.clip(x1, 0, nx-1))
        x2 = int(np.clip(x2, x1+1, nx))
        y1 = int(np.clip(y1, 0, ny-1))
        y2 = int(np.clip(y2, y1+1, ny))

        return x1, x2, y1, y2

    def apply_display_transform(self):
        if self.hsi_cube_yxw is None:
            messagebox.showwarning("Display", "No HSI cube available.")
            return

        cube = self.hsi_cube_yxw.copy()
        mode = self.transform_var.get()

        if mode == "Transpose X/Y":
            cube = np.transpose(cube, (1, 0, 2))
        elif mode == "Flip X":
            cube = np.flip(cube, axis=1)
        elif mode == "Flip Y":
            cube = np.flip(cube, axis=0)
        elif mode == "Rotate 90":
            cube = np.rot90(cube, 1, axes=(0, 1))
        elif mode == "Rotate 180":
            cube = np.rot90(cube, 2, axes=(0, 1))
        elif mode == "Rotate 270":
            cube = np.rot90(cube, 3, axes=(0, 1))

        self.hsi_cube_yxw = cube
        self.cube = cube
        self.regions = []
        self.refresh_region_list()
        self.update_display_image()

    def apply_plot_limits(self, draw=True):
        if self.hsi_cube_yxw is None:
            return

        ny, nx, _ = self.hsi_cube_yxw.shape

        if self.xmin_var.get() and self.xmax_var.get():
            self.ax_img.set_xlim(float(self.xmin_var.get()), float(self.xmax_var.get()))
        else:
            self.ax_img.set_xlim(0, nx-1)

        if self.ymin_var.get() and self.ymax_var.get():
            self.ax_img.set_ylim(float(self.ymax_var.get()), float(self.ymin_var.get()))
        else:
            self.ax_img.set_ylim(ny-1, 0)

        if draw:
            self.canvas.draw_idle()

    def reset_plot_limits(self):
        self.xmin_var.set("")
        self.xmax_var.set("")
        self.ymin_var.set("")
        self.ymax_var.set("")
        self.apply_plot_limits(draw=True)

    def normalize_vec(self, x):
        x = np.asarray(x, dtype=np.float32)
        m = np.nanmax(np.abs(x))
        return x/m if m > 0 else x

    def on_image_click(self, event):
        if event.inaxes != self.ax_img or self.hsi_cube_yxw is None:
            return
        if event.xdata is None or event.ydata is None:
            return

        x = int(np.clip(event.xdata, 0, self.hsi_cube_yxw.shape[1]-1))
        y = int(np.clip(event.ydata, 0, self.hsi_cube_yxw.shape[0]-1))
        spectrum = self.hsi_cube_yxw[y, x, :]

        if self.ref_normalize_var.get():
            spectrum = self.normalize_vec(spectrum)

        self.ax_spec.clear()
        self.ax_spec.plot(self.wavelength_axis, spectrum, label=f"X={x}, Y={y}")
        self.plot_references_if_enabled()
        self.ax_spec.set_title(f"Spectrum at X={x}, Y={y}")
        self.ax_spec.set_xlabel("Wavelength / spectral channel")
        self.ax_spec.set_ylabel("Intensity")
        self.ax_spec.relim()
        self.ax_spec.autoscale_view()
        self.ax_spec.margins(y=0.20)
        self.ax_spec.legend(fontsize=8)
        self.canvas.draw_idle()

    def on_select_area(self, eclick, erelease):
        if self.hsi_cube_yxw is None or eclick.xdata is None or erelease.xdata is None:
            return

        x1, x2 = sorted([int(eclick.xdata), int(erelease.xdata)])
        y1, y2 = sorted([int(eclick.ydata), int(erelease.ydata)])

        x1 = int(np.clip(x1, 0, self.hsi_cube_yxw.shape[1]-1))
        x2 = int(np.clip(x2, 0, self.hsi_cube_yxw.shape[1]))
        y1 = int(np.clip(y1, 0, self.hsi_cube_yxw.shape[0]-1))
        y2 = int(np.clip(y2, 0, self.hsi_cube_yxw.shape[0]))

        if x2 <= x1 or y2 <= y1:
            return

        color = REGION_COLORS[len(self.regions) % len(REGION_COLORS)]
        self.regions.append({
            "name": f"Region {len(self.regions)+1}",
            "x1": x1,
            "x2": x2,
            "y1": y1,
            "y2": y2,
            "color": color
        })

        self.refresh_region_list()
        self.update_display_image()
        self.replot_all_region_spectra()

    def replot_all_region_spectra(self, draw=True):
        if self.hsi_cube_yxw is None:
            self.ax_spec.clear()
            self.plot_references_if_enabled()
            if draw:
                self.canvas.draw_idle()
            return

        self.ax_spec.clear()

        for reg in self.regions:
            selected = self.hsi_cube_yxw[reg["y1"]:reg["y2"], reg["x1"]:reg["x2"], :]
            if selected.size == 0:
                continue

            avg = np.nanmean(selected, axis=(0, 1))
            std = np.nanstd(selected, axis=(0, 1))

            if self.ref_normalize_var.get():
                scale = np.nanmax(np.abs(avg))
                if scale > 0:
                    avg, std = avg/scale, std/scale

            self.ax_spec.plot(self.wavelength_axis, avg, color=reg["color"], label=reg["name"])
            self.ax_spec.fill_between(self.wavelength_axis, avg-std, avg+std, color=reg["color"], alpha=0.15)

        self.plot_references_if_enabled()
        self.ax_spec.set_title("Average spectra from selected X/Y regions")
        self.ax_spec.set_xlabel("Wavelength / spectral channel")
        self.ax_spec.set_ylabel("Intensity")
        self.ax_spec.relim()
        self.ax_spec.autoscale_view()
        self.ax_spec.margins(y=0.25)
        self.ax_spec.legend(fontsize=8)

        if draw:
            self.canvas.draw_idle()

    def plot_references_if_enabled(self):
        if not hasattr(self, "ref_overlay_var"):
            return
        if not self.ref_overlay_var.get():
            return

        for ref in self.references:
            x = ref["wavelength"]
            y = ref["intensity"]
            if self.ref_normalize_var.get():
                y = self.normalize_vec(y)
            self.ax_spec.plot(x, y, "--", linewidth=1.2, label=f"Ref: {ref['name']}")

    def clear_regions(self):
        self.regions = []
        if hasattr(self, "region_list"):
            self.region_list.delete(0, "end")
        if self.hsi_cube_yxw is not None:
            self.update_display_image()

    def refresh_region_list(self):
        self.region_list.delete(0, "end")
        for reg in self.regions:
            self.region_list.insert(
                "end",
                f"{reg['name']}: X {reg['x1']}:{reg['x2']}, Y {reg['y1']}:{reg['y2']}, color={reg['color']}"
            )

    def load_reference_csv(self):
        path = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv"), ("Text files", "*.txt"), ("All files", "*.*")]
        )
        if not path:
            return

        try:
            arr = np.genfromtxt(path, delimiter=",", comments="#")
            if arr.ndim != 2 or arr.shape[1] < 2:
                arr = np.genfromtxt(path, comments="#")
            if arr.ndim != 2 or arr.shape[1] < 2:
                raise ValueError("CSV must contain two columns: wavelength,intensity")

            mask = np.isfinite(arr[:, 0]) & np.isfinite(arr[:, 1])
            arr = arr[mask]

            ref = {
                "name": os.path.basename(path),
                "wavelength": arr[:, 0].astype(np.float32),
                "intensity": arr[:, 1].astype(np.float32)
            }

            self.references.append(ref)
            self.ref_list.insert("end", ref["name"])

            self.ax_spec.clear()
            self.plot_references_if_enabled()
            self.ax_spec.set_title("Loaded spectral references")
            self.ax_spec.set_xlabel("Wavelength")
            self.ax_spec.set_ylabel("Intensity")
            self.ax_spec.relim()
            self.ax_spec.autoscale_view()
            self.ax_spec.legend(fontsize=8)
            self.canvas.draw_idle()

            self.status_var.set(f"Status: reference CSV loaded: {ref['name']}")

        except Exception as e:
            messagebox.showerror("Reference load error", str(e))

    def clear_references(self):
        self.references = []
        self.ref_list.delete(0, "end")
        self.ax_spec.clear()
        self.ax_spec.set_title("References cleared")
        self.ax_spec.set_xlabel("Wavelength / spectral channel")
        self.ax_spec.set_ylabel("Intensity")
        self.canvas.draw_idle()

    def load_wavelength_vector(self):
        if self.hsi_cube_yxw is None:
            messagebox.showwarning("No cube", "Load or generate a cube first, then import wavelength vector.")
            return

        path = filedialog.askopenfilename(
            filetypes=[("CSV/TXT files", "*.csv *.txt"), ("All files", "*.*")]
        )
        if not path:
            return

        try:
            arr = np.genfromtxt(path, delimiter=",", comments="#")
            if arr.ndim > 1:
                arr = arr[:, 0]
            if arr.size != self.hsi_cube_yxw.shape[2]:
                raise ValueError(
                    f"Wavelength vector length {arr.size} does not match cube bands {self.hsi_cube_yxw.shape[2]}"
                )
            self.wavelength_axis = arr.astype(np.float32)
            self.update_display_image()
            self.status_var.set(f"Status: wavelength vector imported: {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Wavelength vector error", str(e))

    def log_ml(self, text):
        self.ml_log.insert("end", text + "\n")
        self.ml_log.see("end")
        self.root.update_idletasks()

    def get_feature_matrix_in_bounds(self):
        bounds = self.get_xy_bounds()
        if bounds is None:
            return None

        x1, x2, y1, y2 = bounds
        cube = self.hsi_cube_yxw[y1:y2, x1:x2, :]
        yy, xx, ww = cube.shape
        X = cube.reshape(-1, ww)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return X, yy, xx, ww, x1, x2, y1, y2

    def get_training_labels_from_regions_in_bounds(self, yy, xx, x0, y0):
        labels = -np.ones((yy, xx), dtype=int)

        for i, reg in enumerate(self.regions):
            rx1 = max(reg["x1"], x0) - x0
            rx2 = min(reg["x2"], x0 + xx) - x0
            ry1 = max(reg["y1"], y0) - y0
            ry2 = min(reg["y2"], y0 + yy) - y0

            if rx2 > rx1 and ry2 > ry1:
                labels[ry1:ry2, rx1:rx2] = i

        flat = labels.reshape(-1)
        mask = flat >= 0
        return flat, mask

    def make_classifier(self, method, level):
        n_estimators = {"Fast": 50, "Medium": 150, "High": 300}.get(level, 150)
        max_iter = {"Fast": 200, "Medium": 500, "High": 1000}.get(level, 500)

        if method == "SVM":
            return make_pipeline(StandardScaler(), SVC(kernel="rbf", gamma="scale"))
        if method == "RF":
            return RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)
        if method == "GB":
            return GradientBoostingClassifier(random_state=42)
        if method == "NN":
            return make_pipeline(
                StandardScaler(),
                MLPClassifier(hidden_layer_sizes=(80, 40), max_iter=max_iter, random_state=42)
            )
        return None

    def run_classification(self):
        if not SKLEARN_AVAILABLE:
            messagebox.showerror("Missing package", "Install scikit-learn first:\n\npip install scikit-learn")
            return
        if self.hsi_cube_yxw is None:
            messagebox.showwarning("No data", "Load or generate an HSI cube first.")
            return

        self.ml_log.delete("1.0", "end")

        method = self.ml_method_var.get()
        n_classes = max(2, int(self.ml_classes_var.get()))
        pca_n = max(1, int(self.ml_pca_var.get()))
        level = self.ml_level_var.get()

        result = self.get_feature_matrix_in_bounds()
        if result is None:
            return

        X, yy, xx, nw, x0, x2, y0, y2 = result

        self.log_ml(f"Analysis bounds: X={x0}:{x2}, Y={y0}:{y2}")
        self.log_ml(f"Data in range: Y={yy}, X={xx}, wavelengths={nw}, pixels={X.shape[0]}")
        self.log_ml(f"Method={method}, classes={n_classes}, PCA={pca_n}, level={level}")

        pca_n = min(pca_n, nw, X.shape[0])
        X_work = make_pipeline(
            StandardScaler(),
            PCA(n_components=pca_n, random_state=42)
        ).fit_transform(X)

        labels_flat, train_mask = self.get_training_labels_from_regions_in_bounds(yy, xx, x0, y0)

        if method == "KMeans" or np.sum(train_mask) < n_classes:
            self.log_ml("Using unsupervised KMeans fallback.")
            model = KMeans(n_clusters=n_classes, random_state=42, n_init=10)
            pred = model.fit_predict(X_work)
            rmse = np.sqrt(mean_squared_error(X_work, model.cluster_centers_[pred]))
            self.log_ml(f"KMeans spectral-feature RMSE = {rmse:.6g}")
        else:
            y_train = labels_flat[train_mask]
            X_train = X_work[train_mask]
            model = self.make_classifier(method, level)
            model.fit(X_train, y_train)
            pred = model.predict(X_work)
            train_pred = model.predict(X_train)
            acc = accuracy_score(y_train, train_pred)
            rmse = np.sqrt(mean_squared_error(y_train.astype(float), train_pred.astype(float)))
            self.log_ml(f"Training pixels in range = {len(y_train)}")
            self.log_ml(f"Training accuracy = {acc:.4f}")
            self.log_ml(f"Label RMSE = {rmse:.6g}")

        self.class_map = pred.reshape(yy, xx)
        self.show_classification_map(self.class_map, int(np.max(pred))+1, x0, x2, y0, y2)
        self.plot_class_median_spectra(x0, x2, y0, y2)
        self.status_var.set("Status: classification complete.")

    def show_classification_map(self, class_map, n_classes, x0, x2, y0, y2):
        self.ax_img.clear()
        self.clear_colorbar()

        im = self.ax_img.imshow(
            class_map,
            origin="upper",
            aspect="auto",
            cmap="tab20",
            extent=[x0, x2-1, y2-1, y0],
            vmin=-0.5,
            vmax=n_classes-0.5
        )

        divider = make_axes_locatable(self.ax_img)
        cax = divider.append_axes("right", size="2.5%", pad=0.12)
        self.colorbar = self.fig.colorbar(im, cax=cax)
        self.colorbar.set_label("Class")

        self.ax_img.set_title("Classified spectral image in active X/Y range")
        self.ax_img.set_xlabel("X spatial pixel")
        self.ax_img.set_ylabel("Y spatial pixel")
        self.apply_plot_limits(draw=False)
        self.redraw_region_rectangles()
        self.canvas.draw_idle()

    def plot_class_median_spectra(self, x0, x2, y0, y2):
        if self.class_map is None or self.hsi_cube_yxw is None:
            return

        crop = self.hsi_cube_yxw[y0:y2, x0:x2, :]
        self.ax_spec.clear()
        classes = np.unique(self.class_map)

        for c in classes:
            spectra = crop[self.class_map == c, :]
            if spectra.size == 0:
                continue

            med = np.nanmedian(spectra, axis=0)
            std = np.nanstd(spectra, axis=0)

            if self.ref_normalize_var.get():
                scale = np.nanmax(np.abs(med))
                if scale > 0:
                    med, std = med/scale, std/scale

            self.ax_spec.plot(self.wavelength_axis, med, label=f"Class {c} median")
            self.ax_spec.fill_between(self.wavelength_axis, med-std, med+std, alpha=0.12)

        self.plot_references_if_enabled()
        self.ax_spec.set_title("Median spectrum of each class in active X/Y range ± error")
        self.ax_spec.set_xlabel("Wavelength / spectral channel")
        self.ax_spec.set_ylabel("Intensity")
        self.ax_spec.relim()
        self.ax_spec.autoscale_view()
        self.ax_spec.margins(y=0.25)
        self.ax_spec.legend(fontsize=8)
        self.canvas.draw_idle()

    def update_dimension_report(self):
        if self.hsi_cube_yxw is None:
            self.dimension_report_var.set("No data loaded.")
            return

        y, x, w = self.hsi_cube_yxw.shape
        report = f"HSI cube mapping:\nY rows = {y}\nX columns = {x}\nWavelength bands = {w}"

        if self.raw_loaded_data is not None:
            report += f"\n\nRaw loaded data shape:\n{self.raw_loaded_data.shape}"

        if self.lookUpX is not None:
            report += f"\n\nlookUp_x length = {len(np.ravel(self.lookUpX))}"
        if self.lookUpY is not None:
            report += f"\nlookUp_y length = {len(np.ravel(self.lookUpY))}"

        report += f"\nnumOfFrame = {self.numOfFrame}\nroiX = {self.roiX}\nroiY = {self.roiY}"
        self.dimension_report_var.set(report)

    def convert_basler_to_yxw_cube(self, baslerObject):
        data = np.asarray(baslerObject.data, dtype=np.float32)

        self.lookUpX = getattr(baslerObject, "lookUp_x", None)
        self.lookUpY = getattr(baslerObject, "lookUp_y", None)
        self.roiX = getattr(baslerObject, "roiX", None)
        self.roiY = getattr(baslerObject, "roiY", None)
        self.numOfFrame = getattr(baslerObject, "numOfFrame", None)

        wavelength = np.ravel(self.lookUpX).astype(np.float32) if self.lookUpX is not None else None
        y_map = np.ravel(self.lookUpY).astype(np.float32) if self.lookUpY is not None else None

        if data.ndim == 3:
            shape = data.shape

            if wavelength is not None and len(wavelength) in shape:
                w_axis = list(shape).index(len(wavelength))
            else:
                w_axis = int(np.argmin(shape))

            remaining = [i for i in range(3) if i != w_axis]

            if y_map is not None and len(y_map) in [shape[i] for i in remaining]:
                y_axis = [i for i in remaining if shape[i] == len(y_map)][0]
                x_axis = [i for i in remaining if i != y_axis][0]
            else:
                if shape[remaining[0]] <= shape[remaining[1]]:
                    y_axis, x_axis = remaining[0], remaining[1]
                else:
                    y_axis, x_axis = remaining[1], remaining[0]

            cube_yxw = np.transpose(data, (y_axis, x_axis, w_axis))

        elif data.ndim == 2:
            cube_yxw = data[:, None, :]
        else:
            raise ValueError(f"Unsupported Basler data shape: {data.shape}")

        y_size, x_size, w_size = cube_yxw.shape

        if wavelength is None or len(wavelength) != w_size:
            wavelength = np.arange(w_size, dtype=np.float32)
        if y_map is None or len(y_map) != y_size:
            y_map = np.arange(y_size, dtype=np.float32)

        x_map = np.arange(x_size, dtype=np.float32)
        return cube_yxw, x_map, y_map, wavelength

    def save_npz(self):
        if self.hsi_cube_yxw is None:
            messagebox.showwarning("Save", "No HSI cube to save.")
            return

        path = filedialog.asksaveasfilename(defaultextension=".npz", filetypes=[("NumPy archive", "*.npz")])
        if not path:
            return

        np.savez_compressed(
            path,
            hsi_cube_yxw=self.hsi_cube_yxw,
            wavelength_axis=self.wavelength_axis,
            x_map=self.x_map,
            y_map=self.y_map,
            regions=json.dumps(self.regions)
        )
        messagebox.showinfo("Saved", f"Saved NPZ:\n{path}")

    def load_npz(self):
        path = filedialog.askopenfilename(filetypes=[("NumPy archive", "*.npz")])
        if not path:
            return

        data = np.load(path, allow_pickle=True)
        self.hsi_cube_yxw = data["hsi_cube_yxw"].astype(np.float32)
        self.raw_loaded_data = self.hsi_cube_yxw.copy()
        self.cube = self.hsi_cube_yxw

        self.wavelength_axis = data["wavelength_axis"].astype(np.float32) if "wavelength_axis" in data else np.arange(self.hsi_cube_yxw.shape[2])
        self.x_map = data["x_map"].astype(np.float32) if "x_map" in data else np.arange(self.hsi_cube_yxw.shape[1])
        self.y_map = data["y_map"].astype(np.float32) if "y_map" in data else np.arange(self.hsi_cube_yxw.shape[0])

        self.regions = []
        if "regions" in data:
            try:
                self.regions = json.loads(str(data["regions"]))
            except Exception:
                pass

        self.refresh_region_list()
        self.update_display_image()
        self.status_var.set(f"Status: NPZ loaded {self.hsi_cube_yxw.shape}")

    def save_cube_file(self):
        if self.hsi_cube_yxw is None:
            messagebox.showwarning("Save", "No HSI cube to save.")
            return

        path = filedialog.asksaveasfilename(defaultextension=".cube", filetypes=[("Basler cube object", "*.cube")])
        if not path:
            return

        obj = baslerData(
            numOfFrame=self.hsi_cube_yxw.shape[0],
            roiX=self.hsi_cube_yxw.shape[1],
            roiY=self.hsi_cube_yxw.shape[0],
            lookUp_x=self.wavelength_axis,
            lookUp_y=self.y_map,
            data=self.hsi_cube_yxw,
            deviceModel=CAMERA_NAME,
            exposureTime=self.exposure_var.get(),
            Gain=self.gain_var.get(),
            averagingTime=self.average_var.get(),
            sizeofDataCube=self.hsi_cube_yxw.shape,
            describtion="Saved as Y-X-wavelength HSI cube by SpectraLab"
        )

        with open(path, "wb") as f:
            pk.dump(obj, f)

        messagebox.showinfo("Saved", f"Saved Basler CUBE object:\n{path}")

    def load_cube_file(self):
        path = filedialog.askopenfilename(filetypes=[("Basler cube object", "*.cube"), ("All files", "*.*")])
        if not path:
            return

        self.status_var.set("Status: waiting... loading Basler CUBE object")
        self.root.update_idletasks()

        with open(path, "rb") as f:
            obj = pk.load(f)

        self.hsi_cube_yxw, self.x_map, self.y_map, self.wavelength_axis = self.convert_basler_to_yxw_cube(obj)
        self.raw_loaded_data = np.asarray(obj.data, dtype=np.float32)
        self.cube = self.hsi_cube_yxw

        self.regions = []
        self.refresh_region_list()
        self.update_display_image()
        self.status_var.set(f"Status: Basler cube converted to Y,X,λ = {self.hsi_cube_yxw.shape}")

    def save_fits_file(self):
        if not ASTROPY_AVAILABLE:
            messagebox.showerror("FITS error", "Install astropy first:\n\npip install astropy")
            return
        if self.hsi_cube_yxw is None:
            messagebox.showwarning("Save", "No HSI cube to save.")
            return

        path = filedialog.asksaveasfilename(defaultextension=".fits", filetypes=[("FITS file", "*.fits")])
        if not path:
            return

        hdu = fits.PrimaryHDU(self.hsi_cube_yxw.astype(np.float32))
        hdu.header["AXISORD"] = "Y,X,W"
        hdu.header["CAMERA"] = CAMERA_NAME
        hdu.writeto(path, overwrite=True)
        messagebox.showinfo("Saved", f"Saved FITS Y-X-wavelength cube:\n{path}")

    def load_fits_file(self):
        if not ASTROPY_AVAILABLE:
            messagebox.showerror("FITS error", "Install astropy first:\n\npip install astropy")
            return

        path = filedialog.askopenfilename(filetypes=[("FITS file", "*.fits"), ("FITS file", "*.fit"), ("All files", "*.*")])
        if not path:
            return

        self.status_var.set("Status: waiting... loading FITS file")
        self.root.update_idletasks()

        with fits.open(path) as hdul:
            data = None
            for hdu in hdul:
                if hdu.data is not None:
                    data = np.asarray(hdu.data, dtype=np.float32)
                    break

        if data is None:
            messagebox.showerror("FITS error", "No image data found.")
            return

        if data.ndim == 3:
            self.hsi_cube_yxw = data
        elif data.ndim == 2:
            self.hsi_cube_yxw = data[:, None, :]
        else:
            messagebox.showerror("FITS error", f"Unsupported FITS shape: {data.shape}")
            return

        self.raw_loaded_data = data.copy()
        self.cube = self.hsi_cube_yxw
        self.x_map = np.arange(self.hsi_cube_yxw.shape[1], dtype=np.float32)
        self.y_map = np.arange(self.hsi_cube_yxw.shape[0], dtype=np.float32)
        self.wavelength_axis = np.arange(self.hsi_cube_yxw.shape[2], dtype=np.float32)

        self.regions = []
        self.refresh_region_list()
        self.update_display_image()
        self.status_var.set(f"Status: FITS loaded as Y,X,λ cube {self.hsi_cube_yxw.shape}")

    def deload_all(self):
        self.scan_running = False
        self.hsi_cube_yxw = None
        self.display_image = None
        self.raw_loaded_data = None
        self.cube = None
        self.x_map = None
        self.y_map = None
        self.wavelength_axis = None
        self.lookUpX = None
        self.lookUpY = None
        self.numOfFrame = None
        self.roiX = None
        self.roiY = None
        self.regions = []
        self.references = []
        self.class_map = None

        if hasattr(self, "region_list"):
            self.region_list.delete(0, "end")
        if hasattr(self, "ref_list"):
            self.ref_list.delete(0, "end")

        self.ax_img.clear()
        self.ax_spec.clear()
        self.clear_colorbar()
        self.dimension_report_var.set("No data loaded.")
        self.status_var.set("Status: all data cleared.")
        self.canvas.draw_idle()

    def on_close(self):
        self.scan_running = False
        self.cam.close()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = SpectraLabPushBroomGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
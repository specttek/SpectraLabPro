#!/usr/bin/env python3
"""
SpectraLab Push-Broom Imaging — SpectTek Co. (M. Raouf)

Data model:
    hsi_cube_yxw[Y_scan, X_spatial, wavelength]

Basler live mapping:
    raw_frame[lambda_pixel, X_spatial] = 1200 × 1600
    line_xw = raw_frame.T = 1600 × 1200
    live cube = scan_steps × 1600 × 1200

SPECIM live mapping:
    line_xw[X_spatial, wavelength]
    live cube = scan_steps × SPECIM_X × SPECIM_BANDS
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
SPECIM_NAME = "SPECIM hyperspectral line camera"
LINKEDIN_URL = "https://www.linkedin.com/in/specttek/"
SENSOR_WIDTH = 1600
SENSOR_HEIGHT = 1200

REGION_COLORS = [
    "red", "lime", "cyan", "yellow",
    "magenta", "orange", "deepskyblue", "white"
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
        return "Basler dummy mode enabled."

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
        except Exception:
            pass

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
        except Exception:
            return None

        return None

    def synthetic_frame(self):
        """
        Simulated Basler detector frame:

            frame[lambda_pixel, X_spatial] = 1200 × 1600
        """
        lam = np.arange(SENSOR_HEIGHT)[:, None]
        x = np.arange(SENSOR_WIDTH)[None, :]

        spectral_axis = (
            0.8 * np.exp(-((lam - 250) / 50) ** 2)
            + 1.2 * np.exp(-((lam - 620) / 90) ** 2)
            + 0.9 * np.exp(-((lam - 980) / 120) ** 2)
        )

        spatial_axis = (
            0.5
            + 0.35 * np.exp(-((x - 450) / 160) ** 2)
            + 0.45 * np.exp(-((x - 1100) / 230) ** 2)
        )

        frame = spectral_axis * spatial_axis + np.random.normal(
            0, 0.03, (SENSOR_HEIGHT, SENSOR_WIDTH)
        )

        frame -= np.nanmin(frame)
        if np.nanmax(frame) > 0:
            frame /= np.nanmax(frame)

        return frame.astype(np.float32)

    def close(self):
        try:
            if self.camera is not None:
                self.camera.Close()
        except Exception:
            pass

        self.camera = None
        self.connected = False


class SpecimCamera:
    """
    SPECIM placeholder adapter.

    Replace connect() and get_line_cube() with SPECIM SDK calls.

    Expected output:
        line_xw[X_spatial, wavelength]
    """

    def __init__(self):
        self.connected = False
        self.x_pixels = 320
        self.bands = 220
        self.wavelength = np.linspace(400, 900, self.bands).astype(np.float32)

    def connect(self):
        self.connected = True
        return "SPECIM placeholder connected. Replace with real SPECIM SDK calls."

    def get_line_cube(self, step=0):
        x = np.arange(self.x_pixels)[:, None]
        wl = self.wavelength[None, :]

        spatial = (
            0.55
            + 0.35 * np.exp(-((x - 90 - 20 * np.sin(step / 12)) / 35) ** 2)
            + 0.30 * np.exp(-((x - 230) / 55) ** 2)
        )

        spectral = (
            0.7 * np.exp(-((wl - 550) / 45) ** 2)
            + 1.1 * np.exp(-((wl - 680) / 65) ** 2)
            + 0.8 * np.exp(-((wl - 820) / 75) ** 2)
        )

        line = spatial * spectral + np.random.normal(
            0, 0.02, (self.x_pixels, self.bands)
        )

        line -= np.nanmin(line)
        if np.nanmax(line) > 0:
            line /= np.nanmax(line)

        return line.astype(np.float32)

    def close(self):
        self.connected = False


class SpectraLabPushBroomGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("SpectraLab Push-Broom Imaging — SpectTek Co.")
        self.root.geometry("1880x1060")

        self.camera_password = "SpectTek"
        self.logo_image = None

        self.basler = BaslerCamera()
        self.specim = SpecimCamera()
        self.active_camera = "Basler"
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

        self.colorbar = None
        self.chip_colorbar = None
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
                "Camera", "Scan", "Dummy", "Regions",
                "Save / Load", "Display", "Analysis", "References"
            ]
        )
        self.section_selector.pack(fill="x", padx=6, pady=6)
        self.section_selector.bind("<<ComboboxSelected>>", lambda e: self.show_section())

        self.left_body = ttk.Frame(left)
        self.left_body.pack(fill="both", expand=True, padx=4, pady=4)

        for name in self.section_selector["values"]:
            self.section_frames[name] = ttk.Frame(self.left_body)

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

    def build_camera_section(self, parent):
        ttk.Label(
            parent,
            text="SpecTek Hyperspectral Imager",
            font=("Arial", 13, "bold")
        ).pack(pady=(10, 4))

        ttk.Label(
            parent,
            text=(
                f"{CAMERA_NAME}\n"
                "Basler raw frame: detector Y = wavelength, detector X = spatial slit\n\n"
                f"{SPECIM_NAME}"
            ),
            justify="center"
        ).pack(pady=(0, 10))

        self.camera_type_var = tk.StringVar(value="Basler")
        ttk.Label(parent, text="Camera source").pack(anchor="w", padx=8)
        ttk.Combobox(
            parent,
            textvariable=self.camera_type_var,
            state="readonly",
            values=["Basler", "SPECIM"]
        ).pack(fill="x", padx=8, pady=4)

        ttk.Button(
            parent,
            text="Connect Selected Camera",
            command=self.connect_camera
        ).pack(fill="x", padx=8, pady=5)

        ttk.Button(
            parent,
            text="Enable Basler Dummy/Test Mode",
            command=self.enable_dummy_mode
        ).pack(fill="x", padx=8, pady=5)

        self.status_var = tk.StringVar(value="Status: not connected")
        ttk.Label(parent, textvariable=self.status_var, wraplength=460).pack(padx=8, pady=10)

        ttk.Separator(parent).pack(fill="x", padx=8, pady=8)

        self.exposure_var = tk.DoubleVar(value=10000.0)
        self.gain_var = tk.DoubleVar(value=0.0)
        self.average_var = tk.IntVar(value=3)

        self.make_entry(parent, "Integration / exposure time [µs]", self.exposure_var)
        self.make_entry(parent, "Gain", self.gain_var)
        self.make_entry(parent, "Frame average", self.average_var)

        ttk.Button(
            parent,
            text="Apply Camera Settings",
            command=self.apply_settings
        ).pack(fill="x", padx=8, pady=8)

        if PIL_AVAILABLE:
            try:
                img = Image.open("logo.png")
                img.thumbnail((220, 220))
                self.logo_image = ImageTk.PhotoImage(img)
                ttk.Label(parent, image=self.logo_image).pack(pady=(15, 10))
            except Exception:
                ttk.Label(parent, text="[logo.png not found]").pack(pady=10)

    def build_scan_section(self, parent):
        ttk.Label(parent, text="Push-Broom Live Scan", font=("Arial", 12, "bold")).pack(pady=10)

        self.steps_var = tk.IntVar(value=100)
        self.delay_var = tk.DoubleVar(value=0.05)
        self.live_update_every_var = tk.IntVar(value=1)

        self.make_entry(parent, "Scan steps / final Y rows", self.steps_var)
        self.make_entry(parent, "Delay per step [s]", self.delay_var)
        self.make_entry(parent, "Live display update every N lines", self.live_update_every_var)

        ttk.Label(
            parent,
            text=(
                "Basler live mapping:\n"
                "raw frame[λ_pixel, X_spatial] = 1200 × 1600\n"
                "line_xλ = frame.T = 1600 × 1200\n"
                "scan cube = steps × 1600 × 1200\n\n"
                "The bottom plot shows the most recent detector chip map:\n"
                "λ × X = 1200 × 1600"
            ),
            wraplength=460,
            justify="left"
        ).pack(anchor="w", padx=8, pady=8)

        ttk.Button(
            parent,
            text="Start Push-Broom Live Scan",
            command=self.start_scan
        ).pack(fill="x", padx=8, pady=10)

        ttk.Button(
            parent,
            text="Stop Scan",
            command=self.stop_scan
        ).pack(fill="x", padx=8, pady=5)

        self.progress = ttk.Progressbar(parent, orient="horizontal", mode="determinate")
        self.progress.pack(fill="x", padx=8, pady=10)

    def build_dummy_section(self, parent):
        ttk.Label(parent, text="Dummy Test Data", font=("Arial", 12, "bold")).pack(pady=10)

        self.dummy_y_var = tk.IntVar(value=120)
        self.dummy_x_var = tk.IntVar(value=160)
        self.dummy_w_var = tk.IntVar(value=220)
        self.dummy_noise_var = tk.DoubleVar(value=0.03)

        self.make_entry(parent, "Dummy Y rows", self.dummy_y_var)
        self.make_entry(parent, "Dummy X columns", self.dummy_x_var)
        self.make_entry(parent, "Dummy wavelength bands", self.dummy_w_var)
        self.make_entry(parent, "Noise level", self.dummy_noise_var)

        ttk.Button(
            parent,
            text="Generate Dummy Y-X-λ Cube",
            command=self.generate_dummy_cube
        ).pack(fill="x", padx=8, pady=10)

    def build_region_section(self, parent):
        ttk.Label(parent, text="Region Comparison", font=("Arial", 12, "bold")).pack(pady=10)

        ttk.Label(
            parent,
            text="Drag on image to add regions. Average spectra are extracted from the full Y-X-λ cube.",
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

        ttk.Button(
            file_box,
            text="Import wavelength vector CSV/TXT",
            command=self.load_wavelength_vector
        ).pack(fill="x", padx=6, pady=5)

        ttk.Button(
            file_box,
            text="Deload / Clear All",
            command=self.deload_all
        ).pack(fill="x", padx=6, pady=5)

        dim_box = ttk.LabelFrame(parent, text="Current Dimensions")
        dim_box.pack(fill="both", expand=True, padx=8, pady=5)

        self.dimension_report_var = tk.StringVar(value="No data loaded.")
        ttk.Label(
            dim_box,
            textvariable=self.dimension_report_var,
            wraplength=460,
            justify="left"
        ).pack(anchor="w", padx=6, pady=5)

    def build_display_section(self, parent):
        ttk.Label(parent, text="Display Controls", font=("Arial", 12, "bold")).pack(pady=8)

        self.display_mode_var = tk.StringVar(value="RGB")

        box = ttk.LabelFrame(parent, text="Display mode")
        box.pack(fill="x", padx=8, pady=5)

        ttk.Radiobutton(
            box,
            text="RGB preview",
            variable=self.display_mode_var,
            value="RGB",
            command=self.update_display_image
        ).pack(anchor="w", padx=8)

        ttk.Radiobutton(
            box,
            text="Single wavelength/channel",
            variable=self.display_mode_var,
            value="Single",
            command=self.update_display_image
        ).pack(anchor="w", padx=8)

        self.single_wave_var = tk.DoubleVar(value=550.0)
        self.red_wave_var = tk.DoubleVar(value=650.0)
        self.green_wave_var = tk.DoubleVar(value=550.0)
        self.blue_wave_var = tk.DoubleVar(value=450.0)
        self.brightness_var = tk.DoubleVar(value=0.0)
        self.contrast_var = tk.DoubleVar(value=1.0)

        for label, var in [
            ("Single wavelength / band", self.single_wave_var),
            ("Red wavelength / band", self.red_wave_var),
            ("Green wavelength / band", self.green_wave_var),
            ("Blue wavelength / band", self.blue_wave_var),
            ("Brightness offset", self.brightness_var),
            ("Contrast scale", self.contrast_var),
        ]:
            self.make_entry(parent, label, var)

        ttk.Button(
            parent,
            text="Apply Display",
            command=self.update_display_image
        ).pack(fill="x", padx=8, pady=5)

        self.transform_var = tk.StringVar(value="Original")
        ttk.Combobox(
            parent,
            textvariable=self.transform_var,
            state="readonly",
            values=["Original", "Transpose X/Y", "Flip X", "Flip Y", "Rotate 90", "Rotate 180", "Rotate 270"]
        ).pack(fill="x", padx=8, pady=5)

        ttk.Button(
            parent,
            text="Apply Orientation",
            command=self.apply_display_transform
        ).pack(fill="x", padx=8, pady=5)

    def build_analysis_section(self, parent):
        ttk.Label(parent, text="ML Classification", font=("Arial", 12, "bold")).pack(pady=8)

        self.ml_method_var = tk.StringVar(value="RF")
        self.ml_classes_var = tk.IntVar(value=3)
        self.ml_pca_var = tk.IntVar(value=10)
        self.ml_level_var = tk.StringVar(value="Medium")

        ttk.Combobox(
            parent,
            textvariable=self.ml_method_var,
            state="readonly",
            values=["RF", "SVM", "GB", "NN", "KMeans"]
        ).pack(fill="x", padx=8, pady=4)

        self.make_entry(parent, "Number of classes", self.ml_classes_var)
        self.make_entry(parent, "PCA components", self.ml_pca_var)

        ttk.Combobox(
            parent,
            textvariable=self.ml_level_var,
            state="readonly",
            values=["Fast", "Medium", "High"]
        ).pack(fill="x", padx=8, pady=4)

        ttk.Button(
            parent,
            text="Run Classification in X/Y Range",
            command=self.run_classification
        ).pack(fill="x", padx=8, pady=5)

        self.ml_log = tk.Text(parent, height=18, wrap="word")
        self.ml_log.pack(fill="both", expand=True, padx=8, pady=8)

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

        for text, var in [
            ("X min", self.xmin_var),
            ("X max", self.xmax_var),
            ("Y min", self.ymin_var),
            ("Y max", self.ymax_var),
        ]:
            ttk.Label(range_bar, text=text).pack(side="left", padx=(8, 2))
            ttk.Entry(range_bar, textvariable=var, width=9).pack(side="left", padx=2)

        ttk.Button(range_bar, text="Apply range", command=self.apply_plot_limits).pack(side="left", padx=10)
        ttk.Button(range_bar, text="Reset", command=self.reset_plot_limits).pack(side="left", padx=4)

        self.fig = Figure(figsize=(15.8, 10.2), dpi=100)

        gs = self.fig.add_gridspec(
            3,
            1,
            height_ratios=[4.0, 1.6, 1.5],
            hspace=0.48,
            left=0.055,
            right=0.93,
            top=0.96,
            bottom=0.07
        )

        self.ax_img = self.fig.add_subplot(gs[0])
        self.ax_spec = self.fig.add_subplot(gs[1])
        self.ax_chip = self.fig.add_subplot(gs[2])

        self.ax_img.set_title("Live RGB push-broom image")
        self.ax_img.set_xlabel("X spatial pixel")
        self.ax_img.set_ylabel("Y scan direction")

        self.ax_spec.set_title("Average / selected spectrum")
        self.ax_spec.set_xlabel("Wavelength / spectral channel")
        self.ax_spec.set_ylabel("Intensity")

        self.ax_chip.set_title("Live detector chip map: wavelength axis vs X pixels")
        self.ax_chip.set_xlabel("X spatial pixel on detector")
        self.ax_chip.set_ylabel("Wavelength pixel / spectral channel")

        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
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

        self.active_camera = self.camera_type_var.get()

        if self.active_camera == "SPECIM":
            self.status_var.set("Status: " + self.specim.connect())
            self.wavelength_axis = self.specim.wavelength.copy()
        else:
            self.status_var.set("Status: " + self.basler.connect())

    def enable_dummy_mode(self):
        self.active_camera = "Basler"
        self.camera_type_var.set("Basler")
        self.status_var.set("Status: " + self.basler.enable_dummy())

    def apply_settings(self):
        if self.active_camera == "SPECIM":
            self.status_var.set("Status: SPECIM selected. Add SDK exposure/gain calls in SpecimCamera.")
            return

        if not self.basler.connected:
            messagebox.showwarning("Camera", "Connect Basler camera first.")
            return

        self.basler.set_exposure_us(self.exposure_var.get())
        self.basler.set_gain(self.gain_var.get())

        self.status_var.set(
            f"Settings applied: exposure={self.exposure_var.get()} µs, gain={self.gain_var.get()}"
        )

    def averaged_frame(self):
        frames = []

        for _ in range(max(1, int(self.average_var.get()))):
            fr = self.basler.get_frame()
            if fr is not None:
                frames.append(fr)
            time.sleep(0.002)

        return np.mean(frames, axis=0) if frames else None

    def extract_basler_line_xw(self, frame):
        """
        Correct Basler push-broom mapping.

        Raw frame:
            frame[lambda_pixel, X_spatial] = 1200 × 1600

        One scan step:
            line_xw[X_spatial, wavelength] = frame.T = 1600 × 1200

        Full scan:
            cube[Y_scan, X_spatial, wavelength]
        """
        frame = np.asarray(frame, dtype=np.float32)
        return frame.T.astype(np.float32)

    def start_scan(self):
        if self.active_camera == "SPECIM":
            if not self.specim.connected:
                messagebox.showwarning("SPECIM", "Connect SPECIM first.")
                return
        else:
            if not self.basler.connected:
                messagebox.showwarning("Basler", "Connect Basler camera or enable dummy mode first.")
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
        update_every = max(1, int(self.live_update_every_var.get()))

        lines = []
        self.status_var.set("Status: live push-broom scan running...")

        for i in range(steps):
            if not self.scan_running:
                break

            if self.active_camera == "SPECIM":
                line_xw = self.specim.get_line_cube(step=i)

                if self.wavelength_axis is None or len(self.wavelength_axis) != line_xw.shape[1]:
                    self.wavelength_axis = self.specim.wavelength.copy()

            else:
                frame = self.averaged_frame()
                if frame is None:
                    continue

                line_xw = self.extract_basler_line_xw(frame)

                if self.lookUpX is not None and len(np.ravel(self.lookUpX)) == line_xw.shape[1]:
                    self.wavelength_axis = np.ravel(self.lookUpX).astype(np.float32)
                elif self.wavelength_axis is None or len(self.wavelength_axis) != line_xw.shape[1]:
                    self.wavelength_axis = np.arange(line_xw.shape[1], dtype=np.float32)

            lines.append(line_xw)

            if (i % update_every == 0) or i == steps - 1:
                live_cube = np.stack(lines, axis=0).astype(np.float32)

                self.hsi_cube_yxw = live_cube
                self.raw_loaded_data = live_cube.copy()
                self.cube = live_cube
                self.x_map = np.arange(live_cube.shape[1], dtype=np.float32)
                self.y_map = np.arange(live_cube.shape[0], dtype=np.float32)

                self.root.after(0, self.update_display_image)

            self.root.after(0, self.progress.configure, {"value": 100 * (i + 1) / steps})
            time.sleep(delay)

        self.scan_running = False

        if lines:
            self.hsi_cube_yxw = np.stack(lines, axis=0).astype(np.float32)
            self.raw_loaded_data = self.hsi_cube_yxw.copy()
            self.cube = self.hsi_cube_yxw
            self.x_map = np.arange(self.hsi_cube_yxw.shape[1], dtype=np.float32)
            self.y_map = np.arange(self.hsi_cube_yxw.shape[0], dtype=np.float32)

            self.root.after(0, self.update_display_image)
            self.status_var.set(f"Status: scan finished. HSI Y,X,λ = {self.hsi_cube_yxw.shape}")

    def nearest_band(self, target):
        if self.wavelength_axis is None:
            return 0

        wl = np.asarray(self.wavelength_axis, dtype=np.float32)

        if np.nanmax(wl) > 100:
            return int(np.argmin(np.abs(wl - target)))

        return int(np.clip(round(target), 0, len(wl) - 1))

    def adjust_bc(self, image):
        out = (image - 0.5) * float(self.contrast_var.get()) + 0.5 + float(self.brightness_var.get())
        return np.clip(out, 0, 1)

    def make_rgb_from_cube(self):
        cube = self.hsi_cube_yxw

        bands = [
            self.nearest_band(float(self.red_wave_var.get())),
            self.nearest_band(float(self.green_wave_var.get())),
            self.nearest_band(float(self.blue_wave_var.get()))
        ]

        rgb = np.stack(
            [cube[:, :, bands[0]], cube[:, :, bands[1]], cube[:, :, bands[2]]],
            axis=2
        )

        lo, hi = np.nanpercentile(rgb, 1), np.nanpercentile(rgb, 99)
        return self.adjust_bc(np.clip((rgb - lo) / (hi - lo + 1e-12), 0, 1))

    def make_single_band_image(self):
        idx = self.nearest_band(float(self.single_wave_var.get()))
        idx = int(np.clip(idx, 0, self.hsi_cube_yxw.shape[2] - 1))

        img = self.hsi_cube_yxw[:, :, idx]

        lo, hi = np.nanpercentile(img, 1), np.nanpercentile(img, 99)
        return self.adjust_bc(np.clip((img - lo) / (hi - lo + 1e-12), 0, 1)), idx

    def update_display_image(self):
        if self.hsi_cube_yxw is None:
            return

        if self.display_mode_var.get() == "Single":
            self.display_image, idx = self.make_single_band_image()
            title = f"Single wavelength/channel | band={idx}, λ={self.wavelength_axis[idx]:.3g}"
            self.show_display_image(title, "gray")
        else:
            self.display_image = self.make_rgb_from_cube()
            self.show_display_image("Live RGB push-broom image", None)

    def clear_colorbar(self):
        if self.colorbar is not None:
            try:
                self.colorbar.remove()
            except Exception:
                pass
            self.colorbar = None

    def clear_chip_colorbar(self):
        if self.chip_colorbar is not None:
            try:
                self.chip_colorbar.remove()
            except Exception:
                pass
            self.chip_colorbar = None

    def show_display_image(self, title, cmap=None):
        ny, nx, nw = self.hsi_cube_yxw.shape

        self.ax_img.clear()
        self.clear_colorbar()

        im = self.ax_img.imshow(
            self.display_image,
            cmap=cmap,
            origin="upper",
            aspect="auto",
            extent=[0, nx - 1, ny - 1, 0]
        )

        if self.display_image.ndim == 2:
            divider = make_axes_locatable(self.ax_img)
            cax = divider.append_axes("right", size="2.5%", pad=0.12)
            self.colorbar = self.fig.colorbar(im, cax=cax)

        self.ax_img.set_title(f"{title} | Y={ny}, X={nx}, λ={nw}")
        self.ax_img.set_xlabel("X spatial pixel")
        self.ax_img.set_ylabel("Y scan direction")

        self.apply_plot_limits(draw=False)
        self.redraw_region_rectangles()

        self.ax_spec.clear()
        self.ax_spec.set_title("Select pixel or region to show spectrum")
        self.ax_spec.set_xlabel("Wavelength / spectral channel")
        self.ax_spec.set_ylabel("Intensity")

        if self.regions:
            self.replot_all_region_spectra(draw=False)
        else:
            self.plot_references_if_enabled()

        self.plot_detector_chip_map()
        self.update_dimension_report()
        self.canvas.draw_idle()

    def get_latest_detector_chip_map(self):
        """
        Return detector chip map as:

            wavelength × X

        Basler:
            latest scan line is X × wavelength = 1600 × 1200
            chip map = wavelength × X = 1200 × 1600

        SPECIM:
            latest scan line is X × wavelength
            chip map = wavelength × X
        """
        if self.hsi_cube_yxw is None:
            return None

        latest_line_xw = self.hsi_cube_yxw[-1, :, :]
        chip_map = latest_line_xw.T

        return chip_map

    def plot_detector_chip_map(self):
        if not hasattr(self, "ax_chip"):
            return

        chip_map = self.get_latest_detector_chip_map()

        self.ax_chip.clear()
        self.clear_chip_colorbar()

        if chip_map is None:
            self.ax_chip.set_title("Live detector chip map: no data")
            self.ax_chip.set_xlabel("X spatial pixel on detector")
            self.ax_chip.set_ylabel("Wavelength pixel / spectral channel")
            return

        n_wave, n_x = chip_map.shape

        im = self.ax_chip.imshow(
            chip_map,
            origin="lower",
            aspect="auto",
            extent=[0, n_x - 1, 0, n_wave - 1],
            cmap="viridis"
        )

        self.ax_chip.set_title(
            f"Live detector chip map | wavelength × X = {n_wave} × {n_x}"
        )
        self.ax_chip.set_xlabel("X spatial pixel on detector")
        self.ax_chip.set_ylabel("Wavelength pixel / spectral channel")

        if self.wavelength_axis is not None and len(self.wavelength_axis) == n_wave:
            ticks = np.linspace(0, n_wave - 1, 5).astype(int)
            labels = [f"{self.wavelength_axis[t]:.0f}" for t in ticks]
            self.ax_chip.set_yticks(ticks)
            self.ax_chip.set_yticklabels(labels)
            self.ax_chip.set_ylabel("Wavelength / nm")

        divider = make_axes_locatable(self.ax_chip)
        cax = divider.append_axes("right", size="2.5%", pad=0.12)
        self.chip_colorbar = self.fig.colorbar(im, cax=cax)
        self.chip_colorbar.set_label("Counts / intensity")

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

        return (
            int(np.clip(x1, 0, nx - 1)),
            int(np.clip(x2, x1 + 1, nx)),
            int(np.clip(y1, 0, ny - 1)),
            int(np.clip(y2, y1 + 1, ny))
        )

    def apply_plot_limits(self, draw=True):
        if self.hsi_cube_yxw is None:
            return

        ny, nx, _ = self.hsi_cube_yxw.shape

        if self.xmin_var.get() and self.xmax_var.get():
            self.ax_img.set_xlim(float(self.xmin_var.get()), float(self.xmax_var.get()))
        else:
            self.ax_img.set_xlim(0, nx - 1)

        if self.ymin_var.get() and self.ymax_var.get():
            self.ax_img.set_ylim(float(self.ymax_var.get()), float(self.ymin_var.get()))
        else:
            self.ax_img.set_ylim(ny - 1, 0)

        if draw:
            self.canvas.draw_idle()

    def reset_plot_limits(self):
        self.xmin_var.set("")
        self.xmax_var.set("")
        self.ymin_var.set("")
        self.ymax_var.set("")
        self.apply_plot_limits()

    def apply_display_transform(self):
        if self.hsi_cube_yxw is None:
            return

        mode = self.transform_var.get()
        cube = self.hsi_cube_yxw.copy()

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
        self.clear_regions()
        self.update_display_image()

    def redraw_region_rectangles(self):
        for reg in self.regions:
            self.ax_img.add_patch(Rectangle(
                (reg["x1"], reg["y1"]),
                reg["x2"] - reg["x1"],
                reg["y2"] - reg["y1"],
                edgecolor=reg["color"],
                facecolor="none",
                linewidth=2
            ))

    def normalize_vec(self, x):
        m = np.nanmax(np.abs(x))
        return x / m if m > 0 else x

    def on_image_click(self, event):
        if event.inaxes != self.ax_img:
            return
        if self.hsi_cube_yxw is None or event.xdata is None or event.ydata is None:
            return

        x = int(np.clip(event.xdata, 0, self.hsi_cube_yxw.shape[1] - 1))
        y = int(np.clip(event.ydata, 0, self.hsi_cube_yxw.shape[0] - 1))

        spec = self.hsi_cube_yxw[y, x, :]

        if self.ref_normalize_var.get():
            spec = self.normalize_vec(spec)

        self.ax_spec.clear()
        self.ax_spec.plot(self.wavelength_axis, spec, label=f"X={x}, Y={y}")
        self.plot_references_if_enabled()

        self.ax_spec.set_title(f"Spectrum at X={x}, Y={y}")
        self.ax_spec.set_xlabel("Wavelength / spectral channel")
        self.ax_spec.set_ylabel("Intensity")
        self.ax_spec.legend(fontsize=8)
        self.ax_spec.relim()
        self.ax_spec.autoscale_view()
        self.ax_spec.margins(y=0.20)

        self.canvas.draw_idle()

    def on_select_area(self, eclick, erelease):
        if self.hsi_cube_yxw is None or eclick.xdata is None or erelease.xdata is None:
            return

        x1, x2 = sorted([int(eclick.xdata), int(erelease.xdata)])
        y1, y2 = sorted([int(eclick.ydata), int(erelease.ydata)])

        x1 = int(np.clip(x1, 0, self.hsi_cube_yxw.shape[1] - 1))
        x2 = int(np.clip(x2, 0, self.hsi_cube_yxw.shape[1]))
        y1 = int(np.clip(y1, 0, self.hsi_cube_yxw.shape[0] - 1))
        y2 = int(np.clip(y2, 0, self.hsi_cube_yxw.shape[0]))

        if x2 <= x1 or y2 <= y1:
            return

        color = REGION_COLORS[len(self.regions) % len(REGION_COLORS)]

        self.regions.append({
            "name": f"Region {len(self.regions) + 1}",
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
            return

        self.ax_spec.clear()

        for reg in self.regions:
            sub = self.hsi_cube_yxw[reg["y1"]:reg["y2"], reg["x1"]:reg["x2"], :]
            if sub.size == 0:
                continue

            avg = np.nanmean(sub, axis=(0, 1))
            std = np.nanstd(sub, axis=(0, 1))

            if self.ref_normalize_var.get():
                s = np.nanmax(np.abs(avg))
                if s > 0:
                    avg, std = avg / s, std / s

            self.ax_spec.plot(self.wavelength_axis, avg, color=reg["color"], label=reg["name"])
            self.ax_spec.fill_between(self.wavelength_axis, avg - std, avg + std, color=reg["color"], alpha=0.15)

        self.plot_references_if_enabled()

        self.ax_spec.set_title("Average spectra from selected X/Y regions")
        self.ax_spec.set_xlabel("Wavelength / spectral channel")
        self.ax_spec.set_ylabel("Intensity")
        self.ax_spec.legend(fontsize=8)
        self.ax_spec.relim()
        self.ax_spec.autoscale_view()
        self.ax_spec.margins(y=0.25)

        if draw:
            self.canvas.draw_idle()

    def clear_regions(self):
        self.regions = []
        if hasattr(self, "region_list"):
            self.region_list.delete(0, "end")

    def refresh_region_list(self):
        self.region_list.delete(0, "end")

        for reg in self.regions:
            self.region_list.insert(
                "end",
                f"{reg['name']}: X {reg['x1']}:{reg['x2']}, Y {reg['y1']}:{reg['y2']}"
            )

    def plot_references_if_enabled(self):
        if not hasattr(self, "ref_overlay_var") or not self.ref_overlay_var.get():
            return

        for ref in self.references:
            y = ref["intensity"]
            if self.ref_normalize_var.get():
                y = self.normalize_vec(y)
            self.ax_spec.plot(ref["wavelength"], y, "--", label=f"Ref: {ref['name']}")

    def load_reference_csv(self):
        path = filedialog.askopenfilename(
            filetypes=[("CSV/TXT", "*.csv *.txt"), ("All files", "*.*")]
        )
        if not path:
            return

        try:
            arr = np.genfromtxt(path, delimiter=",", comments="#")
            if arr.ndim != 2 or arr.shape[1] < 2:
                arr = np.genfromtxt(path, comments="#")

            arr = arr[np.isfinite(arr[:, 0]) & np.isfinite(arr[:, 1])]

            ref = {
                "name": os.path.basename(path),
                "wavelength": arr[:, 0].astype(np.float32),
                "intensity": arr[:, 1].astype(np.float32)
            }

            self.references.append(ref)
            self.ref_list.insert("end", ref["name"])

            self.ax_spec.clear()
            self.plot_references_if_enabled()
            self.ax_spec.legend(fontsize=8)
            self.canvas.draw_idle()

        except Exception as e:
            messagebox.showerror("Reference error", str(e))

    def clear_references(self):
        self.references = []
        self.ref_list.delete(0, "end")
        self.ax_spec.clear()
        self.canvas.draw_idle()

    def load_wavelength_vector(self):
        if self.hsi_cube_yxw is None:
            messagebox.showwarning("No cube", "Load or generate data first.")
            return

        path = filedialog.askopenfilename(
            filetypes=[("CSV/TXT", "*.csv *.txt"), ("All files", "*.*")]
        )
        if not path:
            return

        arr = np.genfromtxt(path, delimiter=",", comments="#")
        if arr.ndim > 1:
            arr = arr[:, 0]

        if arr.size != self.hsi_cube_yxw.shape[2]:
            messagebox.showerror(
                "Length mismatch",
                f"Vector length {arr.size} != bands {self.hsi_cube_yxw.shape[2]}"
            )
            return

        self.wavelength_axis = arr.astype(np.float32)
        self.update_display_image()

    def generate_dummy_cube(self):
        ny = max(2, int(self.dummy_y_var.get()))
        nx = max(2, int(self.dummy_x_var.get()))
        nw = max(10, int(self.dummy_w_var.get()))
        noise = max(0.0, float(self.dummy_noise_var.get()))

        wl = np.linspace(400, 900, nw).astype(np.float32)
        yy, xx = np.mgrid[0:ny, 0:nx]

        cube = np.zeros((ny, nx, nw), dtype=np.float32)

        for i, w in enumerate(wl):
            cube[:, :, i] = (
                0.8 * np.exp(-((w - 550) / 45) ** 2)
                * np.exp(-((xx - nx * .35) ** 2 + (yy - ny * .40) ** 2) / (0.06 * nx * ny))
                + 1.1 * np.exp(-((w - 700) / 70) ** 2)
                * np.exp(-((xx - nx * .65) ** 2 + (yy - ny * .65) ** 2) / (0.08 * nx * ny))
                + np.random.normal(0, noise, (ny, nx))
            )

        cube -= cube.min()
        if cube.max() > 0:
            cube /= cube.max()

        self.hsi_cube_yxw = cube
        self.raw_loaded_data = cube.copy()
        self.cube = cube
        self.x_map = np.arange(nx)
        self.y_map = np.arange(ny)
        self.wavelength_axis = wl

        self.update_display_image()

    def log_ml(self, text):
        self.ml_log.insert("end", text + "\n")
        self.ml_log.see("end")

    def run_classification(self):
        if not SKLEARN_AVAILABLE:
            messagebox.showerror("Missing package", "pip install scikit-learn")
            return
        if self.hsi_cube_yxw is None:
            return

        self.ml_log.delete("1.0", "end")

        x1, x2, y1, y2 = self.get_xy_bounds()
        crop = self.hsi_cube_yxw[y1:y2, x1:x2, :]
        yy, xx, ww = crop.shape

        X = np.nan_to_num(crop.reshape(-1, ww))

        pca_n = min(max(1, int(self.ml_pca_var.get())), ww, X.shape[0])

        Xw = make_pipeline(
            StandardScaler(),
            PCA(n_components=pca_n, random_state=42)
        ).fit_transform(X)

        n_classes = max(2, int(self.ml_classes_var.get()))
        method = self.ml_method_var.get()

        labels = -np.ones((yy, xx), dtype=int)

        for i, r in enumerate(self.regions):
            rx1, rx2 = max(r["x1"], x1) - x1, min(r["x2"], x2) - x1
            ry1, ry2 = max(r["y1"], y1) - y1, min(r["y2"], y2) - y1

            if rx2 > rx1 and ry2 > ry1:
                labels[ry1:ry2, rx1:rx2] = i

        flat = labels.reshape(-1)
        mask = flat >= 0

        if method == "KMeans" or np.sum(mask) < n_classes:
            model = KMeans(n_clusters=n_classes, random_state=42, n_init=10)
            pred = model.fit_predict(Xw)
            rmse = np.sqrt(mean_squared_error(Xw, model.cluster_centers_[pred]))
            self.log_ml(f"KMeans RMSE = {rmse:.6g}")
        else:
            if method == "SVM":
                model = make_pipeline(StandardScaler(), SVC(kernel="rbf", gamma="scale"))
            elif method == "GB":
                model = GradientBoostingClassifier(random_state=42)
            elif method == "NN":
                model = make_pipeline(
                    StandardScaler(),
                    MLPClassifier(hidden_layer_sizes=(80, 40), max_iter=500, random_state=42)
                )
            else:
                model = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)

            model.fit(Xw[mask], flat[mask])
            pred = model.predict(Xw)
            train_pred = model.predict(Xw[mask])

            self.log_ml(f"Training accuracy = {accuracy_score(flat[mask], train_pred):.4f}")
            self.log_ml(f"Label RMSE = {np.sqrt(mean_squared_error(flat[mask], train_pred)):.6g}")

        self.class_map = pred.reshape(yy, xx)
        self.show_classification_map(self.class_map, int(np.max(pred)) + 1, x1, x2, y1, y2)
        self.plot_class_median_spectra(crop)

    def show_classification_map(self, cmap_data, n_classes, x1, x2, y1, y2):
        self.ax_img.clear()
        self.clear_colorbar()

        im = self.ax_img.imshow(
            cmap_data,
            origin="upper",
            aspect="auto",
            cmap="tab20",
            extent=[x1, x2 - 1, y2 - 1, y1],
            vmin=-0.5,
            vmax=n_classes - 0.5
        )

        divider = make_axes_locatable(self.ax_img)
        cax = divider.append_axes("right", size="2.5%", pad=0.12)
        self.colorbar = self.fig.colorbar(im, cax=cax)

        self.ax_img.set_title("Classified spectral image")
        self.canvas.draw_idle()

    def plot_class_median_spectra(self, crop):
        self.ax_spec.clear()

        for c in np.unique(self.class_map):
            spectra = crop[self.class_map == c, :]

            med = np.nanmedian(spectra, axis=0)
            std = np.nanstd(spectra, axis=0)

            if self.ref_normalize_var.get():
                s = np.nanmax(np.abs(med))
                if s > 0:
                    med, std = med / s, std / s

            self.ax_spec.plot(self.wavelength_axis, med, label=f"Class {c}")
            self.ax_spec.fill_between(self.wavelength_axis, med - std, med + std, alpha=0.12)

        self.plot_references_if_enabled()
        self.ax_spec.legend(fontsize=8)
        self.canvas.draw_idle()

    def update_dimension_report(self):
        if self.hsi_cube_yxw is None:
            self.dimension_report_var.set("No data loaded.")
            return

        y, x, w = self.hsi_cube_yxw.shape

        report = (
            f"HSI cube mapping:\n"
            f"Y scan rows = {y}\n"
            f"X spatial pixels = {x}\n"
            f"Wavelength bands = {w}"
        )

        if self.raw_loaded_data is not None:
            report += f"\n\nRaw shape: {self.raw_loaded_data.shape}"

        if self.lookUpX is not None:
            report += f"\nlookUp_x length = {len(np.ravel(self.lookUpX))}"

        self.dimension_report_var.set(report)

    def convert_basler_to_yxw_cube(self, obj):
        data = np.asarray(obj.data, dtype=np.float32)

        self.lookUpX = getattr(obj, "lookUp_x", None)
        self.lookUpY = getattr(obj, "lookUp_y", None)
        self.numOfFrame = getattr(obj, "numOfFrame", None)
        self.roiX = getattr(obj, "roiX", None)
        self.roiY = getattr(obj, "roiY", None)

        if data.ndim == 3:
            if self.lookUpX is not None and len(np.ravel(self.lookUpX)) == data.shape[2]:
                cube = data
                wavelength = np.ravel(self.lookUpX).astype(np.float32)
            else:
                shape = data.shape
                wavelength = np.ravel(self.lookUpX).astype(np.float32) if self.lookUpX is not None else None

                w_axis = list(shape).index(len(wavelength)) if wavelength is not None and len(wavelength) in shape else int(np.argmin(shape))
                remaining = [i for i in range(3) if i != w_axis]

                y_axis, x_axis = remaining[0], remaining[1]
                cube = np.transpose(data, (y_axis, x_axis, w_axis))

                if wavelength is None or len(wavelength) != cube.shape[2]:
                    wavelength = np.arange(cube.shape[2], dtype=np.float32)

        elif data.ndim == 2:
            cube = data.T[None, :, :]
            wavelength = np.arange(cube.shape[2], dtype=np.float32)
        else:
            raise ValueError(f"Unsupported data shape: {data.shape}")

        x_map = np.arange(cube.shape[1], dtype=np.float32)
        y_map = np.arange(cube.shape[0], dtype=np.float32)

        return cube.astype(np.float32), x_map, y_map, wavelength

    def save_npz(self):
        if self.hsi_cube_yxw is None:
            return

        path = filedialog.asksaveasfilename(defaultextension=".npz")
        if path:
            np.savez_compressed(
                path,
                hsi_cube_yxw=self.hsi_cube_yxw,
                wavelength_axis=self.wavelength_axis,
                x_map=self.x_map,
                y_map=self.y_map
            )

    def load_npz(self):
        path = filedialog.askopenfilename(filetypes=[("NPZ", "*.npz")])
        if not path:
            return

        d = np.load(path, allow_pickle=True)

        self.hsi_cube_yxw = d["hsi_cube_yxw"].astype(np.float32)
        self.wavelength_axis = d["wavelength_axis"].astype(np.float32) if "wavelength_axis" in d else np.arange(self.hsi_cube_yxw.shape[2])
        self.x_map = d["x_map"] if "x_map" in d else np.arange(self.hsi_cube_yxw.shape[1])
        self.y_map = d["y_map"] if "y_map" in d else np.arange(self.hsi_cube_yxw.shape[0])

        self.raw_loaded_data = self.hsi_cube_yxw.copy()
        self.cube = self.hsi_cube_yxw

        self.update_display_image()

    def save_cube_file(self):
        if self.hsi_cube_yxw is None:
            return

        path = filedialog.asksaveasfilename(defaultextension=".cube")
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
            describtion="Saved by SpectraLab as Y-X-lambda cube"
        )

        with open(path, "wb") as f:
            pk.dump(obj, f)

    def load_cube_file(self):
        path = filedialog.askopenfilename(
            filetypes=[("CUBE", "*.cube"), ("All files", "*.*")]
        )
        if not path:
            return

        with open(path, "rb") as f:
            obj = pk.load(f)

        self.hsi_cube_yxw, self.x_map, self.y_map, self.wavelength_axis = self.convert_basler_to_yxw_cube(obj)

        self.raw_loaded_data = np.asarray(obj.data, dtype=np.float32)
        self.cube = self.hsi_cube_yxw
        self.regions = []

        self.refresh_region_list()
        self.update_display_image()

    def save_fits_file(self):
        if not ASTROPY_AVAILABLE or self.hsi_cube_yxw is None:
            return

        path = filedialog.asksaveasfilename(defaultextension=".fits")
        if path:
            fits.PrimaryHDU(self.hsi_cube_yxw.astype(np.float32)).writeto(path, overwrite=True)

    def load_fits_file(self):
        if not ASTROPY_AVAILABLE:
            messagebox.showerror("FITS error", "Install astropy.")
            return

        path = filedialog.askopenfilename(
            filetypes=[("FITS", "*.fits *.fit"), ("All files", "*.*")]
        )
        if not path:
            return

        with fits.open(path) as hdul:
            data = None
            for h in hdul:
                if h.data is not None:
                    data = np.asarray(h.data, dtype=np.float32)
                    break

        if data is None:
            return

        self.hsi_cube_yxw = data if data.ndim == 3 else data.T[None, :, :]
        self.raw_loaded_data = data.copy()
        self.cube = self.hsi_cube_yxw
        self.wavelength_axis = np.arange(self.hsi_cube_yxw.shape[2], dtype=np.float32)
        self.x_map = np.arange(self.hsi_cube_yxw.shape[1], dtype=np.float32)
        self.y_map = np.arange(self.hsi_cube_yxw.shape[0], dtype=np.float32)

        self.update_display_image()

    def run_loading_task(self, func):
        self.status_var.set("Status: waiting... loading data")
        self.root.update_idletasks()

        try:
            func()
            self.status_var.set("Status: loading complete.")
        except Exception as e:
            self.status_var.set("Status: loading failed.")
            messagebox.showerror("Loading error", str(e))

    def deload_all(self):
        self.scan_running = False

        self.hsi_cube_yxw = None
        self.display_image = None
        self.raw_loaded_data = None
        self.cube = None
        self.x_map = None
        self.y_map = None
        self.wavelength_axis = None

        self.regions = []
        self.references = []
        self.class_map = None

        if hasattr(self, "region_list"):
            self.region_list.delete(0, "end")
        if hasattr(self, "ref_list"):
            self.ref_list.delete(0, "end")

        self.ax_img.clear()
        self.ax_spec.clear()
        self.ax_chip.clear()

        self.clear_colorbar()
        self.clear_chip_colorbar()

        self.dimension_report_var.set("No data loaded.")
        self.canvas.draw_idle()

    def on_close(self):
        self.scan_running = False
        self.basler.close()
        self.specim.close()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = SpectraLabPushBroomGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()

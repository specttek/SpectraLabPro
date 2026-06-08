# SpectraLab Push-Broom Imaging

Python GUI for live push-broom hyperspectral imaging, developed for **SpectTek Co.**

The software supports Basler area-scan camera acquisition, SPECIM-style hyperspectral line-camera integration, live RGB preview, detector chip visualization, spectral extraction, reference overlay, machine-learning classification, and data export.

---

## Main Data Model

All data are converted to:

```python
hsi_cube_yxw[Y_scan, X_spatial, wavelength]
````

For the Basler daA1600-60um live scan:

```text
Raw frame = 1200 × 1600
frame[lambda_pixel, X_spatial]
```

Converted to:

```python
line_xw = frame.T
```

Then stacked into:

```text
cube = scan_steps × 1600 × 1200
```

So:

```text
Y = push-broom scan direction
X = detector spatial axis
λ = wavelength/spectral detector axis
```

---

## Features

* Password-protected camera connection
* Basler daA1600-60um support through `pypylon`
* Dummy Basler mode for testing without hardware
* SPECIM hyperspectral imager placeholder adapter
* Live push-broom RGB image during scan
* Live detector chip map showing wavelength vs X pixels
* RGB preview from selected spectral bands
* Single wavelength/channel display
* Region selection with average spectra
* Reference spectrum overlay from CSV/TXT
* Machine learning classification:

  * Random Forest
  * SVM
  * Gradient Boosting
  * Neural Network
  * KMeans
* Save/load:

  * `.npz`
  * `.cube`
  * `.fits`
* Import wavelength vector from CSV/TXT
* Export Basler-compatible pickled `baslerData` cube

---

## Camera Password

To connect the camera, use:

```text
Ask SpectTek@gmail.com
```

---

## Supported Cameras

### Basler daA1600-60um

The Basler camera is treated as a spectral detector frame:

```text
Detector Y axis = wavelength pixels
Detector X axis = spatial slit pixels
```

Each acquired frame is converted as:

```python
line_xw = frame.T
```

### SPECIM Hyperspectral Imager

A SPECIM adapter placeholder is included:

```python
class SpecimCamera
```

The expected SPECIM SDK output should be:

```python
line_xw[X_spatial, wavelength]
```

This is stacked directly into:

```python
cube[Y_scan, X_spatial, wavelength]
```

Replace the placeholder methods with real SPECIM SDK calls:

```python
connect()
get_line_cube()
```

---

## Installation

Install Python 3.9 or newer.

Then install dependencies:

```bash
pip install numpy matplotlib pypylon astropy scikit-learn pillow
```

Optional, for Basler camera operation:

```bash
pip install pypylon
```

You also need the Basler Pylon runtime installed from Basler.

---

## Run

Save the code as:

```text
spectralab_pushbroom_basler.py
```

Run:

```bash
python spectralab_pushbroom_basler.py
```

---

## GUI Sections

The left panel contains a section selector:

```text
Camera
Scan
Dummy
Regions
Save / Load
Display
Analysis
References
```

---

## Camera Section

Use this section to:

* Select Basler or SPECIM
* Connect the selected camera
* Enable Basler dummy mode
* Set exposure time
* Set gain
* Set frame averaging
* Display `logo.png`

The file:

```text
logo.png
```

should be placed in the same folder as the Python script.

---

## Scan Section

The scan section controls live push-broom acquisition.

Parameters:

```text
Scan steps / final Y rows
Delay per step
Live display update every N lines
```

During scan, every frame is converted into a hyperspectral line and stacked into the live cube.

Example:

```text
Scan steps = 100
Basler frame = 1200 × 1600
Final cube = 100 × 1600 × 1200
```

---

## Live Plot Panels

The right side contains three plots.

### 1. Live RGB Push-Broom Image

Displays the live hyperspectral image using selected RGB wavelength bands.

### 2. Average / Selected Spectrum

Displays:

* Spectrum from clicked pixel
* Average spectrum from selected region
* Reference spectra
* Class median spectra after ML classification

### 3. Detector Chip Map

Displays the most recent detector frame as:

```text
wavelength × X
```

For Basler:

```text
1200 × 1600
```

For SPECIM:

```text
SPECIM_BANDS × SPECIM_X
```

---

## Display Controls

Display modes:

```text
RGB preview
Single wavelength/channel
```

RGB band defaults:

```text
Red   = 650
Green = 550
Blue  = 450
```

These values may represent physical wavelengths or spectral channel indices.

---

## Wavelength Vector

A wavelength vector can be imported from CSV/TXT.

Example:

```text
400
401
402
403
...
900
```

The vector length must match the number of spectral bands.

Example:

```text
Cube shape = 100 × 1600 × 1200
Wavelength vector length must be 1200
```

---

## Region Selection

Drag a rectangle on the image.

The GUI calculates:

```python
average_spectrum = mean(cube[y1:y2, x1:x2, :])
std_spectrum     = std(cube[y1:y2, x1:x2, :])
```

Each region is plotted with a different color.

---

## Reference Spectra

Load reference spectra from CSV/TXT.

Expected format:

```csv
wavelength,intensity
400,0.1
500,0.4
600,0.8
700,0.3
```

or:

```text
400 0.1
500 0.4
600 0.8
700 0.3
```

References can be overlaid and normalized to maximum intensity.

---

## Machine Learning Classification

The Analysis section supports:

```text
RF
SVM
GB
NN
KMeans
```

The active X/Y plot range is used for classification.

Outputs:

* Classified color map
* RMSE
* Training accuracy, for supervised methods
* Median spectrum per class
* Error region per class

If no labeled regions are selected, the software automatically uses KMeans.

---

## File Formats

### NPZ

Stores:

```text
hsi_cube_yxw
wavelength_axis
x_map
y_map
```

### CUBE

Pickled `baslerData` object.

Stores:

```text
data
lookUp_x
lookUp_y
roiX
roiY
numOfFrame
metadata
```

### FITS

Stores hyperspectral cube data through Astropy.

---

## Recommended Workflow

### Test without camera

1. Open the GUI.
2. Go to `Camera`.
3. Press `Enable Basler Dummy/Test Mode`.
4. Go to `Scan`.
5. Set scan steps, for example:

```text
50
```

6. Press:

```text
Start Push-Broom Live Scan
```

The GUI will generate a live simulated Basler scan.

---

### Test with dummy cube

1. Go to `Dummy`.
2. Set:

```text
Y rows = 120
X columns = 160
Wavelength bands = 220
```

3. Press:

```text
Generate Dummy Y-X-λ Cube
```

---

### Real Basler scan

1. Connect the Basler camera.
2. Press `Connect Selected Camera`.
3. Enter password:

```text
Ask SpectTek@gmail.com
```

4. Apply exposure, gain, and averaging.
5. Go to `Scan`.
6. Start push-broom scan.

---

### SPECIM integration

Replace the placeholder methods:

```python
SpecimCamera.connect()
SpecimCamera.get_line_cube()
```

with SDK-specific commands.

The method must return:

```python
line_xw[X_spatial, wavelength]
```

---

## Suggested Repository Structure

```text
spectralab-pushbroom/
│
├── spectralab_pushbroom_basler.py
├── README.md
├── requirements.txt
├── logo.png
├── examples/
│   ├── wavelength_vector.csv
│   ├── reference_spectrum.csv
│   └── demo_cube.npz
└── data/
    └── sample_cubes/
```

---

## requirements.txt

```text
numpy
matplotlib
pypylon
astropy
scikit-learn
pillow
```

---

## Notes

* For Basler hardware, install Basler Pylon runtime.
* The current SPECIM section is a software adapter template and must be connected to the real SPECIM SDK.

---

## Developer

```text
SpectTek Co.
CEO(M. Raouf)
```

LinkedIn:

```text
https://www.linkedin.com/in/specttek/
```




## Acknowledgment

This project was created from the PhD work of Leiden University student Fatemeh Fazel Hesar, with her dedicated effort. 
https://www.mdpi.com/2218-1997/12/4/93

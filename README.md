# SpectraLab Push-Broom Imaging — SpectTek Co.

**SpectraLab Push-Broom Imaging** is a Python GUI for hyperspectral push-broom imaging using a Basler area-scan detector, with support for loading, visualizing, saving, analyzing, and classifying hyperspectral data cubes.

The software is designed around the internal data model:

```python
hsi_cube_yxw[Y, X, wavelength]
```

This means every loaded or generated dataset is converted into a consistent **Y spatial × X spatial × wavelength** hyperspectral cube. RGB images, single-band images, selected-region spectra, spectral references, and machine-learning classification are all derived from this cube.

---

## Main Features

### Camera and dummy mode

The interface supports the **Basler daA1600-60um** area-scan camera through `pypylon`.

Camera information:

```text
Basler daA1600-60um
1600 × 1200 pixels
Monochrome
USB 3.0
Global shutter
4.5 µm × 4.5 µm pixels
```

The GUI also includes a **dummy mode**, allowing the full workflow to be tested without a connected camera.

---

## Data Model

All hyperspectral data are converted to:

```python
hsi_cube_yxw[Y, X, wavelength]
```

where:

```text
Y = spatial row
X = spatial column
wavelength = spectral channel or physical wavelength
```

This avoids confusion between different file formats and detector layouts.

The GUI can display:

```text
RGB mapped image
Single wavelength image
Classification map
Selected-region spectra
Reference spectra
Median spectra of classified classes
```

---

## Supported File Formats

### NPZ

The `.npz` format stores:

```text
hsi_cube_yxw
wavelength_axis
x_map
y_map
regions
```

This is the recommended internal format for saving processed data.

---

### Basler CUBE object

The `.cube` file is saved and loaded as a pickled `baslerData` object.

The required class is included directly in the code:

```python
class baslerData:
    def __init__(self,numOfFrame=None, roiX=None,roiY=None, lookUp_x=None, lookUp_y=None,
                 refUsed=None, darkRefUsed=False, data=None, scannerType=None,
                 matFileLen=None, refData=None, deviceModel=None, dateAndTime=None,
                 exposureTime=None, frameRate=None, Gain=None, Gamma=None,
                 desScannerType=None, scanLength=None, averagingTime=None,
                 refFrameUsed=None, sizeofDataCube=None, describtion=None, cropped=None):
        ...
```

The loader reads:

```python
baslerObject.data
baslerObject.lookUp_x
baslerObject.lookUp_y
baslerObject.numOfFrame
baslerObject.roiX
baslerObject.roiY
```

and converts the data into:

```python
hsi_cube_yxw[Y, X, wavelength]
```

---

### FITS

The GUI supports FITS loading and saving through `astropy`.

Saved FITS files contain:

```text
PrimaryHDU data = hsi_cube_yxw
AXISORD = Y,X,W
CAMERA = Basler daA1600-60um
```

If a loaded FITS file is 2D, it is interpreted as:

```python
Y × wavelength
```

and converted to:

```python
Y × 1 × wavelength
```

---

### Spectral wavelength vector

A separate wavelength vector can be imported from `.csv` or `.txt`.

The file should contain one column:

```text
400
401
402
...
900
```

or a first column containing the wavelength values.

The number of values must match the number of spectral bands in the cube.

---

## GUI Layout

The interface is divided into a **left control column** and a **right plot panel**.

The left column uses a section selector instead of many small crowded tabs. Available sections are:

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

## Right Plot Panel

The right panel contains:

```text
Top: X/Y plot range controls
Middle: spectral image / RGB image / classification map
Bottom: spectrum plot
```

The X/Y plot range controls are used for both visualization and machine-learning classification.

Example:

```text
X min = 10
X max = 120
Y min = 20
Y max = 150
```

When classification is run, only pixels inside this active X/Y range are analyzed.

---

## Display Modes

### RGB preview

The RGB preview maps three selected wavelengths to red, green, and blue channels.

Default values:

```text
Red   = 650 nm
Green = 550 nm
Blue  = 450 nm
```

Brightness and contrast can be adjusted using:

```text
Brightness offset
Contrast scale
```

---

### Single wavelength image

A single wavelength or spectral channel can be displayed as a grayscale image.

The user can select:

```text
Single wavelength target / nm
Single channel index fallback
```

If the wavelength axis is physical, the nearest wavelength is used. Otherwise, the channel index is used.

A color bar is displayed outside the image on the right.

---

## Region Selection

The user can drag rectangles on the image to define regions.

Each selected region is stored with:

```text
X range
Y range
Color
Region name
```

For each selected region, the software calculates:

```python
average_spectrum = mean(cube[y1:y2, x1:x2, :])
std_spectrum     = std(cube[y1:y2, x1:x2, :])
```

The spectrum panel shows:

```text
Average spectrum
Error band
Reference spectra, if enabled
```

---

## Spectral References

The References section allows loading spectral reference files in CSV or TXT format.

Expected format:

```csv
wavelength,intensity
400,0.12
401,0.15
402,0.17
...
```

or simple whitespace-separated format:

```text
400 0.12
401 0.15
402 0.17
```

After loading, references are immediately displayed in the spectral plot.

Options:

```text
Overlay references
Normalize selected/reference to max
Clear references
```

When normalization is enabled, both selected spectra and reference spectra are scaled to their maximum value.

---

## Machine Learning Classification

The Analysis section supports several classification modes:

```text
RF      Random Forest
SVM     Support Vector Machine
GB      Gradient Boosting
NN      Neural Network
KMeans  Unsupervised clustering
```

Classification is run only inside the active X/Y plot range.

The user can choose:

```text
Number of classes
PCA components / spectral level
Model level: Fast, Medium, High
```

---

## Supervised and Unsupervised Classification

### Supervised mode

If selected regions are available, they are used as training labels.

Each selected region becomes one class label.

Example:

```text
Region 1 → Class 0
Region 2 → Class 1
Region 3 → Class 2
```

The model is trained on pixels inside those regions and then applied to all pixels inside the active X/Y range.

---

### Unsupervised fallback

If no training regions exist, or if the selected method is `KMeans`, the software runs unsupervised clustering.

This is useful for quick exploration when no labels are available.

---

## ML Outputs

After classification, the GUI displays:

```text
Classified color map
Color bar outside the image
Log output
RMSE
Training accuracy, when supervised
Median spectrum of each class
Error band for each class
```

The classified map appears in the spectral image panel.

The median spectra of classes appear in the spectrum panel.

---

## Installation

Create a Python environment and install the dependencies:

```bash
pip install numpy matplotlib pypylon astropy scikit-learn
```

If the camera is not available, the software still works in dummy mode.

---

## Running the Software

Save the code as:

```bash
spectralab_pushbroom_basler.py
```

Run:

```bash
python spectralab_pushbroom_basler.py
```

---

## Recommended Workflow

### 1. Start the GUI

```bash
python spectralab_pushbroom_basler.py
```

---

### 2. Use dummy data or load real data

For testing:

```text
Dummy → Generate Dummy Y-X-W Cube
```

For real data:

```text
Save / Load → Load CUBE
Save / Load → Load FITS
Save / Load → Load NPZ
```

---

### 3. Import wavelength vector if needed

```text
Save / Load → Import wavelength vector CSV/TXT
```

The wavelength vector must match the number of spectral bands.

---

### 4. Adjust display

```text
Display → RGB preview
Display → Single wavelength/channel
Display → Brightness / contrast
Display → Orientation
```

---

### 5. Select regions

Drag rectangles on the image.

The spectrum panel will show the average spectrum of each selected region.

---

### 6. Load spectral references

```text
References → Load Reference CSV
```

References appear directly in the spectrum panel.

---

### 7. Run classification

Set the X/Y range at the top of the plot panel.

Then:

```text
Analysis → Run Classification in X/Y Range
```

The classified map and class median spectra will be shown.

---

## File Format Examples

### Wavelength vector CSV

```csv
400
401
402
403
404
```

---

### Spectral reference CSV

```csv
wavelength,intensity
400,0.10
450,0.25
500,0.62
550,0.91
600,0.74
650,0.40
700,0.22
```

---

## Notes on Basler CUBE Loading

The software assumes that `.cube` files are pickled `baslerData` objects.

Loading is performed with:

```python
with open(path, "rb") as f:
    obj = pickle.load(f)
```

Then:

```python
obj.data
obj.lookUp_x
obj.lookUp_y
obj.roiX
obj.roiY
obj.numOfFrame
```

are used to reconstruct the hyperspectral cube.

The final converted cube is always:

```python
hsi_cube_yxw[Y, X, wavelength]
```

---

## Troubleshooting

### Camera not found

The GUI automatically switches to dummy mode if no Basler camera is detected.

---

### FITS loading error

Install Astropy:

```bash
pip install astropy
```

---

### ML classification not available

Install scikit-learn:

```bash
pip install scikit-learn
```

---

### CUBE loading fails

Check that the `.cube` file is a pickled `baslerData` object and not raw binary.

---

### Wavelength vector length mismatch

The wavelength vector must have exactly the same number of entries as the number of spectral bands.

Example:

```text
Cube shape = 120 × 80 × 220
Wavelength vector length must be 220
```

---

## Output Products

The software can produce:

```text
RGB preview image
Single wavelength image
Average region spectra
Reference overlays
Classification map
Class median spectra
NPZ data cube
Basler CUBE object
FITS data cube
```

---

## Project Identity

Developed for:

```text
SpectTek Co.
```

LinkedIn:

```text
https://www.linkedin.com/in/specttek/
```

---

## Suggested Repository Structure

```text
spectralab-pushbroom/
│
├── spectralab_pushbroom_basler.py
├── README.md
├── requirements.txt
├── examples/
│   ├── wavelength_vector.csv
│   ├── reference_spectrum.csv
│   └── dummy_output.npz
└── data/
    └── sample_cube_files/
```

---
<img width="1774" height="887" alt="Pipeline_Spectrolab_hyper" src="https://github.com/user-attachments/assets/f1287e10-4cf9-4cd8-a04b-7b9ea25d698c" />


## Suggested `requirements.txt`

```text
numpy
matplotlib
pypylon
astropy
scikit-learn
```
---

## Acknowledgment

This project was created from the PhD work of Leiden University student Fatemeh Fazel Hesar, with her dedicated effort. 
https://www.mdpi.com/2218-1997/12/4/93


---

## License

Add your preferred license here, for example:

```text
MIT License
```

or

```text
Proprietary — SpectTek Co.
```

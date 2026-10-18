# Tatuoca Magnetogram Digitization Pipeline: Architecture & Code Guide

## 1. Overview and Problem Context

Historical magnetograms recorded on photographic paper at the **Tatuoca Magnetic Observatory (TTB)** represent 50+ years of continuous analog geomagnetic observations (H, D, and Z components). Extracting discrete physical observations from scanned analog records requires isolating trace curves and baseline references from paper degradation, non-uniform photographic exposure, mold, time-mark breaks, and overlapping lines.

This pipeline performs the **Image Processing & Feature Extraction Phase**, converting raw scans into structured physical measurements relative to baseline references.

---

## 2. End-to-End Processing Flow

+-----------------------------------------------------------+
| 0. Raw Scanned Magnetogram (Grayscale / BGR)              |
+-----------------------------------------------------------+
|
v
+-----------------------------------------------------------+
| 1. Image Cropping & Spatial Calibration (10 px = 1.0 mm)  |
+-----------------------------------------------------------+
|
v
+-----------------------------------------------------------+
| 2. Preprocessing: Adaptive Gaussian Thresholding + Blur   |
+-----------------------------------------------------------+
|
v
+-----------------------------------------------------------+
| 3. Feature Detection: Canny Edge Filter                   |
+-----------------------------------------------------------+
|
v
+-----------------------------------------------------------+
| 4. Thinning / Skeletonization (Single-Pixel Traces)       |
+-----------------------------------------------------------+
|
v
+-----------------------------------------------------------+
| 5. Spatial Clustering: DBSCAN Component Disambiguation    |
+-----------------------------------------------------------+
|
v
+-----------------------------------------------------------+
| 6. Coordinate Extraction & Physical Calibration (CSV/TXT) |
+-----------------------------------------------------------+

---

## 3. Pipeline Stages & Algorithmic Mechanics

### Step 1: Image Cutter & Calibration

* **Objective:** Remove non-essential borders (metadata headers, non-data margins) and align the spatial resolution with standard photographic variometer recordings.
* **Spatial Scale:** Calibrated to **10 pixels = 1.0 mm** ($0.1\text{ mm/pixel}$).
* **Coordinate System:** Standard image array convention:
  * **Origin $(0, 0)$:** Top-Left corner of the cropped ROI.
  * **Row index ($r$ / $y$):** Vertical axis, increasing downwards.
  * **Column index ($c$ / $x$):** Horizontal axis (time axis), increasing rightwards.

### Step 2: Adaptive Gaussian Thresholding & Smoothing

* **Objective:** Binarize degraded analog paper while handling spatial luminance variations caused by age, mold, and non-uniform chemical exposure.
* **Mechanics:**
  * **Adaptive Gaussian Thresholding:** Calculates the local threshold $T(x, y)$ as a Gaussian-weighted sum over a local neighborhood window minus a constant $C$:
    $$T(x, y) = \sum_{i} \sum_{j} G(i, j) \cdot I(x+i, y+j) - C$$
    This separates weak, fading trace lines from background noise without globally clipping darker degraded regions.
  * **Gaussian Blur:** Low-pass convolution smoothing reduces high-frequency speckle noise and fills microscopic pinholes in faint curves, enhancing continuity.

### Step 3: Canny Edge Detection

* **Objective:** Detect precise gradient transitions defining the contours of the trace and baseline curves while filtering out low-contrast artifacts.
* **Mechanics:** Computes spatial intensity gradients via Sobel operators, suppresses non-maximum gradient responses to thin edges, and uses hysteresis thresholding ($T_{\text{low}}, T_{\text{high}}$) to preserve continuous weak edges connected to strong edges.

### Step 4: Skeletonization

* **Objective:** Collapse multi-pixel trace and baseline edge contours down to 1-pixel-wide medial lines (skeletons), yielding single $(r, c)$ representations per feature.
* **Mechanics:** Scans column-by-column (or via morphological thinning kernels) to identify white pixel spans across trace boundaries and resolves their midpoints/topological centers, preserving connectivity while eliminating stroke-width variance.

### Step 5: DBSCAN Clustering (Trace Disambiguation)

* **Objective:** Group discrete skeleton coordinates into distinct semantic entities (Baseline $B_H, B_D, B_Z$ and Component Curves $H, D, Z$) and classify stray noise points as outliers.
* **Why DBSCAN?**
  * Does not require pre-specifying cluster counts ($k$).
  * Adapts naturally to arbitrary, non-linear geometric trajectories (long continuous curves).
  * Effectively tags disconnected photographic spots, dust, and time-mark artifacts as noise (`label = -1`).
* **Parameters:**
  * `eps` ($\epsilon$): Maximum neighborhood search radius in pixel space.
  * `min_samples`: Minimum number of pixels required to constitute a dense trace core.

### Step 6: Baseline Alignment & Physical Scaling

* **Objective:** Compute vertical deviations between the component trace and its reference baseline at identical temporal positions (columns).
* **Calculations:**
  1. For a given horizontal column $c$:
     * Trace row coordinate: $r_{\text{trace}}(c)$
     * Mean baseline row coordinate: $\bar{r}_{\text{base}} = \frac{1}{N}\sum_{i=1}^{N} r_{\text{base}}(c_i)$ (or localized segment mean if baseline drift is present).
  2. Vertical separation:
     $$\Delta r(c) = \vert{}r_{\text{trace}}(c) - \bar{r}_{\text{base}}\vert{} \quad \text{(in pixels)}$$
  3. Conversion to millimeters:
     $$\Delta d_{\text{mm}}(c) = \frac{\Delta r(c)}{10.0}$$

---

## 4. Output Data Structure

The processed coordinates are exported to structured text/CSV tables formatted with 4 fundamental columns:

| Column | Field Name | Description | Units |
| :---: | :--- | :--- | :---: |
| 1 | `Row_Trace` | Vertical pixel position of the component trace curve ($r$) | Pixels |
| 2 | `Row_Baseline_Mean` | Mean vertical pixel position of the associated baseline ($\bar{r}_{\text{base}}$) | Pixels |
| 3 | `Column` | Horizontal coordinate along the time axis ($c$) | Pixels |
| 4 | `Distance_mm` | Vertical distance between component trace and baseline ($\Delta d$) | mm |

*Note: In downstream geophysics modules, $\Delta d_{\text{mm}}$ is multiplied by the variometer scale value $\epsilon_E$ (e.g., in $\text{nT/mm}$ or $\text{min/mm}$) and added to the absolute baseline value $E_0$ to produce physical magnetic field units ($E(t) = E_0 + \epsilon_E \cdot \Delta d(t)$).*

---

## 5. Typical Code Module Mapping

When exploring the `linuxdev` branch, you will encounter scripts matching these functional responsibilities:

├── crop.py / cutter.py       --> Handles ROI extraction and spatial resampling (10 px = 1 mm)

├── threshold.py / filters.py --> OpenCV Gaussian Adaptive Threshold, Blur, and Canny filters

├── skeleton.py               --> Column-scanning / morphological thinning functions

├── cluster.py / dbscan.py    --> Sklearn DBSCAN clustering, label sorting, component separation

└── extract.py / exporter.py  --> Column matching, baseline distance calculation, CSV/TXT formatting

## How to run

To run the process over your magnetogram images, extract.py is the intended entry point for executing the automated batch processing pipeline, while the Jupyter notebooks serve specific development and inspection roles:

    extract.py: The production CLI script designed to automate the pipeline without manual notebook execution. It loops through target images in your directory, applies the filter chain (crop, Gaussian threshold/blur, Canny, skeletonization, DBSCAN), and exports coordinate files.

    extract.ipynb: The interactive notebook counterpart to extract.py. Use this if you want to run the pipeline cell-by-cell to visualize intermediate stages (thresholded masks, edge maps, and cluster assignments) for debugging parameter sensitivity.

    curve.ipynb: An exploratory/analysis notebook specifically focused on plotting, curve reconstruction, baseline drift inspection, or calibrating specific cluster curves after extraction.
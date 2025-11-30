# Borobudur Experiment: 2025 Data Preparation Phase

**Project:** Digital Relief Restoration and Analysis
**Phase:** 1 of 2
**Year:** 2025
**Location:** Semarang, Central Java, Indonesia

---

## 1. Project Overview

This document provides a comprehensive guide for the 2025 Data Preparation phase of the Borobudur Experiment. The primary objective for this year is to produce a high-quality, large-scale, and meticulously annotated image dataset of the Borobudur temple's bas-reliefs. This dataset will be the foundational element for the machine learning model development scheduled for 2026. All activities will be coordinated from our base in Semarang.

**Project Status (As of September 27, 2025):** The project is currently in the final stages of Q3, focusing on wrapping up the primary photography and beginning the intensive annotation process.

---

## 2. Team Roles and Responsibilities

* **Project Lead:** Professor team from Universitas Negeri Solo and Unika Soegijapranata
* **Field Photography Team (UNS):** Responsible for on-site image acquisition at Borobudur Temple.
* **Data Processing (Unika):** Responsible for image stitching, pre-processing, and data management.
* **Annotation Team (Unika):** Responsible for the manual annotation (bounding boxes and segmentation) of the processed images.

---

## 3. Detailed 2025 Timeline

| Quarter | Months          | Key Objectives                                                              | Status      |
| :------ | :-------------- | :-------------------------------------------------------------------------- | :---------- |
| **Q1** | Jan - Mar 2025  | **Planning & Procurement:** Finalize methodology, procure equipment.        | `Completed` |
| **Q2** | Apr - Jun 2025  | **Initial Photography:** Begin on-site image capture, establish workflow.   | `Completed` |
| **Q3** | Jul - Sep 2025  | **Intensive Photography & Initial Processing:** Complete image acquisition. | `In Progress` |
| **Q4** | Oct - Dec 2025  | **Intensive Annotation & Finalization:** Complete all annotations and splits. | `Upcoming`  |

---

## 4. Step-by-Step Execution Plan

### Step 1: Relief Photography (Q2-Q3)

* **Objective:** Capture high-resolution, overlapping images of the entire surface of the temple reliefs.
* **Equipment:**
    * **Camera:** **##Need to be Revised** : 2 x Sony A7R IV (or similar high-resolution full-frame cameras).
    * **Lens:** **##Need to be Revised** 50mm prime lens to minimize distortion.
    * **Lighting:** **##Need to be Revised** 2 x portable LED panel kits with diffusers for consistent, soft lighting.
    * **Tools:** **##Need to be Revised** Tripod, remote shutter, color checker passport for white balance calibration.
* **Procedure:** **(Let it Stay, Blank since UNS will Fill this out)
    1.  **Setup:** Mount the camera on a tripod at a fixed distance from the relief wall.
        - Distance : 
            1.  Camera From 
    2.  **Calibration:** Capture a shot with the color checker at the beginning of each session.
    3.  **Capture:** Shoot in RAW format, manual mode **##Need to be revised**(ISO 100, f/8, consistent shutter speed). Move the camera rig systematically, ensuring a 30-40% overlap for each frame.
    4.  **Data Management:** Name files according to a strict convention: `FrameNumber{Number_for_multiple_shot}.ARW` (e.g., `1.ARW` and `2a.ARW` for first shot of 3 shot 1 panel, the next will be `2b.ARW` and `2c.ARW` ).
    5.  **Backup:** Create on-site backups to two separate external SSDs daily.

### Step 2: Image Stitching (Q3-Q4)

* **Objective:** Combine individual photographs into large, seamless panoramic images.
* **Software:**
    * **Primary:** PTGui for its advanced control points and masking features.
    * **Alternative:** Custom Python scripts utilizing OpenCV's stitcher class for batch processing.
* **Procedure:**
    1.  **Batch Processing:** Ingest RAW files, apply color correction and lens profiles in Adobe Lightroom, and export as 16-bit TIFFs.
    2.  **Stitching:**
        * Load a sequence of images for a single panel into PTGui.
        * Run SIFT/SURF for automatic control point generation. Manually add control points in areas with low texture if needed.
        * Optimize the alignment and lens parameters.
        * Render the final panorama as a full-resolution TIFF file.
    3.  **Quality Control:** Visually inspect each stitched panorama for artifacts, ghosting, or misalignments. Rework if necessary.

### Step 3: Pre-processing (Q4)

* **Objective:** Standardize the stitched images for model training.
* **Tools:** Python script using `OpenCV` and `Pillow` libraries.
* **Procedure:**
    1.  **Grayscale Conversion:** Load the stitched TIFF and convert it to grayscale using `cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)`.
    2.  **Normalization:** Normalize pixel values to a floating-point range of [0, 1] by dividing the image array by 255.0.
    3.  **Save:** Save the processed images as PNGs to a designated folder.

### Step 4: Annotation (Q3-Q4)

* **Objective:** Manually label features of interest to create ground-truth data.
* **Platform:** CVAT (Computer Vision Annotation Tool) for its collaborative features and support for both bounding boxes and polygons.
* **Procedure:**
    1.  **Setup Project:** Create a new project in CVAT and upload the pre-processed images.
    2.  **Define Labels:** Establish a clear, documented set of labels (e.g., `human_figure`, `animal_figure`, `stupa`, `vegetation`, `damage_crack`, `damage_missing`).
    3.  **Bounding Box Task:** Assign images to annotators to draw rectangular boxes around every instance of the defined labels.
    4.  **Segmentation Task:** Assign the same images to different annotators to draw precise polygon masks, especially for `human_figure` and `animal_figure` classes.
    5.  **Review:** A senior annotator or the project lead will review a sample of annotations each week to ensure consistency and quality.

### Step 5: Dataset Splitting (End of Q4)

* **Objective:** Divide the annotated dataset into training, validation, and testing sets.
* **Methodology:** A Python script will be used to perform stratified sampling, ensuring that the proportion of each object class is maintained across the splits.
* **Final Directory Structure:**
    ```
    /borobudur_dataset/
    ├── /images/
    │   ├── /train/
    │   ├── /val/
    │   └── /test/
    └── /labels/
        ├── /train/
        ├── /val/
        └── /test/
    ```
* **Split Ratios:** The final split will be **80% Training**, **10% Validation**, and **10% Testing**.

---

## 5. Key Deliverables for 2025

By December 31, 2025, the Data Preparation phase will deliver:

1.  A complete archive of high-resolution RAW images from the field photography sessions.
2.  A full set of stitched, panoramic relief images in high-resolution TIFF format.
3.  The final, pre-processed dataset of grayscale PNG images.
4.  A complete set of annotations (both bounding box and segmentation) in YOLO format, linked to the processed dataset.
5.  The final, split dataset, organized into `train`, `val`, and `test` directories, ready for the 2026 Model Development phase.
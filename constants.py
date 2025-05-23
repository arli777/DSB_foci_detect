"""
constants.py

Module defining global constants for cell and IRIF image analysis.

This module centralizes all tunable parameters used across the image processing
pipeline, including thresholds for cell and foci sizing, filtering parameters,
labeling conventions, and drawing settings for visualization of results.

Constants:
    CELL_MIN_AREA (int): Minimum allowed cell area in pixels.
    CELL_MAX_AREA (int): Maximum allowed cell area in pixels.
    FOCI_MIN_AREA (int): Minimum allowed IRIF (foci) area in pixels.
    S (int): Sigma value for Gaussian smoothing in cell mask generation.
    Q (int): Quantization level for cell mask thresholding.
    OL (float): Minimum voting percentage threshold for merging foci labels.
    N (int): Enlargement factor for IRIF drawing.
    TS (int): Text size for IRIF drawing annotations.
    BG_L (int): Label value reserved for background pixels.
    TH_MORPH (int): Minimum neighbor count threshold for morphological operations.
"""


# specific to cell sizes
CELL_MIN_AREA = 300                         # cell minimal area in pixels
CELL_MAX_AREA = 30000                       # cell maximal area in pixels

# specific to IRIF sizes
FOCI_MIN_AREA = 5                           # IRIF minimal area in pixels

# specific to cell filtering
S1 = 50                                     # background removal size for min filter
S2 = 20                                     # background removal size for gauss filter
S = 15                                      # sigma for cell_mask_S
Q = 8                                       # quantise for cell_mask_Q

# specific to foci filtering
OL = 0.4                                    # threshold of minimal voting percentage for label merging

# specific to results drawing
N = 5                                       # enlargement of IRIF drawing
TS = 10                                     # text size for IRIF drawing

# specific to labeling
BG_L = -1                                   # background label

# specific to morphology
TH_MORPH = 2                                # threshold for minimal count of neighbors

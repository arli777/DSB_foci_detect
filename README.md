# DSB_foci_detect

a codebase of a batchelor thesis that detects, 
quantizes ionizing radiation-induced foci (IRIFs), 
forming around DNA double stranded breaks (DSBs).

takes czi files, with IRIFs fluorescent images, image requirements:
- S (scenes) dimension: ≥1
- Z (axial) dimension: ≥1
- XY (lateral) dimensions: must be consistent across all scenes
- C (channel) dimension: ≥2 DAPI and IRIFs markers (≥1)
- missing dimensions: M (mosaic) and T (time)
  
## Files:
- [requirements.txt](requirements.txt)] - Python package requirements
- [constants.py](constants.py) - defining global constants for cell and IRIF image analysis
- [data_matrix.py](data_matrix.py) - DataMatrix class: representing and processing multidimensional image data using Laplacian filtering
- [read_czi.py](read_czi.py) - CZI class: loading and processing czi file
- [cell_segment.py](cell_segment.py) - image segmentation utilities for cell detection and division; CellSegment class: connected cell segment representation
- [foci_segment.py](foci_segment.py) - FociSegment class: segmenting and grouping foci within cell region
- [cell_foci_segment.py](cell_foci_segment.py) - CellFociSegment class: wraps CellSegment and a mapping from channel identifiers to FociSegment
- [segments.py](segments.py) - Segments class: wraps CellFociSegment classes for all cells in particular scene
- [drawing.py](drawing.py) - visualisation and png save utilities
- [detect.py](detect.py) - Detection class: detection pipeline combining CZI file reading, cell segmentation, and foci extraction

## usual pipeline
```
from detect import Detect

d = Detect("path/to/file.czi")
d.detect()
d.save_mat("path/to/results")
d.save_draw("path/to/results")
```

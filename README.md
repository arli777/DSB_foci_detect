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

### Python package requirements with specific version
| Package        | Version        |
| -------------- | -------------- |
| `numpy`        | `~=2.2.6`      |
| `matplotlib`   | `~=3.10.3`     |
| `scikit-image` | `~=0.25.2`     |
| `scipy`        | `~=1.15.3`     |
| `czifile`      | `~=2019.7.2.1` |
| `networkx`     | `~=3.4.2`      |
  
## Files:
- [find_pillar.py](find_pillar.py) finds pillar coordinates; finds goalposts from two lists of pillars (red and blue)
- [transformations.py](transformations.py) transforms matrix into list point cloud without nan values and rotates; visualises point cloud
- [vision.py](vision.py) draws pillar boundaries with colors; thread display camera feed; thread: find goal posts of pillars (red and blue)
- [take_picture.py](take_picture.py) saves .jpg picture from rgb camera
- [take_data.py](take_data.py) saves all data into .mat file and corresponding rgb image into .jpg file
- [params.txt](params.txt) note of color measurements

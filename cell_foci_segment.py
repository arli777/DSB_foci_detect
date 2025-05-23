"""
cell_foci_segment.py

Module providing the CellFociSegment class to associate foci segmentation results
with individual cell segments. Manages a mapping from channel identifiers to
FociSegment instances, extracting foci within a given CellSegment region.

Classes:
    CellFociSegment: Wraps a CellSegment and computes FociSegment per channel.
"""

import numpy as np
from foci_segment import FociSegment
from cell_segment import CellSegment


class CellFociSegment:
    """
    Associates foci segmentation with a specific cell region.

    Maintains a CellSegment instance and a dictionary mapping channel indices
    to their corresponding FociSegment results. Foci are computed on demand.

    Attributes:
        cell_seg (CellSegment): The cell region of interest.
        foci_seg_dict (dict[int, FociSegment]): Cache of FociSegment objects by channel.
    """

    def __init__(self, cell_seg: CellSegment):
        """
        Initialize with a given cell segment.

        Args:
            cell_seg (CellSegment): CellSegment defining the ROI for foci detection.
        """
        self.cell_seg = cell_seg
        self.foci_seg_dict = {}

    def foci_segment(self, c: int, img: np.ndarray, lap: np.ndarray):
        """
        Compute or the FociSegment for a given channel in this cell.

        If segmentation for channel `c` has not yet been performed, extracts the
        subvolume of `img` and `lap` corresponding to the cell's bounding box,
        instantiates a FociSegment, and caches it.

        Args:
            c (int): Channel index for which to segment foci.
            img (ndarray[Z,H,W,C]): Full image stack.
            lap (ndarray[Z,H,W,C]): Full Laplacian stack.
        """
        if c not in self.foci_seg_dict.keys():
            im = img[:, self.cell_seg.tl[0]:self.cell_seg.br[0] + 1, self.cell_seg.tl[1]:self.cell_seg.br[1] + 1]
            lp = lap[:, self.cell_seg.tl[0]:self.cell_seg.br[0] + 1, self.cell_seg.tl[1]:self.cell_seg.br[1] + 1]
            self.foci_seg_dict[c] = FociSegment(im, lp, self.cell_seg)

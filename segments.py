"""
segments.py

Module for extracting cell segments from an image and managing per-cell foci segmentation.
Defines the Segments class to detect cell regions, label them, and associate each
with a CellFociSegment for downstream foci analysis.

Classes:
    Segments: Detects cell segments in a 2D image, labels them, and wraps each in a CellFociSegment.
"""

import numpy as np
from cell_foci_segment import CellFociSegment
from cell_segment import CellSegment, cells_segments
from constants import BG_L


class Segments:
    """
    Detects and organizes cell regions within a single-channel image.

    This class finds connected cell boundary segments, labels each pixel by cell ID,
    and constructs a CellFociSegment object for each detected cell to enable
    foci segmentation within that cell.

    Attributes:
        cells (list[CellFociSegment]): List of per-cell foci segmentation managers.
        n_cells (int): Number of detected cells.
        labeled_cells (ndarray[int]): 2D array with same shape as input image, where each
            pixel is assigned its cell index or BG_L for background.
    """

    def __init__(self, image: np.ndarray):
        """
        Initialize and immediately extract cell segments from the image.

        Args:
            image (ndarray[H, W]): 2D single-channel image array.
        """
        self.cells: list[CellFociSegment] = []
        self.n_cells: int = 0
        self.labeled_cells: np.ndarray = BG_L * np.ones_like(image, dtype=int)
        self.extract_segments(image)

    def extract_segments(self, image: np.ndarray):
        """
        Find cell boundary segments and initialize CellFociSegment for each.

        Uses `cells_segments` to compute pixel-coordinate lists for each cell.
        Builds a labeled image where pixel values correspond to cell indices,
        and creates a CellFociSegment for each segment.

        Args:
            image (ndarray[H, W]): 2D image in which to detect cells.

        Side Effects:
            - Populates `self.cells` with CellFociSegment instances.
            - Fills `self.labeled_cells` with cell labels.
            - Updates `self.n_cells` to the number of segments found.
        """
        segments = cells_segments(image)
        for i in range(len(segments)):
            self.cells.append(CellFociSegment(CellSegment(segments[i])))
            self.labeled_cells[segments[i][:, 0], segments[i][:, 1]] = i
        self.n_cells = len(self.cells)

    def get_dict(self, z, channels) -> dict:
        """
        Serialize the Segments state and all cell-level foci results.

        Constructs a dictionary capturing the number of cells, the labeled cell image,
        and merged labeled images for all cells at once.

        Args:
            z (int): dimension of Z axis.
            channels (list[int]): list of channel axes.

        Returns:
            dict: {
                'cells_n': int number of detected cells,
                'cells_labels': ndarray[int] labeled cell image,
                'channels': dict[int, dict:
                    {
                    'foci_merged_n': list[int],
                    'foci_z_labels': ndarray[int],
                    'foci_merged_labels': ndarray[int]
                    }
                ]
            }
        """
        result = {}
        result['cells_n'] = self.n_cells
        result['cells_labels'] = self.labeled_cells
        result['channels'] = {}

        for c in channels:
            labels_merged = BG_L * np.ones_like(self.labeled_cells)
            labels_z = np.stack([BG_L * np.ones_like(self.labeled_cells) for _ in range(z)])
            labels_merged_n = []
            for cell in self.cells:
                pts = cell.cell_seg.pts
                pts_i = cell.cell_seg.pts_i
                labels_merged[pts[:, 0], pts[:, 1]] = cell.foci_seg_dict[c].merged_labels[pts_i[:, 0], pts_i[:, 1]]
                labels_i = cell.foci_seg_dict[c].get_labels_z_stack()
                labels_z[:, pts[:, 0], pts[:, 1]] = labels_i[:, pts_i[:, 0], pts_i[:, 1]]
                labels_merged_n.append(cell.foci_seg_dict[c].n)
            result['channels'][c] = {'foci_merged_n': labels_merged_n,
                                     'foci_z_labels': labels_z,
                                     'foci_merged_labels': labels_merged}
        return result

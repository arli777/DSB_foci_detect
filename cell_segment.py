"""
cell_segment.py

Module providing image segmentation utilities for cell detection and division.
Includes normalization, mask generation (Gaussian and quantization-based), boundary
pairing, segment division, and a CellSegment class to represent individual segments.

Functions:
    normalise_img(image): Normalise image intensities to [0,1].
    pairing_matrix(segments_out, segments_in): Build inclusion matrix between segment sets.
    cells_mask_S(image, sigma): Generate binary mask of cells via Gaussian filtering.
    cells_mask_Q(image, quantise, th): Generate binary mask of cells via quantization.
    divide_segments(segments_out, segments_in): Split outer segments by nearest inner segment.
    cells_segments(image, quantise, sigma): Extract outer cell boundary segments.

Classes:
    CellSegment: Represents a connected cell segment with bounding box and mask.
"""

import numpy as np
import scipy.ndimage as spi

from skimage.morphology import dilation, erosion, remove_small_objects, remove_small_holes, closing, disk
from skimage.measure import label

from scipy.spatial import cKDTree

from constants import S, Q, CELL_MIN_AREA, CELL_MAX_AREA


def normalise_img(image: np.ndarray) -> np.ndarray:
    """
    Normalize an image to the [0, 1] range.

    Args:
        image (ndarray): Input image array of any numeric type.

    Returns:
        ndarray: Floating-point image scaled so min maps to 0.0 and max to 1.0.
    """
    im = image.copy().astype(float)
    im -= im.min()
    im /= im.max()
    return im


def pairing_matrix(segments_out: list[np.ndarray], segments_in: list[np.ndarray]) -> np.ndarray:
    """
    Compute a binary inclusion matrix between inner and outer segments.

    pairing_matrix[i, j] == 1 indicates that the i-th `segments_in` lies entirely
    within the j-th `segments_out` bounding box.

    Args:
        segments_out (list of ndarray): Each array is shape (N,2) coordinates of an outer segment.
        segments_in (list of ndarray): Each array is shape (M,2) coordinates of an inner segment.

    Returns:
        ndarray[int]: Matrix of shape (len(segments_in), len(segments_out)) with 0/1 entries.
    """
    n_in = len(segments_in)
    n_out = len(segments_out)

    tl_in = np.vstack([np.min(segments_in[i], axis=0) for i in range(n_in)])
    br_in = np.vstack([np.max(segments_in[i], axis=0) for i in range(n_in)])

    tl_out = np.vstack([np.min(segments_out[i], axis=0) for i in range(n_out)])
    br_out = np.vstack([np.max(segments_out[i], axis=0) for i in range(n_out)])

    pairing_mx = np.zeros((n_in, n_out), dtype=int)

    for i in range(len(segments_out)):
        is_tl_out_smaller = np.all((tl_in >= tl_out[[i], :]).reshape(-1, 2), axis=1)
        is_br_out_bigger = np.all((br_in <= br_out[[i], :]).reshape(-1, 2), axis=1)
        segment_in_is_within_out = np.all(np.vstack((is_tl_out_smaller, is_br_out_bigger)), axis=0).astype(int)
        pairing_mx[:, i] += segment_in_is_within_out

    return pairing_mx


def cells_mask_S(image: np.ndarray, sigma=S) -> np.ndarray:
    """
    Generate a binary mask of cell regions using Gaussian smoothing.

    Applies a Difference-of-Gaussian-like approach, thresholding, morphological
    cleaning, and hole removal to isolate cell interiors (excluding halos).

    Args:
        image (ndarray[int]): 2D single-channel image with intensity range [0,255].
        sigma (float): Standard deviation for Gaussian filter.

    Returns:
        ndarray[bool]: Binary mask of detected cell regions.
    """
    im = normalise_img(image)
    im = im - spi.gaussian_filter(im, sigma=sigma)
    im[im < 0] = 0
    im = normalise_img(im)

    # find threshold at first rising histogram bin
    hist, bins = np.histogram(im, bins=256, range=(0, 1))
    th_idx = np.where((hist[:-1] - hist[1:]) < 0)[0][0] - 1
    th = bins[th_idx]

    salem3 = disk(3)

    mask = im > th

    # remove holes
    mask = remove_small_holes(mask, area_threshold=CELL_MAX_AREA)

    # remove small objects that could be atached
    mask = erosion(mask, salem3)
    mask = remove_small_objects(mask, min_size=CELL_MIN_AREA)
    mask = dilation(mask, salem3)

    # close open cells
    mask = closing(mask, salem3)
    mask = remove_small_holes(mask, area_threshold=CELL_MAX_AREA)

    # remove strings
    mask = erosion(mask, disk(3))
    mask = dilation(mask, disk(3))

    return mask


def cells_mask_Q(image: np.ndarray, quantise=Q, th=0) -> np.ndarray:
    """
    Generate a binary mask of cell regions using intensity quantization.

    Args:
        image (ndarray): 2D single-channel image with intensity range [0,255].
        quantise (int): Number of quantization levels.
        th (int): Threshold level in quantised units.

    Returns:
        ndarray[bool]: Binary mask of detected cell regions including slight halo.
    """

    step = (np.max(image) - np.min(image)) // quantise
    im = (image - np.min(image)) // step
    mask = im > th
    mask = remove_small_holes(mask, area_threshold=CELL_MAX_AREA)
    mask = remove_small_objects(mask, min_size=CELL_MIN_AREA)
    mask = closing(mask, disk(3))
    mask = remove_small_objects(mask, min_size=CELL_MIN_AREA)
    return mask


def divide_segments(segments_out, segments_in) -> list[np.ndarray]:
    """
    Divide outer segments by assigning pixels to nearest inner segment when multiple overlap.

    Segments with exactly one inner inclusion are returned unchanged. For segments
    containing multiple inner segments, a 1-nearest-neighbor split is performed using
    a KD-tree per inner segment.

    Args:
        segments_out (list of ndarray): Outer segment point lists, each array is shape (N,2) coordinates of an outer segment.
        segments_in (list of ndarray): Inner segment point lists, each array is shape (M,2) coordinates of an inner segment.

    Returns:
        list of ndarray: New list of split segments combining single and multi-case.
    """
    pairing_mx = pairing_matrix(segments_out, segments_in)
    segment_in_within_out_count = np.sum(pairing_mx, axis=0)

    # segments with single inner inclusion
    single_segments_idx = np.where(segment_in_within_out_count == 1)[0]
    single_segments = [segments_out[i] for i in single_segments_idx]

    # segments needing split
    multi_segment_idx = np.where(segment_in_within_out_count > 1)[0]
    multi_segments = []
    for i in multi_segment_idx:
        segment_out_i = segments_out[i]
        segments_in_i_idx = np.where(pairing_mx[:, i] > 0)[0]

        labels_i = np.empty(segment_out_i.shape[0], dtype=int)
        min_dists_i = np.full(segment_out_i.shape[0], np.inf)

        k = 0
        for j in segments_in_i_idx:
            tree_j = cKDTree(segments_in[j])
            dists_j, _ = tree_j.query(segment_out_i)
            mask_j = dists_j < min_dists_i
            labels_i[mask_j] = k
            k += 1
            min_dists_i[mask_j] = dists_j[mask_j]

        # collect split segments
        for k in range(len(segments_in_i_idx)):
            idx_k = np.where(labels_i == k)[0]
            multi_segments.append(segment_out_i[idx_k, :])

    return single_segments + multi_segments


def cells_segments(image: np.ndarray, quantise=Q, sigma=S) -> list[np.ndarray]:
    """
    Extract coordinate lists of outer cell boundary segments from an image.

    Combines quantization and Gaussian masks to detect inner and outer regions,
    labels connected components, and divides overlapping regions.

    Args:
        image (ndarray): 2D single-channel image.
        quantise (int): Quantization levels for `cells_mask_Q`.
        sigma (float): Gaussian sigma for `cells_mask_S`.

    Returns:
        list of ndarray[int]: Each element is an (N,2) array of pixel coordinates for one segment.
    """
    mask_in = np.logical_or(cells_mask_Q(image, quantise=quantise, th=1), cells_mask_S(image, sigma=sigma))
    label_in = label(mask_in)
    segments_in = [np.column_stack(np.where(label_in == i)) for i in range(1, np.max(label_in) + 1)]

    mask_out = np.logical_or(cells_mask_Q(image, quantise=quantise, th=0),
                             mask_in)  # every inner mask should be inside another outer mask
    label_out = label(mask_out)
    segments_out = [np.column_stack(np.where(label_out == i)) for i in range(1, np.max(label_out) + 1)]

    segments = divide_segments(segments_out, segments_in)
    segments = [seg for seg in segments if seg.size >= CELL_MIN_AREA]
    return segments


class CellSegment:
    """
    Represents a connected cell segment with bounding box and binary mask.

    Attributes:
        pts (ndarray[int]): Coordinates of segment pixels.
        n_pts (int): Number of pixels in the segment.
        tl (ndarray[int]): Top-left coordinate of bounding box.
        br (ndarray[int]): Bottom-right coordinate of bounding box.
        center (ndarray[float]): Mean coordinate of segment pixels.
        pts_i (ndarray[int]): Coordinates of segment pixels inert to bounding box
        res (ndarray[int]): Size of bounding box (height, width).
        mask (ndarray[bool]): Binary mask of the segment within its bbox.
    """

    def __init__(self, pts: np.ndarray):
        """
        Initialize a CellSegment from pixel coordinates.

        Args:
            pts (ndarray[int]): Array of shape (N,2) of (row, col) pixel indices.
        """
        self.pts: np.ndarray = pts
        self.n_pts: int = pts.shape[0]
        self.tl: np.ndarray = np.min(pts, axis=0)
        self.br: np.ndarray = np.max(pts, axis=0)

        self.center: np.ndarray = np.mean(pts, axis=0)

        self.pts_i: np.ndarray = pts - self.tl
        self.res: np.ndarray = self.br - self.tl + 1

        self.mask: np.ndarray = np.zeros(self.res, dtype=bool)
        self.get_bin_mask()

    def get_bin_mask(self):
        """
        Build the binary mask of the segment within its bounding box.

        The mask is a 2D boolean array of shape `res`, with True at segment pixels.
        """
        self.mask[self.pts_i[:, 0], self.pts_i[:, 1]] = True

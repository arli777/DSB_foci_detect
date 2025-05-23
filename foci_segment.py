"""
foci_segment.py

Module for segmenting and grouping foci within cell regions using
Laplacian-of-Gaussian masks, morphological operations, and graph-based merging.
Provides utilities to create binary masks, label connected components,
build inclusion graphs across Z-layers, and postprocess merged foci regions.

Functions:
    get_float_mask(bin_mask): Smooth binary mask to float mask via Gaussian.
    get_masked_img(img, pts_i, mask): Apply mask to 3D image stack, filling outside with local minima.
    fraction_erosion(bin_im, a, b): Erode binary image on upscaled grid and downsample.
    sep_label(bin_mask): Separate connected regions and split by inner boundaries.
    get_individual_labels(lap_gauss_mask): Extract per-layer labels and index mappings.
    labels_to_pts(labels, ranges): Convert label arrays to point-coordinate lists.
    get_graph(labels, ranges, l_pts, root): Build directed inclusion graph of segments.
    graph_2_groups(G): Group leaf-to-root paths in the inclusion graph.
    groups_to_labels(labels, l_pts, foci_groups): Merge groups into weighted label image.
    fill_by_neighbor_and_connect(binary_img, threshold): Fill and connect regions by neighbor count.
    merged_labels_postprocess(merged_labels, n): Postprocess merged labels with size ordering.
    labels_2_circles(merged_labels): Compute circle approximations (center, radius) for labeled regions.

Classes:
    FociSegment: Encapsulates extraction, connectivity, and merging of foci segments within a cell.
"""

import numpy as np
import skimage as ski
import scipy.ndimage as spi
from skimage.measure import label
import networkx as nx

from cell_segment import CellSegment, divide_segments
from constants import OL, BG_L, TH_MORPH


def get_float_mask(bin_mask: np.ndarray) -> np.ndarray:
    """
    Smooth a binary mask into a float-valued mask via Gaussian blur.

    Args:
        bin_mask (ndarray[bool]): 2D binary mask.

    Returns:
        ndarray[float]: Smoothed mask of same shape, values in range [0., 1.].
    """
    return spi.gaussian_filter(bin_mask.astype(float), sigma=1)


def get_masked_img(img: np.ndarray, pts_i: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Apply a binary/float mask to each Z-plane of a 3D image stack.

    Outside the mask, pixel values are replaced by the minimum value over the masked points
    per plane.

    Args:
        img (ndarray): 3D image array shape (Z, H, W).
        pts_i (ndarray[int]): Coordinates of points within mask.
        mask (ndarray[float or bool]): 2D mask shape (H, W).

    Returns:
        ndarray[int]: Masked image stack of same shape as `img`.
    """
    img_min = np.ones_like(img) * np.min(img[:, pts_i[:, 0], pts_i[:, 1]], axis=1)[:, None, None]
    img_masked = (img * mask[None, :, :] + img_min * (1 - mask)[None, :, :]).astype(int)
    return img_masked


def fraction_erosion(bin_im: np.ndarray, a: int, b: int) -> np.ndarray:
    """
    Perform erosion on an upscaled binary image and downsample by max-pooling.

    Args:
        bin_im (ndarray[bool]): Input binary image.
        a (int): Structuring element radius for erosion.
        b (int): Upscaling factor.

    Returns:
        ndarray[bool]: Downsampled eroded image of original shape.
    """
    kernel = np.ones((b, b), dtype=bool)
    upscaled = np.kron(bin_im, kernel)
    upscaled = ski.morphology.erosion(upscaled, ski.morphology.disk(a))
    downscaled = np.max(upscaled.reshape((bin_im.shape[0], b, bin_im.shape[1], b)), axis=(1, 3))
    return downscaled


def sep_label(bin_mask: np.ndarray) -> np.ndarray:
    """
    Separate connected regions and split using fraction erosion segments.

    Args:
        bin_mask (ndarray[bool]): Binary mask of regions.

    Returns:
        ndarray[int]: Labeled image after separating connected segments.
    """
    labels_prim = label(bin_mask, connectivity=1)
    segments = [np.column_stack(np.where(labels_prim == i)).reshape((-1, 2)) for i in range(1, np.max(labels_prim) + 1)]

    labels_sec = label(fraction_erosion(bin_mask, 1, 2), connectivity=1)
    # there are any
    if np.max(labels_sec) > 0:
        segments_in = [np.column_stack(np.where(labels_sec == i)).reshape((-1, 2)) for i in
                       range(1, np.max(labels_sec) + 1)]
        segments = divide_segments(segments, segments_in)

    labels_div = np.zeros_like(labels_prim, dtype=int)
    for i in range(len(segments)):
        labels_div[segments[i][:, 0], segments[i][:, 1]] = i + 1
    return labels_div


def get_individual_labels(lap_gauss_mask: np.ndarray) -> (np.ndarray, list[int], int, np.ndarray, dict[int, int]):
    """
    Extract and reindex individual segment labels across Z-layers.

    Iterates through each Z-plane mask, separates segments,
    offsets label values to maintain unique IDs,
    and builds mapping arrays.

    Args:
        lap_gauss_mask (ndarray[bool], shape (Z, H, W)):
            Boolean mask stack per Z-plane.

    Returns:
        labels (ndarray[int]): Stacked label images with background = BG_L.
        ranges (list[int]): Cumulative label index boundaries per layer.
        n_seg (int): Total number of segments.
        idx_l (ndarray[int]): Mapping from label ID to layer index.
        z_dict (dict): Mapping from Z-plane index to label stack index or -1 if empty.
    """
    labels = []
    ranges = [0]
    n_seg = 0

    z_dict = {}

    for i, mask in enumerate(lap_gauss_mask):
        if not np.any(mask):
            z_dict[i] = -1
            continue  # leave out empty masks
        z_dict[i] = len(labels)
        mask_i = sep_label(mask) + BG_L
        mask_i[mask_i > BG_L] += n_seg
        labels.append(mask_i)
        n_seg = int(np.max(mask_i) + 1)
        ranges.append(n_seg)

    if len(labels) == 0: return None, None, 0, None, z_dict

    idx_l = np.zeros(n_seg, dtype=int)
    for l in range(len(labels)): idx_l[ranges[l]:ranges[l + 1]] = l

    labels = np.stack(labels)

    return labels, ranges, n_seg, idx_l, z_dict


def labels_to_pts(labels: np.ndarray, ranges: list[int]) -> list[np.ndarray]:
    """
    Convert labeled images to coordinate lists for each segment.

    Args:
        labels (ndarray[int], shape (L, H, W)): Label stack over L layers.
        ranges (list[int]): Label index boundaries for each layer.

    Returns:
        list of ndarray[int]: Each array is (N,2) coordinates of one segment.
    """
    segments = []
    for l in range(labels.shape[0]):
        for i in range(ranges[l], ranges[l + 1]):
            segments.append(np.column_stack(np.where(labels[l, :, :] == i)))
    return segments


def get_graph(labels: np.ndarray, ranges: list[int], l_pts: list[np.ndarray], root: int):
    """
    Build a directed graph of segment inclusions across layers.

    Nodes represent segment IDs; edges connect parent to child segments when
    their pixel sets overlap in projection.

    Args:
        labels (ndarray[int]): Stack of labeled layers.
        ranges (list[int]): Label boundaries per layer.
        l_pts (list of ndarray): Coordinate lists for each segment.
        root (int): ID to use for root node (background).

    Returns:
        DiGraph: NetworkX directed graph of segment connectivity.
    """
    G = nx.DiGraph()
    G.add_node(root)  # add -1 root node
    for i in range(ranges[0], ranges[1]):
        G.add_node(i)  # add the nodes from the first layer
        G.add_edge(root, i)

    for l in range(1, labels.shape[0]):
        # draw graph leaves
        labels_l = np.ones_like(labels[l, :, :]) * root
        if l == 1:
            labels_l = labels[0, :, :]
        else:
            leaves = [node for node in G.nodes if G.out_degree(node) == 0]
            for leaf in leaves:
                labels_l[l_pts[leaf][:, 0], l_pts[leaf][:, 1]] = leaf

        for i in range(ranges[l], ranges[l + 1]):
            values_i = labels_l[l_pts[i][:, 0], l_pts[i][:, 1]]

            unique_values = np.unique(values_i[values_i != root])  # get what values are overlaping
            n_val = unique_values.size

            if n_val <= 1:
                G.add_node(i)
                G.add_edge(root if n_val == 0 else unique_values[0], i)

    return G


def graph_2_groups(G: nx.DiGraph):
    """
    Extract leaf-to-root paths as groups from the inclusion graph.

    Args:
        G (DiGraph): Directed graph of segment connectivity.

    Returns:
        list of lists: Each sublist is sequence of node IDs from leaf up to branch.
    """
    groups = []
    leaves = [node for node in G.nodes if G.out_degree(node) == 0]

    for leaf in leaves:
        g_l = [leaf]
        cur_node = leaf
        while True:
            predecessor = list(G.predecessors(cur_node))[0]
            if G.out_degree(predecessor) == 1 & G.in_degree(predecessor) == 1:
                g_l.append(predecessor)
                cur_node = predecessor

            else:
                break
        groups.append(g_l)

    return groups


def groups_to_labels(labels: np.ndarray, l_pts: list[np.ndarray], foci_groups: list[list[int]]) -> np.ndarray:
    """
    Merge segment groups into a labeled image via voting threshold.

    Args:
        labels (ndarray[int]): Original label stack.
        l_pts (list of ndarray): Coordinates per segment ID.
        foci_groups (list of lists): Groupings of segment IDs.

    Returns:
        ndarray[int]: 2D label image with merged group IDs, background=BG_L.
    """

    weighted_labels = []

    for group in foci_groups:
        labels_i = np.zeros_like(labels[0, :, :], dtype=float)
        for g in group:
            labels_i[l_pts[g][:, 0], l_pts[g][:, 1]] += 1
        labels_i[labels_i / len(group) < OL] = 0
        weighted_labels.append(labels_i)

    weighted_labels = np.stack(weighted_labels)

    active_mask = np.sum(weighted_labels, axis=0) > 0
    argmax_labels = BG_L * np.ones_like(active_mask, dtype=int)
    argmax_labels[active_mask] = np.argmax(weighted_labels, axis=0)[active_mask]

    return argmax_labels


def fill_by_neighbor_and_connect(binary_img: np.ndarray, threshold: int = TH_MORPH) -> np.ndarray:
    """
    Fill holes and connect regions based on neighbor counts and morphology.

    Args:
        binary_img (ndarray[bool]): Input binary image.
        threshold (int): Minimum neighbor count to fill a pixel.

    Returns:
        ndarray[bool]: Processed binary image.
    """
    # fill holes
    bin_i = ski.morphology.remove_small_holes(binary_img)

    # get how many unconnected parts
    label_i = label(bin_i, connectivity=1) + BG_L
    n = np.max(label_i) - BG_L

    # draw connecting polygone/line
    if n > 1:
        r = []
        c = []
        for j in range(n):
            c_j = np.round(np.mean(np.column_stack(np.where(label_i == j)).reshape((-1, 2)), axis=0)).astype(int)
            r.append(c_j[0])
            c.append(c_j[1])

        r, c = np.array(r), np.array(c)
        rr, cc = ski.draw.line(r[0], c[0], r[1], c[1]) if n == 2 else ski.draw.polygon(r, c, bin_i.shape)

        bin_shape = np.zeros_like(bin_i, dtype=bool)
        bin_shape[rr, cc] = True

        bin_shape = ski.morphology.dilation(bin_shape, ski.morphology.disk(1))

        bin_i = (bin_i | bin_shape)

    # count white neighbors in 4 neighborhood
    kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])

    for i in range(3, threshold - 1, -1):
        neighbor_count = spi.convolve(bin_i.astype(np.uint8), kernel, mode='constant', cval=0)
        bin_i = (neighbor_count >= i) | bin_i
    return bin_i


def merged_labels_postprocess(merged_labels: np.ndarray, n: int) -> np.ndarray:
    """
    Reassign processed merged labels by descending region size.

    Args:
        merged_labels (ndarray[int]): Label image after merging groups.
        n (int): Number of label classes.

    Returns:
        ndarray[int]: Relabeled image with largest region = 0, next = 1, etc.
    """
    labels = BG_L * np.ones_like(merged_labels, dtype=int)

    bin_labels = []

    for i in range(n):
        bin_labels.append(fill_by_neighbor_and_connect(merged_labels == i))
    bin_labels = np.stack(bin_labels)

    # order by size from biggest to smallest
    sizes = np.sum(bin_labels, axis=(1, 2))
    sorted_indices = np.argsort(-sizes)

    for i in range(n):
        labels[bin_labels[sorted_indices[i]]] = i

    return labels


def labels_2_circles(merged_labels: np.ndarray) -> tuple[list, list]:
    """
    Compute circle approximations (center, radius) for each labeled region.

    Args:
        merged_labels (ndarray[int]): Label image.

    Returns:
        centers (list[tuple]): (row, col) center of each region.
        radii (list[float]): Radius for each region.
    """
    centers = []
    radius = []

    if np.all(merged_labels == BG_L):
        return centers, radius

    for i in np.unique(merged_labels[merged_labels > BG_L]):
        pts_i = np.column_stack(np.where(merged_labels == i)).reshape((-1, 2))
        centers.append(np.mean(pts_i, axis=0))
        radius.append(np.max(np.linalg.norm(pts_i - centers[-1], axis=1)))
    return centers, radius


class FociSegment:
    """
    Encapsulates extraction, connectivity, and merging of foci segments within a cell.

    Attributes:
        masked_img (ndarray): Masked image stack for the cell.
        sharpest_z (int): Z-index with highest Laplacian response.
        lap_gauss_mask (ndarray[bool]): Gaussian-smoothed Laplacian mask stack.
        labels (ndarray[int]): Label stack of individual detected segments.
        merged_labels (ndarray[int]): Final merged label image.
        n (int): Number of merged foci.
    """

    def __init__(self, img: np.ndarray, lap: np.ndarray, seg: CellSegment):
        """
        Initialize by extracting and connecting foci segments.

        Args:
            img (ndarray, shape (Z,H,W)): Original pixel stack.
            lap (ndarray, shape (Z,H,W)): Laplacian stack.
            seg (CellSegment): Cell segment defining region of interest.
        """
        self.masked_img: np.ndarray = np.zeros_like(img, dtype=int)
        self.sharpest_z: int = -1
        self.lap_gauss_mask: np.ndarray = np.zeros_like(img, dtype=bool)
        self.labels: np.ndarray = None
        self.z_dict: dict[int, int] = {}
        self.merged_labels: np.ndarray = BG_L * np.ones_like(img[0], dtype=int)
        self.n: int = 0

        self.extract_segments(img, lap, seg)
        self.connect_segments()

    def extract_segments(self, img: np.ndarray, lap: np.ndarray, seg: CellSegment):
        """
        Generate masked image and binary Laplacian-Gaussian masks.

        Args:
            img (ndarray): Original image stack.
            lap (ndarray): Laplacian of image stack.
            seg (CellSegment): CellSegment region.
        """
        self.masked_img = get_masked_img(img, seg.pts_i, get_float_mask(seg.mask))
        self.sharpest_z = np.argmax(np.sum(np.abs(lap[:, seg.pts_i[:, 0], seg.pts_i[:, 1]]), axis=1))

        lap_gauss = np.stack([spi.gaussian_filter(lp, sigma=1) for lp in lap])
        self.lap_gauss_mask = np.stack([(
                ski.morphology.dilation(
                    ski.morphology.erosion(l_mask, ski.morphology.disk(1)), ski.morphology.disk(1)) & seg.mask) for
            l_mask in (lap_gauss < 0)])

    def connect_segments(self):
        """
        Perform graph-based grouping and postprocessing of foci labels.
        """
        self.labels, ranges, n_segments, idx_l, self.z_dict = get_individual_labels(self.lap_gauss_mask)

        if self.labels is None:
            return

        l_pts = labels_to_pts(self.labels, ranges)
        root = -1

        G = get_graph(self.labels, ranges, l_pts, root)
        foci_groups = graph_2_groups(G)
        merged_labels_raw = groups_to_labels(self.labels, l_pts, foci_groups)
        self.n = np.max(merged_labels_raw) - BG_L
        merged_labels = merged_labels_postprocess(merged_labels_raw, self.n)
        self.merged_labels = merged_labels

    def get_labels_z_stack(self):
        """
        Reorganises self.labels, so that each z has its detection, including empty.

        Returns:
            labels (ndarray[int]): Label stack.
        """
        labels = []
        empty_label = BG_L * np.ones_like(self.merged_labels)
        for z in range(self.lap_gauss_mask.shape[0]):
            l = self.z_dict[z]
            labels.append(empty_label if l == -1 else self.labels[l])
        labels = np.stack(labels)
        return labels

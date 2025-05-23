"""
drawing.py

Visualization layer for the cell and foci detection results.
Draws and saves segmentations in whole image:
cell segmentations, foci_segmentations on each z and merged result.

Functions:
    get_path: Ensures a directory exists, creates if necessary.
    get_fig: Initializes a matplotlib figure sized to an image.
    set_axes: Configures axis limits and hides figure axes.
    save_draw_cells: Draws labels on image and writes labels id.
    save_draw_cells_foci_counts: Draws labeled cells on image and writes number of segmented foci.
    save_draw_from_Detect: Draws and saves detections
"""

import numpy as np
import skimage

from cell_segment import normalise_img
from constants import BG_L, TS

import os
import matplotlib.pyplot as plt
from matplotlib import cm


def get_path(path: str) -> str:
    """
    Ensures that a given directory path exists.

    Args:
        path (str): The directory path to check or create.

    Returns:
        str: The verified or newly created path.
    """
    if not os.path.exists(path): os.mkdir(path)
    return path


def get_fig(max1, max0, n=1):
    """
    Creates a matplotlib figure with dimensions proportional to image shape.

    Args:
        max1 (int): Width of the image.
        max0 (int): Height of the image.
        n (int): Scaling factor (default is 1).

    Returns:
        tuple: A tuple (fig, ax) representing the matplotlib figure and axis.
    """
    return plt.subplots(figsize=(n * max1 / 100, n * max0 / 100), dpi=100)


def set_axes(max1: int, max0: int, ax: plt.Axes):
    """
    Configures the axes to match image dimensions and hides axes.

    Args:
        max1 (int): Image width.
        max0 (int): Image height.
        ax (matplotlib.axes.Axes): The axes object to configure.
    """
    ax.set_xlim(0, max1 - 1)
    ax.set_ylim(0, max0 - 1)
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)


def generate_n_colors(n: int, colormap: str = 'gist_rainbow') -> list:
    """
    Generate N distinct RGB colors using a colormap.

    Args:
        n (int): Number of colors to generate.:
        colormap (str, optional): Colormap to use. Defaults to 'gist_rainbow'.:

    Returns:
        list of colors.
    """
    cmap = cm.get_cmap(colormap, n)
    return [cmap(i)[:3] for i in range(n)]


def get_colors_for_labels(colors: list, labels: np.ndarray, bgl: int = 0) -> list:
    """
    pick from colors only those that are present in labels.
    Args:
        colors (list): full list of colors, len(colors) >= max(labels-bgl)
        labels (np.ndarray): image labels
        bgl (int): background label, default 0

    Returns:
        list of selected colors.
    """

    c = []
    for l in np.unique(labels):
        if l == bgl: continue
        l_i = l - bgl - 1
        c.append(colors[l_i])
    return c


######################################Draw & Save detection

def save_draw_cells(image: np.ndarray, labels: np.ndarray, n: int, f_path: str, colors=None):
    """
    Draws labeled cells on an image.

    Args:
        image (ndarray): Image with cells.
        labels (ndarray): Cells labels.
        n (int): Magnification of image.
        f_path (str): Path to save image.
        colors (list): Full list of colors.
    """
    if colors is None:
        colors = list()
    max0, max1 = image.shape[:2]
    b = TS // (n * 2) + 1

    if colors is None or len(colors)==0:
        c = generate_n_colors(np.max(labels)-BG_L)
    else:
        c = get_colors_for_labels(colors, labels, BG_L)

    labeled_image = skimage.color.label2rgb(label=labels, image=normalise_img(image), alpha=0.15, bg_label=BG_L, colors=c)

    fig, ax = get_fig(max1, max0, n)
    ax.imshow(labeled_image)
    for region_label in np.unique(labels):
        if region_label == BG_L: continue
        # multiple labels acros multiple blobs can happen
        mask_label = skimage.measure.label(labels == region_label)
        for i in np.unique(mask_label):
            if i == 0: continue
            coords = np.argwhere(mask_label == i)
            y, x = coords.mean(axis=0)
            y = min(max(b, y), max0 - 1 - b)
            x = min(max(b, x), max1 - 1 - b)
            ax.text(x, y, str(region_label), color='k', fontsize=TS, ha='center', va='center')
    set_axes(max1, max0, ax)
    plt.savefig(f_path, bbox_inches='tight', pad_inches=0)
    plt.close()


def save_draw_cells_foci_counts(image: np.ndarray, labels: np.ndarray, foci_counts: list, f_path: str):
    """
    Draws labeled cells on an image.

    Args:
        image (ndarray): Image with cells.
        labels (ndarray): Cells labels.
        foci_counts (list[int]): number of foci per cell.
        f_path (str): Path to save image.
    """
    max0, max1 = image.shape[:2]
    b = TS // 2 + 2

    colors = generate_n_colors(np.max(labels) + 1)

    labeled_image = skimage.color.label2rgb(label=labels, image=normalise_img(image), alpha=0.05, bg_label=BG_L,
                                            colors=colors)

    fig, ax = get_fig(max1, max0)
    ax.imshow(labeled_image)
    for i in np.unique(labels):
        if i == BG_L: continue
        coords = np.argwhere(labels == i)
        y, x = coords.mean(axis=0)
        y = min(max(b, y), max0 - 1 - b)
        x = min(max(b, x), max1 - 1 - b)
        ax.text(x, y, str(foci_counts[i]), color='w', fontsize=TS, ha='center', va='center')
    set_axes(max1, max0, ax)
    plt.savefig(f_path, bbox_inches='tight', pad_inches=0)
    plt.close()

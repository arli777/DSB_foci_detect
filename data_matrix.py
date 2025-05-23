"""
data_matrix.py

Module for representing and processing multidimensional image data using Laplacian filtering.
Defines the DataMatrix class to manage raw image stacks and compute
layer-wise Laplacian transformations for further analysis.

Classes:
    DataMatrix: Holds 5D image data and computes Laplacian on each layer.
"""

import numpy as np
import scipy.ndimage as spi


class DataMatrix:
    """
    Encapsulates a multidimensional image dataset and computes Laplacian-filtered layers.

    This class accepts a raw integer-valued data array of shape (S, Z, Y, X, C) where:
        S: number of scenes,
        Z: number of depth slices,
        Y, X: spatial dimensions (height, width),
        C: number of channels (e.g., 2 or 3).

    It computes the 2D Laplacian on each (Y, X) plane for each sample, slice, and channel,
    supporting cases where the first channel may be skipped for two-channel data.

    Attributes:
        dims (dict): Mapping with keys 'S', 'Z', 'Y', 'X', 'C' describing data shape.
        data (ndarray[int]): Input data cast to integers, data.shape[4] is always 3
        data_lap (ndarray[int]): Laplacian-filtered result, same shape as `data`.
    """

    def __init__(self, data: np.ndarray, dims: dict):
        """
        Initializes the DataMatrix with raw data and dimension metadata.

        Args:
            data (array-like): Input image data convertible to an integer NumPy array
                with shape (S, Z, Y, X, C), where C=3
            dims (dict): Mapping of dimension names to sizes:
                - 'S': number of scenes
                - 'Z': number of depth slices
                - 'Y': image height in pixels
                - 'X': image width in pixels
                - 'C': number of channels (2 or 3), may differ from data.shape[4]

        Raises:
            ValueError: If `data.shape[:-1]` does not match the provided `dims`.
        """
        expected_shape = (dims['S'], dims['Z'], dims['Y'], dims['X'])
        if tuple(data.shape[:-1]) != expected_shape:
            raise ValueError(f"Data shape[:-1] {data.shape[:-1]} does not match dims {expected_shape}")
        if dims['C'] not in [2, 3]:
            raise ValueError(f"Wrong number of channels: {dims['C']}")
        if data.shape[-1] != 3:
            raise ValueError(f"Data does not have 3 channels: data shape[-1] {data.shape[-1]}")
        self.dims: dict = dims
        self.data: np.ndarray[int] = data.astype(int)
        self.data_lap: np.ndarray[int] = np.zeros_like(self.data)
        self.data_layers()  # compute Laplacian layers

    def data_layers(self):
        """
        Compute the Laplacian filter on each 2D plane of the data array.

        Applies a 2D Laplace operator on the (Y, X) plane for every
        combination of scene (S), depth slice (Z), and channel (C).
        """

        # determine starting channel index (skip channel 0 for 2-channel data)
        c0 = 0 if self.dims['C'] == self.data.shape[-1] else self.data.shape[-1] - self.dims['C']

        for s in range(self.dims['S']):
            for z in range(self.dims['Z']):
                for c in range(c0, self.dims['C'] + c0):
                    # apply 2D Laplacian on Y,X plane
                    self.data_lap[s, z, :, :, c] = spi.laplace(self.data[s, z, :, :, c])

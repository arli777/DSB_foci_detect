"""
read_czi.py

Module for loading and processing Zeiss CZI microscopy files.
Provides the CZI class to parse metadata, extract multi‑scene image data,
convert to standard arrays, and visualize or save image planes.

Constants:
    SCALING (dict): Unit labels for each axis (e.g., 'X': 'm', 'T': 's').

Classes:
    CZI: Represents a CZI file, exposes metadata, image data as a DataMatrix,
         and methods for plotting and saving image slices.
"""

from pathlib import Path
from czifile import CziFile, DirectoryEntryDV
from xml.etree import ElementTree as ET
import os
import numpy as np
from matplotlib import pyplot as plt
from data_matrix import DataMatrix

SCALING = {'X': 'm', 'Y': 'm', 'Z': 'm', 'T': 's'}  # physical units for axes


class CZI:
    """
    Reader and container for Zeiss .czi microscopy files.

    Based on napari-czifile2, this class loads image data and XML metadata,
    splits into scenes, maps channels, and wraps the pixel data in a DataMatrix
    for further analysis.

    Attributes:
        f_name (str): Filename without extension.
        acquisition_time (str): ISO timestamp of image acquisition.
        metadata_xml (Element): Root of parsed XML metadata tree.
        axes (str): Ordered axis names of the data (e.g. 'SZYXC').
        dims (dict): Sizes for each axis name.
        scales (dict): Physical scaling factors per axis.
        channel_names (list[str]): Names of channels in original file order.
        DAPI (int): Index of DAPI channel in image arrays.
        FOCI (list[int]): Indices of foci channels.
        channel_dict (dict[int,str]): Mapping channel index to name.
        DM (DataMatrix): DataMatrix instance holding image data.
    """

    def __init__(self, path: Path):
        """
        Initialize and immediately read a .czi file.

        Args:
            path (Path): Filesystem path to the .czi file to load.
        """
        self.f_name: str = ''  # name of file without '.czi': str
        self.acquisition_time: str = ''  # "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'"
        self.metadata_xml: ET.Element = None  # full metadata Element instance
        self.axes: str = ''  # axes names in order
        self.dims: dict = {}  # dims[f"{axis name}] = axis dimension, dict:str->int
        self.scales: dict = {}  # scales[f"{axis name}"] = physical scale, dict:str->float
        self.channel_names: list[str] = []  # list of channel names in order
        self.DAPI: int = -1
        self.FOCI: list[int] = []
        self.channel_dict: dict[int, str] = {}  #
        self.DM: DataMatrix = None  # DataMatrix object
        self.read_czi(path)

    def read_czi(self, path: Path):
        """
        Load pixel data and metadata from a CZI file into this instance.

        Reads the raw image array (squeezing unused axes), parses XML metadata,
        infers axes ordering, extracts channel names and physical scales, and
        splits data into scenes.

        Args:
            path (Path): Path to the .czi file.
        """

        with CziFile(path) as czi_file:
            # set file name
            self.f_name = os.path.basename(path).replace('.czi', '')

            # parse XML metadata
            self.metadata_xml = ET.fromstring(czi_file.metadata())


            # load image data, exclude last axis '0'
            data=czi_file.asarray()
            data = np.squeeze(data, axis=-1)

            # ensure scene axis 'S'
            self.axes = str(czi_file.axes)[:-1]

            if 'S' not in self.axes:
                self.axes = 'S' + self.axes
                data = np.expand_dims(data, axis=0)

            # record dims and scales
            for i in range(len(self.axes)):
                self.dims[self.axes[i]] = data.shape[i]
                scale_element = self.metadata_xml.find(
                    f'.//Metadata/Scaling/Items/Distance[@Id="{self.axes[i]}"]/Value')
                if scale_element is not None:
                    self.scales[self.axes[i]] = float(scale_element.text)

            # channel names from metadata
            if "C" in self.axes:
                channel_elements = self.metadata_xml.findall(
                    ".//Metadata/Information/Image/Dimensions/Channels/Channel")
                if len(channel_elements) == self.dims["C"]:
                    self.channel_names = [c.attrib.get("Name", c.attrib["Id"]) for c in channel_elements]

            # get acquisition time
            self.acquisition_time = self.metadata_xml.find(".//Metadata/Information/Image/AcquisitionDateAndTime").text

            # get scenes bb and DM
            self.extract_scenes(data, czi_file.filtered_subblock_directory)

            # make axes SZYXC to correspond with self. DM
            self.axes = 'SZYXC'

    def extract_scenes(self, data: np.ndarray, fsd: list[DirectoryEntryDV]):
        """
        Partition raw array into per-scene subvolumes and build DataMatrix.

        Determines XY bounding boxes for each scene from the filtered subblock
        directory entries, crops the raw stack accordingly, maps channels to RGB
        positions, and constructs the DataMatrix object.

        Args:
            data (ndarray): Raw image array with axes SCZYX.
            fsd (list[DirectoryEntryDV]): Directory entries describing subblocks.
        """
        if self.dims['S'] == 1:
            scenes_bb = [{'X': (0, self.dims['X']), 'Y': (0, self.dims['Y'])}]
        else:
            scenes_bb = []
            S = 0
            for frame in fsd:
                # dims SCZYX0
                axes = frame.axes
                s, c, z, x, y = axes.index('S'), axes.index('C'), axes.index('Z'), axes.index('X'), axes.index('Y')
                start, shape = frame.start, frame.shape
                # each scene has the same bb in YX (invariable to C and Z axes) -> take the first scene bb accessible
                if start[s] == S and start[c] == start[z] == 0:
                    scenes_bb.append({'X': (start[x], shape[x]), 'Y': (start[y], shape[y])})
                    S += 1

        # crop and reformat each scene
        data_scenes = []
        for s in range(self.dims['S']):
            x_start, y_start = scenes_bb[s]['X'][0], scenes_bb[s]['Y'][0]
            x_dim, y_dim = scenes_bb[s]['X'][1], scenes_bb[s]['Y'][1]
            # change dimensions to fit current matrix reshaping (all scenes has the same resolution)
            if s == 0: self.dims['X'], self.dims['Y'] = x_dim, y_dim
            x_end = x_start + x_dim
            y_end = y_start + y_dim

            # original DM: axes CZYX -> transpose to ZYXC, channels: 'TexRe', 'AF488', 'DAPI' -> R,G,B
            self.DAPI = 2
            # cut out scenes as matrix
            if self.dims['C'] == 2:
                self.FOCI = [1]
                self.channel_dict = {0: '', 1: self.channel_names[0], 2: self.channel_names[1]}
                # 2chan matrix to 3chan image matrix
                image = np.zeros((self.dims['Z'], y_dim, x_dim, 3), dtype=np.uint8)
                image[:, :, :, 1:] = np.transpose(data[s, :, :, y_start:y_end, x_start:x_end], (1, 2, 3, 0)).astype(
                    np.uint8)
            else:
                self.FOCI = [0, 1]
                self.channel_dict = {i: self.channel_names[i] for i in range(self.dims['C'])}
                image = np.transpose(data[s, :, :, y_start:y_end, x_start:x_end], (1, 2, 3, 0)).astype(np.uint8)
            data_scenes.append(image)

        # make image matrix with shape (S, Z, Y, X, C)
        self.DM = DataMatrix(np.stack(data_scenes, axis=0), self.dims)

    def __repr__(self):
        """
        Return summary string of CZI object.
        Includes filename, dimensions, channel names, scales, and acquisition time.
        """
        txt = (f'CZI(f_name: {self.f_name + '.czi'},'
               f'\n\t dimensions: ({', '.join([f'{ax}: {self.dims[ax]}' for ax in self.axes])}),'
               f'\n\t channel_names: ({', '.join(self.channel_names)}),'
               f'\n\t scales: ({', '.join([f'{ax}: {self.scales[ax]} {SCALING[ax]}' for ax in self.scales.keys()])}),'
               f'\n\t acquisition_time: {self.acquisition_time})')
        return txt

    def str_metadata(self):
        """
        Return the raw XML metadata as a UTF-8 string.

        Returns:
            str: XML metadata.
        """
        return ET.tostring(self.metadata_xml, encoding='utf-8').decode('utf-8')

    def plot_data(self):
        """
        Display each Z-plane of every scene sequentially.

        Opens a matplotlib figure for each (scene, z) slice; blocks until a key
        is pressed, then closes the figure.
        """
        i = 1
        for s in range(self.dims['S']):
            for z in range(self.dims['Z']):
                fig = plt.figure(i)
                i += 1
                ax = fig.add_subplot(1, 1, 1)
                ax.imshow(self.DM.data[s, z, :, :, :])
                ax.set_title(
                    f'channels: {self.channel_names if len(self.channel_names) == 3 else ['_'].extend(self.channel_names)} to RGB')
                fig.suptitle(f'scene: {s}, Z:{z}')
                fig.show()
                fig.waitforbuttonpress()
                plt.close(fig)

    def plot_max_data_z(self):
        """
        Display maximum-intensity projection across Z for each scene.

        For each scene, computes the pixel-wise maximum over Z slices
        and displays as an RGB image.
        """
        i = 1
        for s in range(self.dims['S']):
            fig = plt.figure(i)
            i += 1
            ax = fig.add_subplot(1, 1, 1)
            im = np.max(self.DM.data[s, :, :, :, :], axis=0)
            ax.imshow(im)
            ax.set_title(
                f'channels: {self.channel_names if len(self.channel_names) == 3 else ['_'].extend(self.channel_names)} to RGB')
            fig.suptitle(f'scene: {s}, max in Z')
            fig.show()
            fig.waitforbuttonpress()
            plt.close(fig)

    def save_data_z_layers(self, dir_path: Path):
        """
        Save each Z-plane of every scene as PNG files in a structured directory.

        Creates a folder named after the CZI file under `dir_path`, then a
        `z_layers` subfolder, then per-scene subdirectories `sXXX`. Within each,
        saves images named `{filename}_sXXX_zYYY.png`.

        Args:
            dir_path (Path): Directory in which to create output folders.
        """
        name = self.f_name

        path1 = os.path.join(dir_path, name)
        if not os.path.exists(path1): os.mkdir(path1)

        path2 = os.path.join(path1, 'z_layers')
        if not os.path.exists(path2): os.mkdir(path2)

        for s in range(self.dims['S']):

            path_s = os.path.join(path2, 's' + str(s).zfill(3))
            if not os.path.exists(path_s): os.mkdir(path_s)

            for z in range(self.dims['Z']):
                plt.imsave(os.path.join(path_s, name + '_s' + str(s).zfill(3) + '_z' + str(z).zfill(3) + '.png'),
                           np.flipud(self.DM.data[s, z, :, :, :].astype(np.uint8)))
            plt.imsave(os.path.join(path_s, name + '_s' + str(s).zfill(3) + '_z_max.png'),
                       np.flipud(np.max(self.DM.data[s, :, :, :, :], axis=0).astype(np.uint8)))

    def get_dict(self) -> dict:
        """
        Serialize the CZI state into a dictionary.

        Returns:
            dict: {
                'name': filename,
                'data': ndarray[int] (S,Z,Y,X,C),
                'channel_dict': dict[int,str],
                'axes': str,
                'acquisition_time': str,
                'scales': dict[str,float]
            }
        """
        return {
            'name': self.f_name,
            'data': self.DM.data,
            'channel_dict': self.channel_dict,
            'axes': self.axes,
            'acquisition_time': self.acquisition_time,
            'scales': self.scales
        }

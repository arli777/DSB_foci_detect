"""
detect.py

High-level detection pipeline combining CZI file reading, cell segmentation, and foci extraction.
Defines the Detect class to orchestrate loading microscopy data, identifying cells in each scene,
and segmenting foci within each cell per channel.

Classes:
    Detect: Manages the end-to-end workflow from file input to per-cell foci segmentation.
"""

import numpy as np
from read_czi import CZI
from segments import Segments
from constants import BG_L, N, S1, S2
import os
from scipy.io import savemat
import scipy.ndimage as spi
from drawing import get_path, save_draw_cells, save_draw_cells_foci_counts, generate_n_colors
import time

WRITE_TIME = True


class Detect:
    """
    Orchestrates cell and foci detection on Zeiss CZI microscopy datasets.

    Attributes:
        czi (CZI): Loaded CZI file object containing image data and metadata.
        cell_segments_list (list[Segments]): Per-scene Segments instances holding detected cells.
        detection_complete (bool): Whether detection is complete.
        cells_extracted (bool): Whether cells extracted.
    """

    def __init__(self, path):
        """
        Initialize the detection pipeline with a .czi file path.

        Args:
            path (Path or str): Filesystem path to the CZI file to process.
        """
        self.czi = CZI(path)

        print(self.czi)

        self.cell_segments_list = []  # index is according to scene
        self.detection_complete = False
        self.cells_extracted = False

    def extract_cells(self):
        """
        Segment cell nuclei in each scene using the DAPI channel maximum projection.

        For each scene index `s`, computes a 2D maximum-intensity projection over Z for the DAPI channel,
        deletes DC offset, and initializes a Segments instance to detect cells. Stores results in `cell_segments_list`.
        """
        if self.cells_extracted: return
        t_all = time.time()
        for s in range(self.czi.dims['S']):
            t_start = time.time()

            DAPI_image = np.max(self.czi.DM.data[s, :, :, :, self.czi.DAPI], axis=0)
            DAPI_image = DAPI_image - spi.gaussian_filter(spi.minimum_filter(DAPI_image, size=S1), sigma = S2)

            self.cell_segments_list.append(Segments(DAPI_image))

            if WRITE_TIME:
                print(f"\tCell segmentation on scene {s} took {time.time() - t_start:.2f} seconds.")

        if WRITE_TIME:
            print(f"Cell segmentation took {time.time() - t_all:.2f} seconds.")

        self.cells_extracted = True

    def get_canvas(self, s: int, c: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Retrieve raw and Laplacian image stacks for a given scene and channel.

        Args:
            s (int): Scene index.
            c (int): Channel index.
        Returns:
            tuple of two ndarray[Z,H,W]: (raw image stack, Laplacian stack).
        """
        return self.czi.DM.data[s, :, :, :, c], self.czi.DM.data_lap[s, :, :, :, c]

    def extract_foci_s_c(self, s: int, c: int):
        """
        Segment foci within all cells of a specific scene and channel.

        Args:
            s (int): Scene index.
            c (int): Channel index to process.
        """
        if not self.cells_extracted:
            raise Exception('No cells extracted.')

        t_start = time.time()

        img, lap = self.get_canvas(s, c)
        for cf_seg in self.cell_segments_list[s].cells:
            cf_seg.foci_segment(c, img, lap)

        if WRITE_TIME:
            print(
                f"\t\tFoci segmentation on scene {s}, channel {self.czi.channel_dict[c]} took {time.time() - t_start:.2f} seconds.")

    def extract_foci(self, s_list: list[int], c_list: list[int]):
        """
        Run foci segmentation only on selected scenes and channels.

        Args:
            s_list (list[int]): Scene indices.
            c_list (list[int]): Channel indices.
        """
        for s in s_list:
            t_start = time.time()
            for c in c_list:
                self.extract_foci_s_c(s, c)
            if WRITE_TIME:
                print(f"\tFoci segmentation on scene {s} took {time.time() - t_start:.2f} seconds.")

    def detect_s_c(self, s_list: list[int], c_list: list[int]):
        """
        Execute detection workflow only on selected scenes and channels.

        Args:
            s_list (list[int]): Scene indices:
            c_list (list[int]): Channel indices:
        """
        self.extract_cells()
        self.extract_foci(s_list, c_list)

    def detect(self):
        """
        Execute the full detection workflow: cells then foci.
        """
        t_start = time.time()
        self.detect_s_c(list(range(self.czi.dims['S'])), self.czi.FOCI)
        self.detection_complete = True
        if WRITE_TIME:
            print(f"Detection took {time.time() - t_start:.2f} seconds.")

    def save_draw(self, dir_path):
        """
        Saves cell segmentation and foci detection visualization images from Detect object.

        This function processes each scene in the Detect object.
         It generates and saves images of:
          - cell segmentations
          - foci per Z-slice
          - merged foci
          - foci count per cell

        Images are saved under a structured directory hierarchy:
            {dir_path}/{czi_file_name}/draw/sXXX/

        Args:
            dir_path (str): Root directory path where the output folders and images will be saved.

        Notes:
            - The maximum projection across Z is used for cell and merged foci visualization.
            - Output file names and folder structure are auto-generated based on the scene index and channel.
        """
        if not self.detection_complete:
            raise RuntimeError('Detection pipeline not complete.')

        name = self.czi.f_name

        path1 = get_path(os.path.join(dir_path, name))
        path2 = get_path(os.path.join(path1, 'draw'))

        for s in range(self.czi.dims['S']):
            path_s = get_path(os.path.join(path2, 's' + str(s).zfill(3)))
            seg_dict = self.cell_segments_list[s].get_dict(z=self.czi.dims['Z'], channels=self.czi.FOCI)

            # draw cell
            image = np.max(self.czi.DM.data[s, :, :, :, self.czi.DAPI], axis=0)
            save_draw_cells(image, seg_dict['cells_labels'], 1,
                            os.path.join(path_s, name + '_s' + str(s).zfill(3) + '_cells.png'))

            for c in self.czi.FOCI:
                chan_name = self.czi.channel_dict[c]
                labels_z = seg_dict['channels'][c]['foci_z_labels']
                image_c_z = self.czi.DM.data[s, :, :, :, c]

                max_label = np.max(labels_z) - BG_L
                colors = generate_n_colors(max_label)

                # draw foci on each z layer
                for z in range(self.czi.dims['Z']):
                    save_draw_cells(image_c_z[z], labels_z[z], N, os.path.join(path_s, name + '_s' + str(s).zfill(
                        3) + '_' + chan_name + '_foci_z' + str(z).zfill(3) + '.png'), colors)

                # draw merged foci
                save_draw_cells(np.max(image_c_z, axis=0), seg_dict['channels'][c]['foci_merged_labels'], N,
                                os.path.join(path_s,
                                             name + '_s' + str(s).zfill(3) + '_' + chan_name + '_foci_merged.png'))

                # draw foci_counts
                save_draw_cells_foci_counts(np.max(image_c_z, axis=0), seg_dict['cells_labels'],
                                            seg_dict['channels'][c]['foci_merged_n'],
                                            os.path.join(path_s, name + '_s' + str(s).zfill(
                                                3) + '_' + chan_name + '_foci_merged_n.png'))

    def save_mat(self, dir_path):
        """
        Saves cell segmentation and foci detection in mat files.

        This function processes each scene in the Detect object.
         It saves .mat files:
          - {dir_path}/mat/{f_name}_sXX_cells.mat: cell labels under 'cells'
          - {dir_path}/mat/{f_name}_sXX_cXX.mat:
                z-stack foci labels under 'foci_z_labels'
                merged foci labels under 'foci_merged_labels'

        Args:
            dir_path (str): Root directory path where the output folders and images will be saved.
        """
        if not self.detection_complete:
            raise RuntimeError('Detection pipeline not complete.')

        name = self.czi.f_name

        path1 = get_path(os.path.join(dir_path, 'mat'))

        for s in range(self.czi.dims['S']):
            f_name = name + '_s' + str(s).zfill(2)

            seg_dict = self.cell_segments_list[s].get_dict(z=self.czi.dims['Z'], channels=self.czi.FOCI)

            savemat(os.path.join(path1, f_name + '_cells.mat'), {'cells': seg_dict['cells_labels']})

            for c in self.czi.FOCI:
                savemat(os.path.join(path1, f_name + '_ch' + str(c).zfill(2) + '.mat'),
                        {'z_labels': seg_dict['channels'][c]['foci_z_labels'],
                         'merged_labels': seg_dict['channels'][c]['foci_merged_labels']})

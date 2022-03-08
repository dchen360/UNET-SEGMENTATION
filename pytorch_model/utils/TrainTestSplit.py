#!/usr/bin/env python3
# coding: utf-8

"""
Classes and helper functions for reading and splitting data in NiFTI files.
"""

# pylint: disable=no-name-in-module
# pylint erroneously fails to recognize function from_numpy in torch.

import sys
# from typing import Tuple
import torch
import os
from torch import from_numpy, stack
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

def train_test_split(dataset):
    """
    Split a ImageDataset into two groups (training and testing) based on
    information specified within the metadata.csv.
    (dataset: "ImageDataset") -> Tuple["ImageDataset", "ImageDataset"]:
    Parameters
    ----------
    dataset: the ImageDataset to split.
    Returns
    -------
    A tuple containing two NpyDataset with training and testing data,
    respectively.
    """

    root = dataset.root
    metadata = dataset.metadata
    train_mask = metadata['is_Train']
    # select the images that are used for training
    dataset_mask = metadata['Dataset_Label'] == "CT_20-03-24"
    # select only Dataset_Label = CT_20-03-24 to shorten training time
    transform = dataset.transform
    slice_cols = dataset.slice_cols

    return (
        ImageDataset(metadata.loc[(train_mask & dataset_mask), :], root, slice_cols, transform),
        ImageDataset(metadata.loc[((~train_mask) & dataset_mask), :], root, slice_cols, transform)
    )


class ImageDataset(Dataset):
    """
    Dataset responsible for reading in a metadata CSV or DataFrame specifying
    file information for a given hierarchy of data.
    Parameters
    ----------
    metadata:     CSV or DataFrame containing the metadata information.
    root:         Base filepath used by the relative filepaths contained
                  within the metadata file.
    transform:    Pytorch transforms dynamically applied to the loaded
                  images within the Dataset. Optional.
    slice_cols:   The names of the columns within the metadata file to use
                  for slice images, organized in a list. Slices will be
                  stacked by the order of their appearance in the list.
                  A single string may be used in place of a list.
    Metadata Columns
    -------
    CT_image_path:  Relative path to the CT image from the parent path, parent path
                    being the directory 'Spring2022' folder is at.
    lung_mask_path: Relative path to the lung_mask image from the parent path, parent path
                    being the directory 'Spring2022' folder is at.
    covid_infection_mask_path: Relative path to the lung_mask image from the parent path, parent path
                               being the directory 'Spring2022' folder is at.
    Dataset_Label:  Name of the folder containing images with the given label.
    is_Train:       Indicating if the image is used as training or testing dataset.

    Output
    ------
    Data are output from the DataLoader as a Python dictionary with the following
    key-value pairs.
    image:        A key containing a single Tensor containing all of the images
                  loaded in the current batch.
    label:        A Tensor containing all of the labels for the current batch
                  of images.
    Corresponding image information is located at the same relative position
    within each value of every key-value pair. In other words, the image data
    and label for an image at index X are at index X within both the image
    and label Tensor.
    Created by Peter Lais on 09/21/2021.
    Revised by Danni Chen on 02/03/2022.
    """

    def __init__(self, metadata, root, slice_cols, transform=None, verbose=False):
        # Check if root exists.
        if not os.path.isdir(root):
            sys.exit('ImageDataset: Root does not exist.')

        # Load metadata (path_to_csv or dataframe).
        if isinstance(metadata, pd.core.frame.DataFrame):
            metadata_df = metadata.copy()
        else:
            metadata_df = pd.read_csv(metadata)

        # Optional transforms on top of tensor-ization.
        self.transform = transform
        # Metadata attribute
        self.metadata = metadata_df
        # Image directory
        self.root = root
        # Verbosity setting
        self.verbose = verbose
        # Col to use for slices.
        self.slice_cols = slice_cols if isinstance(slice_cols, list) else [slice_cols]

    def __len__(self):
        # Number of rows of metadata dataframe.
        return len(self.metadata)

    def __getitem__(self, idx):
        # Extract relevant information.
        # lung_mask_path would be the label.
        image_row = self.metadata.iloc[idx]
        if self.verbose:
            print(image_row)
        # label = image_row['Label']

        CT_images = []
        lung_masks = []

        for slice_col in self.slice_cols:
            CT_image_path = os.path.join(self.root, image_row['CT_image_path'])
            # Load image into numpy array and convert to Tensor.
            CT_images.append(from_numpy(np.load(CT_image_path)))

            lung_mask_path = os.path.join(self.root, image_row['lung_mask_path'])
            lung_masks.append(from_numpy(np.load(lung_mask_path)))

        CT_images = stack(CT_images, dim=0) if len(CT_images) != 1 else CT_images[0]
        lung_masks = stack(lung_masks, dim=0) if len(lung_masks) != 1 else lung_masks[0]

        # Custom transforms.
        if self.transform:
            CT_images = self.transform(CT_images)
        if self.transform:
            lung_masks = self.transform(lung_masks)

        # Return image and label.
        return {'image': CT_images, 'lung_mask': lung_masks}

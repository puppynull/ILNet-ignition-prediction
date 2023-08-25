import os
import numpy as np
import torch
import numpy as np
import gdal
import matplotlib.pyplot as plt
from torchvision.transforms import Resize
from network.segformer import SegFormer



def get_tif_data( path="", geo=False, fliter=False):

    dataset = gdal.Open(path)
    if dataset == None:
        print(path + " load file error!")

    if geo:
        geo_projection = dataset.GetProjection()
        geo_transform = dataset.GetGeoTransform()

    data_w = dataset.RasterXSize
    data_h = dataset.RasterYSize

    band = dataset.GetRasterBand(1)

    data_raster = band.ReadAsArray(0, 0, data_w, data_h)

    if fliter:
        index = data_raster > 1e5

        data_raster[index] = 0

    del dataset

    return data_raster.astype(np.float32)
def get_boundary_pos( grid):
    pred_boundary = np.zeros((1, 2))
    standard = np.zeros(grid.shape[0])

    for i in range(grid.shape[0]):
        if ((i != 0 and grid[i, :].any() != standard.any() and grid[i - 1, :].any() == standard.any()) or
                (i == grid.shape[0] - 1 and grid[i,:].any() != standard.any()) or(
                grid[i, :].any() != standard.any() and grid[i + 1, :].any() == standard.any()) or (i == 0 and grid[0,:].any() != standard.any())):

            index = np.where(grid[i, :] == 1)

            index = index[0].reshape((index[0].shape[0], 1))

            y = np.zeros((index.shape[0], 1))
            y = y + i

            index_new = np.append(y, index, axis=1)

            pred_boundary = np.append(pred_boundary, index_new, axis=0)

        elif (grid[i, :].any() != standard.any()):
            index = np.where(grid[i, :] == 1)[0]

            index_left = np.min(index).reshape((1, 1))
            index_right = np.max(index).reshape((1, 1))
            index_total = np.append(index_left, index_right, axis=0)
            y = np.zeros((2, 1))
            y = y + i
            index_total = np.append(y, index_total, axis=1)
            pred_boundary = np.append(pred_boundary, index_total, axis=0)

    pred_boundary = np.delete(pred_boundary, 0, axis=0)

    return pred_boundary

# import itertools
# import math
# from typing import List

# from tqdm import tqdm
# import nvidia.dali.fn as fn
# import nvidia.dali.types as types
# from nvidia.dali import pipeline_def
# from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
# import torch
# from torch import nn
# from torch.types import Device, Tensor
# from torchvision import models
# import numpy as np

# from util.filelist import write_filelist_to_disk

# # TODO: use os.tmpdir
# TMP_FILE_LIST_PATH = "./filelist.txt"
# FILE_READER_NAME = "file_reader"


# @pipeline_def
# def simple_pipeline(grid_size, tile_size, path_to_file_list):
#     jpegs, labels = fn.readers.file(
#         file_list=path_to_file_list, random_shuffle=False, name=FILE_READER_NAME
#     )
#     images = fn.decoders.image(jpegs, device="mixed")

#     grid_image_size = grid_size * tile_size
#     # scale to GRID_SIZE*TILE_SIZE
#     scaled_images = fn.resize(
#         images.gpu(),
#         # practically no speed difference between cubic antialiased and NN non-antialiased?
#         min_filter=types.DALIInterpType.INTERP_CUBIC,
#         antialias=True,
#         size=(grid_image_size, grid_image_size),
#     )

#     return scaled_images, labels
import os
from os import path
from multiprocessing import Pool
import cv2
import numpy as np

from osgeo import gdal
from tqdm import tqdm

from util.io_utils import find_all_files_with_extension_recursively

SOURCE_IMAGES = "/mnt/e/mml/orto"
DESTINATION_IMAGES = "/mnt/e/resized"


def basename_no_ext(p: str) -> str:
    return path.splitext(path.basename(p))[0]


def read_image(path: str):
    with gdal.Open(path) as ds:
        buf = ds.ReadAsArray(
            xoff=0,
            yoff=0,
            xsize=12000,
            ysize=12000,
            buf_xsize=244 * 12,
            buf_ysize=244 * 12,
        )
    buf = buf.transpose((1, 2, 0))  # (C, H, W) to (H, W, C)
    return cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)


def split_image_to_tiles(buf: np.ndarray, image_prefix: str, dest_path: str):
    for y in range(12):
        for x in range(12):
            y_offset = y * 244
            x_offset = x * 244
            tile = buf[y_offset : y_offset + 244, x_offset : x_offset + 244, :]

            output_file = f"{image_prefix}_y{y}_x{x}.jpg"
            cv2.imwrite(
                path.join(dest_path, output_file), tile, [cv2.IMWRITE_JPEG_QUALITY, 100]
            )


def process_image(source_path: str, dest_path: str):
    buf = read_image(source_path)

    os.makedirs(dest_path, exist_ok=True)
    split_image_to_tiles(buf, basename_no_ext(source_path), dest_path)


def main():
    # create output dir
    os.makedirs(DESTINATION_IMAGES, exist_ok=True)

    imagelist = list(find_all_files_with_extension_recursively(SOURCE_IMAGES, ".jp2"))
    imagelist = imagelist[0:1]

    tasks = [
        (fname, path.join(DESTINATION_IMAGES, basename_no_ext(fname)))
        for fname in imagelist
    ]

    # find common prefix so we can show the actually changing part of the path
    # in tqdm's desc
    common_prefix = path.commonprefix([x[0] for x in tasks])
    print(f"common prefix: {common_prefix}")

    # # use python multiprocessing istarmap to process images with process_image
    # with Pool(1) as pool:
    #     for _ in tqdm(pool.istarmap(process_image, tasks), total=len(tasks)):
    #         pass

    # same as above but with a single process for debugging:
    pbar = tqdm(total=len(tasks))
    for source_path, dest_path in tasks:
        # show only the changing part of the path in tqdm's desc
        pbar.set_description(source_path[len(common_prefix) :])
        process_image(source_path, dest_path)
        pbar.update(1)


if __name__ == "__main__":
    main()

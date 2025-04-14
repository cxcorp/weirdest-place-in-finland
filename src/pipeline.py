import itertools
import os
from typing import List, Generator

from tqdm import tqdm, trange
import torch
from torch import nn
from torch.types import Device, Tensor
from torchvision import models, tv_tensors
from torchvision.transforms import v2
import numpy as np
from PIL import Image
from multiprocessing import Pool
from osgeo import gdal


from util.io_utils import find_all_files_with_extension_recursively


FILE_READER_NAME = "file_reader"


def calculate_histograms(tiles: Tensor):
    B, C, _, _ = tiles.shape
    histograms = torch.zeros((B, C, 32), dtype=torch.uint8, device=tiles.device)

    for b in range(B):
        for c in range(C):
            histograms[b, c] = torch.histc(
                tiles[b, c].reshape(-1), bins=32, min=0, max=255
            )

    return histograms


def load_resnet(device: Device):
    # default pretrained model: https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html#torchvision.models.ResNet50_Weights
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.eval()

    embedding_extractor = nn.Sequential(
        # remove classification layer to just get the 2048 embedding
        *list(model.children())[:-1]
    )
    embedding_extractor.to(device)

    return embedding_extractor


# def visualize_tiles(tiles):
#     import matplotlib.pyplot as plt
#     from torchvision.utils import make_grid
#
#     grid_img = make_grid(tiles, nrow=12)
#     # back from (C, H, W) -> (H, W, C)
#     grid_img_np = grid_img.permute(1, 2, 0).cpu().numpy()
#     plt.figure(figsize=(12, 24))
#     plt.imshow(grid_img_np)
#     plt.axis("off")
#     plt.show()

transforms = v2.Compose(
    [
        # basically models.ResNet50_Weights.IMAGENET1K_V2.transforms
        # without the resize and crop
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=True
        ),
    ]
)


def read_images(paths: str):
    # tensor of shape (B, C, H, W, dtype=torch.uint8)
    print("Read images")
    output_tensor = torch.empty((len(paths), 3, 224, 224), dtype=torch.uint8)
    for i, path in enumerate(paths):
        image = Image.open(path)
        image = v2.ToImage()(image)
        output_tensor[i] = image
    return output_tensor


def read_image(path: str, device: Device = None):
    ds = gdal.Open(path)
    arr = ds.ReadAsArray(
        xoff=0,
        yoff=0,
        xsize=ds.RasterXSize,
        ysize=ds.RasterYSize,
        buf_xsize=ds.RasterXSize,
        buf_ysize=ds.RasterYSize,
        buf_type=gdal.GDT_Byte,
    )

    arr = torch.from_numpy(arr).contiguous()
    return tv_tensors.Image(arr, device=device)


def basename_no_ext(p: str) -> str:
    return os.path.splitext(os.path.basename(p))[0]


def run_pipeline(input_file_paths: List[str], batch_size_resnet: int = 12 * 12):
    resnet_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedding_extractor = load_resnet(resnet_device)

    # with Pool(8) as mp_pool:
    #     images_iterator = mp_pool.imap_unordered(
    #         read_image, input_file_paths, chunksize=8
    #     )
    #     images_iterator = itertools.batched(images_iterator, batch_size_resnet)

    images_iterator: Generator[tuple[tv_tensors.Image, str], None, None] = (
        (read_image(path, resnet_device), path) for path in input_file_paths
    )

    pbar = tqdm(total=len(input_file_paths))
    for image, path in images_iterator:

        pbar.set_description("Datamangel")
        # image is a tensor of the full 2688x2688 image in CHW format
        # send it to GPU and efficiently split to 12*12 tiles of size 224x224
        batch = image.unfold(1, 224, 224).unfold(2, 224, 224)  # (3, 12, 12, 224, 224)
        batch = batch.permute(1, 2, 0, 3, 4).reshape(
            -1, 3, 224, 224
        )  # (144, 3, 224, 224)

        batch_size = batch.shape[0]

        # calculate all embeddings for this batch
        with torch.no_grad():
            histograms = calculate_histograms(batch)
            # resnet presets transforms: to normalized floats
            batch = transforms(batch)
            embeddings = embedding_extractor(batch)

        histograms = histograms.cpu().numpy()
        # (BATCH_SIZE, 2048, 1, 1) -> (BATCH_SIZE, 2048)
        all_embeddings = np.squeeze(embeddings.cpu().numpy(), axis=(2, 3))
        grid_points = [({"x": i % 12, "y": i // 12}) for i in range(batch_size)]

        pbar.set_description("Database")
        yield path, list(zip(grid_points, all_embeddings, histograms))

        pbar.update(1)  # first value is batch size

    pbar.close()

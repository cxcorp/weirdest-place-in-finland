import itertools
import math
from typing import List

from tqdm import tqdm
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali import pipeline_def
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
import torch
from torch import nn
from torchvision import models
import numpy as np

# todo use os.tmpdir
TMP_FILE_LIST_PATH = "./filelist.txt"
FILE_READER_NAME = "file_reader"


def write_filelist_to_disk(files: List[str], filelist_path: str):
    # set label to be same as index on this list
    files_with_label = [[file, str(i)] for i, file in enumerate(files)]
    contents = "\n".join([" ".join(pair) for pair in files_with_label])

    with open(filelist_path, "w", encoding="utf-8") as fp:
        fp.write(contents)


@pipeline_def
def simple_pipeline(grid_size, tile_size, path_to_file_list):
    jpegs, labels = fn.readers.file(
        file_list=path_to_file_list, random_shuffle=False, name=FILE_READER_NAME
    )
    images = fn.decoders.image(jpegs, device="mixed")

    grid_image_size = grid_size * tile_size
    # scale to GRID_SIZE*TILE_SIZE
    scaled_images = fn.resize(
        images.gpu(),
        min_filter=types.DALIInterpType.INTERP_NN,
        antialias=False,
        size=(grid_image_size, grid_image_size),
    )

    return scaled_images, labels


def tile_image(image_batch, grid_size, tile_size):
    # This function is full-on ChatGPT, sorry.

    pbar = tqdm(total=4, desc="Tiling image")

    # image_batch: torch tensor of shape (B, H, W, C) from DALI pipeline
    B, _, _, C = image_batch.shape
    # Permute to (B, C, H, W) for resnet (it wants CHW)
    images = image_batch.permute(0, 3, 1, 2)
    pbar.update(1)

    # Reshape: first, split H and W into grid dimensions
    # images.shape -> (B, C, GRID_SIZE, TILE_SIZE, GRID_SIZE, TILE_SIZE)
    images = images.reshape(B, C, grid_size, tile_size, grid_size, tile_size)
    pbar.update(1)

    # Permute to gather the grid dimensions next to each other: (B, GRID_SIZE, GRID_SIZE, C, TILE_SIZE, TILE_SIZE)
    images = images.permute(0, 2, 4, 1, 3, 5)
    pbar.update(1)

    # Flatten the first three dimensions so that each tile becomes an individual image:
    # new shape: (B * GRID_SIZE * GRID_SIZE, C, TILE_SIZE, TILE_SIZE)
    tiles = images.reshape(B * grid_size * grid_size, C, tile_size, tile_size)
    pbar.update(1)
    pbar.close()

    return tiles


def load_resnet(device):
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


def run_pipeline(
    input_file_paths: List[str],
    batch_size_resize: int = 1,
    batch_size_resnet: int = 16,
    grid_size: int = 12,
    tile_size: int = 244,
    dali_num_threads: int = 8,
    dali_device_id: int = 0,
):
    resnet_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedding_extractor = load_resnet(resnet_device)

    write_filelist_to_disk(input_file_paths, TMP_FILE_LIST_PATH)

    pipe = simple_pipeline(
        grid_size,
        tile_size,
        TMP_FILE_LIST_PATH,
        batch_size=batch_size_resize,
        num_threads=dali_num_threads,
        device_id=dali_device_id,
    )
    dali_iter = DALIGenericIterator(
        pipe,
        ["scaled_images", "labels"],
        reader_name=FILE_READER_NAME,
        auto_reset=False,
        last_batch_policy=LastBatchPolicy.DROP,
    )

    total_tiles_per_image = grid_size * grid_size

    mean = None
    std = None
    pbar = tqdm(
        # LastBatchPolicy.DROP -> if there are leftover images that don't fill
        # the batch, they're ignored
        total=len(input_file_paths) - (len(input_file_paths) % batch_size_resize),
        desc="Reading and resizing images",
    )
    for batch in dali_iter:
        # shape: (B, H, W, C) on the GPU or CPU depending on pipeline
        images = batch[0]["scaled_images"]
        labels = batch[0]["labels"].flatten()

        # Update progress bar
        pbar.update(images.shape[0])  # first value is batch size
        tqdm.write(f"Shape: {repr(images.shape)}, labels: {repr(labels.tolist())}")

        # (B, C, H, W)
        tiles = tile_image(images, grid_size=grid_size, tile_size=tile_size)
        # tiles now has each image from `images` tiled so that e.g.
        # tiles[0:144]   -> labels[0]
        # tiles[144:288] -> labels[1]
        # etc.
        # and labels[0] is the index of the input file in input_file_paths
        # so to get the filename of the source file for tiles[144:288]
        # you get input_file_paths[labels[1]]

        # (0-255) uint8 to float 0-1

        if mean is None or std is None:
            # standard imagenet normalization coefficients
            mean = torch.tensor([0.485, 0.456, 0.406], device=tiles.device).view(
                1, -1, 1, 1
            )
            std = torch.tensor([0.229, 0.224, 0.225], device=tiles.device).view(
                1, -1, 1, 1
            )

        # calculate all embeddings for this batch
        tqdm.write("Calculating embeddings")
        all_embeddings = []
        for idx_batch in itertools.batched(
            tqdm(range(tiles.shape[0]), desc="Calculating embeddings", leave=False),
            batch_size_resnet,
        ):
            tile_batch = tiles[list(idx_batch)]  # (16, C, H, W)

            # normalize according to "standard ImageNet normalization"
            tile_batch = tile_batch.float().div(255.0)
            tile_batch -= mean
            tile_batch /= std

            with torch.no_grad():
                embeddings = embedding_extractor(tile_batch)

            numpy_embeddings = embeddings.cpu().numpy()
            # (BATCH_SIZE, 2048, 1, 1) -> (BATCH_SIZE, 2048)
            numpy_embeddings = np.squeeze(numpy_embeddings, axis=(2, 3))
            all_embeddings.append(numpy_embeddings)

        all_embeddings = np.concat(all_embeddings)

        # recover image labels
        for i, embedding in enumerate(all_embeddings):
            label_idx_idx = math.floor(i / total_tiles_per_image)
            label_idx = labels[label_idx_idx]
            label = input_file_paths[label_idx]

            stride = i % total_tiles_per_image
            tile_y = math.floor(stride / grid_size)
            tile_x = stride - tile_y * grid_size

            yield label, embedding, {"y": tile_y, "x": tile_x}

    pbar.close()

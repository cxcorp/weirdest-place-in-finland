from collections.abc import Iterable
import os
import time
import random

import numpy as np
import matplotlib.pyplot as plt
import cv2
import numpy as np
from osgeo import gdal
import sys
import pyarrow.dataset as ds
from tqdm import tqdm, trange
import torch
import torchvision.models
import pandas as pd

from util.results_helpers import (
    dump_run_results_to_parquet,
    read_run_results_from_parquet,
    connect_to_db,
)


# Image.MAX_IMAGE_PIXELS = 12000 * 12000  # disable decompression bomb warning

BATCH_SIZE_RESIZE = 4
# batch entire image at a time, we seem to have the VRAM for it
BATCH_SIZE_RESNET = 144

GRID_SIZE = 12
# resnet50 size
TILE_SIZE = 224

EMBEDDING_SIZE = 512

PARQUETS_PATH = "./parquets/maxvit"
RESULTS_PATH = "./results/maxvit-autoencoder"


def most_usual_and_usual_according_to_euclidean_dist(all_embeddings):
    mean_embedding = np.mean(all_embeddings, axis=0)
    distances = np.linalg.norm(all_embeddings - mean_embedding, axis=1)

    most_usual_index = np.argmin(distances)
    most_unusual_index = np.argmax(distances)
    return most_usual_index, most_unusual_index


def most_usual_and_usual_according_to_Mahalanobis_dist(all_embeddings):
    mean_embedding = np.mean(all_embeddings, axis=0)
    covariance = np.cov(all_embeddings, rowvar=False)

    epsilon = 1e-6
    covariance += np.eye(covariance.shape[0]) * epsilon

    inv_covariance = np.linalg.inv(covariance)
    diff = all_embeddings - mean_embedding
    mahal_distances = np.sqrt(np.sum(diff @ inv_covariance * diff, axis=1))

    most_usual_index = np.argmin(mahal_distances)
    most_unusual_index = np.argmax(mahal_distances)

    return most_usual_index, most_unusual_index


def normalize_histogram(h: list):
    h = np.array(h).astype(np.float32)
    total = h.sum()
    if total == 0:
        return h
    return h / total


def chi_square_distance(h1, h2, eps=1e-10):
    return np.sum(((h1 - h2) ** 2) / (h1 + h2 + eps))


def get_multiple_rois_from_image(
    file_path: str, grid_size: int, rois: Iterable[tuple[int, int]]
):
    with gdal.Open(file_path) as d:
        assert d.RasterXSize == d.RasterYSize
        assert d.RasterXSize == 12 * TILE_SIZE
        arr = d.ReadAsArray(
            xoff=0,
            yoff=0,
            xsize=d.RasterXSize,
            ysize=d.RasterYSize,
            buf_xsize=d.RasterXSize,
            buf_ysize=d.RasterYSize,
            buf_type=gdal.GDT_Byte,
        )
    # (C, H, W) -> (H, W, C)
    im = arr.transpose(1, 2, 0)

    height, width, channels = im.shape
    assert (height, width) == (12 * TILE_SIZE, 12 * TILE_SIZE) and channels == 3
    assert (height % grid_size) == 0

    px_grid_size = height // grid_size

    result = dict()
    for grid_x, grid_y in rois:
        start_x = int(grid_x) * px_grid_size
        start_y = int(grid_y) * px_grid_size

        roi = im[
            start_y : start_y + px_grid_size,
            start_x : start_x + px_grid_size,
        ]

        result[(grid_x, grid_y)] = roi

    return result


def get_image_roi(file_path: str, grid_x: str, grid_y: str, grid_size: int):
    im = cv2.imread(file_path)
    cv2.cvtColor(im, cv2.COLOR_BGR2RGB, dst=im)

    height, width, channels = im.shape
    assert (height, width) == (12 * TILE_SIZE, 12 * TILE_SIZE) and channels == 3

    assert (height % grid_size) == 0
    effective_grid_size = height // grid_size

    start_x = effective_grid_size * grid_x
    start_y = effective_grid_size * grid_y

    return im[
        start_y : start_y + effective_grid_size, start_x : start_x + effective_grid_size
    ]


def get_image(file_path: str):
    im = cv2.imread(file_path)
    cv2.cvtColor(im, cv2.COLOR_BGR2RGB, dst=im)

    height, width, channels = im.shape
    assert (height, width) == (TILE_SIZE, TILE_SIZE) and channels == 3

    return im


def dump_to_parquet(run_id: int):
    print("connect to db")
    conn = connect_to_db()
    try:
        dump_run_results_to_parquet(conn, run_id, f"./run_{run_id}.parquet")
    finally:
        conn.close()


def read_from_parquet(run_id: int):
    return read_run_results_from_parquet(f"./run_{run_id}.parquet")


def read_from_parquet_dataset(parquets_path: str):
    dataset = ds.dataset(parquets_path, format="parquet")
    rowcount: int = dataset.count_rows()

    file_paths: list[str] = list()
    histograms = np.empty((rowcount, 96), dtype=np.uint8)
    embeddings = np.empty((rowcount, EMBEDDING_SIZE), dtype=np.float32)
    grid_x = np.empty((rowcount,), dtype=np.uint8)
    grid_y = np.empty((rowcount,), dtype=np.uint8)

    i = 0
    pbar = tqdm(total=rowcount, desc="Reading parquets")
    for batch in dataset.to_batches():
        file_paths.extend(batch["file_path"].to_pylist())
        start_idx = i
        end_idx = i + batch.num_rows
        histograms[start_idx:end_idx] = np.stack(
            batch["histogram"].to_numpy(zero_copy_only=False)
        )
        embeddings[start_idx:end_idx] = np.stack(
            batch["embedding"].to_numpy(zero_copy_only=False)
        )
        grid_x[start_idx:end_idx] = np.stack(
            batch["grid_x"].to_numpy(zero_copy_only=False)
        )
        grid_y[start_idx:end_idx] = np.stack(
            batch["grid_y"].to_numpy(zero_copy_only=False)
        )

        i += batch.num_rows
        pbar.update(batch.num_rows)

    pbar.close()

    gridpoints = np.stack((grid_x, grid_y), axis=1)

    print(f"Read {histograms.shape[0]} rows from parquet")
    return file_paths, histograms, embeddings, gridpoints


def main2():
    print("read")
    start_time = time.perf_counter()
    file_paths, histograms, embeddings, gridpoints = read_from_parquet_dataset(
        PARQUETS_PATH
    )
    end_time = time.perf_counter()
    print(f"read took {end_time - start_time:.2f} seconds")

    print("filter away empty images")
    valid_indices = [
        i for i, h in enumerate(histograms) if not np.all(np.array(h) == 0)
    ]
    file_paths = [file_paths[i] for i in valid_indices]
    histograms = [histograms[i] for i in valid_indices]
    embeddings = np.array([embeddings[i] for i in valid_indices], dtype=np.float32)
    gridpoints = [gridpoints[i] for i in valid_indices]

    # print("normalize histograms")
    histograms = [normalize_histogram(h) for h in histograms]
    histograms = np.array(histograms)

    # embeddings and histograms are now both (ROW_COUNT, n) - add the histograms
    # to the end of the embeddings so we have (ROW_COUNT, n + 96)
    # embeddings = np.concatenate((embeddings, histograms), axis=1)

    # print("fit PCA")
    # X_reduced = PCA(n_components=1024).fit_transform(embeddings)
    # print(X_reduced.shape)

    plot_ääripäät(embeddings, histograms, file_paths, gridpoints)

    # print("import umap")
    # import umap
    # import umap.plot
    # print("umap fitting")
    # mapper = umap.UMAP(metric="cosine", n_neighbors=100).fit(embeddings)

    # umap.plot.points(mapper)
    # plt.show()


def calculate_variables_for_mahal(vector):
    mean = np.mean(vector, axis=0)

    covariance = np.cov(vector, rowvar=False)
    epsilon = 1e-6
    covariance += np.eye(covariance.shape[0]) * epsilon
    inv_covariance = np.linalg.inv(covariance)
    diff = vector - mean
    return inv_covariance, diff


def plot_ääripäät(embeddings, histograms, paths, gridpoints):
    # print("norm")
    # embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # calculate Mahalanobis distances
    # embeddings = np.concatenate((embeddings, histograms), axis=1)
    import sklearn.datasets
    import sklearn.neighbors

    embeddings = np.concatenate((embeddings, histograms), axis=1)

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader

    class VectorDataset(Dataset):
        def __init__(self, vectors):
            self.data = torch.from_numpy(vectors.astype(np.float32))

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            # Autoencoder: the target is the input itself.
            return self.data[idx]

    batch_size = 256
    dataset = VectorDataset(embeddings)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    class Autoencoder(nn.Module):
        def __init__(self, input_dim=608, latent_dim=64):
            """
            input_dim : dimensionality of input data (608)
            latent_dim: dimensionality of latent space
            """
            super(Autoencoder, self).__init__()
            # Define the encoder network
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Linear(256, latent_dim),
            )
            # Define the decoder network
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 256),
                nn.ReLU(),
                nn.Linear(256, input_dim),
                # Optionally: if your data is normalized, you might use a Sigmoid()
                # activation to constrain outputs between 0 and 1.
            )

        def forward(self, x):
            latent = self.encoder(x)
            recon = self.decoder(latent)
            return recon

    latent_dim = 64  # or another value that makes sense
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Autoencoder(input_dim=608, latent_dim=latent_dim)
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 50  # you can adjust the number of epochs
    pbar = trange(num_epochs)
    for epoch in pbar:
        epoch_loss = 0.0
        model.train()

        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()

            # Forward pass: get reconstruction
            recon = model(batch)

            # Compute reconstruction loss
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch.size(0)

        epoch_loss /= len(dataset)
        pbar.set_description(f"loss: {epoch_loss:.6f}")

    # Switch model to evaluation mode
    model.eval()
    reconstruction_errors = []

    del dataloader

    # Calculate reconstruction errors for all embeddings
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            batch = batch.to(device)
            recon = model(batch)
            errors = torch.mean((recon - batch) ** 2, dim=1)  # MSE per sample
            reconstruction_errors.extend(errors.cpu().numpy())

    reconstruction_errors = np.array(reconstruction_errors)

    # Sort indices based on reconstruction errors
    indices_sorted = np.argsort(reconstruction_errors)

    IMAGES_PER_GROUP_TO_SHOW = 200
    most_usual_indices = indices_sorted[:IMAGES_PER_GROUP_TO_SHOW]
    most_unusual_indices = indices_sorted[-IMAGES_PER_GROUP_TO_SHOW:]

    # embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # rng = np.random.default_rng(seed=42)
    # sample_indexes = rng.choice(embeddings.shape[0], 100_000, replace=False)
    # embeddings = embeddings[sample_indexes]
    # # histograms = histograms[sample_indexes]
    # gridpoints = [gridpoints[i] for i in sample_indexes]
    # paths = [paths[i] for i in sample_indexes]

    #### los = sklearn.neighbors.LocalOutlierFactor(
    ####     metric="minkowski", n_neighbors=75, contamination=0.001, n_jobs=31
    #### ).fit(embeddings)

    #### # los.negative_outlier_factor_ contains outlier scores: higher is more normal, lower is less normal
    #### outlier_scores = los.negative_outlier_factor_
    #### indices_sorted = np.argsort(outlier_scores)

    #### IMAGES_PER_GROUP_TO_SHOW = 200
    #### most_usual_indices = indices_sorted[:IMAGES_PER_GROUP_TO_SHOW]
    #### most_unusual_indices = indices_sorted[-IMAGES_PER_GROUP_TO_SHOW:]

    #### most_unusual_indices = sample_indexes[outlier_scores == -1]

    # print("done")
    # return

    # em_inv_covariance, em_diff = calculate_variables_for_mahal(embeddings)
    # # hist_inv_covariance, hist_diff = calculate_variables_for_mahal(histograms)

    # batch_size = 100_000
    # num_samples: int = embeddings.shape[0]
    # embedding_mahal_distances = np.empty(num_samples, dtype=np.float32)
    # # histogram_mahal_distances = np.empty(num_samples, dtype=np.float32)

    # for start_idx in trange(0, num_samples, batch_size, desc="Calculating distances"):
    #     end_idx = min(start_idx + batch_size, num_samples)
    #     em_batch_diff = em_diff[start_idx:end_idx]
    #     embedding_mahal_distances[start_idx:end_idx] = np.sqrt(
    #         np.sum(em_batch_diff @ em_inv_covariance * em_batch_diff, axis=1)
    #     )
    #     # hist_batch_diff = hist_diff[start_idx:end_idx]
    #     # histogram_mahal_distances[start_idx:end_idx] = np.sqrt(
    #     #     np.sum(hist_batch_diff @ hist_inv_covariance * hist_batch_diff, axis=1)
    #     # )
    # distances = embedding_mahal_distances
    # # now combine the two distances by averaging them
    # # distances = (0.9 * embedding_mahal_distances) + (0.1 * histogram_mahal_distances)

    # IMAGES_PER_GROUP_TO_SHOW = 200  # 5 * 7

    # indices_sorted = np.argsort(distances)
    # most_usual_indices = indices_sorted[:IMAGES_PER_GROUP_TO_SHOW]
    # most_unusual_indices = indices_sorted[-IMAGES_PER_GROUP_TO_SHOW:]

    # unusual_tensors = torch.from_numpy(embeddings[most_unusual_indices])
    # with torch.no_grad():
    #     model = torchvision.models.resnet50(
    #         weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2
    #     )
    #     model.eval()
    #     extractor = torch.nn.Sequential(list(model.children())[-1])

    #     embeds = unusual_tensors[:, :-96]  # remove histogram

    #     result = extractor(embeds)
    #     predictions = result.softmax(dim=1)
    #     class_ids = predictions.argmax(dim=1)

    #     predictions_per_i = [
    #         predictions[image_i, idx.item()].item()
    #         for image_i, idx in enumerate(class_ids)
    #     ]
    #     labels_per_i = [
    #         torchvision.models.ResNet50_Weights.IMAGENET1K_V2.meta["categories"][i]
    #         for i in class_ids.tolist()
    #     ]

    # classifications = list(zip(labels_per_i, predictions_per_i))

    # # free model
    # del extractor
    # del model

    # # run torch.softmax on the most unusual embeddings (embeddings[most_unusual_indices])
    # # to effectively get classifications

    # read images en bulk with get_multiple_rois_from_image()
    image_path_to_rois = dict()
    for i in most_usual_indices:
        grid_x, grid_y = gridpoints[i]
        filepath = paths[i]
        if filepath not in image_path_to_rois:
            image_path_to_rois[filepath] = set()
        image_path_to_rois[filepath].add((grid_x, grid_y))
    for i in most_unusual_indices:
        grid_x, grid_y = gridpoints[i]
        filepath = paths[i]
        if filepath not in image_path_to_rois:
            image_path_to_rois[filepath] = set()
        image_path_to_rois[filepath].add((grid_x, grid_y))

    print("read images from disk")
    image_path_to_images = dict()
    for filepath, rois in tqdm(image_path_to_rois.items(), desc="Reading images"):
        image_path_to_images[filepath] = get_multiple_rois_from_image(
            filepath, GRID_SIZE, rois
        )

    def gather_images_from_indices(indices):
        output = []
        for index_i, i in enumerate(indices):
            grid_x, grid_y = gridpoints[i]
            filepath = paths[i]
            roi = image_path_to_images[filepath][(grid_x, grid_y)]

            # label is basename without extension combined with grid coordinates
            label = os.path.splitext(os.path.basename(filepath))[0]
            label += f" ({grid_x}, {grid_y})"
            # label += (
            #     f" ({classifications[index_i][0]}, {classifications[index_i][1]:.2f})"
            # )
            output.append((label, roi))
        return output

    most_usuals = gather_images_from_indices(most_usual_indices)
    most_unusuals = gather_images_from_indices(most_unusual_indices)

    print("write results to disk")
    output_dir = RESULTS_PATH
    os.makedirs(output_dir + "/usual", exist_ok=True)
    os.makedirs(output_dir + "/unusual", exist_ok=True)
    for i, (label, img) in enumerate(most_usuals):
        cv2.imwrite(
            os.path.join(output_dir, "usual", f"{i:03}_{label}.png"),
            cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
        )
    for i, (label, img) in enumerate(most_unusuals):
        cv2.imwrite(
            os.path.join(output_dir, "unusual", f"{i:03}_{label}.png"),
            cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
        )


def plot_all(images, title, ncols=4):
    for i, (path, img) in enumerate(images):
        plt.subplot(
            len(images) // ncols + (1 if len(images) % ncols != 0 else 0), ncols, i + 1
        )
        plt.imshow(img)
        plt.axis("off")
        plt.title(os.path.basename(path), fontsize=8)

    plt.tight_layout(pad=0.2)
    plt.suptitle(title)


def real_main():
    if len(sys.argv) <= 1:
        print("No command provided. Use 'plot' or 'parquet <id>'.")
        return 1

    command = sys.argv[1]
    if command == "plot":
        main2()
        return 0

    if command == "parquet" and len(sys.argv) > 2:
        run_id = int(sys.argv[2])
        dump_to_parquet(run_id)
        return 0

    print("Unknown command. Use 'plot' or 'parquet <id>'.")
    return 1


if __name__ == "__main__":
    main2()
    # sys.exit(real_main())
